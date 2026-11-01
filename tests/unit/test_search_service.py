# SPDX-License-Identifier: Apache-2.0
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.config import Settings
from app.db.sqlite import run_migrations
from app.search.base import SearchResult
from app.search.service import SearchService, compute_cache_key, normalize_results
from app.security.ratelimit import TokenBucket


def test_compute_cache_key():
    k1 = compute_cache_key("Python LangGraph", 5, "any")
    k2 = compute_cache_key("  python langgraph  ", 5, "any")
    k3 = compute_cache_key("python langgraph", 10, "any")
    k4 = compute_cache_key("python langgraph", 5, "week")

    assert k1 == k2
    assert k1 != k3
    assert k1 != k4


def test_normalize_results():
    long_snippet = "a" * 350
    raw = [
        SearchResult(
            title="Result 1",
            url="https://example.com/page?utm_source=twitter&id=1",
            snippet=long_snippet,
        ),
        # Duplicate host + path
        SearchResult(
            title="Result 1 duplicate",
            url="https://example.com/page?utm_medium=email",
            snippet="Duplicate path",
        ),
        # Invalid scheme
        SearchResult(
            title="Bad scheme",
            url="ftp://example.com/file.txt",
            snippet="FTP file",
        ),
        # Valid distinct result
        SearchResult(
            title="Result 2",
            url="https://example.com/other-page",
            snippet="Short snippet",
        ),
    ]

    cleaned = normalize_results(raw, max_results=5)
    assert len(cleaned) == 2
    assert cleaned[0].title == "Result 1"
    # utm_source stripped
    assert "utm_source" not in cleaned[0].url
    assert "id=1" in cleaned[0].url
    # snippet truncated to 300 chars ending with ...
    assert len(cleaned[0].snippet) == 300
    assert cleaned[0].snippet.endswith("...")

    assert cleaned[1].title == "Result 2"
    assert cleaned[1].snippet == "Short snippet"


@pytest.mark.asyncio
async def test_search_cache_hit_and_miss(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        search_providers="ddgs",
    )
    await run_migrations(settings.app_db_path)
    service = SearchService(settings)

    mock_provider = MagicMock()
    mock_provider.name = "mock"
    mock_provider.is_available.return_value = True
    mock_provider.search = AsyncMock(
        return_value=[
            SearchResult(
                title="LangGraph",
                url="https://langchain.com/langgraph",
                snippet="Agent framework",
            )
        ]
    )
    service.providers["ddgs"] = mock_provider

    # 1. Cache miss
    resp1 = await service.search("langgraph", max_results=5)
    assert resp1.ok is True
    assert resp1.provider == "mock"
    assert len(resp1.results) == 1
    assert mock_provider.search.await_count == 1

    # 2. Cache hit (mock provider should not be called again)
    resp2 = await service.search("langgraph", max_results=5)
    assert resp2.ok is True
    assert resp2.provider == "cache"
    assert len(resp2.results) == 1
    assert resp2.results[0].title == "LangGraph"
    assert mock_provider.search.await_count == 1


@pytest.mark.asyncio
async def test_provider_fallback_chain(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        search_providers="ddgs,searxng,brave",
        searxng_url="http://127.0.0.1:8888",
        brave_api_key="test-brave-key",
    )
    await run_migrations(settings.app_db_path)
    service = SearchService(settings)

    p_ddgs = MagicMock()
    p_ddgs.name = "ddgs"
    p_ddgs.is_available.return_value = True
    p_ddgs.search = AsyncMock(side_effect=RuntimeError("DDGS rate limit"))

    p_searxng = MagicMock()
    p_searxng.name = "searxng"
    p_searxng.is_available.return_value = True
    p_searxng.search = AsyncMock(side_effect=TimeoutError("SearXNG timeout"))

    p_brave = MagicMock()
    p_brave.name = "brave"
    p_brave.is_available.return_value = True
    p_brave.search = AsyncMock(
        return_value=[
            SearchResult(title="Brave Result", url="https://brave.com/search", snippet="Brave snippet")
        ]
    )

    service.providers["ddgs"] = p_ddgs
    service.providers["searxng"] = p_searxng
    service.providers["brave"] = p_brave

    resp = await service.search("test query", max_results=5)
    assert resp.ok is True
    assert resp.provider == "brave"
    assert len(resp.results) == 1
    assert resp.results[0].title == "Brave Result"

    assert p_ddgs.search.await_count == 1
    assert p_searxng.search.await_count == 1
    assert p_brave.search.await_count == 1


@pytest.mark.asyncio
async def test_all_providers_fail(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        search_providers="ddgs,searxng",
        searxng_url="http://127.0.0.1:8888",
    )
    await run_migrations(settings.app_db_path)
    service = SearchService(settings)

    p_ddgs = MagicMock()
    p_ddgs.name = "ddgs"
    p_ddgs.is_available.return_value = True
    p_ddgs.search = AsyncMock(side_effect=RuntimeError("DDGS failed"))

    p_searxng = MagicMock()
    p_searxng.name = "searxng"
    p_searxng.is_available.return_value = True
    p_searxng.search = AsyncMock(side_effect=RuntimeError("SearXNG failed"))

    service.providers["ddgs"] = p_ddgs
    service.providers["searxng"] = p_searxng

    resp = await service.search("test query", max_results=5)
    assert resp.ok is False
    assert resp.provider == "none"
    assert resp.code == "search_unavailable"
    assert len(resp.results) == 0


@pytest.mark.asyncio
async def test_token_bucket_wait_reports_exhaustion():
    """H-07: wait() must fail instead of pretending a token was granted."""
    bucket = TokenBucket(rate_per_minute=1, capacity=1)
    assert await bucket.acquire() is True

    assert await bucket.wait(max_wait_seconds=0.05) is False

    with pytest.raises(ValueError):
        # Allocation must refuse non-positive rates rather than looping forever.
        TokenBucket(rate_per_minute=0)


@pytest.mark.asyncio
async def test_search_rate_limit_returns_typed_error(tmp_path):
    """H-07: an exhausted search bucket returns code='rate_limited'."""
    settings = Settings(data_dir=tmp_path, search_providers="ddgs")
    await run_migrations(settings.app_db_path)
    limiter = TokenBucket(rate_per_minute=1, capacity=1)
    assert await limiter.acquire() is True
    service = SearchService(settings, limiter=limiter)

    p_ddgs = MagicMock()
    p_ddgs.name = "ddgs"
    p_ddgs.is_available.return_value = True
    p_ddgs.search = AsyncMock(return_value=[])
    service.providers["ddgs"] = p_ddgs

    resp = await service.search("rate limited query")

    assert resp.ok is False
    assert resp.code == "rate_limited"
    assert p_ddgs.search.await_count == 0


@pytest.mark.asyncio
async def test_provider_http_error_falls_through(tmp_path):
    """H-08: a 429 from provider #1 must not abort the fallback chain."""
    settings = Settings(
        data_dir=tmp_path,
        search_providers="ddgs,searxng",
        searxng_url="http://127.0.0.1:8888",
    )
    await run_migrations(settings.app_db_path)
    service = SearchService(settings)

    request = httpx.Request("GET", "https://api.example.com/search")
    response = httpx.Response(429, request=request)
    status_error = httpx.HTTPStatusError("rate limited", request=request, response=response)

    p_ddgs = MagicMock()
    p_ddgs.name = "ddgs"
    p_ddgs.is_available.return_value = True
    p_ddgs.search = AsyncMock(side_effect=status_error)

    p_searxng = MagicMock()
    p_searxng.name = "searxng"
    p_searxng.is_available.return_value = True
    p_searxng.search = AsyncMock(
        return_value=[
            SearchResult(title="Fallback", url="https://example.org/page", snippet="ok")
        ]
    )

    service.providers["ddgs"] = p_ddgs
    service.providers["searxng"] = p_searxng

    resp = await service.search("fallback query")

    assert resp.ok is True
    assert resp.provider == "searxng"
    assert resp.results[0].title == "Fallback"

