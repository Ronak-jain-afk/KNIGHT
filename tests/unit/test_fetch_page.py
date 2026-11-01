# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import pytest
import respx

from app.config import get_settings
from app.db import cache
from app.db.sqlite import run_migrations
from app.tools.fetch_page import (
    _fetch_page_async,
    clamp_max_chars,
    fetch_page,
    read_capped_body,
)
from app.utils.async_bridge import run_async

PUBLIC_DNS_INFO = [(2, 1, 6, "", ("93.184.216.34", 443))]


@pytest.fixture(autouse=True)
async def setup_db(tmp_path, monkeypatch):
    monkeypatch.setenv("KNIGHT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    settings = get_settings()
    await run_migrations(settings.app_db_path)
    yield
    get_settings.cache_clear()


def test_fetch_page_ssrf_blocked():
    # Direct loopback IP
    res1 = fetch_page.invoke({"url": "http://127.0.0.1:8080"})
    assert res1.startswith("ERROR: blocked_address")

    # AWS metadata IP
    res2 = fetch_page.invoke({"url": "http://169.254.169.254/latest/meta-data/"})
    assert res2.startswith("ERROR: blocked_address")

    # Private IP
    res3 = fetch_page.invoke({"url": "http://192.168.1.1/"})
    assert res3.startswith("ERROR: blocked_address")

    # Disallowed scheme
    res4 = fetch_page.invoke({"url": "ftp://example.com/file.txt"})
    assert res4.startswith("ERROR: blocked_address")


@respx.mock
def test_fetch_page_success_and_cache():
    html_content = (
        "<!DOCTYPE html><html><head><title>Test Article</title></head>"
        "<body><article><h1>Main Header</h1>"
        "<p>This is a test article body that contains enough text content "
        "to satisfy the Trafilatura extraction requirements. "
        "It provides information about AI agents, local models, and LangGraph architecture. "
        "Here are more details about the system implementation and testing verification steps.</p>"
        "</article></body></html>"
    )

    route = respx.get("https://example.com/article").respond(
        status_code=200,
        headers={"content-type": "text/html; charset=utf-8"},
        text=html_content,
    )

    with patch("socket.getaddrinfo", return_value=PUBLIC_DNS_INFO):
        # 1. First fetch: fetches over network
        out1 = fetch_page.invoke({"url": "https://example.com/article"})
        assert "[1] Test Article" in out1
        assert "URL: https://example.com/article" in out1
        assert "<untrusted_web_content>" in out1
        assert "Main Header" in out1 or "test article" in out1
        assert route.call_count == 1

        # 2. Second fetch: hits SQLite page cache
        out2 = fetch_page.invoke({"url": "https://example.com/article"})
        assert "[1] Test Article" in out2
        assert route.call_count == 1  # Network route was not called again!


@respx.mock
def test_fetch_page_redirect_to_blocked():
    # https://example.com/redirect redirects to 127.0.0.1
    respx.get("https://example.com/redirect").respond(
        status_code=302,
        headers={"location": "http://127.0.0.1:8080/admin"},
    )

    with patch("socket.getaddrinfo", return_value=PUBLIC_DNS_INFO):
        out = fetch_page.invoke({"url": "https://example.com/redirect"})
        assert out.startswith("ERROR: blocked_address")


@respx.mock
def test_fetch_page_unsupported_content_type():
    respx.get("https://example.com/doc.pdf").respond(
        status_code=200,
        headers={"content-type": "application/pdf"},
        content=b"%PDF-1.4...",
    )

    with patch("socket.getaddrinfo", return_value=PUBLIC_DNS_INFO):
        out = fetch_page.invoke({"url": "https://example.com/doc.pdf"})
        assert "ERROR: unsupported_content_type" in out


@respx.mock
def test_fetch_page_payload_too_large():
    # Return header content-length exceeding 2MB
    respx.get("https://example.com/large.html").respond(
        status_code=200,
        headers={"content-type": "text/html", "content-length": "3000000"},
        content=b"A" * 100,
    )

    with patch("socket.getaddrinfo", return_value=PUBLIC_DNS_INFO):
        out = fetch_page.invoke({"url": "https://example.com/large.html"})
        assert "ERROR: payload_too_large" in out


def test_clamp_max_chars():
    """H-05: the model cannot request an output larger than the configured cap."""
    assert clamp_max_chars(10_000_000, 6000) == 6000
    assert clamp_max_chars(10, 6000) == 500
    assert clamp_max_chars(1000, 6000) == 1000


class _ChunkedResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.consumed = 0

    async def aiter_bytes(self):
        for chunk in self.chunks:
            self.consumed += len(chunk)
            yield chunk


@pytest.mark.asyncio
async def test_read_capped_body_stops_at_limit():
    """H-03: the body is abandoned as soon as the byte cap is exceeded."""
    response = _ChunkedResponse([b"a" * 10, b"b" * 10, b"c" * 10])

    result = await read_capped_body(response, max_bytes=15)

    assert result is None
    assert response.consumed == 20  # third chunk was never consumed


@respx.mock
def test_fetch_page_caps_output_and_cached_body(monkeypatch):
    """H-05/H-06: output and cache rows honour the configured character cap."""
    monkeypatch.setenv("KNIGHT_FETCH_MAX_CHARS", "600")
    get_settings.cache_clear()

    respx.get("https://example.com/huge").respond(
        status_code=200,
        headers={"content-type": "text/html; charset=utf-8"},
        text="<html><body><article>hello</article></body></html>",
    )

    with (
        patch("socket.getaddrinfo", return_value=PUBLIC_DNS_INFO),
        patch("app.tools.fetch_page.extract_markdown", return_value="x" * 5000),
    ):
        out = fetch_page.invoke({"url": "https://example.com/huge", "max_chars": 10_000_000})

        settings = get_settings()
        cached = run_async(cache.get_page_cache(settings.app_db_path, "https://example.com/huge"))

    body = out.split("<untrusted_web_content>\n", 1)[1].split(
        "\n</untrusted_web_content>", 1
    )[0]
    assert len(body) == 600
    assert cached is not None
    assert len(cached["markdown"]) == 600


@respx.mock
def test_extraction_runs_off_the_event_loop():
    """H-04: trafilatura extraction must not block the streaming event loop."""
    import threading

    worker_threads: list[threading.Thread] = []

    def fake_extract(html: str, final_url: str) -> str:
        worker_threads.append(threading.current_thread())
        return "extracted body text " * 20

    respx.get("https://example.com/article").respond(
        status_code=200,
        headers={"content-type": "text/html; charset=utf-8"},
        text="<html><body><article>hello</article></body></html>",
    )

    with (
        patch("socket.getaddrinfo", return_value=PUBLIC_DNS_INFO),
        patch("app.tools.fetch_page.extract_markdown", side_effect=fake_extract),
    ):
        out = fetch_page.invoke({"url": "https://example.com/article"})

    assert "extracted body text" in out
    assert worker_threads
    assert worker_threads[0] is not threading.main_thread()


@pytest.mark.asyncio
async def test_fetch_page_returns_rate_limited_error(monkeypatch):
    """H-07: an exhausted fetch bucket yields a typed error, not a silent fetch."""
    import importlib

    fetch_page_module = importlib.import_module("app.tools.fetch_page")

    class Exhausted:
        async def wait(self, max_wait_seconds: float = 0.0) -> bool:
            return False

    monkeypatch.setattr(fetch_page_module, "fetch_limiter", Exhausted())

    out = await _fetch_page_async("https://example.com/article")

    assert out.startswith("ERROR: rate_limited")

