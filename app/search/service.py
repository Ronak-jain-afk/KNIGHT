# SPDX-License-Identifier: Apache-2.0
import hashlib
import logging
from typing import Literal
from urllib.parse import urlparse

import httpx

from app.agent.sources import normalize_url
from app.config import Settings
from app.db import cache
from app.search.base import BaseSearchProvider, SearchResponse, SearchResult
from app.search.brave_provider import BraveProvider
from app.search.ddgs_provider import DDGSProvider
from app.search.searxng_provider import SearXNGProvider
from app.security.ratelimit import TokenBucket, get_outbound_semaphore, search_limiter

logger = logging.getLogger(__name__)

SEARCH_RATE_LIMIT_MAX_WAIT_SECONDS = 5.0


def compute_cache_key(query: str, max_results: int, freshness: str) -> str:
    raw = f"{query.strip().lower()}|{max_results}|{freshness}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def normalize_results(raw_results: list[SearchResult], max_results: int) -> list[SearchResult]:
    seen_urls: set[str] = set()
    cleaned: list[SearchResult] = []

    for res in raw_results:
        clean_url = normalize_url(res.url)
        parsed = urlparse(clean_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            continue

        dedup_key = f"{parsed.netloc}{parsed.path}"
        if dedup_key in seen_urls:
            continue
        seen_urls.add(dedup_key)

        snippet = res.snippet.strip()
        if len(snippet) > 300:
            snippet = snippet[:297] + "..."

        cleaned.append(
            SearchResult(
                title=res.title.strip() or clean_url,
                url=clean_url,
                snippet=snippet,
            )
        )
        if len(cleaned) >= max_results:
            break

    return cleaned


class SearchService:
    def __init__(
        self,
        settings: Settings,
        limiter: TokenBucket | None = None,
    ) -> None:
        self.settings = settings
        self.limiter = limiter or search_limiter
        self.providers: dict[str, BaseSearchProvider] = {
            "ddgs": DDGSProvider(),
            "searxng": SearXNGProvider(settings.searxng_url),
            "brave": BraveProvider(settings.brave_api_key),
        }

    async def search(
        self,
        query: str,
        max_results: int = 5,
        freshness: Literal["any", "day", "week", "month", "year"] = "any",
    ) -> SearchResponse:
        cache_key = compute_cache_key(query, max_results, freshness)

        # 1. Cache lookup
        cached_payload = await cache.get_search_cache(self.settings.app_db_path, cache_key)
        if cached_payload:
            cached_results = [
                SearchResult(title=r["title"], url=r["url"], snippet=r["snippet"])
                for r in cached_payload
            ]
            return SearchResponse(ok=True, results=cached_results, provider="cache")

        # Enforce (not just log) the search rate limit (H-07).
        if not await self.limiter.wait(
            max_wait_seconds=SEARCH_RATE_LIMIT_MAX_WAIT_SECONDS
        ):
            logger.warning("Search rate limit reached (10 searches/min)")
            return SearchResponse(
                ok=False, results=[], provider="none", code="rate_limited"
            )

        # 2. Try configured providers in order
        configured_order = self.settings.parsed_search_providers

        for provider_name in configured_order:
            provider = self.providers.get(provider_name)
            if not provider or not provider.is_available():
                continue

            try:
                async with get_outbound_semaphore():
                    raw_results = await provider.search(query, max_results, freshness)

                cleaned = normalize_results(raw_results, max_results)
                if cleaned:
                    # Save to cache (24h TTL)
                    payload = [
                        {"title": r.title, "url": r.url, "snippet": r.snippet}
                        for r in cleaned
                    ]
                    await cache.set_search_cache(
                        self.settings.app_db_path,
                        cache_key,
                        payload,
                        ttl_hours=self.settings.search_cache_ttl_hours,
                    )
                    return SearchResponse(ok=True, results=cleaned, provider=provider.name)
            except httpx.HTTPError as exc:
                # Provider HTTP failures (401/429/5xx/timeouts) must fall through
                # to the next provider instead of aborting the chain (H-08).
                logger.warning("Search provider '%s' HTTP error: %s", provider_name, exc)
                continue
            except (RuntimeError, TimeoutError, OSError, ValueError) as exc:
                logger.warning("Search provider '%s' failed: %s", provider_name, exc)
                continue

        # 3. All providers failed
        return SearchResponse(ok=False, results=[], provider="none", code="search_unavailable")
