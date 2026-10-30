import asyncio
import logging
from typing import Literal

from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

from app.search.base import BaseSearchProvider, SearchResult

logger = logging.getLogger(__name__)

BACKENDS = ["duckduckgo", "bing", "mojeek", "wikipedia"]

FRESHNESS_MAP = {
    "day": "d",
    "week": "w",
    "month": "m",
    "year": "y",
}


class DDGSProvider(BaseSearchProvider):
    @property
    def name(self) -> str:
        return "ddgs"

    def is_available(self) -> bool:
        return True

    def _sync_search_backend(
        self,
        query: str,
        backend: str,
        max_results: int,
        timelimit: str | None,
    ) -> list[SearchResult]:
        results: list[SearchResult] = []
        with DDGS() as ddgs_client:
            kwargs = {"max_results": max_results, "backend": backend}
            if timelimit:
                kwargs["timelimit"] = timelimit

            raw_items = ddgs_client.text(query, **kwargs)
            for item in raw_items:
                url = item.get("href") or item.get("url", "")
                title = item.get("title", "")
                snippet = item.get("body") or item.get("snippet", "")
                if url:
                    results.append(SearchResult(title=title, url=url, snippet=snippet))
        return results

    async def search(
        self,
        query: str,
        max_results: int = 5,
        freshness: Literal["any", "day", "week", "month", "year"] = "any",
    ) -> list[SearchResult]:
        timelimit = FRESHNESS_MAP.get(freshness)

        for backend in BACKENDS:
            try:
                # Wrap sync ddgs call with 10s timeout
                results = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._sync_search_backend,
                        query,
                        backend,
                        max_results,
                        timelimit,
                    ),
                    timeout=10.0,
                )
                if results:
                    return results
            except (RatelimitException, TimeoutException, TimeoutError) as exc:
                logger.warning("DDGS backend %s failed for query '%s': %s", backend, query, exc)
                continue
            except (DDGSException, OSError, ValueError, RuntimeError) as exc:
                logger.debug("DDGS backend %s error for query '%s': %s", backend, query, exc)
                continue

        return []
