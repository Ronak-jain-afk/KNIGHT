import logging
from typing import Literal

import httpx

from app.search.base import BaseSearchProvider, SearchResult

logger = logging.getLogger(__name__)


class SearXNGProvider(BaseSearchProvider):
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    @property
    def name(self) -> str:
        return "searxng"

    def is_available(self) -> bool:
        return bool(self.base_url)

    async def search(
        self,
        query: str,
        max_results: int = 5,
        freshness: Literal["any", "day", "week", "month", "year"] = "any",
    ) -> list[SearchResult]:
        if not self.is_available():
            return []

        params = {
            "q": query,
            "format": "json",
        }
        if freshness != "any":
            params["time_range"] = freshness

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{self.base_url}/search", params=params)
            resp.raise_for_status()
            data = resp.json()

            results: list[SearchResult] = []
            for item in data.get("results", [])[:max_results]:
                url = item.get("url")
                title = item.get("title", "")
                snippet = item.get("content", "")
                if url:
                    results.append(SearchResult(title=title, url=url, snippet=snippet))
            return results
