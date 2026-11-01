# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Literal

import httpx

from app.search.base import BaseSearchProvider, SearchResult

logger = logging.getLogger(__name__)

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

BRAVE_FRESHNESS = {
    "day": "pd",
    "week": "pw",
    "month": "pm",
    "year": "py",
}


class BraveProvider(BaseSearchProvider):
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key.strip()

    @property
    def name(self) -> str:
        return "brave"

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def search(
        self,
        query: str,
        max_results: int = 5,
        freshness: Literal["any", "day", "week", "month", "year"] = "any",
    ) -> list[SearchResult]:
        if not self.is_available():
            return []

        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key,
        }
        params = {
            "q": query,
            "count": str(max_results),
        }
        if freshness in BRAVE_FRESHNESS:
            params["freshness"] = BRAVE_FRESHNESS[freshness]

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(BRAVE_ENDPOINT, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()

            results: list[SearchResult] = []
            web_results = data.get("web", {}).get("results", [])
            for item in web_results[:max_results]:
                url = item.get("url")
                title = item.get("title", "")
                snippet = item.get("description", "")
                if url:
                    results.append(SearchResult(title=title, url=url, snippet=snippet))
            return results
