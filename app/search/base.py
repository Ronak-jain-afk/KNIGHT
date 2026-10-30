from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


@dataclass
class SearchResponse:
    ok: bool
    results: list[SearchResult]
    provider: str
    code: str | None = None


class BaseSearchProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""

    @abstractmethod
    def is_available(self) -> bool:
        """Whether provider credentials / config are configured."""

    @abstractmethod
    async def search(
        self,
        query: str,
        max_results: int = 5,
        freshness: Literal["any", "day", "week", "month", "year"] = "any",
    ) -> list[SearchResult]:
        """Perform search query and return raw list of results."""
