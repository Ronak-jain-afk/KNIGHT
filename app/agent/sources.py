# SPDX-License-Identifier: Apache-2.0
import contextvars
from datetime import UTC, datetime
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

current_sources: contextvars.ContextVar["SourceCollector | None"] = contextvars.ContextVar(
    "current_sources", default=None
)


def normalize_url(url: str) -> str:
    """Normalize URL by stripping fragments, utm/tracking parameters, and trailing slash."""
    parsed = urlparse(url.strip())
    # Strip tracking queries
    tracking_prefixes = ("utm_", "fbclid", "gclid", "ref", "source")
    query_params = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=True)
        if not any(k.lower().startswith(prefix) for prefix in tracking_prefixes)
    ]
    clean_query = urlencode(query_params)
    clean_path = parsed.path.rstrip("/") if parsed.path != "/" else "/"

    normalized = urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            clean_path,
            parsed.params,
            clean_query,
            "",  # strip fragment
        )
    )
    return normalized


class SourceCollector:
    """Per-turn collector of citation URLs with stable 1-based indices."""

    def __init__(self) -> None:
        self._url_to_idx: dict[str, int] = {}
        self._items: list[dict] = []

    def add(
        self,
        url: str,
        title: str | None = None,
        snippet: str | None = None,
        **kwargs: object,
    ) -> int:
        norm_url = normalize_url(url)
        if norm_url not in self._url_to_idx:
            idx = len(self._url_to_idx) + 1
            self._url_to_idx[norm_url] = idx
            item: dict[str, object] = {
                "idx": idx,
                "url": norm_url,
                "title": title or norm_url,
                "fetched_at": datetime.now(UTC).isoformat(),
            }
            if snippet:
                item["snippet"] = snippet
            self._items.append(item)
        return self._url_to_idx[norm_url]

    def items(self) -> list[dict]:
        return list(self._items)
