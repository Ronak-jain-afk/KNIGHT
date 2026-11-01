# SPDX-License-Identifier: Apache-2.0
from typing import Literal

from langchain_core.tools import tool

from app.agent.sources import current_sources
from app.config import get_settings
from app.search.service import SearchService
from app.utils.async_bridge import run_async


async def _web_search_async(
    query: str,
    max_results: int = 5,
    freshness: Literal["any", "day", "week", "month", "year"] = "any",
) -> str:
    settings = get_settings()
    service = SearchService(settings)
    resp = await service.search(query, max_results=max_results, freshness=freshness)

    if not resp.ok:
        return f"ERROR: search_failed: {resp.code or 'unavailable'}"

    if not resp.results:
        return f"No results found for query '{query}'."

    collector = current_sources.get()
    lines = ["Results (cite these numbers):"]

    for r in resp.results:
        idx = collector.add(r.url, r.title) if collector else len(lines)
        lines.append(f"[{idx}] {r.title} — {r.url}\n    {r.snippet}")

    return "\n".join(lines)


@tool
def web_search(
    query: str,
    max_results: int = 5,
    freshness: Literal["any", "day", "week", "month", "year"] = "any",
) -> str:
    """Search the public World Wide Web for current online information.
    Do NOT use this tool for local files, codebases, directories, or filesystem paths on the user's computer.

    Args:
        query: The online search query, 2-12 words.
        max_results: How many results to return (1-10).
        freshness: Restrict to recent results (any, day, week, month, year).
    Returns:
        "Results (cite these numbers):\\n[1] Title — https://url\\n    snippet\\n..."
        or "ERROR: <reason>".
    """
    try:
        return run_async(_web_search_async(query, max_results, freshness))
    except (RuntimeError, TimeoutError, OSError, ValueError) as exc:
        return f"ERROR: search_failed: {exc}"
