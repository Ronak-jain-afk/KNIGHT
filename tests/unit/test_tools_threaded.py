# SPDX-License-Identifier: Apache-2.0
import concurrent.futures
from unittest.mock import AsyncMock, patch

from app.agent.sources import SourceCollector, current_sources
from app.search.base import SearchResponse, SearchResult
from app.tools.fetch_page import fetch_page
from app.tools.web_search import web_search


def test_concurrent_threaded_tool_invocations():
    """T-07: Exercise sync tools from concurrent worker threads (>= 7 calls).

    Sync tools (web_search, fetch_page, memory) bridge into asyncio via run_async().
    This test verifies that calling .invoke() from 8 concurrent worker threads
    safely shares the async bridge without deadlocking or failing.
    """
    mock_search_resp = SearchResponse(
        ok=True,
        provider="mock",
        results=[
            SearchResult(
                url="https://example.com/info",
                title="Example Info",
                snippet="A useful snippet of information.",
            )
        ],
    )

    collector = SourceCollector()
    token = current_sources.set(collector)

    num_threads = 8

    def run_search_worker(idx: int) -> str:
        # Called on a ThreadPoolExecutor worker thread
        return web_search.invoke({"query": f"query {idx}", "max_results": 3})

    try:
        with (
            patch("app.search.service.SearchService.search", new=AsyncMock(return_value=mock_search_resp)),
            concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor,
        ):
            futures = [executor.submit(run_search_worker, i) for i in range(num_threads)]
            results = [f.result(timeout=10.0) for f in futures]

        assert len(results) == num_threads
        for res in results:
            assert "Results (cite these numbers):" in res
            assert "https://example.com/info" in res
    finally:
        current_sources.reset(token)


def test_concurrent_threaded_fetch_page_invocations():
    """T-07: Exercise fetch_page tool from 8 concurrent worker threads."""
    num_threads = 8

    def run_fetch_worker(idx: int) -> str:
        return fetch_page.invoke({"url": f"https://example.com/article-{idx}", "max_chars": 500})

    with (
        patch(
            "app.tools.fetch_page._fetch_page_async",
            new=AsyncMock(return_value="<untrusted_web_content>\nTest page content\n</untrusted_web_content>"),
        ),
        concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor,
    ):
        futures = [executor.submit(run_fetch_worker, i) for i in range(num_threads)]
        results = [f.result(timeout=10.0) for f in futures]

    assert len(results) == num_threads
    for res in results:
        assert "<untrusted_web_content>" in res
