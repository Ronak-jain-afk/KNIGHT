"""Regression tests for C-01/C-02: one shared async bridge, no duplicate helpers."""

import asyncio
import threading
from pathlib import Path

from app.security.ratelimit import TokenBucket, get_outbound_semaphore
from app.utils.async_bridge import get_bridge_loop, run_async, shutdown_async_bridge

# Created at import time on purpose: this mirrors the production module-level
# primitives that used to bind to the first event loop and then fail.
_test_bucket = TokenBucket(rate_per_minute=120)

APP_ROOT = Path(__file__).resolve().parents[2] / "app"


def test_run_async_handles_concurrent_threads_on_one_loop():
    """C-01: >= 7 concurrent sync->async bridges must all succeed on one loop."""
    results: list[int] = []
    errors: list[BaseException] = []
    loops: set[asyncio.AbstractEventLoop] = set()
    lock = threading.Lock()

    async def work(index: int) -> int:
        loops.add(asyncio.get_running_loop())
        assert await _test_bucket.acquire() is True
        async with get_outbound_semaphore():
            await asyncio.sleep(0.02)
        return index

    def worker(index: int) -> None:
        try:
            value = run_async(work(index))
        except BaseException as exc:  # noqa: BLE001 - surfaced via assertion
            with lock:
                errors.append(exc)
            return
        with lock:
            results.append(value)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert errors == []
    assert sorted(results) == list(range(8))
    assert loops == {get_bridge_loop()}


def test_run_async_reuses_the_bridge_loop_across_batches():
    """A second batch must not create a new loop (the regression in C-01)."""

    async def noop() -> int:
        return 42

    assert run_async(noop()) == 42
    first_loop = get_bridge_loop()
    assert run_async(noop()) == 42
    assert get_bridge_loop() is first_loop


def test_run_async_refuses_to_run_on_the_bridge_loop():
    async def nested() -> None:
        run_async(asyncio.sleep(0))

    try:
        run_async(nested())
    except RuntimeError as exc:
        assert "bridge loop" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("run_async() from the bridge loop must raise")


def test_bridge_shutdown_is_idempotent():
    async def noop() -> None:
        return None

    run_async(noop())
    shutdown_async_bridge()
    shutdown_async_bridge()


def test_there_is_exactly_one_run_async_helper():
    """C-02: the three duplicated `_run_async` copies are gone."""
    definitions = [
        path
        for path in APP_ROOT.rglob("*.py")
        if "def _run_async" in path.read_text(encoding="utf-8")
    ]
    assert definitions == []
