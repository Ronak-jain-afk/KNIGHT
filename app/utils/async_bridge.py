"""Single shared event loop for synchronous LangChain tools.

Sync tools (``@tool`` functions) are invoked by LangChain from worker threads via
``run_in_executor``. Historically each call created a brand-new event loop with
``asyncio.run()``, which broke any module-level ``asyncio`` primitive (semaphores,
locks) created at import time: those primitives bind to the first loop that has to
wait, and a later loop raises ``RuntimeError: ... is bound to a different event
loop``.

This module owns one long-lived loop thread. Every sync->async bridge goes through
``run_async`` so all async primitives see the same loop for the process lifetime.
"""

from __future__ import annotations

import asyncio
import atexit
import threading
from collections.abc import Coroutine
from concurrent.futures import Future
from typing import Any

_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_loop_lock = threading.Lock()


def _run_loop_forever(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


def get_bridge_loop() -> asyncio.AbstractEventLoop:
    """Return the process-wide bridge loop, starting it on first use."""
    global _loop, _thread

    loop = _loop
    if loop is not None and loop.is_running():
        return loop

    with _loop_lock:
        loop = _loop
        if loop is not None and loop.is_running():
            return loop

        new_loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=_run_loop_forever,
            args=(new_loop,),
            name="knight-async-bridge",
            daemon=True,
        )
        thread.start()

        _loop = new_loop
        _thread = thread
        return new_loop


def run_async[T](coro: Coroutine[Any, Any, T], timeout: float | None = None) -> T:
    """Run ``coro`` on the shared bridge loop and block until it finishes."""
    loop = get_bridge_loop()

    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None

    if running is loop:
        coro.close()
        raise RuntimeError(
            "run_async() called from the bridge loop; awaiting it would deadlock."
        )
    if running is not None:
        # Never call .result() from an event-loop thread: it would deadlock that
        # loop. Execute on an isolated thread instead.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="knight-sync-bridge"
        ) as pool:
            return pool.submit(run_async, coro, timeout).result()

    future: Future[T] = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return future.result(timeout)
    except BaseException:
        future.cancel()
        raise


def shutdown_async_bridge(timeout: float = 5.0) -> None:
    """Stop the bridge loop. Safe to call multiple times (used by tests/exit)."""
    global _loop, _thread

    with _loop_lock:
        loop, thread = _loop, _thread
        _loop, _thread = None, None

    if loop is None:
        return

    if loop.is_running():
        loop.call_soon_threadsafe(loop.stop)

    if thread is not None and thread.is_alive() and thread is not threading.current_thread():
        thread.join(timeout)

    if not loop.is_closed():
        try:
            loop.close()
        except RuntimeError:  # pragma: no cover - already closed by the loop thread
            pass


atexit.register(shutdown_async_bridge)
