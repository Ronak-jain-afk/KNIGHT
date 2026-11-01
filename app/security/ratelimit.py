# SPDX-License-Identifier: Apache-2.0
import asyncio
import threading
import time
import weakref

OUTBOUND_CONCURRENCY = 6


class TokenBucket:
    """Token bucket rate limiter.

    Uses a ``threading.Lock`` rather than ``asyncio.Lock`` so the bucket is not
    bound to any particular event loop. The critical section contains no awaits
    and completes in microseconds, so it cannot block the loop meaningfully.
    """

    def __init__(self, rate_per_minute: float, capacity: float | None = None) -> None:
        if rate_per_minute <= 0:
            raise ValueError("rate_per_minute must be positive")
        self.rate = rate_per_minute / 60.0  # tokens per second
        self.capacity = capacity or rate_per_minute
        self.tokens = self.capacity
        self.last_update = time.time()
        self._lock = threading.Lock()

    async def acquire(self) -> bool:
        with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            self.last_update = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)

            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            return False

    async def wait(self, max_wait_seconds: float = 10.0) -> bool:
        """Wait until a token is available, or return ``False`` after ``max_wait_seconds``.

        Callers must act on the return value: a ``False`` result means the request
        should be rejected instead of silently proceeding without a token.
        """
        if await self.acquire():
            return True

        deadline = time.monotonic() + max(0.0, max_wait_seconds)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            delay = min(self.retry_after_seconds(), remaining)
            await asyncio.sleep(max(delay, 0.05))
            if await self.acquire():
                return True

    def retry_after_seconds(self) -> float:
        """Seconds until the next token is available."""
        with self._lock:
            if self.tokens >= 1.0:
                return 0.0
            return max(0.0, (1.0 - self.tokens) / self.rate)


# Per-loop semaphores: a single module-level Semaphore binds to the first loop that
# waits on it, which breaks once a second loop uses it (see app/utils/async_bridge.py).
_outbound_semaphores: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, asyncio.Semaphore
] = weakref.WeakKeyDictionary()


def get_outbound_semaphore() -> asyncio.Semaphore:
    """Return the outbound concurrency semaphore for the current event loop."""
    loop = asyncio.get_running_loop()
    semaphore = _outbound_semaphores.get(loop)
    if semaphore is None:
        semaphore = asyncio.Semaphore(OUTBOUND_CONCURRENCY)
        _outbound_semaphores[loop] = semaphore
    return semaphore


# 10 searches/min, 20 fetches/min, 6 concurrent outbound sockets
search_limiter = TokenBucket(rate_per_minute=10)
fetch_limiter = TokenBucket(rate_per_minute=20)
