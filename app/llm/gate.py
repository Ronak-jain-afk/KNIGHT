# SPDX-License-Identifier: Apache-2.0
"""Tracks in-flight inference so a model swap can drain instead of killing streams."""

from __future__ import annotations

import asyncio


class InferenceGate:
    """Reference count of active agent runs with an idle event."""

    def __init__(self) -> None:
        self._active = 0
        self._idle = asyncio.Event()
        self._idle.set()

    @property
    def active(self) -> int:
        return self._active

    def enter(self) -> None:
        self._active += 1
        self._idle.clear()

    def leave(self) -> None:
        self._active = max(0, self._active - 1)
        if self._active == 0:
            self._idle.set()

    async def wait_idle(self, timeout: float) -> bool:
        """Wait until no run is active. Returns False on timeout."""
        if self._active == 0:
            return True
        try:
            await asyncio.wait_for(self._idle.wait(), timeout=timeout)
            return True
        except TimeoutError:
            return False


inference_gate = InferenceGate()
