"""Regression tests for M-11 (Windows RAM telemetry) and M-12 (GPU cache race)."""

import asyncio
import ctypes
import time
from types import SimpleNamespace

import pytest

from app.llm import hardware


def test_windows_ram_reads_global_memory_status(monkeypatch):
    """M-11: Windows must report real physical memory, not `available: False`."""

    class FakeKernel32:
        @staticmethod
        def GlobalMemoryStatusEx(pointer) -> int:
            # Layout: dwLength(4), dwMemoryLoad(4), ullTotalPhys(8), ullAvailPhys(8).
            values = ctypes.cast(pointer, ctypes.POINTER(ctypes.c_uint64))
            values[1] = 16 * 1024**3
            values[2] = 8 * 1024**3
            return 1

    monkeypatch.setattr(
        ctypes, "windll", SimpleNamespace(kernel32=FakeKernel32()), raising=False
    )
    monkeypatch.setattr(hardware.platform, "system", lambda: "Windows")

    result = hardware.get_ram_telemetry()

    assert result["available"] is True
    assert result["total_mb"] == 16 * 1024
    assert result["used_mb"] == 8 * 1024
    assert result["percent"] == 50.0


def test_windows_ram_reports_unavailable_when_api_fails(monkeypatch):
    class FailingKernel32:
        @staticmethod
        def GlobalMemoryStatusEx(pointer) -> int:
            del pointer
            return 0

    monkeypatch.setattr(
        ctypes, "windll", SimpleNamespace(kernel32=FailingKernel32()), raising=False
    )
    monkeypatch.setattr(hardware.platform, "system", lambda: "Windows")

    assert hardware.get_ram_telemetry() == {"available": False}


@pytest.mark.asyncio
async def test_concurrent_gpu_detection_runs_once(monkeypatch):
    """M-12: N concurrent callers share one nvidia-smi probe."""
    monkeypatch.setattr(hardware, "_last_gpu_check", 0.0)
    monkeypatch.setattr(hardware, "_cached_gpu_info", {"available": False})

    calls = 0

    def fake_query() -> dict:
        nonlocal calls
        calls += 1
        time.sleep(0.05)
        return {"available": True, "name": "Fake GPU"}

    monkeypatch.setattr(hardware, "_query_gpu_info", fake_query)

    results = await asyncio.gather(*(hardware.get_gpu_telemetry() for _ in range(5)))

    assert calls == 1
    assert all(result == {"available": True, "name": "Fake GPU"} for result in results)
