# SPDX-License-Identifier: Apache-2.0
import asyncio
import logging
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)

_last_gpu_check: float = 0
_cached_gpu_info: dict[str, Any] = {"available": False}
_gpu_lock: asyncio.Lock | None = None
_gpu_lock_loop: asyncio.AbstractEventLoop | None = None


def _get_gpu_lock() -> asyncio.Lock:
    """Return a lock bound to the current loop (never create one at import)."""
    global _gpu_lock, _gpu_lock_loop
    loop = asyncio.get_running_loop()
    if _gpu_lock is None or _gpu_lock_loop is not loop:
        _gpu_lock = asyncio.Lock()
        _gpu_lock_loop = loop
    return _gpu_lock


def _parse_int(val: str, default: int = 0) -> int:
    try:
        return round(float(val.strip()))
    except (ValueError, TypeError):
        return default


def _query_gpu_info() -> dict[str, Any]:
    """Blocking nvidia-smi probe, isolated so it can be run in a thread."""
    try:
        smi_bin = shutil.which("nvidia-smi") or "nvidia-smi"
        if not shutil.which("nvidia-smi") and platform.system().lower() == "windows":
            for candidate in (
                r"C:\Windows\System32\nvidia-smi.exe",
                r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
            ):
                if Path(candidate).exists():
                    smi_bin = candidate
                    break

        cmd = [
            smi_bin,
            "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=4.0, check=False)
        if res.returncode == 0 and res.stdout.strip():
            line = res.stdout.strip().split("\n")[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                used = _parse_int(parts[1])
                total = _parse_int(parts[2])
                pct = round((used / total) * 100, 1) if total > 0 else 0.0
                util = _parse_int(parts[3])
                temp = _parse_int(parts[4])
                return {
                    "available": True,
                    "name": parts[0],
                    "vram_used_mb": used,
                    "vram_total_mb": total,
                    "vram_percent": pct,
                    "utilization_percent": util,
                    "temperature_c": temp,
                }
    except (subprocess.SubprocessError, OSError, ValueError) as exc:
        logger.debug("GPU query failed: %s", exc)
    return {"available": False}


async def get_gpu_telemetry() -> dict[str, Any]:
    """Return cached GPU telemetry; concurrent first calls detect only once."""
    global _last_gpu_check, _cached_gpu_info
    async with _get_gpu_lock():
        now = time.time()
        # Cache for 4 seconds
        if now - _last_gpu_check < 4.0:
            return _cached_gpu_info

        _cached_gpu_info = await asyncio.to_thread(_query_gpu_info)
        _last_gpu_check = time.time()
        return _cached_gpu_info


def _windows_ram_telemetry() -> dict[str, Any]:
    """Read real total/available RAM on Windows via GlobalMemoryStatusEx."""
    import ctypes

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_uint32),
            ("dwMemoryLoad", ctypes.c_uint32),
            ("ullTotalPhys", ctypes.c_uint64),
            ("ullAvailPhys", ctypes.c_uint64),
            ("ullTotalPageFile", ctypes.c_uint64),
            ("ullAvailPageFile", ctypes.c_uint64),
            ("ullTotalVirtual", ctypes.c_uint64),
            ("ullAvailVirtual", ctypes.c_uint64),
            ("ullAvailExtendedVirtual", ctypes.c_uint64),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")

    total_mb = int(status.ullTotalPhys // (1024 * 1024))
    avail_mb = int(status.ullAvailPhys // (1024 * 1024))
    used_mb = total_mb - avail_mb
    pct = round((used_mb / total_mb) * 100, 1) if total_mb > 0 else 0.0
    return {
        "available": True,
        "total_mb": total_mb,
        "used_mb": used_mb,
        "percent": pct,
    }


def _proc_meminfo_telemetry() -> dict[str, Any]:
    with open("/proc/meminfo") as f:
        lines = f.readlines()
    data: dict[str, str] = {}
    for line in lines:
        parts = line.split(":")
        if len(parts) == 2:
            data[parts[0].strip()] = parts[1].strip()
    total_kb = int(data.get("MemTotal", "0").split()[0])
    avail_kb = int(data.get("MemAvailable", "0").split()[0])
    used_kb = total_kb - avail_kb
    total_mb = round(total_kb / 1024)
    used_mb = round(used_kb / 1024)
    pct = round((used_kb / total_kb) * 100, 1) if total_kb > 0 else 0.0
    return {
        "available": True,
        "total_mb": total_mb,
        "used_mb": used_mb,
        "percent": pct,
    }


def get_ram_telemetry() -> dict[str, Any]:
    try:
        if platform.system().lower() == "windows":
            return _windows_ram_telemetry()
        return _proc_meminfo_telemetry()
    except (OSError, ValueError) as exc:
        logger.debug("RAM query failed: %s", exc)
    return {"available": False}


async def get_hardware_status(settings: Settings) -> dict[str, Any]:
    gpu = await get_gpu_telemetry()
    ram = get_ram_telemetry()

    # Query llama-server ping and active slots
    server_root = settings.llama_base_url.rstrip("/").removesuffix("/v1")
    headers = {}
    api_key = settings.effective_llama_api_key
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    server_status = {"reachable": False, "latency_ms": None, "slots_busy": 0}
    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            resp = await client.get(f"{server_root}/health", headers=headers)
            lat = round((time.time() - t0) * 1000, 1)
            if resp.status_code in (200, 503):
                health = resp.json() if resp.status_code == 200 else {}
                server_status = {
                    "reachable": True,
                    "latency_ms": lat,
                    "slots_busy": health.get("slots_processing", 0),
                }
    except (httpx.HTTPError, OSError):
        pass

    from app.llm.manager import ModelManager

    return {
        "gpu": gpu,
        "ram": ram,
        "context": {
            "max_tokens": settings.context_tokens,
            "model_name": ModelManager.get_active_model(settings),
        },
        "server": server_status,
    }
