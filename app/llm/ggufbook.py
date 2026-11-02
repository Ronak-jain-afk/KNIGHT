# SPDX-License-Identifier: Apache-2.0
import asyncio
import logging
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, ClassVar, Literal
from urllib.parse import urljoin, urlparse

import httpx

from app.config import Settings
from app.llm.hardware import get_gpu_telemetry, get_ram_telemetry
from app.security.net_guard import (
    BlockedAddressError,
    InvalidURLError,
    NetGuard,
    PinnedSyncNetworkBackend,
)
from app.tools.local_files import format_size

logger = logging.getLogger(__name__)

HF_API_BASE = "https://huggingface.co/api"
HF_HUB_BASE = "https://huggingface.co"

MAX_REDIRECTS = 3
GGUF_MAGIC = b"GGUF"
DISK_SAFETY_MARGIN_BYTES = 256 * 1024 * 1024
UNKNOWN_SIZE_RESERVE_BYTES = 1024 * 1024 * 1024

CURATED_TRENDING_MODELS = [
    {
        "id": "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
        "name": "Qwen2.5 Coder 7B Instruct",
        "author": "bartowski",
        "description": "State-of-the-art open-source code generation model from Alibaba.",
        "is_curated": True,
        "tags": ["gguf", "coding", "qwen", "7b"],
    },
    {
        "id": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "name": "Llama 3.2 3B Instruct",
        "author": "bartowski",
        "description": "Meta's highly optimized lightweight 3B model, ideal for fast local inference.",
        "is_curated": True,
        "tags": ["gguf", "llama", "3b", "fast"],
    },
    {
        "id": "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF",
        "name": "DeepSeek R1 Distill Qwen 7B",
        "author": "bartowski",
        "description": "DeepSeek R1 reasoning capabilities distilled into a fast 7B architecture.",
        "is_curated": True,
        "tags": ["gguf", "reasoning", "deepseek", "7b"],
    },
    {
        "id": "bartowski/gemma-2-2b-it-GGUF",
        "name": "Gemma 2 2B Instruct",
        "author": "bartowski",
        "description": "Google's high-efficiency 2B model that runs entirely in small VRAM.",
        "is_curated": True,
        "tags": ["gguf", "google", "gemma", "2b"],
    },
    {
        "id": "bartowski/Phi-3.5-mini-instruct-GGUF",
        "name": "Phi 3.5 Mini Instruct (3.8B)",
        "author": "bartowski",
        "description": "Microsoft's compact model with strong multi-turn reasoning and math capabilities.",
        "is_curated": True,
        "tags": ["gguf", "microsoft", "phi", "3.8b"],
    },
]


def extract_quant_type(filename: str) -> str:
    """Extract quantization identifier (e.g. Q4_K_M, Q8_0, IQ3_M) from GGUF filename."""
    match = re.search(
        r"[-._]([qQiIfFbB][0-9a-zA-Z_]+)\.gguf$", filename, re.IGNORECASE
    )
    if match:
        return match.group(1).upper()
    return "GGUF"


def evaluate_hardware_fit(
    size_bytes: int | None,
    gpu: dict[str, Any],
    ram: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate whether a model fits in GPU VRAM or host RAM, and provide warnings."""
    if size_bytes is None or size_bytes <= 0:
        return {
            "status": "unknown",
            "badge": "Unknown Size",
            "label": "Unknown Size",
            "color": "secondary",
            "can_fit_gpu": None,
            "warning": "Model file size is not reported by repository metadata.",
            "detail": "Unknown size: ensure your hardware has sufficient RAM/VRAM before downloading.",
            "explanation": "Unknown size: ensure your hardware has sufficient RAM/VRAM before downloading.",
        }

    size_mb = size_bytes / (1024 * 1024)
    gpu_total = (
        gpu.get("vram_total_mb", 0)
        if (gpu and gpu.get("available") and isinstance(gpu.get("vram_total_mb"), int))
        else 0
    )
    ram_total = (
        ram.get("total_mb", 0)
        if (ram and ram.get("available") and isinstance(ram.get("total_mb"), int))
        else 0
    )

    ctx_overhead = 1200  # ~1.2 GB context KV cache allowance at 16k tokens

    if gpu_total > 0 and (size_mb + ctx_overhead <= gpu_total):
        return {
            "status": "optimal",
            "badge": "Full GPU",
            "label": "Full GPU",
            "color": "success",
            "can_fit_gpu": True,
            "warning": None,
            "detail": f"Fits fully in {gpu_total} MB VRAM with room for 16k context. Fast GPU inference.",
            "explanation": f"Fits fully in {gpu_total} MB VRAM with room for 16k context. Fast GPU inference.",
        }
    elif gpu_total > 0 and (size_mb <= gpu_total):
        return {
            "status": "tight",
            "badge": "Tight VRAM",
            "label": "Tight VRAM",
            "color": "warning",
            "can_fit_gpu": True,
            "warning": "Fits in VRAM, but large context windows may spill into system RAM.",
            "detail": f"Fits in {gpu_total} MB VRAM with minimal context overhead.",
            "explanation": f"Fits in {gpu_total} MB VRAM with minimal context overhead.",
        }
    elif ram_total > 0 and (size_mb + 2000 <= ram_total):
        return {
            "status": "partial_cpu",
            "badge": "Partial / CPU",
            "label": "Partial / CPU",
            "color": "warning",
            "can_fit_gpu": False,
            "warning": f"Model ({round(size_mb / 1024, 1)} GB) exceeds GPU VRAM ({gpu_total} MB). It will offload partially and use Host RAM ({round(ram_total / 1024, 1)} GB). Inference will be noticeably slower.",
            "detail": f"Exceeds GPU VRAM. Will balance between GPU and {round(ram_total / 1024, 1)} GB system RAM.",
            "explanation": f"Exceeds GPU VRAM. Will balance between GPU and {round(ram_total / 1024, 1)} GB system RAM.",
        }
    else:
        return {
            "status": "oom_danger",
            "badge": "Insufficient RAM",
            "label": "Insufficient RAM",
            "color": "danger",
            "can_fit_gpu": False,
            "warning": f"Model ({round(size_mb / 1024, 1)} GB) exceeds available Host RAM ({round(ram_total / 1024, 1)} GB). Loading this model will likely cause an Out-Of-Memory crash.",
            "detail": f"Dangerous: Exceeds total host RAM of {round(ram_total / 1024, 1)} GB.",
            "explanation": f"Dangerous: Exceeds total host RAM of {round(ram_total / 1024, 1)} GB.",
        }


async def search_hf_gguf_models(
    query: str = "", limit: int = 15
) -> list[dict[str, Any]]:
    """Search Hugging Face models API for GGUF repos, or return curated models."""
    q = query.strip()
    if not q:
        return CURATED_TRENDING_MODELS

    url = f"{HF_API_BASE}/models"
    params = {
        "search": q,
        "filter": "gguf",
        "sort": "downloads",
        "direction": "-1",
        "limit": str(limit),
    }

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            res = await client.get(url, params=params)
            if res.status_code == 200:
                data = res.json()
                results = []
                for item in data:
                    repo_id = item.get("id") or item.get("modelId", "")
                    if not repo_id:
                        continue
                    author = repo_id.split("/")[0] if "/" in repo_id else "unknown"
                    results.append(
                        {
                            "id": repo_id,
                            "name": repo_id.split("/")[-1],
                            "author": author,
                            "downloads": item.get("downloads", 0),
                            "likes": item.get("likes", 0),
                            "last_modified": item.get("lastModified", ""),
                            "tags": item.get("tags", []),
                        }
                    )
                return results
    except (httpx.HTTPError, OSError) as exc:
        logger.warning("HF search failed: %s", exc)

    # Fallback to local filtering of curated models if offline
    return [
        m
        for m in CURATED_TRENDING_MODELS
        if q.lower() in m["name"].lower() or q.lower() in m["id"].lower()
    ]


async def get_hf_model_variants(
    repo_id: str, settings: Settings
) -> dict[str, Any] | None:
    """Fetch repository files, filter .gguf files, and calculate hardware fit."""
    clean_repo = repo_id.strip()
    models_dir = settings.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    gpu = await get_gpu_telemetry()
    ram = get_ram_telemetry()

    url = f"{HF_API_BASE}/models/{clean_repo}"
    repo_meta: dict[str, Any] = {}
    siblings: list[dict[str, Any]] = []

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(url)
            if res.status_code == 200:
                repo_meta = res.json()
                siblings = repo_meta.get("siblings", [])
            elif res.status_code == 404:
                return None
    except (httpx.HTTPError, OSError) as exc:
        logger.warning("HF repo lookup failed for %s: %s", clean_repo, exc)
        return None

    variants = []
    for f in siblings:
        rpath = f.get("rfilename", "")
        if not rpath.lower().endswith(".gguf"):
            continue

        filename = Path(rpath).name
        # Size might be in lfs metadata or direct size
        lfs = f.get("lfs")
        raw_size = lfs.get("size") if isinstance(lfs, dict) else f.get("size")
        size_bytes: int | None = (
            int(raw_size)
            if (raw_size and isinstance(raw_size, (int, float)) and raw_size > 0)
            else None
        )

        is_downloaded = (models_dir / filename).exists()
        quant = extract_quant_type(filename)
        fit = evaluate_hardware_fit(size_bytes, gpu, ram)
        download_url = f"{HF_HUB_BASE}/{clean_repo}/resolve/main/{rpath}"

        variants.append(
            {
                "filename": filename,
                "quant_type": quant,
                "size_bytes": size_bytes,
                "size_formatted": (
                    format_size(size_bytes) if size_bytes is not None else "Unknown"
                ),
                "is_downloaded": is_downloaded,
                "already_downloaded": is_downloaded,
                "download_url": download_url,
                "hardware_fit": fit,
            }
        )

    # Sort variants by size ascending (None sizes sorted last)
    variants.sort(key=lambda v: (v["size_bytes"] is None, v["size_bytes"] or 0))

    return {
        "repo_id": clean_repo,
        "author": clean_repo.split("/")[0] if "/" in clean_repo else "unknown",
        "name": clean_repo.split("/")[-1],
        "downloads": repo_meta.get("downloads", 0),
        "likes": repo_meta.get("likes", 0),
        "variants": variants,
    }


class DownloadURLError(ValueError):
    """Raised when a download URL fails validation (SSRF / allowlist)."""


class DownloadLimitError(ValueError):
    """Raised when a download would exceed the configured size/disk limits."""


class _Cancelled(Exception):
    """Internal signal: the task was cancelled while streaming."""


def _host_allowed(host: str, allowlist: list[str]) -> bool:
    for entry in allowlist:
        if entry.startswith("."):
            if host.endswith(entry):
                return True
        elif host == entry or host.endswith("." + entry):
            return True
    return False


def validate_download_url(url: str, settings: Settings, *, resolve: bool = True) -> str:
    """Validate a GGUF download URL.

    Enforces https, an explicit host allowlist, no embedded credentials, an allowed
    port, and (when ``resolve`` is true) that the host does not resolve to a
    loopback/private/link-local/CGNAT address.
    """
    raw = (url or "").strip()
    if not raw:
        raise DownloadURLError("download_url cannot be empty.")

    parsed = urlparse(raw)
    if parsed.scheme.lower() != "https":
        raise DownloadURLError("download_url must use https.")

    if parsed.username or parsed.password:
        raise DownloadURLError("download_url must not contain credentials.")

    try:
        host = (parsed.hostname or "").lower()
        port = parsed.port
    except ValueError as exc:
        raise DownloadURLError("download_url is malformed.") from exc

    if not host:
        raise DownloadURLError("download_url must contain a hostname.")

    allowlist = settings.parsed_download_hosts
    if not _host_allowed(host, allowlist):
        raise DownloadURLError(
            f"Host '{host}' is not an allowed model host ({', '.join(allowlist)})."
        )

    if port not in (None, 443):
        raise DownloadURLError("download_url must use the default HTTPS port.")

    if resolve:
        try:
            NetGuard.validate_url(raw)
        except InvalidURLError as exc:
            raise DownloadURLError(str(exc)) from exc
        except BlockedAddressError as exc:
            raise DownloadURLError(str(exc)) from exc

    return raw


def _open_validated_stream(
    client: httpx.Client, url: str, settings: Settings
) -> httpx.Response:
    """Open a stream, validating every redirect hop and pinning DNS to prevent TOCTOU SSRF."""
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        clean_url = validate_download_url(current, settings)
        hostname, _port, pinned_ip = NetGuard.resolve_url(clean_url)
        pool = getattr(client._transport, "_pool", None)
        if pool is not None:
            pool._network_backend = PinnedSyncNetworkBackend({hostname: pinned_ip})

        request = client.build_request("GET", clean_url)
        response = client.send(request, stream=True)

        if response.is_redirect:
            location = response.headers.get("location")
            response.close()
            if not location:
                raise DownloadURLError("Redirect response is missing a Location header.")
            current = urljoin(clean_url, location)
            continue
        return response

    raise DownloadURLError(f"Too many redirects (max {MAX_REDIRECTS}).")


def _ensure_disk_space(models_dir: Path, expected_bytes: int) -> None:
    free = shutil.disk_usage(models_dir).free
    required = (expected_bytes or UNKNOWN_SIZE_RESERVE_BYTES) + DISK_SAFETY_MARGIN_BYTES
    if required > free:
        raise DownloadLimitError(
            f"Not enough free disk space: need {format_size(required)}, have {format_size(free)}."
        )


def _verify_gguf(path: Path) -> None:
    try:
        with open(path, "rb") as fh:
            magic = fh.read(len(GGUF_MAGIC))
    except OSError as exc:
        raise DownloadURLError(f"Could not read downloaded file: {exc}") from exc

    if magic != GGUF_MAGIC:
        raise DownloadURLError("Downloaded file is not a valid GGUF model (bad magic bytes).")


class DownloadTask:
    def __init__(
        self,
        task_id: str,
        repo_id: str,
        filename: str,
        url: str,
        size_bytes: int,
    ) -> None:
        self.id = task_id
        self.repo_id = repo_id
        self.filename = filename
        self.url = url
        self.size_bytes = size_bytes
        self.downloaded_bytes = 0
        self.speed_mbps = 0.0
        self.eta_seconds: int | None = None
        self.status: Literal[
            "queued", "downloading", "completed", "failed", "cancelled"
        ] = "queued"
        self.error: str | None = None
        self.created_at = time.time()
        self._async_task: asyncio.Task[None] | None = None

    def to_dict(self) -> dict[str, Any]:
        pct = (
            round((self.downloaded_bytes / self.size_bytes) * 100, 1)
            if self.size_bytes > 0
            else 0.0
        )
        return {
            "id": self.id,
            "task_id": self.id,
            "repo_id": self.repo_id,
            "filename": self.filename,
            "size_bytes": self.size_bytes,
            "total_bytes": self.size_bytes,
            "size_formatted": format_size(self.size_bytes),
            "downloaded_bytes": self.downloaded_bytes,
            "downloaded_formatted": format_size(self.downloaded_bytes),
            "percentage": pct,
            "progress_pct": pct,
            "speed_mbps": round(self.speed_mbps, 2),
            "speed_mb_s": round(self.speed_mbps, 2),
            "eta_seconds": self.eta_seconds,
            "status": self.status,
            "error": self.error,
        }


def _sync_download_worker(task: DownloadTask, settings: Settings) -> None:
    models_dir = settings.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    part_path = models_dir / f"{task.filename}.part"
    target_path = models_dir / task.filename

    task.status = "downloading"
    last_calc_time = time.time()
    last_calc_bytes = 0
    total_bytes = 0
    completed = False
    max_bytes = settings.model_download_max_bytes

    try:
        with httpx.Client(
            timeout=httpx.Timeout(connect=15.0, read=60.0, write=60.0, pool=60.0),
            follow_redirects=False,
            trust_env=False,
        ) as client:
            response = _open_validated_stream(client, task.url, settings)
            try:
                if response.status_code != 200:
                    raise DownloadURLError(
                        f"HTTP {response.status_code} while fetching model file."
                    )

                header_len = response.headers.get("content-length")
                if header_len and header_len.isdigit():
                    task.size_bytes = int(header_len)

                if task.size_bytes > max_bytes:
                    raise DownloadLimitError(
                        f"Model file is {format_size(task.size_bytes)}, "
                        f"above the {format_size(max_bytes)} download limit."
                    )

                _ensure_disk_space(models_dir, task.size_bytes)

                with open(part_path, "wb") as f:
                    for chunk in response.iter_bytes(chunk_size=512 * 1024):
                        if task.status == "cancelled":
                            raise _Cancelled()

                        total_bytes += len(chunk)
                        if total_bytes > max_bytes:
                            raise DownloadLimitError(
                                f"Download exceeded the {format_size(max_bytes)} limit."
                            )

                        f.write(chunk)
                        task.downloaded_bytes = total_bytes

                        now = time.time()
                        elapsed_since_calc = now - last_calc_time
                        if elapsed_since_calc >= 1.0:
                            delta_bytes = task.downloaded_bytes - last_calc_bytes
                            mbps = (delta_bytes / (1024 * 1024)) / elapsed_since_calc
                            task.speed_mbps = mbps

                            remaining_bytes = max(0, task.size_bytes - task.downloaded_bytes)
                            bytes_per_sec = delta_bytes / elapsed_since_calc
                            if bytes_per_sec > 0:
                                task.eta_seconds = int(remaining_bytes / bytes_per_sec)

                            last_calc_time = now
                            last_calc_bytes = task.downloaded_bytes

                    f.flush()
                    os.fsync(f.fileno())
            finally:
                response.close()

        _verify_gguf(part_path)
        if task.status == "cancelled":
            raise _Cancelled()

        os.replace(part_path, target_path)

        if task.status == "cancelled":
            try:
                target_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise _Cancelled()

        task.status = "completed"
        task.eta_seconds = 0
        completed = True
        logger.info("Successfully downloaded GGUF model: %s", task.filename)

    except _Cancelled:
        task.status = "cancelled"
    except (DownloadURLError, DownloadLimitError) as exc:
        if task.status != "cancelled":
            logger.warning("Rejected download of %s: %s", task.filename, exc)
            task.status = "failed"
            task.error = str(exc)
    except (httpx.HTTPError, OSError) as exc:
        if task.status != "cancelled":
            logger.warning("Download of %s failed: %s", task.filename, exc)
            task.status = "failed"
            task.error = "Download failed (network or filesystem error)."
    except Exception:
        if task.status != "cancelled":
            logger.exception("Unexpected failure downloading %s", task.filename)
            task.status = "failed"
            task.error = "Download failed unexpectedly."
    finally:
        if not completed:
            try:
                part_path.unlink(missing_ok=True)
            except OSError:
                pass


class DownloadManager:
    """Manages streaming background GGUF model downloads."""

    _tasks: ClassVar[dict[str, DownloadTask]] = {}
    _lock = asyncio.Lock()

    @classmethod
    def list_tasks(cls) -> list[dict[str, Any]]:
        return [
            t.to_dict()
            for t in sorted(
                cls._tasks.values(), key=lambda x: x.created_at, reverse=True
            )
        ]

    @classmethod
    def get_task(cls, task_id: str) -> DownloadTask | None:
        return cls._tasks.get(task_id)

    @classmethod
    async def start_download(
        cls,
        repo_id: str,
        filename: str,
        download_url: str,
        size_bytes: int,
        settings: Settings,
    ) -> DownloadTask:
        async with cls._lock:
            # Check if already downloading this file
            for t in cls._tasks.values():
                if t.filename == filename and t.status in (
                    "queued",
                    "downloading",
                ):
                    return t

            # Bound task registry: evict older terminal tasks when size exceeds 50
            if len(cls._tasks) >= 50:
                terminal_ids = [
                    tid for tid, t in cls._tasks.items()
                    if t.status in ("completed", "failed", "cancelled")
                ]
                terminal_ids.sort(key=lambda tid: cls._tasks[tid].created_at)
                for tid in terminal_ids[: len(cls._tasks) - 49]:
                    cls._tasks.pop(tid, None)

            clean_url = validate_download_url(download_url, settings)

            task_id = f"dl_{uuid.uuid4().hex[:8]}"
            task = DownloadTask(
                task_id=task_id,
                repo_id=repo_id,
                filename=filename,
                url=clean_url,
                size_bytes=size_bytes,
            )
            cls._tasks[task_id] = task

            # Spawn background download worker in thread
            task._async_task = asyncio.create_task(
                asyncio.to_thread(_sync_download_worker, task, settings)
            )
            return task

    @classmethod
    async def cancel_download(
        cls, task_id: str, settings: Settings
    ) -> bool:
        task = cls._tasks.get(task_id)
        if not task:
            return False

        if task.status in ("queued", "downloading"):
            task.status = "cancelled"
            if task._async_task and not task._async_task.done():
                task._async_task.cancel()
            return True
        return False
