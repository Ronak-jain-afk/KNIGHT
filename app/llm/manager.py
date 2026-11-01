# SPDX-License-Identifier: Apache-2.0
import asyncio
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI

from app.agent.graph import build_agent
from app.config import Settings
from app.llm.client import get_chat_model
from app.llm.gate import inference_gate
from app.llm.llama_args import build_llama_argv, find_llama_bin, llama_env
from app.tools.local_files import format_size
from app.utils.processes import IS_WINDOWS, terminate_process_tree

logger = logging.getLogger(__name__)

# How long a model swap waits for in-flight streaming runs to finish.
DRAIN_TIMEOUT_SECONDS = 60.0


def list_gguf_models(settings: Settings, active_model: str | None = None) -> list[dict[str, Any]]:
    """Scan models directory for GGUF files and return metadata."""
    models_dir = settings.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    gguf_files = sorted(models_dir.glob("*.gguf"))
    current_active = active_model or ModelManager.get_active_model(settings)

    models: list[dict[str, Any]] = []
    for f in gguf_files:
        try:
            sz = f.stat().st_size
            mod_iso = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(f.stat().st_mtime)
            )
        except OSError:
            sz = 0
            mod_iso = "unknown"

        # Check if active by filename or stem
        is_active = (f.name == current_active) or (f.stem == current_active)

        models.append(
            {
                "name": f.name,
                "stem": f.stem,
                "size_bytes": sz,
                "size_formatted": format_size(sz),
                "modified_at": mod_iso,
                "is_active": is_active,
            }
        )
    return models


async def wait_for_server_health(
    server_root: str,
    api_key: str,
    timeout: int = 60,
    proc: subprocess.Popen | None = None,
) -> bool:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    start_time = time.time()
    while time.time() - start_time < timeout:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{server_root}/health", headers=headers)
                if resp.status_code == 200:
                    return True
        except (httpx.HTTPError, OSError):
            pass
        await asyncio.sleep(1.0)
    return False


def resolve_model_path(settings: Settings, model_filename: str) -> Path:
    """Resolve a model filename inside the models directory and confine it there.

    Raises ValueError for empty names, separators, non-GGUF files, and any path that
    escapes ``settings.models_dir`` (``..``, absolute paths, symlinks).
    """
    name = model_filename.strip()
    if not name:
        raise ValueError("model_name cannot be empty.")
    if Path(name).name != name or any(sep in name for sep in ("/", "\\", "\x00")):
        raise ValueError("model_name must be a bare filename; path separators are forbidden.")

    models_dir = settings.models_dir.resolve()
    candidate = (models_dir / name).resolve()

    if not candidate.is_relative_to(models_dir):
        raise ValueError("model_name resolves outside the models directory.")
    if candidate.suffix.lower() != ".gguf":
        raise ValueError("model_name must point to a .gguf file.")
    return candidate


class ModelManager:
    """Manages active GGUF inference model swapping."""

    _lock = asyncio.Lock()
    _active_server_proc: subprocess.Popen | None = None
    _active_model: str | None = None

    @classmethod
    def get_active_model(cls, settings: Settings) -> str:
        if cls._active_model:
            return cls._active_model

        models_dir = settings.models_dir
        if models_dir.exists():
            # 1. Match settings.model_name explicitly against GGUF files in models_dir
            for f in models_dir.glob("*.gguf"):
                if f.name == settings.model_name or f.stem == settings.model_name:
                    cls._active_model = f.name
                    return cls._active_model

            # 2. Check for default model file
            default_path = models_dir / "Qwen3.5-4B-UD-Q4_K_XL.gguf"
            if default_path.exists():
                cls._active_model = default_path.name
                return cls._active_model

            # 3. Fallback to first available GGUF model in models_dir
            ggufs = sorted(models_dir.glob("*.gguf"))
            if ggufs:
                cls._active_model = ggufs[0].name
                return cls._active_model

        return settings.model_name

    @classmethod
    async def switch_model(
        cls,
        settings: Settings,
        model_filename: str,
        app: FastAPI,
    ) -> dict[str, Any]:
        async with cls._lock:
            try:
                model_path = resolve_model_path(settings, model_filename)
            except ValueError as exc:
                return {"ok": False, "error": "invalid_model_name", "detail": str(exc)}

            if not model_path.exists() or not model_path.is_file():
                return {
                    "ok": False,
                    "error": "model_not_found",
                    "detail": f"Model file '{model_filename}' does not exist in {settings.models_dir}",
                }

            # 1. Drain in-flight turns instead of killing active streams.
            if not await inference_gate.wait_idle(DRAIN_TIMEOUT_SECONDS):
                logger.warning(
                    "Refusing to swap model while %d run(s) are active",
                    inference_gate.active,
                )
                return {
                    "ok": False,
                    "error": "model_swapping",
                    "detail": "A response is still being generated. Try again in a moment.",
                }

            # 2. Stop only the child process this manager started.
            proc = cls._active_server_proc
            cls._active_server_proc = None
            await asyncio.to_thread(terminate_process_tree, proc, 10.0)
            await asyncio.sleep(0.5)

            # 3. Spawn a new llama-server using the canonical argv.
            try:
                llama_bin = find_llama_bin()
            except FileNotFoundError as exc:
                return {"ok": False, "error": "llama_binary_not_found", "detail": str(exc)}

            server_root = settings.llama_base_url.rstrip("/").removesuffix("/v1")
            port = int(server_root.rsplit(":", 1)[-1].split("/")[0])
            api_key = settings.llama_api_key.strip()
            if not api_key:
                try:
                    api_key = settings.llama_api_key_file.read_text(encoding="utf-8").strip()
                except OSError:
                    api_key = ""

            try:
                argv = build_llama_argv(
                    llama_bin=llama_bin,
                    model_path=model_path,
                    port=port,
                    api_key=api_key,
                    settings=settings,
                )
            except ValueError as exc:
                return {"ok": False, "error": "invalid_launch_config", "detail": str(exc)}

            popen_kwargs: dict[str, Any] = {
                "env": llama_env(llama_bin),
                "cwd": str(llama_bin.parent),
            }
            if IS_WINDOWS:
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                popen_kwargs["start_new_session"] = True

            cls._active_server_proc = await asyncio.to_thread(
                subprocess.Popen, argv, **popen_kwargs
            )

            # 4. Wait for the new server to be ready.
            ready = await wait_for_server_health(
                server_root,
                settings.llama_api_key,
                timeout=90,
                proc=cls._active_server_proc,
            )

            if not ready:
                logger.error("Failed to start llama-server with model %s", model_filename)
                terminate_process_tree(cls._active_server_proc, 5.0)
                cls._active_server_proc = None
                return {
                    "ok": False,
                    "error": "model_load_timeout",
                    "detail": f"Timed out waiting for {model_filename} to load.",
                }

            # 5. Record the active model outside Settings (never mutate cached config)
            #    and rebuild the agent so the new process is used.
            cls._active_model = model_path.name
            app.state.active_model = model_path.name

            checkpointer = getattr(app.state, "checkpointer", None)
            streaming_model = get_chat_model(settings, streaming=True)
            blocking_model = get_chat_model(settings, streaming=False)

            app.state.agent = build_agent(
                settings=settings,
                checkpointer=checkpointer,
                streaming_model=streaming_model,
                blocking_model=blocking_model,
            )

            logger.info("Successfully switched model to: %s", model_path.name)
            return {
                "ok": True,
                "model_name": model_path.name,
                "message": f"Active model successfully changed to {model_path.name}",
            }
