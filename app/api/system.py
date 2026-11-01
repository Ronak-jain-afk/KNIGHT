# SPDX-License-Identifier: Apache-2.0
import asyncio
import os
import platform
import signal
import sqlite3
from pathlib import Path
from typing import Annotated

import aiosqlite
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from app.api.schemas import ConfigResponse, HealthResponse
from app.config import Settings, get_settings
from app.llm.client import check_llama_health
from app.llm.hardware import get_hardware_status

router = APIRouter(prefix="/api", tags=["system"])


async def check_db_health(db_path: Path, timeout: float = 2.0) -> bool:
    """Run a real ``SELECT 1`` against a SQLite file with a hard timeout."""
    if not db_path.exists():
        # Nothing has been created yet; there is no corrupted/locked file to report.
        return True
    try:
        async with asyncio.timeout(timeout):
            async with aiosqlite.connect(str(db_path)) as db:
                await db.execute("SELECT 1")
        return True
    except (OSError, sqlite3.Error, TimeoutError):
        return False


@router.get("/health", response_model=HealthResponse)
async def get_health(
    settings: Annotated[Settings, Depends(get_settings)],
) -> HealthResponse:
    llama_health = await check_llama_health(settings)
    memory_status = "disabled"
    if settings.memory_enabled:
        from app.memory.embeddings import get_embedding_service

        emb_service = get_embedding_service(settings)
        memory_status = "ok" if emb_service.is_available() else "degraded"

    app_db_ok = await check_db_health(settings.app_db_path)
    checkpoint_db_ok = await check_db_health(settings.checkpoint_db_path)

    return HealthResponse(
        app="ok",
        llama=llama_health,
        db="ok" if app_db_ok and checkpoint_db_ok else "error",
        memory=memory_status,
    )


@router.get("/config", response_model=ConfigResponse)
async def get_safe_config(
    settings: Annotated[Settings, Depends(get_settings)],
) -> ConfigResponse:
    return ConfigResponse(
        model_name=settings.model_name,
        context_tokens=settings.context_tokens,
        enable_thinking=settings.enable_thinking,
        search_providers=settings.parsed_search_providers,
        temperature=settings.temperature,
        top_p=settings.top_p,
        top_k=settings.top_k,
        max_output_tokens=settings.max_output_tokens,
        presence_penalty=settings.presence_penalty,
        max_message_chars=settings.max_message_chars,
        max_attachment_bytes=settings.max_attachment_bytes,
    )


@router.get("/system/hardware")
async def get_hardware(
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict:
    """Return live host and GPU hardware telemetry."""
    return await get_hardware_status(settings)


def request_graceful_shutdown() -> None:
    """Ask the process to stop through its normal signal handlers.

    Windows has no graceful SIGTERM: ``os.kill`` terminates the process, so we
    raise SIGINT instead, which uvicorn handles by draining in-flight work.
    """
    if platform.system().lower() == "windows":
        try:
            signal.raise_signal(signal.SIGINT)
            return
        except (AttributeError, ValueError, OSError):
            pass
    os.kill(os.getpid(), signal.SIGTERM)


@router.post("/shutdown")
async def shutdown_server(
    settings: Annotated[Settings, Depends(get_settings)],
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    if not settings.dev:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Shutdown endpoint only available in dev mode (KNIGHT_DEV=true).",
        )
    # Defer until the response has been sent so the client sees a clean reply.
    background_tasks.add_task(request_graceful_shutdown)
    return {"status": "shutting down"}
