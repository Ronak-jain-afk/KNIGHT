# SPDX-License-Identifier: Apache-2.0
import asyncio
import contextlib
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.agent.graph import build_agent
from app.agent.streaming import drain_background_tasks
from app.api.asr import router as asr_router
from app.api.chat import router as chat_router
from app.api.conversations import router as conversations_router
from app.api.documents import router as documents_router
from app.api.ggufbook import router as ggufbook_router
from app.api.memories import router as memories_router
from app.api.models import router as models_router
from app.api.profiles import router as profiles_router
from app.api.system import router as system_router
from app.asr.service import ASRService
from app.config import Settings, get_settings
from app.db.cache import purge_expired_cache_loop
from app.db.sqlite import close_db_pools, init_checkpoint_db, run_migrations
from app.llm.client import get_chat_model
from app.llm.manager import ModelManager
from app.logging_setup import setup_logging
from app.memory.repository import get_memory_repository
from app.security.middleware import (
    AuthTokenMiddleware,
    OriginCheckMiddleware,
    SecurityHeadersMiddleware,
)
from app.security.vendor_guard import verify_vendor_manifest

logger = logging.getLogger(__name__)

_TRACING_ENV_VARS = ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING")
_FALSY = {"", "0", "false", "no", "off"}


def assert_tracing_disabled() -> None:
    """Refuse to start with off-box tracing enabled (local-first ground rule).

    We do not overwrite the operator's environment silently; a misconfiguration is
    reported explicitly instead.
    """
    for name in _TRACING_ENV_VARS:
        value = os.environ.get(name, "").strip().lower()
        if value not in _FALSY:
            raise RuntimeError(
                f"{name}={value!r} is not allowed. Knight is local-first and must not "
                "send traces off-box; unset it or set it to 'false'."
            )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    assert_tracing_disabled()
    verify_vendor_manifest()

    settings = get_settings()
    setup_logging(settings)

    # 1. Database setup
    await run_migrations(settings.app_db_path)
    await init_checkpoint_db(settings.checkpoint_db_path)

    # 2. Memory repository setup
    if settings.memory_enabled:
        await get_memory_repository(settings).initialize()

    # Expired search/page cache rows are purged on a timer, not just at startup.
    purge_stop = asyncio.Event()
    purge_task = asyncio.create_task(
        purge_expired_cache_loop(
            settings.app_db_path,
            interval_seconds=settings.cache_purge_interval_seconds,
            max_rows=settings.cache_max_rows,
            stop_event=purge_stop,
        ),
        name="cache-purge",
    )

    # 3. Checkpointer and Agent graph build
    async with AsyncSqliteSaver.from_conn_string(str(settings.checkpoint_db_path)) as checkpointer:
        await checkpointer.setup()

        streaming_model = get_chat_model(settings, streaming=True)
        blocking_model = get_chat_model(settings, streaming=False)

        agent = build_agent(
            settings=settings,
            checkpointer=checkpointer,
            streaming_model=streaming_model,
            blocking_model=blocking_model,
        )

        app.state.settings = settings
        app.state.checkpointer = checkpointer
        app.state.agent = agent
        app.state.active_model = ModelManager.get_active_model(settings)
        app.state.asr_service = ASRService(settings)

        try:
            yield
        finally:
            purge_stop.set()
            purge_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await purge_task
            # Wait for background memory extraction so nothing is lost on shutdown.
            await drain_background_tasks()
            await close_db_pools()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application. Importing this module has no side effects."""
    application = FastAPI(
        title="Knight",
        description="Local-first LangGraph agent with llama.cpp",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Hardening middlewares (settings resolved lazily inside dispatch).
    application.add_middleware(SecurityHeadersMiddleware)
    application.add_middleware(OriginCheckMiddleware, settings=settings)
    application.add_middleware(AuthTokenMiddleware, settings=settings)

    application.include_router(system_router)
    application.include_router(chat_router)
    application.include_router(conversations_router)
    application.include_router(memories_router)
    application.include_router(documents_router)
    application.include_router(models_router)
    application.include_router(profiles_router)
    application.include_router(ggufbook_router)
    application.include_router(asr_router)

    # Mount static web UI at root (after API routers)
    web_dir = Path(__file__).parent / "web"
    if web_dir.exists():
        favicon_path = web_dir / "favicon.svg"
        if favicon_path.exists():
            @application.get("/favicon.ico", include_in_schema=False)
            async def favicon_ico() -> FileResponse:
                return FileResponse(favicon_path, media_type="image/svg+xml")

        application.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")

    return application


def __getattr__(name: str) -> Any:
    """Lazily expose ``app`` for ``uvicorn app.main:app`` without import-time setup."""
    if name == "app":
        application = create_app()
        globals()["app"] = application
        return application
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
