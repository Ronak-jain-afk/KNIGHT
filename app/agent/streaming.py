# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import Request
from langchain_core.messages import ToolMessage
from langgraph.graph.state import CompiledStateGraph

from app.agent.middleware import loop_guard_state
from app.agent.run_context import (
    current_memories,
    current_model_customization,
    current_system_prompt_override,
)
from app.agent.sources import SourceCollector, current_sources
from app.config import Settings
from app.db import runs, sources
from app.llm.gate import inference_gate
from app.llm.manager import ModelManager
from app.logging_setup import current_run_id, current_thread_id

logger = logging.getLogger(__name__)

GENERIC_ERROR_MESSAGE = (
    "The agent could not complete this turn. Check the server logs for the request id."
)

# Strong references to fire-and-forget tasks so they are not garbage collected
# mid-flight. Drained on shutdown (see app.main.lifespan).
_background_tasks: set[asyncio.Task[Any]] = set()


def track_background_task(task: asyncio.Task[Any]) -> None:
    """Keep a strong reference until the task finishes."""
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def drain_background_tasks(timeout: float = 10.0) -> None:
    """Wait for background tasks (memory extraction) to finish on shutdown."""
    pending = {t for t in _background_tasks if not t.done()}
    if not pending:
        return
    _done, still_pending = await asyncio.wait(pending, timeout=timeout)
    for task in still_pending:
        task.cancel()
    if still_pending:
        await asyncio.gather(*still_pending, return_exceptions=True)


def sse_event(event: str, data: dict[str, Any]) -> str:
    """Format single SSE frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _record_success(
    settings: Settings,
    thread_id: str,
    last_assistant_id: str,
    turn_sources: list[dict[str, Any]],
    memories: list[dict[str, Any]] | None,
) -> None:
    if turn_sources:
        await sources.save_message_sources(
            settings.app_db_path, thread_id, last_assistant_id, turn_sources
        )

    if memories:
        try:
            from app.memory.repository import get_memory_repository

            repo = get_memory_repository(settings)
            used_ids = [m["id"] for m in memories if isinstance(m, dict) and "id" in m]
            if used_ids:
                await repo.record_usage(used_ids)
        except (RuntimeError, ValueError, OSError, TimeoutError) as exc:
            logger.warning("Failed to record memory usage: %s", exc)


async def _extract_memories(
    agent: CompiledStateGraph[Any, Any, Any, Any],
    config: dict[str, Any],
    thread_id: str,
) -> None:
    try:
        from app.memory.extractor import extract_memories_background

        state = await agent.aget_state(config)
        msgs = state.values.get("messages", [])
        await extract_memories_background(msgs, source_thread_id=thread_id)
    except (RuntimeError, ValueError, OSError, TimeoutError, AttributeError, KeyError) as exc:
        logger.warning("Memory auto-extract failed: %s", exc)


async def stream_agent_run(
    agent: CompiledStateGraph[Any, Any, Any, Any],
    settings: Settings,
    thread_id: str,
    run_id: str,
    user_message: str,
    mode: str = "agent",
    memories: list[dict[str, Any]] | None = None,
    http_request: Request | None = None,
    system_prompt_override: str | None = None,
    model_customization: dict[str, Any] | None = None,
    model_name: str | None = None,
) -> AsyncIterator[str]:
    """Execute LangGraph agent stream and yield SSE events.

    The generator is cancellation-safe: frames are only yielded from the normal
    execution path, never from ``finally``. Terminal run state is persisted in
    ``finally`` under ``asyncio.shield`` so client disconnects/cancellation still
    leave the run row in a terminal status instead of ``running``.
    """
    current_run_id.set(run_id)
    current_thread_id.set(thread_id)

    collector = SourceCollector()
    current_sources.set(collector)
    loop_guard_state.set({"seen": set(), "count": 0})

    # Per-turn values live in context variables only; never in graph inputs, which
    # LangGraph checkpoints into thread state (see app/agent/run_context.py).
    context_tokens = [
        (current_memories, current_memories.set(memories or [])),
        (
            current_system_prompt_override,
            current_system_prompt_override.set(system_prompt_override),
        ),
        (
            current_model_customization,
            current_model_customization.set(model_customization),
        ),
    ]

    start_time = time.time()
    active_model = model_name or ModelManager.get_active_model(settings)
    await runs.start_run(settings.app_db_path, run_id, thread_id, active_model)

    if settings.log_prompts:
        logger.info("Executing prompt [run=%s thread=%s]: %s", run_id, thread_id, user_message)

    tool_start_times: dict[str, float] = {}
    last_assistant_id: str = f"msg_{run_id}"
    chunks_generated = 0
    completion_tokens: int | None = None
    prompt_tokens: int | None = None
    run_status = "ok"
    error_code: str | None = None

    inputs: dict[str, Any] = {
        "messages": [{"role": "user", "content": user_message}],
        "mode": mode,
    }
    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    inference_gate.enter()
    try:
        # 1. First frame: meta. Inside the try so an immediate disconnect still
        #    finalizes the run and releases the inference gate.
        meta_payload: dict[str, Any] = {
            "conversation_id": thread_id,
            "run_id": run_id,
            "model": active_model,
        }
        if model_customization:
            meta_payload["model_customization"] = model_customization

        yield sse_event("meta", meta_payload)

        stream_iter = agent.astream(
            inputs,
            config,
            stream_mode=["messages", "updates"],
            version="v2",
        )

        async for event in stream_iter:
            # v2 emits {"type", "ns", "data"}; keep the tuple fallback for
            # lightweight test doubles that predate the v2 contract.
            if isinstance(event, dict) and {"type", "data"} <= event.keys():
                stream_mode = event["type"]
                payload = event["data"]
            else:
                stream_mode, payload = event

            # Check for client disconnect
            if http_request and await http_request.is_disconnected():
                logger.info("Client disconnected during turn [thread=%s run=%s]", thread_id, run_id)
                run_status = "cancelled"
                break

            if stream_mode == "messages":
                chunk, metadata = payload
                usage = getattr(chunk, "usage_metadata", None)
                if isinstance(usage, dict):
                    output = usage.get("output_tokens")
                    input_count = usage.get("input_tokens")
                    if isinstance(output, int):
                        completion_tokens = output
                    if isinstance(input_count, int):
                        prompt_tokens = input_count
                response_metadata = getattr(chunk, "response_metadata", None)
                if isinstance(response_metadata, dict):
                    token_usage = response_metadata.get("token_usage")
                    if isinstance(token_usage, dict):
                        output = token_usage.get("completion_tokens")
                        input_count = token_usage.get("prompt_tokens")
                        if isinstance(output, int):
                            completion_tokens = output
                        if isinstance(input_count, int):
                            prompt_tokens = input_count

                if metadata.get("langgraph_node") == "model":
                    text = getattr(chunk, "content", "")
                    if text:
                        chunks_generated += 1
                        yield sse_event("token", {"text": text})

            elif stream_mode == "updates":
                if "model" in payload:
                    for msg in payload["model"].get("messages", []):
                        if getattr(msg, "id", None):
                            last_assistant_id = msg.id
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                call_id = tc["id"]
                                tool_start_times[call_id] = time.time()
                                yield sse_event(
                                    "tool_start",
                                    {
                                        "id": call_id,
                                        "name": tc["name"],
                                        "args": tc.get("args", {}),
                                    },
                                )
                elif "tools" in payload:
                    for msg in payload["tools"].get("messages", []):
                        if isinstance(msg, ToolMessage):
                            call_id = msg.tool_call_id
                            st = tool_start_times.pop(call_id, time.time())
                            duration_ms = int((time.time() - st) * 1000)
                            ok = msg.status != "error"
                            summary = str(msg.content)[:120]
                            yield sse_event(
                                "tool_end",
                                {
                                    "id": call_id,
                                    "name": msg.name or "tool",
                                    "ok": ok,
                                    "summary": summary,
                                    "duration_ms": duration_ms,
                                },
                            )

        # Normal path only: no yields from finally, which would break on
        # GeneratorExit ("async generator ignored GeneratorExit").
        if run_status == "ok":
            turn_sources = collector.items()
            if turn_sources:
                yield sse_event("sources", {"items": turn_sources})

            yield sse_event(
                "done",
                {
                    "message_id": last_assistant_id,
                    "elapsed_ms": int((time.time() - start_time) * 1000),
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "chunks": chunks_generated,
                    },
                    "status": "ok",
                },
            )

    except (GeneratorExit, asyncio.CancelledError):
        run_status = "cancelled"
        raise
    except Exception:
        logger.exception("Error during agent execution [thread=%s run=%s]", thread_id, run_id)
        run_status = "error"
        error_code = "execution_error"
        # Never leak internal exception text (paths, SQL, hosts) to the client.
        yield sse_event(
            "error",
            {
                "code": error_code,
                "message": GENERIC_ERROR_MESSAGE,
                "request_id": run_id,
                "retryable": True,
            },
        )
    finally:
        elapsed_ms = int((time.time() - start_time) * 1000)
        final_status = run_status if run_status in ("ok", "error", "cancelled") else "error"

        async def _finalize() -> None:
            try:
                if final_status == "ok":
                    await _record_success(
                        settings, thread_id, last_assistant_id, collector.items(), memories
                    )
                await runs.finish_run(
                    settings.app_db_path,
                    run_id,
                    status=final_status,
                    error_code=error_code,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    elapsed_ms=elapsed_ms,
                )
                if final_status == "ok" and settings.memory_auto_extract:
                    track_background_task(
                        asyncio.create_task(
                            _extract_memories(agent, config, thread_id),
                            name=f"memory-extract-{run_id}",
                        )
                    )
            except (RuntimeError, ValueError, OSError, TimeoutError) as exc:
                logger.warning("Failed to finalize run %s: %s", run_id, exc)

        try:
            await asyncio.shield(_finalize())
        except BaseException:
            logger.warning("Run finalization interrupted [run=%s]", run_id, exc_info=True)

        inference_gate.leave()

        for var, token in reversed(context_tokens):
            try:
                var.reset(token)
            except (ValueError, RuntimeError):
                pass
