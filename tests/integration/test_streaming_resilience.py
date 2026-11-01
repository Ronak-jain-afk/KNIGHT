# SPDX-License-Identifier: Apache-2.0
"""Regression tests for streaming resilience (C-03/C-04/C-05/C-06/C-19)."""

import asyncio
import json
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langgraph.checkpoint.memory import MemorySaver
from pydantic import Field

from app.agent.graph import build_agent
from app.agent.streaming import (
    _background_tasks,
    drain_background_tasks,
    stream_agent_run,
    track_background_task,
)
from app.config import Settings
from app.db import runs
from app.db.sqlite import run_migrations
from app.llm.gate import inference_gate


class EndlessStreamingAgent:
    """Minimal stand-in for a compiled graph: emits tokens forever."""

    async def astream(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        stream_mode: Any = None,
        **kwargs: Any,
    ):
        del inputs, config, stream_mode, kwargs
        for index in range(10_000):
            yield "messages", (
                AIMessageChunk(content=f"t{index} "),
                {"langgraph_node": "model"},
            )


class FailingAgent:
    async def astream(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        stream_mode: Any = None,
        **kwargs: Any,
    ):
        del inputs, config, stream_mode, kwargs
        raise RuntimeError(
            "SECRET internal failure at /srv/private/knight.sqlite host=10.0.0.7"
        )
        yield  # pragma: no cover - makes this an async generator


class V2UsageStreamingAgent:
    """Emits v2 dict events and real provider usage metadata."""

    def __init__(self) -> None:
        self.seen_version: str | None = None

    async def astream(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        stream_mode: Any = None,
        **kwargs: Any,
    ):
        del inputs, config, stream_mode
        self.seen_version = kwargs.get("version")
        yield {
            "type": "messages",
            "ns": [],
            "data": (AIMessageChunk(content="Hel"), {"langgraph_node": "model"}),
        }
        yield {
            "type": "messages",
            "ns": [],
            "data": (
                AIMessageChunk(
                    content="lo",
                    usage_metadata={
                        "input_tokens": 11,
                        "output_tokens": 7,
                        "total_tokens": 18,
                    },
                ),
                {"langgraph_node": "model"},
            ),
        }
        yield {
            "type": "updates",
            "ns": [],
            "data": {
                "model": {"messages": [AIMessage(content="Hello", id="msg-usage")]}
            },
        }


class CapturingChatModel(BaseChatModel):
    seen_system: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "capturing-chat-model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any):
        del messages, stop, run_manager, kwargs
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

    def _stream(self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any):
        del stop, run_manager, kwargs
        system_text = "\n".join(
            str(msg.content) for msg in messages if isinstance(msg, SystemMessage)
        )
        self.seen_system.append(system_text)
        yield ChatGenerationChunk(message=AIMessageChunk(content="ok"))


@pytest.fixture
def stream_settings(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        memory_auto_extract=False,
    )
    settings.ensure_dirs()
    return settings


@pytest.fixture(autouse=True)
async def migrate(stream_settings):
    await run_migrations(stream_settings.app_db_path)


@pytest.mark.asyncio
async def test_client_disconnect_marks_run_cancelled(stream_settings):
    """C-03/C-04: closing the generator neither raises nor leaves a running row."""
    before = inference_gate.active
    generator = stream_agent_run(
        EndlessStreamingAgent(),
        stream_settings,
        "thread-cancel",
        "run-cancel",
        "hello",
    )

    frames: list[str] = []
    async for frame in generator:
        frames.append(frame)
        if len(frames) == 3:
            break

    # Must not raise "async generator ignored GeneratorExit".
    await generator.aclose()

    body = "".join(frames)
    assert "event: meta" in body
    assert "event: done" not in body

    row = await runs.get_run(stream_settings.app_db_path, "run-cancel")
    assert row is not None
    assert row["status"] == "cancelled"
    assert inference_gate.active == before


@pytest.mark.asyncio
async def test_disconnect_before_first_token_still_finalizes(stream_settings):
    """C-04: closing at the meta frame must not leave a running row."""
    before = inference_gate.active
    generator = stream_agent_run(
        EndlessStreamingAgent(),
        stream_settings,
        "thread-meta",
        "run-meta",
        "hello",
    )

    first = await generator.__anext__()
    assert "event: meta" in first
    await generator.aclose()

    row = await runs.get_run(stream_settings.app_db_path, "run-meta")
    assert row is not None
    assert row["status"] == "cancelled"
    assert inference_gate.active == before


@pytest.mark.asyncio
async def test_internal_error_does_not_leak_details(stream_settings):
    """C-06: only a generic message plus the run id may reach the client."""
    frames = [
        frame
        async for frame in stream_agent_run(
            FailingAgent(),
            stream_settings,
            "thread-error",
            "run-error",
            "boom",
        )
    ]
    body = "".join(frames)

    assert "event: error" in body
    for leaked in ("SECRET", "/srv/private", "knight.sqlite", "10.0.0.7"):
        assert leaked not in body
    assert "run-error" in body

    row = await runs.get_run(stream_settings.app_db_path, "run-error")
    assert row is not None
    assert row["status"] == "error"
    assert row["error_code"] == "execution_error"


@pytest.mark.asyncio
async def test_per_turn_context_does_not_leak_into_next_turn(stream_settings):
    """C-19: memories/override from turn 1 must not apply to turn 2."""
    model = CapturingChatModel()
    blocking = CapturingChatModel()
    agent = build_agent(stream_settings, MemorySaver(), model, blocking)

    async def drain(generator):
        async for _ in generator:
            pass

    await drain(
        stream_agent_run(
            agent,
            stream_settings,
            "thread-context",
            "run-ctx-1",
            "first",
            memories=[
                {
                    "id": "m1",
                    "text": "USER LIKES TEA",
                    "kind": "preference",
                    "score": 0.9,
                }
            ],
            system_prompt_override="SECRET-PERSONA",
        )
    )
    await drain(
        stream_agent_run(
            agent,
            stream_settings,
            "thread-context",
            "run-ctx-2",
            "second",
        )
    )

    assert len(model.seen_system) >= 2
    turn_one, turn_two = model.seen_system[0], model.seen_system[-1]
    assert "SECRET-PERSONA" in turn_one
    assert "USER LIKES TEA" in turn_one
    assert "SECRET-PERSONA" not in turn_two
    assert "USER LIKES TEA" not in turn_two


@pytest.mark.asyncio
async def test_background_tasks_are_referenced_and_drained():
    """C-05: extraction tasks stay referenced until they finish."""
    started = asyncio.Event()
    finished = asyncio.Event()

    async def job() -> None:
        started.set()
        await asyncio.sleep(0.05)
        finished.set()

    task = asyncio.create_task(job())
    track_background_task(task)
    assert task in _background_tasks

    await started.wait()
    await drain_background_tasks(timeout=2)
    await asyncio.sleep(0)

    assert finished.is_set()
    assert task not in _background_tasks


@pytest.mark.asyncio
async def test_v2_events_and_real_token_usage(stream_settings):
    """M-13/M-14: request v2 events and report provider token usage, not chunks."""
    agent = V2UsageStreamingAgent()
    frames = [
        frame
        async for frame in stream_agent_run(
            agent,
            stream_settings,
            "thread-usage",
            "run-usage",
            "hi",
        )
    ]
    body = "".join(frames)

    assert agent.seen_version == "v2"
    assert "event: token" in body
    assert "Hello" not in body  # only streamed token deltas reach the client

    done_frame = next(frame for frame in frames if "event: done" in frame)
    done_data = json.loads(done_frame.split("data: ", 1)[1].split("\n\n", 1)[0])
    assert done_data["usage"] == {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "chunks": 2,
    }

    row = await runs.get_run(stream_settings.app_db_path, "run-usage")
    assert row is not None
    assert row["status"] == "ok"
    assert row["prompt_tokens"] == 11
    assert row["completion_tokens"] == 7
