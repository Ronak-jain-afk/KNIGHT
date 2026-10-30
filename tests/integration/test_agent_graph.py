import json
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from pydantic import Field

from app.agent.graph import build_agent
from app.agent.sources import current_sources
from app.agent.streaming import stream_agent_run
from app.config import Settings
from app.db.sqlite import run_migrations


class ScriptedChatModel(BaseChatModel):
    responses: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-chat-model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self

    def _generate(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        msg = self.responses.pop(0) if self.responses else AIMessage(content="Final answer")
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _stream(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> Any:
        msg = self.responses.pop(0) if self.responses else AIMessage(content="Final answer")
        if getattr(msg, "tool_calls", None):
            chunks = [
                {"name": tc["name"], "args": json.dumps(tc.get("args", {})), "id": tc["id"], "index": i}
                for i, tc in enumerate(msg.tool_calls)
            ]
            yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=chunks))
        else:
            for word in str(msg.content).split(" "):
                yield ChatGenerationChunk(message=AIMessageChunk(content=word + " "))


class ToolCapturingModel(ScriptedChatModel):
    captured_tools: list[str] = Field(default_factory=list)

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        self.captured_tools = [getattr(t, "name", str(t)) for t in tools]
        return self


@pytest.fixture
async def agent_settings(tmp_path):
    s = Settings(
        data_dir=tmp_path / "data",
        summary_trigger_tokens=100,
        llama_api_key="test-llama-key",
        memory_auto_extract=False,
    )
    s.ensure_dirs()
    await run_migrations(s.app_db_path)
    return s


@pytest.mark.asyncio
async def test_agent_graph_tool_calling(agent_settings):
    call_msg = AIMessage(
        content="",
        tool_calls=[{"name": "current_datetime", "args": {}, "id": "call_1"}],
    )
    answer_msg = AIMessage(content="The current time is verified.")

    model = ScriptedChatModel(responses=[call_msg, answer_msg])
    blocking = ScriptedChatModel(responses=[AIMessage(content="Summary")])
    checkpointer = MemorySaver()

    agent = build_agent(agent_settings, checkpointer, model, blocking)

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What is the time?"}], "mode": "agent"},
        config={"configurable": {"thread_id": "t-1"}},
    )

    messages = result["messages"]
    assert any(isinstance(m, ToolMessage) for m in messages)
    assert messages[-1].content == "The current time is verified."


@pytest.mark.asyncio
async def test_agent_graph_loop_guard(agent_settings):
    call_msg_1 = AIMessage(
        content="",
        tool_calls=[{"name": "current_datetime", "args": {}, "id": "call_a"}],
    )
    call_msg_2 = AIMessage(
        content="",
        tool_calls=[{"name": "current_datetime", "args": {}, "id": "call_b"}],
    )
    final_msg = AIMessage(content="Resolved despite repeat attempt.")

    model = ScriptedChatModel(responses=[call_msg_1, call_msg_2, final_msg])
    blocking = ScriptedChatModel(responses=[AIMessage(content="Summary")])
    checkpointer = MemorySaver()

    agent = build_agent(agent_settings, checkpointer, model, blocking)

    events: list[dict[str, Any]] = []
    async for sse in stream_agent_run(
        agent=agent,
        settings=agent_settings,
        thread_id="t-2",
        run_id="run-2",
        user_message="Time twice?",
    ):
        for line in sse.strip().split("\n"):
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

    assert any("already called with these arguments" in str(e) for e in events)


@pytest.mark.asyncio
async def test_agent_graph_chat_mode_policy(agent_settings):
    answer_msg = AIMessage(content="Chatting without web search.")
    capturing_model = ToolCapturingModel(responses=[answer_msg])
    blocking = ScriptedChatModel(responses=[AIMessage(content="Summary")])
    checkpointer = MemorySaver()

    agent = build_agent(agent_settings, checkpointer, capturing_model, blocking)

    await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Hi!"}], "mode": "chat"},
        config={"configurable": {"thread_id": "t-3"}},
    )

    assert "web_search" not in capturing_model.captured_tools
    assert "fetch_page" not in capturing_model.captured_tools
    assert "calculate" in capturing_model.captured_tools


@pytest.mark.asyncio
async def test_stream_agent_run_sources_via_contextvar(agent_settings):
    """T-06: drive the real stream_agent_run pipeline with a fake model;

    assert sources arrive via the contextvar without manual state seeding.
    """
    @tool
    def search_with_source(query: str) -> str:
        """Search tool that records a source via the active contextvar."""
        collector = current_sources.get()
        assert collector is not None, "contextvar current_sources must be active"
        collector.add(url="https://example.com/result", title="Test Result")
        return "Found result"

    call_msg = AIMessage(
        content="",
        tool_calls=[{"name": "search_with_source", "args": {"query": "test"}, "id": "call_s1"}],
    )
    answer_msg = AIMessage(content="According to [1], result found.")

    model = ScriptedChatModel(responses=[call_msg, answer_msg])
    blocking = ScriptedChatModel(responses=[AIMessage(content="Summary")])
    checkpointer = MemorySaver()

    agent = build_agent(
        agent_settings,
        checkpointer,
        model,
        blocking,
        tools=[search_with_source],
    )

    sources_frames: list[dict[str, Any]] = []
    async for sse in stream_agent_run(
        agent=agent,
        settings=agent_settings,
        thread_id="t-contextvar-1",
        run_id="run-contextvar-1",
        user_message="Find something",
    ):
        if "event: sources" in sse:
            for line in sse.splitlines():
                if line.startswith("data: "):
                    sources_frames.append(json.loads(line[6:]))

    assert len(sources_frames) == 1
    items = sources_frames[0]["items"]
    assert len(items) == 1
    assert items[0]["url"] == "https://example.com/result"
    assert items[0]["title"] == "Test Result"
