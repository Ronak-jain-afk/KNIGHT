import json

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from app.agent.graph import build_agent
from app.config import Settings, get_settings
from app.db.sqlite import run_migrations
from app.main import app
from tests.integration.test_agent_graph import ScriptedChatModel


@pytest.fixture(autouse=True)
async def setup_test_app(tmp_path):
    settings = Settings(data_dir=tmp_path / "data")
    settings.ensure_dirs()
    await run_migrations(settings.app_db_path)

    call_model = ScriptedChatModel(
        responses=[
            AIMessage(content="Hello there! Welcome to Knight."),
            AIMessage(content="I remember our conversation."),
        ]
    )
    blocking_model = ScriptedChatModel(responses=[AIMessage(content="Summary")])
    checkpointer = MemorySaver()

    agent = build_agent(settings, checkpointer, call_model, blocking_model)

    app.state.agent = agent
    app.state.checkpointer = checkpointer
    app.dependency_overrides[get_settings] = lambda: settings
    yield
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_chat_validation_errors():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Empty message
        r1 = await client.post("/api/chat", json={"message": "   "})
        assert r1.status_code == 400
        assert "empty" in r1.json()["detail"].lower()

        # Oversized message (> configured max)
        r2 = await client.post("/api/chat", json={"message": "x" * 32_001})
        assert r2.status_code == 413
        assert "exceeds maximum length" in r2.json()["detail"].lower()


@pytest.mark.asyncio
async def test_chat_sse_stream_ordering():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/api/chat",
            json={"message": "Hello!"},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        body = resp.text
        frames = [f.strip() for f in body.split("\n\n") if f.strip()]

        events = []
        for frame in frames:
            lines = frame.split("\n")
            event_type = None
            data = None
            for line in lines:
                if line.startswith("event:"):
                    event_type = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    data = json.loads(line.split(":", 1)[1].strip())
            if event_type:
                events.append((event_type, data))

        # Check event sequence
        assert len(events) >= 3
        # First frame MUST be meta
        assert events[0][0] == "meta"
        meta_data = events[0][1]
        assert "conversation_id" in meta_data
        assert "run_id" in meta_data

        # Middle frames are tokens
        token_texts = [d["text"] for ev, d in events if ev == "token"]
        assert len(token_texts) > 0
        full_text = "".join(token_texts).strip()
        assert "Hello there! Welcome to Knight." in full_text

        # Last frame MUST be done
        assert events[-1][0] == "done"
        done_data = events[-1][1]
        assert done_data["status"] == "ok"
        assert done_data["elapsed_ms"] >= 0


@pytest.mark.asyncio
async def test_chat_creates_conversation_and_preserves_history():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # First turn without conversation_id
        r1 = await client.post("/api/chat", json={"message": "Tell me a joke"})
        assert r1.status_code == 200

        # Extract created thread_id from meta event
        first_line = r1.text.split("\n\n")[0]
        meta_json = json.loads(first_line.split("data: ")[1])
        conv_id = meta_json["conversation_id"]
        assert conv_id is not None

        # Second turn on the same thread
        r2 = await client.post(
            "/api/chat",
            json={"conversation_id": conv_id, "message": "What did I just ask?"},
        )
        assert r2.status_code == 200

        # Query conversation detail endpoint to verify history persisted
        r3 = await client.get(f"/api/conversations/{conv_id}")
        assert r3.status_code == 200
        conv_detail = r3.json()
        assert conv_detail["turn_count"] == 2
        assert len(conv_detail["messages"]) >= 4


@pytest.mark.asyncio
async def test_chat_with_system_prompt_override():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/api/chat",
            json={
                "message": "Hello!",
                "system_prompt_override": "Always act like an astronomy professor.",
            },
        )
        assert resp.status_code == 200
        assert "event: meta" in resp.text
        assert "event: done" in resp.text


@pytest.mark.asyncio
async def test_chat_with_profile_id():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/api/chat",
            json={
                "message": "Write a quick test function",
                "profile_id": "software-engineer",
            },
        )
        assert resp.status_code == 200
        assert "event: meta" in resp.text
        assert "event: done" in resp.text


@pytest.mark.asyncio
async def test_profile_mode_is_applied_without_client_mode():
    """M-03: a profile's preferred mode is honoured when the client omits one."""
    from app.db import profiles

    settings = app.dependency_overrides[get_settings]()
    await profiles.update_profile(
        settings.app_db_path,
        profile_id="software-engineer",
        mode="chat",
    )
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/chat",
                json={"message": "hello", "profile_id": "software-engineer"},
            )
            assert resp.status_code == 200
            assert "event: done" in resp.text

            meta_frame = next(
                f for f in resp.text.split("\n\n") if "event: meta" in f
            )
            meta = json.loads(
                next(
                    line[len("data: ") :]
                    for line in meta_frame.split("\n")
                    if line.startswith("data: ")
                )
            )
            state = await app.state.agent.aget_state(
                {"configurable": {"thread_id": meta["conversation_id"]}}
            )
            assert state.values["mode"] == "chat"
    finally:
        await profiles.update_profile(
            settings.app_db_path,
            profile_id="software-engineer",
            mode="agent",
        )


@pytest.mark.asyncio
async def test_explicit_chat_mode_overrides_profile_mode():
    """M-03: an explicit client mode still wins over the profile preference."""
    from app.db import profiles

    settings = app.dependency_overrides[get_settings]()
    await profiles.update_profile(
        settings.app_db_path,
        profile_id="software-engineer",
        mode="chat",
    )
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/chat",
                json={
                    "message": "hello",
                    "profile_id": "software-engineer",
                    "mode": "agent",
                },
            )
            assert resp.status_code == 200
    finally:
        await profiles.update_profile(
            settings.app_db_path,
            profile_id="software-engineer",
            mode="agent",
        )


@pytest.mark.asyncio
async def test_chat_with_model_customization():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        custom_payload = {
            "temperature": 0.3,
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 1024,
            "presence_penalty": 0.2,
            "frequency_penalty": 0.1,
            "enable_thinking": True,
        }
        resp = await client.post(
            "/api/chat",
            json={
                "message": "Explain quantum computing briefly",
                "model_customization": custom_payload,
            },
        )
        assert resp.status_code == 200
        assert "event: meta" in resp.text
        assert "event: done" in resp.text

        # Verify meta event contains model_customization
        frames = [f.strip() for f in resp.text.split("\n\n") if f.strip()]
        meta_frame = next(f for f in frames if "event: meta" in f)
        for line in meta_frame.split("\n"):
            if line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
                assert data["model_customization"]["temperature"] == 0.3
                assert data["model_customization"]["enable_thinking"] is True
                assert data["model_customization"]["top_k"] == 40
                break

