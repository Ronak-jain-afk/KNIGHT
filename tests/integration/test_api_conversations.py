import pytest
from httpx import ASGITransport, AsyncClient

from app.config import Settings, get_settings
from app.db.sqlite import run_migrations
from app.main import app


@pytest.fixture(autouse=True)
async def setup_db(tmp_path):
    settings = Settings(data_dir=tmp_path / "data")
    settings.ensure_dirs()
    await run_migrations(settings.app_db_path)
    app.dependency_overrides[get_settings] = lambda: settings
    yield
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_conversations_api_lifecycle():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Create empty conversation
        create_res = await client.post(
            "/api/conversations",
            json={"title": "Custom Test Topic"},
        )
        assert create_res.status_code == 201
        created = create_res.json()
        conv_id = created["id"]
        assert created["title"] == "Custom Test Topic"

        # List conversations
        list_res = await client.get("/api/conversations")
        assert list_res.status_code == 200
        items = list_res.json()
        assert any(c["id"] == conv_id for c in items)

        # Rename conversation
        patch_res = await client.patch(
            f"/api/conversations/{conv_id}",
            json={"title": "Updated Topic Title"},
        )
        assert patch_res.status_code == 200
        assert patch_res.json()["title"] == "Updated Topic Title"

        # Verify detail
        get_res = await client.get(f"/api/conversations/{conv_id}")
        assert get_res.status_code == 200
        assert get_res.json()["title"] == "Updated Topic Title"

        # Delete conversation
        del_res = await client.delete(f"/api/conversations/{conv_id}")
        assert del_res.status_code == 200
        assert del_res.json()["status"] == "deleted"

        # Confirm 404 after deletion
        get_after = await client.get(f"/api/conversations/{conv_id}")
        assert get_after.status_code == 404


@pytest.mark.asyncio
async def test_conversation_forking():
    from unittest.mock import AsyncMock, MagicMock

    from langchain_core.messages import AIMessage, HumanMessage

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Create root conversation
        res = await client.post("/api/conversations", json={"title": "Root Tree"})
        assert res.status_code == 201
        root_id = res.json()["id"]

        # Mock agent state for root_id
        mock_agent = MagicMock()
        m1 = HumanMessage(id="msg_1", content="Turn 1 User")
        m2 = AIMessage(id="msg_2", content="Turn 1 AI")
        m3 = HumanMessage(id="msg_3", content="Turn 2 User")
        m4 = AIMessage(id="msg_4", content="Turn 2 AI")

        mock_state = MagicMock()
        mock_state.values = {"messages": [m1, m2, m3, m4]}

        async def fake_get_state(config):
            tid = config.get("configurable", {}).get("thread_id")
            if tid == root_id:
                return mock_state
            return MagicMock(values={"messages": []})

        mock_agent.aget_state = AsyncMock(side_effect=fake_get_state)
        mock_agent.aupdate_state = AsyncMock()

        original_agent = getattr(app.state, "agent", None)
        app.state.agent = mock_agent

        try:
            # 2. Fork at turn 1 (message_id = msg_2)
            fork_res = await client.post(
                f"/api/conversations/{root_id}/fork",
                json={"message_id": "msg_2", "title": "Exploration Branch"},
            )
            assert fork_res.status_code == 201
            branch = fork_res.json()
            assert branch["title"] == "Exploration Branch"
            assert branch["parent_conversation_id"] == root_id
            assert branch["forked_from_message_id"] == "msg_2"
            assert len(branch["messages"]) == 2
            assert branch["messages"][0]["id"] == "msg_1"
            assert branch["messages"][1]["id"] == "msg_2"

            # Check aupdate_state was called on the new thread
            assert mock_agent.aupdate_state.called
            call_args = mock_agent.aupdate_state.call_args
            assert call_args[0][0]["configurable"]["thread_id"] == branch["id"]
            assert len(call_args[0][1]["messages"]) == 2

            # 3. Fork entire conversation (message_id omitted)
            full_fork_res = await client.post(
                f"/api/conversations/{root_id}/fork",
                json={},
            )
            assert full_fork_res.status_code == 201
            full_branch = full_fork_res.json()
            assert full_branch["parent_conversation_id"] == root_id
            assert len(full_branch["messages"]) == 4

            # 4. Error cases
            bad_msg_res = await client.post(
                f"/api/conversations/{root_id}/fork",
                json={"message_id": "non_existent_msg_id"},
            )
            assert bad_msg_res.status_code == 400

            bad_conv_res = await client.post(
                "/api/conversations/non-existent-conv/fork",
                json={},
            )
            assert bad_conv_res.status_code == 404
        finally:
            app.state.agent = original_agent


@pytest.mark.asyncio
async def test_conversation_pinning():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Create two conversations
        res1 = await client.post("/api/conversations", json={"title": "Chat Alpha"})
        assert res1.status_code == 201
        id1 = res1.json()["id"]

        res2 = await client.post("/api/conversations", json={"title": "Chat Beta"})
        assert res2.status_code == 201
        id2 = res2.json()["id"]

        # 2. Pin Chat Alpha
        pin_res = await client.post(f"/api/conversations/{id1}/pin", json={"pinned": True})
        assert pin_res.status_code == 200
        assert pin_res.json()["pinned"] is True

        # 3. List and verify pinned Alpha comes first
        list_res = await client.get("/api/conversations")
        assert list_res.status_code == 200
        items = list_res.json()
        assert items[0]["id"] == id1
        assert items[0]["pinned"] is True
        assert items[1]["id"] == id2
        assert items[1]["pinned"] is False

        # 4. Unpin Chat Alpha
        unpin_res = await client.post(f"/api/conversations/{id1}/pin", json={"pinned": False})
        assert unpin_res.status_code == 200
        assert unpin_res.json()["pinned"] is False

        # 5. Pin nonexistent conversation returns 404
        not_found = await client.post("/api/conversations/missing-xyz/pin", json={"pinned": True})
        assert not_found.status_code == 404


@pytest.mark.asyncio
async def test_conversation_detail_reports_pinned_state():
    """M-01: GET detail returns the stored pinned flag, not the model default."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        created = await client.post("/api/conversations", json={"title": "Pinned Detail"})
        conv_id = created.json()["id"]

        await client.post(f"/api/conversations/{conv_id}/pin", json={"pinned": True})
        detail = await client.get(f"/api/conversations/{conv_id}")
        assert detail.status_code == 200
        assert detail.json()["pinned"] is True

        await client.post(f"/api/conversations/{conv_id}/pin", json={"pinned": False})
        detail = await client.get(f"/api/conversations/{conv_id}")
        assert detail.json()["pinned"] is False


@pytest.mark.asyncio
async def test_conversation_title_is_bounded_on_create_and_update():
    """M-02: the same 100-char bound applies to create and update."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        oversized = "x" * 10_000
        create = await client.post("/api/conversations", json={"title": oversized})
        assert create.status_code == 422

        created = await client.post("/api/conversations", json={"title": "ok"})
        conv_id = created.json()["id"]
        update = await client.patch(
            f"/api/conversations/{conv_id}", json={"title": oversized}
        )
        assert update.status_code == 422


@pytest.mark.asyncio
async def test_conversation_archive_endpoint():
    """M-16: PATCH archived works end-to-end and hides the row from default lists."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        created = await client.post("/api/conversations", json={"title": "Archive Me"})
        conv_id = created.json()["id"]

        archived = await client.patch(
            f"/api/conversations/{conv_id}", json={"archived": True}
        )
        assert archived.status_code == 200
        assert archived.json()["archived"] is True

        default_list = await client.get("/api/conversations")
        assert all(c["id"] != conv_id for c in default_list.json())

        all_list = await client.get("/api/conversations?include_archived=true")
        assert any(c["id"] == conv_id for c in all_list.json())

        restore = await client.patch(
            f"/api/conversations/{conv_id}", json={"archived": False}
        )
        assert restore.status_code == 200
        assert restore.json()["archived"] is False

        default_list = await client.get("/api/conversations")
        assert any(c["id"] == conv_id for c in default_list.json())


@pytest.mark.asyncio
async def test_patch_conversation_not_found_returns_404():
    """L-03: PATCH on nonexistent conversation returns 404, never a partial fallback."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.patch("/api/conversations/nonexistent-id", json={"title": "new title"})
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Conversation not found"
