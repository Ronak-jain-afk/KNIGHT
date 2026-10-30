import pytest
from httpx import ASGITransport, AsyncClient

from app.config import get_settings
from app.db.sqlite import init_checkpoint_db, run_migrations
from app.main import app
from app.memory.repository import get_memory_repo_dep, get_memory_repository
from app.tools.memory_tools import recall, remember


@pytest.fixture(autouse=True)
async def setup_app_db(tmp_path, monkeypatch):
    monkeypatch.setenv("KNIGHT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    settings = get_settings()

    await run_migrations(settings.app_db_path)
    await init_checkpoint_db(settings.checkpoint_db_path)

    # Reset repository singleton for clean state
    import app.memory.repository as mem_repo_module

    mem_repo_module._memory_repository = None
    repo = get_memory_repository(settings)
    await repo.initialize()

    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_memory_repo_dep] = lambda: repo

    yield

    app.dependency_overrides.clear()
    get_settings.cache_clear()
    mem_repo_module._memory_repository = None


@pytest.mark.asyncio
async def test_api_memories_crud():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # 1. Initially empty
        resp1 = await client.get("/api/memories")
        assert resp1.status_code == 200
        assert resp1.json() == []

        # 2. Add memory
        resp2 = await client.post(
            "/api/memories",
            json={
                "text": "User prefers dark mode in editors.",
                "kind": "preference",
                "importance": 4,
            },
        )
        assert resp2.status_code == 201
        assert resp2.json()["status"] == "Saved."

        # 3. List memories
        resp3 = await client.get("/api/memories")
        assert resp3.status_code == 200
        memories = resp3.json()
        assert len(memories) == 1
        assert memories[0]["text"] == "User prefers dark mode in editors."
        assert memories[0]["kind"] == "preference"
        assert memories[0]["importance"] == 4
        mem_id = memories[0]["id"]

        # 4. Reject secret
        resp4 = await client.post(
            "/api/memories",
            json={
                "text": "User API token is sk-1234567890abcdef123456",
                "kind": "fact",
            },
        )
        assert resp4.status_code == 400
        assert "looks like a secret" in resp4.json()["detail"]

        # 4b. Update memory (PATCH)
        resp_patch = await client.patch(
            f"/api/memories/{mem_id}",
            json={"text": "User prefers high-contrast dark mode.", "importance": 5},
        )
        assert resp_patch.status_code == 200
        assert resp_patch.json()["ok"] is True

        resp_check = await client.get("/api/memories")
        assert resp_check.status_code == 200
        updated = resp_check.json()
        assert updated[0]["text"] == "User prefers high-contrast dark mode."
        assert updated[0]["importance"] == 5

        # 4c. Patch non-existent memory -> 404
        resp_bad = await client.patch("/api/memories/nonexistent123", json={"importance": 2})
        assert resp_bad.status_code == 404

        # 5. Delete memory
        resp5 = await client.delete(f"/api/memories/{mem_id}")
        assert resp5.status_code == 200
        assert resp5.json()["ok"] is True

        # 6. Verify deleted
        resp6 = await client.get("/api/memories")
        assert resp6.status_code == 200
        assert len(resp6.json()) == 0


@pytest.mark.asyncio
async def test_remember_and_recall_tools():
    # 1. remember tool
    out1 = remember.invoke(
        {
            "text": "User lives in Vancouver, Canada.",
            "kind": "fact",
            "importance": 3,
        }
    )
    assert out1 == "Saved."

    # 2. duplicate remember
    out2 = remember.invoke(
        {
            "text": "  User lives in Vancouver, Canada! ",
            "kind": "fact",
            "importance": 3,
        }
    )
    assert out2 == "Already known."

    # 3. secret reject
    out3 = remember.invoke(
        {
            "text": "User password is supersecretpassword123",
            "kind": "fact",
        }
    )
    assert out3 == "Skipped: looks like a secret."

    # 4. recall tool
    recall_json = recall.invoke({"query": "Vancouver", "k": 3})
    assert "memories" in recall_json
    assert "Vancouver" in recall_json
