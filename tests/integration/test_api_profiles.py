# SPDX-License-Identifier: Apache-2.0
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
async def test_profiles_api_lifecycle():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. List profiles
        res = await client.get("/api/profiles")
        assert res.status_code == 200
        items = res.json()
        assert len(items) >= 4
        ids = [p["id"] for p in items]
        assert "general" in ids
        assert "software-engineer" in ids

        # 2. Get specific built-in profile
        get_res = await client.get("/api/profiles/software-engineer")
        assert get_res.status_code == 200
        data = get_res.json()
        assert data["name"] == "Software Engineer"
        assert data["is_builtin"] is True

        # 3. Cannot delete built-in profile
        del_builtin = await client.delete("/api/profiles/software-engineer")
        assert del_builtin.status_code == 400

        # 4. Create custom profile
        create_res = await client.post(
            "/api/profiles",
            json={
                "name": "Documentation Writer",
                "icon": "📚",
                "description": "Writes clear, user-centric technical documentation.",
                "system_prompt": "You are a professional technical writer. Produce clean Markdown documentation.",
                "mode": "chat",
            },
        )
        assert create_res.status_code == 201
        custom = create_res.json()
        assert custom["name"] == "Documentation Writer"
        assert custom["icon"] == "📚"
        assert custom["is_builtin"] is False
        custom_id = custom["id"]

        # 5. Patch custom profile
        patch_res = await client.patch(
            f"/api/profiles/{custom_id}",
            json={"name": "Lead Technical Writer", "mode": "agent"},
        )
        assert patch_res.status_code == 200
        assert patch_res.json()["name"] == "Lead Technical Writer"
        assert patch_res.json()["mode"] == "agent"

        # 6. Delete custom profile
        del_res = await client.delete(f"/api/profiles/{custom_id}")
        assert del_res.status_code == 200
        assert del_res.json()["status"] == "deleted"

        # 7. 404 on deleted profile
        get_deleted = await client.get(f"/api/profiles/{custom_id}")
        assert get_deleted.status_code == 404
