from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import get_settings
from app.main import app


@pytest.mark.asyncio
async def test_get_models_list():
    settings = get_settings()
    models_dir = settings.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    dummy_model = models_dir / "test-model-v1.gguf"
    dummy_model.write_bytes(b"dummy gguf content")

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/models")
            assert resp.status_code == 200
            data = resp.json()
            assert "models" in data
            assert "active_model" in data
            names = [m["name"] for m in data["models"]]
            assert "test-model-v1.gguf" in names
    finally:
        if dummy_model.exists():
            dummy_model.unlink()


@pytest.mark.asyncio
async def test_switch_model_validation():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Empty model name
        resp = await client.post("/api/models/switch", json={"model_name": "  "})
        assert resp.status_code == 400

        # Path traversal
        resp = await client.post(
            "/api/models/switch", json={"model_name": "../secret.gguf"}
        )
        assert resp.status_code == 400
        assert "path traversal" in resp.json()["detail"].lower()

        # Non-existent file
        resp = await client.post(
            "/api/models/switch", json={"model_name": "non_existent_file_xyz.gguf"}
        )
        assert resp.status_code == 400
        assert "does not exist" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_switch_model_success_mocked():
    settings = get_settings()
    dummy_model = settings.models_dir / "test-swap.gguf"
    dummy_model.write_bytes(b"dummy gguf content")

    try:
        with patch("app.llm.manager.ModelManager.switch_model", new_callable=AsyncMock) as mock_switch:
            mock_switch.return_value = {
                "ok": True,
                "model_name": "test-swap.gguf",
                "message": "Switched successfully",
            }
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/models/switch", json={"model_name": "test-swap.gguf"}
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["ok"] is True
                assert data["model_name"] == "test-swap.gguf"
    finally:
        if dummy_model.exists():
            dummy_model.unlink()


@pytest.mark.asyncio
async def test_hardware_telemetry_endpoint():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/api/system/hardware")
        assert resp.status_code == 200
        data = resp.json()
        assert "gpu" in data
        assert "ram" in data
        assert "context" in data
        assert "server" in data
        assert "max_tokens" in data["context"]
