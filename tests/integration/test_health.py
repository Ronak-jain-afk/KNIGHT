# SPDX-License-Identifier: Apache-2.0
import pytest
from httpx import ASGITransport, AsyncClient

from app.api import system as system_module
from app.config import Settings, get_settings
from app.main import app


@pytest.mark.asyncio
async def test_health_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["app"] == "ok"
        assert data["db"] == "ok"
        assert "memory" in data
        assert "llama" in data
        assert data["llama"]["reachable"] is False


@pytest.mark.asyncio
async def test_config_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_name"] == "qwen3.5-4b"
        assert data["context_tokens"] == 16384
        assert data["enable_thinking"] is False
        assert "ddgs" in data["search_providers"]
        assert data["max_message_chars"] == 32_000
        assert data["max_attachment_bytes"] == 2_000_000


@pytest.mark.asyncio
async def test_check_db_health_reports_unopenable_database(tmp_path):
    """M-24: a path that exists but cannot be opened must report an error."""
    broken = tmp_path / "not-a-database"
    broken.mkdir()

    assert await system_module.check_db_health(broken) is False


@pytest.mark.asyncio
async def test_health_endpoint_reports_db_error_when_probe_fails(tmp_path, monkeypatch):
    """M-24: the endpoint reflects the probe result instead of hardcoding "ok"."""
    settings = Settings(data_dir=tmp_path / "data")
    settings.ensure_dirs()
    app.dependency_overrides[get_settings] = lambda: settings

    async def failing_probe(db_path, timeout: float = 2.0) -> bool:
        del db_path, timeout
        return False

    monkeypatch.setattr(system_module, "check_db_health", failing_probe)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/health")
            assert resp.status_code == 200
            assert resp.json()["db"] == "error"
    finally:
        app.dependency_overrides.clear()


def test_windows_shutdown_raises_sigint(monkeypatch):
    """M-25: Windows has no graceful SIGTERM, so the handler raises SIGINT."""
    raised: list[int] = []
    monkeypatch.setattr(system_module.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        system_module.signal, "raise_signal", lambda sig: raised.append(sig)
    )
    monkeypatch.setattr(
        system_module.os,
        "kill",
        lambda pid, sig: pytest.fail("os.kill must not be used on Windows"),
    )

    system_module.request_graceful_shutdown()

    assert raised == [system_module.signal.SIGINT]


def test_posix_shutdown_uses_sigterm(monkeypatch):
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(system_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        system_module.os, "kill", lambda pid, sig: calls.append((pid, sig))
    )

    system_module.request_graceful_shutdown()

    assert calls == [(system_module.os.getpid(), system_module.signal.SIGTERM)]
