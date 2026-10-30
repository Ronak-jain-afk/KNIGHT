import asyncio
import socket

import pytest
import respx
from httpx import ASGITransport, AsyncClient

from app.config import Settings, get_settings
from app.db.sqlite import run_migrations
from app.main import app


def _fake_getaddrinfo(*args, **kwargs):
    """Resolve every hostname to a fixed public address (offline test runs)."""
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]


@pytest.fixture(autouse=True)
async def setup_db(tmp_path):
    settings = Settings(data_dir=tmp_path / "data")
    settings.ensure_dirs()
    await run_migrations(settings.app_db_path)
    app.dependency_overrides[get_settings] = lambda: settings
    yield settings
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_search_models_curated_and_remote():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # 1. Empty query returns curated trending models
        res = await client.get("/api/ggufbook/search")
        assert res.status_code == 200
        curated = res.json()
        assert len(curated) >= 4
        assert any("Qwen" in m["name"] for m in curated)

        # 2. Remote search query with respx mock
        with respx.mock:
            respx.get("https://huggingface.co/api/models").respond(
                status_code=200,
                json=[
                    {
                        "id": "bartowski/Custom-Test-GGUF",
                        "downloads": 50000,
                        "likes": 120,
                        "tags": ["gguf", "llama"],
                    }
                ],
            )
            res2 = await client.get("/api/ggufbook/search?q=custom")
            assert res2.status_code == 200
            data2 = res2.json()
            assert len(data2) == 1
            assert data2[0]["id"] == "bartowski/Custom-Test-GGUF"


@pytest.mark.asyncio
async def test_get_model_details_and_hardware_fit():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        with respx.mock:
            respx.get(
                "https://huggingface.co/api/models/test-org/test-model-GGUF"
            ).respond(
                status_code=200,
                json={
                    "id": "test-org/test-model-GGUF",
                    "downloads": 12000,
                    "likes": 45,
                    "siblings": [
                        {"rfilename": "README.md", "size": 1200},
                        {
                            "rfilename": "test-model-Q4_K_M.gguf",
                            "lfs": {"size": 2_100_000_000},
                        },
                        {
                            "rfilename": "test-model-Q8_0.gguf",
                            "lfs": {"size": 4_200_000_000},
                        },
                    ],
                },
            )

            res = await client.get(
                "/api/ggufbook/models/test-org/test-model-GGUF"
            )
            assert res.status_code == 200
            data = res.json()
            assert data["name"] == "test-model-GGUF"
            assert len(data["variants"]) == 2

            v1 = data["variants"][0]
            assert v1["filename"] == "test-model-Q4_K_M.gguf"
            assert v1["quant_type"] == "Q4_K_M"
            assert "hardware_fit" in v1
            assert "badge" in v1["hardware_fit"]


@pytest.mark.asyncio
async def test_download_lifecycle_and_validation(setup_db, monkeypatch):
    settings = setup_db
    monkeypatch.setattr(
        "app.security.net_guard.socket.getaddrinfo", _fake_getaddrinfo
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # 1. Validation: non-gguf file
        r1 = await client.post(
            "/api/ggufbook/download",
            json={
                "repo_id": "test-org/repo",
                "filename": "model.bin",
                "download_url": "https://huggingface.co/test",
            },
        )
        assert r1.status_code == 400

        # 2. Validation: path traversal
        r2 = await client.post(
            "/api/ggufbook/download",
            json={
                "repo_id": "test-org/repo",
                "filename": "../secret.gguf",
                "download_url": "https://huggingface.co/test",
            },
        )
        assert r2.status_code == 400

        # 3. Successful streaming download mock
        with respx.mock:
            download_url = (
                "https://huggingface.co/test-org/repo/resolve/main/test-q4.gguf"
            )
            respx.get(download_url).respond(
                status_code=200,
                headers={"content-length": "1024"},
                content=b"GGUF_TEST_BYTES_" * 64,
            )

            r3 = await client.post(
                "/api/ggufbook/download",
                json={
                    "repo_id": "test-org/repo",
                    "filename": "test-q4.gguf",
                    "download_url": download_url,
                    "size_bytes": 1024,
                },
            )
            assert r3.status_code == 202
            task = r3.json()
            task_id = task["id"]
            assert task["filename"] == "test-q4.gguf"

            # Wait briefly for worker to complete
            for _ in range(30):
                await asyncio.sleep(0.1)
                poll = await client.get("/api/ggufbook/downloads")
                tasks = poll.json()
                matching = [t for t in tasks if t["id"] == task_id]
                if matching and matching[0]["status"] in (
                    "completed",
                    "failed",
                ):
                    break

            poll = await client.get("/api/ggufbook/downloads")
            tasks = poll.json()
            assert any(
                t["id"] == task_id and t["status"] == "completed" for t in tasks
            )

            # Target file exists
            assert (settings.models_dir / "test-q4.gguf").exists()

            # 4. Downloading existing file returns already_exists
            r4 = await client.post(
                "/api/ggufbook/download",
                json={
                    "repo_id": "test-org/repo",
                    "filename": "test-q4.gguf",
                    "download_url": download_url,
                },
            )
            assert r4.json()["status"] == "already_exists"


@pytest.mark.asyncio
async def test_download_rejects_cross_site_request(setup_db):
    """M-26: a cross-site POST cannot start a model download (default config)."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        res = await client.post(
            "/api/ggufbook/download",
            json={
                "repo_id": "test-org/repo",
                "filename": "evil.gguf",
                "download_url": "https://huggingface.co/test-org/repo/resolve/main/evil.gguf",
            },
            headers={"Sec-Fetch-Site": "cross-site"},
        )

    assert res.status_code == 403
    assert "Cross-origin" in res.json()["detail"]


@pytest.mark.asyncio
async def test_download_rejects_ssrf_urls(setup_db):
    """C-07: loopback / non-allowlisted hosts must be rejected before any connection."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        for index, url in enumerate(
            [
                "http://127.0.0.1:8080/x.gguf",
                "http://[::1]/x.gguf",
                "https://127.0.0.1/x.gguf",
                "https://evil.example/x.gguf",
                "https://huggingface.co.evil.example/x.gguf",
            ]
        ):
            res = await client.post(
                "/api/ggufbook/download",
                json={
                    "repo_id": "test-org/repo",
                    "filename": f"ssrf-{index}.gguf",
                    "download_url": url,
                },
            )
            assert res.status_code == 400, url
            assert "Invalid download_url" in res.json()["detail"]

