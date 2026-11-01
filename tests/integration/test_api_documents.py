# SPDX-License-Identifier: Apache-2.0
import pytest
from httpx import ASGITransport, AsyncClient

from app.api.documents import read_capped_body, sanitize_filename
from app.config import get_settings
from app.db.sqlite import run_migrations
from app.main import app


@pytest.fixture(autouse=True)
async def setup_documents_env(tmp_path, monkeypatch):
    monkeypatch.setenv("KNIGHT_DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    settings = get_settings()
    settings.ensure_dirs()
    await run_migrations(settings.app_db_path)
    app.dependency_overrides[get_settings] = lambda: settings
    yield
    app.dependency_overrides.clear()
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_documents_crud_and_upload():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # 1. Initially empty
        resp1 = await client.get("/api/documents")
        assert resp1.status_code == 200
        assert resp1.json() == []

        # 2. Create document via JSON
        resp2 = await client.post(
            "/api/documents",
            json={
                "filename": "notes.md",
                "content": "# Meeting Notes\n- Discuss local agent features.",
            },
        )
        assert resp2.status_code == 201
        data2 = resp2.json()
        assert data2["status"] == "created"
        assert data2["name"] == "notes.md"

        # 3. List documents
        resp3 = await client.get("/api/documents")
        assert resp3.status_code == 200
        docs = resp3.json()
        assert len(docs) == 1
        assert docs[0]["name"] == "notes.md"
        assert docs[0]["extension"] == ".md"

        # 4. Upload raw document
        resp4 = await client.post(
            "/api/documents/upload?filename=script.py",
            content=b"print('hello world')\n",
        )
        assert resp4.status_code == 201
        assert resp4.json()["name"] == "script.py"

        # 5. Verify 2 documents exist
        resp5 = await client.get("/api/documents")
        assert len(resp5.json()) == 2

        # 6. Delete document
        resp6 = await client.delete("/api/documents/notes.md")
        assert resp6.status_code == 200
        assert resp6.json()["status"] == "deleted"

        # 7. Verify deletion
        resp7 = await client.get("/api/documents")
        assert len(resp7.json()) == 1
        assert resp7.json()[0]["name"] == "script.py"


@pytest.mark.asyncio
async def test_documents_traversal_sanitization():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/api/documents",
            json={
                "filename": "../../evil.txt",
                "content": "malicious content",
            },
        )
        assert resp.status_code == 201
        # Path.name sanitized to evil.txt
        assert resp.json()["name"] == "evil.txt"


@pytest.mark.asyncio
async def test_documents_char_count_is_characters_not_bytes():
    """H-20: multibyte UTF-8 content reports a real character count."""
    content = "# Café 日本語\n"
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        create = await client.post(
            "/api/documents",
            json={"filename": "unicode.txt", "content": content},
        )
        assert create.status_code == 201

        listing = await client.get("/api/documents")

    item = next(d for d in listing.json() if d["name"] == "unicode.txt")
    assert item["char_count"] == len(content)
    assert item["size_bytes"] == len(content.encode("utf-8"))
    assert item["char_count"] < item["size_bytes"]


def test_sanitize_filename_handles_windows_reserved_names():
    """H-21: reserved device names and trailing dots/spaces are made safe."""
    assert sanitize_filename("CON.txt") == "_CON.txt"
    assert sanitize_filename("com1.log") == "_com1.log"
    assert sanitize_filename("name. ") == "name"
    assert sanitize_filename("aux") == "_aux"
    assert sanitize_filename("bad:name?.txt") == "bad_name_.txt"
    assert sanitize_filename("...") == "document.txt"


class _FakeRequest:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.consumed = 0

    async def stream(self):
        for chunk in self.chunks:
            self.consumed += len(chunk)
            yield chunk


@pytest.mark.asyncio
async def test_read_capped_body_aborts_before_buffering_everything():
    """H-19: an oversized upload is rejected as soon as the cap is crossed."""
    request = _FakeRequest([b"a" * 3, b"b" * 3, b"c" * 3])

    assert await read_capped_body(request, max_bytes=5) is None
    assert request.consumed == 6  # the last chunk was never read


@pytest.mark.asyncio
async def test_upload_uses_configured_attachment_limit(monkeypatch):
    """M-04: uploads honour `settings.max_attachment_bytes`, not a hardcoded cap."""
    settings = get_settings()
    monkeypatch.setattr(settings, "max_attachment_bytes", 16)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/api/documents/upload?filename=large.txt",
            content=b"x" * 64,
        )

    assert resp.status_code == 413
    assert "limit" in resp.json()["detail"].lower()

