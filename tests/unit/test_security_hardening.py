# SPDX-License-Identifier: Apache-2.0
import json
import logging

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.logging_setup import RedactingFilter
from app.security.middleware import (
    AuthTokenMiddleware,
    OriginCheckMiddleware,
    SecurityHeadersMiddleware,
)
from app.security.vendor_guard import verify_vendor_manifest


def create_test_app(settings: Settings) -> FastAPI:
    test_app = FastAPI()
    test_app.add_middleware(SecurityHeadersMiddleware)
    test_app.add_middleware(OriginCheckMiddleware, settings=settings)
    test_app.add_middleware(AuthTokenMiddleware, settings=settings)

    @test_app.get("/api/health")
    async def health():
        return {"app": "ok"}

    @test_app.get("/api/data")
    async def data():
        return {"data": "secret"}

    @test_app.get("/")
    async def index():
        return "index"

    return test_app


@pytest.mark.asyncio
async def test_security_headers():
    settings = Settings(auth_token="")
    test_app = create_test_app(settings)

    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://127.0.0.1:8000"
    ) as client:
        resp = await client.get("/api/health")
        assert resp.status_code == 200
        assert resp.headers["X-Frame-Options"] == "DENY"
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert resp.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
        assert "default-src 'self'" in resp.headers["Content-Security-Policy"]


@pytest.mark.asyncio
async def test_origin_check_blocks_disallowed_origin():
    settings = Settings(port=8000, auth_token="")
    test_app = create_test_app(settings)

    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://127.0.0.1:8000"
    ) as client:
        # Cross-origin evil domain
        resp_bad = await client.get(
            "/api/health",
            headers={"Origin": "https://evil-attacker.com"},
        )
        assert resp_bad.status_code == 403
        assert "forbidden" in resp_bad.json()["detail"].lower()

        # Port-less loopback origin (e.g. port 80/443 default) must be rejected when port is 8000 (KN-19)
        resp_portless = await client.get(
            "/api/health",
            headers={"Origin": "http://127.0.0.1"},
        )
        assert resp_portless.status_code == 403

        # Same origin allowed
        resp_ok = await client.get(
            "/api/health",
            headers={"Origin": "http://127.0.0.1:8000"},
        )
        assert resp_ok.status_code == 200


@pytest.mark.asyncio
async def test_auth_token_middleware():
    settings = Settings(auth_token="secret-token-12345")
    test_app = create_test_app(settings)

    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://127.0.0.1:8000"
    ) as client:
        # 1. Public health endpoint allowed without token
        resp_health = await client.get("/api/health")
        assert resp_health.status_code == 200

        # 2. Public index allowed without token
        resp_root = await client.get("/")
        assert resp_root.status_code == 200

        # 3. Protected endpoint without token -> 401
        resp_no_token = await client.get("/api/data")
        assert resp_no_token.status_code == 401

        # 4. Protected endpoint with wrong token -> 401
        resp_wrong = await client.get(
            "/api/data",
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp_wrong.status_code == 401

        # 4b. Protected endpoint with non-ASCII token -> 401 (KN-18)
        resp_non_ascii = await client.get(
            "/api/data",
            headers={"Authorization": b"Bearer token-\xff"},
        )
        assert resp_non_ascii.status_code == 401

        # 5. Protected endpoint with correct token -> 200
        resp_correct = await client.get(
            "/api/data",
            headers={"Authorization": "Bearer secret-token-12345"},
        )
        assert resp_correct.status_code == 200
        assert resp_correct.json() == {"data": "secret"}


@pytest.mark.asyncio
async def test_schema_endpoints_require_auth():
    """C-21: /docs, /openapi.json, and /redoc are part of the protected surface."""
    settings = Settings(auth_token="secret-token-12345")
    test_app = create_test_app(settings)

    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://127.0.0.1:8000"
    ) as client:
        for path in ("/docs", "/openapi.json", "/redoc"):
            response = await client.get(path)
            assert response.status_code == 401, path

        authorized = await client.get(
            "/openapi.json",
            headers={"Authorization": "Bearer secret-token-12345"},
        )
        assert authorized.status_code == 200


def test_token_comparison_is_constant_time():
    """C-22: bearer tokens are compared with secrets.compare_digest."""
    import inspect

    from app.security.middleware import AuthTokenMiddleware

    source = inspect.getsource(AuthTokenMiddleware.dispatch)
    assert "compare_digest" in source
    assert "== token" not in source


def test_redacting_logger():
    filter_ = RedactingFilter()

    record1 = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Request headers: Authorization: Bearer my-secret-token-xyz",
        args=(),
        exc_info=None,
    )
    filter_.filter(record1)
    assert "Bearer [REDACTED]" in record1.msg
    assert "my-secret-token-xyz" not in record1.msg

    record2 = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Calling OpenAI with key sk-abcdef1234567890abcdef",
        args=(),
        exc_info=None,
    )
    filter_.filter(record2)
    assert "[REDACTED]" in record2.msg
    assert "sk-abcdef1234567890abcdef" not in record2.msg


def test_vendor_manifest_integrity_success():
    """L-06: real vendor assets match the recorded manifest digests and versions."""
    verified = verify_vendor_manifest()
    assert "marked.min.js" in verified
    assert "purify.min.js" in verified
    assert verified["marked.min.js"] == "15.0.7"
    assert verified["purify.min.js"] == "3.2.4"


def test_vendor_manifest_detects_tampered_asset(tmp_path):
    """L-06: any modification to a vendor file raises RuntimeError."""
    # Setup a mock vendor dir
    vdir = tmp_path / "vendor"
    vdir.mkdir()
    asset = vdir / "marked.min.js"
    asset.write_text("console.log('tampered');", encoding="utf-8")

    manifest = {
        "marked.min.js": {
            "version": "12.0.2",
            "sha256": "15fabce5b65898b32b03f5ed25e9f891a729ad4c0d6d877110a7744aa847a894",
        }
    }
    (vdir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="integrity mismatch"):
        verify_vendor_manifest(vdir)
