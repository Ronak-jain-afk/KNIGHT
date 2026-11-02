# SPDX-License-Identifier: Apache-2.0
import secrets
from collections.abc import Awaitable, Callable
from urllib.parse import urlparse

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.config import Settings, get_settings

# Paths that expose the API surface and must be protected when a token is set.
PROTECTED_SCHEMA_PATHS = frozenset({"/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json"})


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; "
            "connect-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'"
        )
        if "token" in request.query_params or request.url.path in ("/", "/index.html"):
            response.headers["Cache-Control"] = "no-store"
        return response


class _SettingsMixin:
    """Resolve settings lazily so importing the app does not read the environment."""

    def __init__(self, app, settings: Settings | None = None) -> None:
        super().__init__(app)  # type: ignore[call-arg]
        self._settings = settings

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            self._settings = get_settings()
        return self._settings


class OriginCheckMiddleware(_SettingsMixin, BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Modern browsers mark cross-site requests even when an intermediary
        # strips the Origin header (defence in depth for model download/swap).
        fetch_site = request.headers.get("sec-fetch-site", "").strip().lower()
        if fetch_site in ("cross-site", "same-site"):
            return JSONResponse(
                status_code=403,
                content={"detail": "Cross-origin requests are forbidden."},
            )

        origin = request.headers.get("origin")
        if origin:
            parsed = urlparse(origin)
            allowed_hosts = ("127.0.0.1", "localhost")
            effective_port = parsed.port or (443 if parsed.scheme == "https" else 80)
            if (
                parsed.scheme not in ("http", "https")
                or parsed.hostname not in allowed_hosts
                or effective_port != self.settings.port
            ):
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Cross-origin requests are forbidden."},
                )

        return await call_next(request)


class AuthTokenMiddleware(_SettingsMixin, BaseHTTPMiddleware):
    @staticmethod
    def requires_auth(path: str) -> bool:
        return path.startswith("/api/") or path in PROTECTED_SCHEMA_PATHS

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        token = self.settings.auth_token.strip()
        path = request.url.path

        # Public: health probe and the static UI bundle.
        if not token or path == "/api/health" or not self.requires_auth(path):
            return await call_next(request)

        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid authorization header."},
            )

        provided_token = auth_header.removeprefix("Bearer ").strip()
        try:
            is_valid = secrets.compare_digest(
                provided_token.encode("utf-8"),
                token.encode("utf-8"),
            )
        except (TypeError, UnicodeEncodeError):
            is_valid = False

        if not is_valid:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid authorization token."},
            )

        return await call_next(request)
