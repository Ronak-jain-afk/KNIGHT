# SPDX-License-Identifier: Apache-2.0
import asyncio
import logging
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request, status
from fastapi.responses import JSONResponse

from app.asr.service import ASRDisabledError, ASRService
from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/asr", tags=["asr"])

MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10 MB limit


def get_asr_service(
    request: Request, settings: Annotated[Settings, Depends(get_settings)]
) -> ASRService:
    """Retrieve or create ASRService from app.state."""
    if hasattr(request.app.state, "asr_service") and request.app.state.asr_service:
        return request.app.state.asr_service
    service = ASRService(settings)
    request.app.state.asr_service = service
    return service


def _error(
    status_code: int,
    code: str,
    request_id: str,
    detail: str | None = None,
) -> JSONResponse:
    """Return a stable error envelope without leaking internal exception text."""
    content: dict[str, Any] = {"error": code, "request_id": request_id}
    if detail:
        content["detail"] = detail
    return JSONResponse(
        status_code=status_code,
        content=content,
        headers={"X-Request-ID": request_id},
    )


async def _read_capped_request(request: Request, max_bytes: int) -> bytes | None:
    """Read a raw request body in chunks, returning ``None`` once over the cap (H-19)."""
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > max_bytes:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


@router.get("/status")
async def get_asr_status(
    service: Annotated[ASRService, Depends(get_asr_service)],
) -> dict:
    """Return on-device Whistle ASR capability and caching status."""
    return service.get_status()


@router.post("/transcribe")
async def transcribe_audio(
    request: Request,
    service: Annotated[ASRService, Depends(get_asr_service)],
    language: Annotated[
        str | None, Query(description="Optional language code (e.g. 'en')")
    ] = None,
) -> Any:
    """Transcribe an uploaded WAV audio file into text."""
    request_id = uuid.uuid4().hex

    if not service.settings.asr_enabled:
        return _error(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "asr_disabled",
            request_id,
            "ASR is disabled in this configuration.",
        )

    content_type = request.headers.get("content-type", "").lower()
    audio_bytes: bytes = b""
    selected_lang = language

    # Check Content-Length upfront before reading or parsing body (KN-07)
    content_length_header = request.headers.get("content-length")
    if content_length_header:
        try:
            cl = int(content_length_header)
            if cl > MAX_AUDIO_BYTES + 65536:
                return _error(
                    status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    "audio_too_large",
                    request_id,
                    f"Audio file exceeds the {MAX_AUDIO_BYTES // 1024 // 1024} MB limit.",
                )
        except (ValueError, TypeError):
            pass

    try:
        body = await _read_capped_request(request, MAX_AUDIO_BYTES + 65536)
        if body is None:
            return _error(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                "audio_too_large",
                request_id,
                f"Audio file exceeds the {MAX_AUDIO_BYTES // 1024 // 1024} MB limit.",
            )

        if "multipart/form-data" in content_type:
            # Replay the safely capped body for the multipart parser
            request._body = body
            form = await request.form()
            file_item = form.get("file")
            if file_item is None or not hasattr(file_item, "read"):
                return _error(
                    status.HTTP_400_BAD_REQUEST,
                    "missing_audio",
                    request_id,
                    "Expected a multipart 'file' field.",
                )
            audio_bytes = await file_item.read(MAX_AUDIO_BYTES + 1)
            if not selected_lang and "language" in form:
                selected_lang = str(form.get("language"))
        else:
            audio_bytes = body
    except Exception:
        logger.exception("Failed to read ASR request body (request_id=%s)", request_id)
        return _error(
            status.HTTP_400_BAD_REQUEST, "invalid_request", request_id
        )

    if len(audio_bytes) > MAX_AUDIO_BYTES:
        return _error(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            "audio_too_large",
            request_id,
            f"Audio file exceeds the {MAX_AUDIO_BYTES // 1024 // 1024} MB limit.",
        )

    if not audio_bytes:
        return _error(
            status.HTTP_400_BAD_REQUEST, "missing_audio", request_id, "No audio data received."
        )

    try:
        return await service.transcribe(audio_bytes, language=selected_lang)
    except ValueError as exc:
        # Validation messages are produced by us and contain no internal state.
        return _error(
            status.HTTP_400_BAD_REQUEST,
            "invalid_audio",
            request_id,
            str(exc),
        )
    except ASRDisabledError:
        return _error(status.HTTP_503_SERVICE_UNAVAILABLE, "asr_disabled", request_id)
    except RuntimeError as exc:
        logger.warning("ASR unavailable (request_id=%s): %s", request_id, exc)
        return _error(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "asr_unavailable",
            request_id,
            "ASR is not available right now.",
        )
    except Exception:
        logger.exception("Unexpected error during ASR transcription (request_id=%s)", request_id)
        return _error(status.HTTP_500_INTERNAL_SERVER_ERROR, "asr_failed", request_id)


@router.post("/download")
async def download_model(
    service: Annotated[ASRService, Depends(get_asr_service)],
) -> Any:
    """Download the whistle.cact model from Hugging Face."""
    request_id = uuid.uuid4().hex

    if not service.settings.asr_enabled:
        return _error(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "asr_disabled",
            request_id,
            "ASR is disabled in this configuration.",
        )

    try:
        path = await asyncio.to_thread(service.download_whistle_model)
        size_bytes = path.stat().st_size if path.exists() else None
        return {
            "status": "downloaded",
            "ok": True,
            "message": "Whistle model downloaded successfully",
            "path": str(path),
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / (1024 * 1024), 2) if size_bytes else None,
        }
    except Exception:
        logger.exception("Failed to download Whistle model (request_id=%s)", request_id)
        return _error(
            status.HTTP_502_BAD_GATEWAY,
            "download_failed",
            request_id,
            "Failed to download the ASR model.",
        )
