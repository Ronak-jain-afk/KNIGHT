# SPDX-License-Identifier: Apache-2.0
import io
import struct
import wave
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


def _asr_service():
    service = getattr(app.state, "asr_service", None)
    if service is None:
        from app.asr.service import ASRService
        from app.config import get_settings

        service = ASRService(get_settings())
        app.state.asr_service = service
    return service


def make_test_wav(
    sample_rate: int = 16000,
    num_channels: int = 1,
    duration_sec: float = 1.0,
    sampwidth: int = 2,
) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        num_frames = int(sample_rate * duration_sec)
        data = struct.pack(f"<{num_frames}h", *[0] * num_frames)
        wf.writeframes(data)
    return buf.getvalue()


@pytest.mark.asyncio
async def test_api_asr_status():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/api/asr/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "enabled" in data
        assert "model_repo" in data
        assert "supported_languages" in data
        assert "en" in data["supported_languages"]


@pytest.mark.asyncio
async def test_api_asr_transcribe_invalid_wav():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        files = {"file": ("corrupt.wav", b"not-a-valid-riff-file-data", "audio/wav")}
        resp = await client.post("/api/asr/transcribe", files=files)
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"] == "invalid_audio"
        assert body["request_id"]
        assert "request_id" in body


@pytest.mark.asyncio
async def test_api_asr_transcribe_unsupported_language():
    wav_bytes = make_test_wav(sample_rate=16000, num_channels=1, duration_sec=1.0)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        files = {"file": ("voice.wav", wav_bytes, "audio/wav")}
        data = {"language": "valyrian"}
        resp = await client.post("/api/asr/transcribe", files=files, data=data)
        assert resp.status_code == 400
        assert resp.json()["error"] == "invalid_audio"
        assert "unsupported language" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_api_asr_transcribe_success_mocked():
    wav_bytes = make_test_wav(sample_rate=16000, num_channels=1, duration_sec=2.0)
    asr_service = getattr(app.state, "asr_service", None)
    assert asr_service is not None

    mock_result = {
        "status": "ok",
        "transcript": "Enable thinking mode and summarize the research",
        "duration_sec": 2.0,
        "language": "en",
        "words": [],
    }

    with patch.object(asr_service, "transcribe", new=AsyncMock(return_value=mock_result)):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            files = {"file": ("voice.wav", wav_bytes, "audio/wav")}
            data = {"language": "en"}
            resp = await client.post("/api/asr/transcribe", files=files, data=data)
            assert resp.status_code == 200
            res = resp.json()
            assert res["status"] == "ok"
            assert res["transcript"] == "Enable thinking mode and summarize the research"
            assert res["duration_sec"] == 2.0
            assert res["language"] == "en"


@pytest.mark.asyncio
async def test_api_asr_transcribe_raw_bytes_success_mocked():
    wav_bytes = make_test_wav(sample_rate=16000, num_channels=1, duration_sec=1.5)
    asr_service = getattr(app.state, "asr_service", None)
    assert asr_service is not None

    mock_result = {
        "transcript": "Direct WAV stream transcription",
        "duration_sec": 1.5,
        "language": "en",
        "words": [],
    }

    with patch.object(asr_service, "transcribe", new=AsyncMock(return_value=mock_result)):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/asr/transcribe?language=en",
                content=wav_bytes,
                headers={"Content-Type": "audio/wav"},
            )
            assert resp.status_code == 200
            res = resp.json()
            assert res["transcript"] == "Direct WAV stream transcription"


@pytest.mark.asyncio
async def test_api_asr_download_mocked(tmp_path):
    asr_service = getattr(app.state, "asr_service", None)
    assert asr_service is not None

    dummy_model_file = tmp_path / "whistle.cact"
    dummy_model_file.write_bytes(b"0" * 16_900_000)

    with patch.object(
        asr_service, "download_whistle_model", return_value=dummy_model_file
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/api/asr/download")
            assert resp.status_code == 200
            res = resp.json()
            assert res["status"] == "downloaded"
            assert res["size_bytes"] == 16900000


@pytest.mark.asyncio
async def test_api_asr_disabled_returns_503():
    """H-10: disabling ASR must be enforced at the HTTP boundary."""
    asr_service = _asr_service()
    wav_bytes = make_test_wav(sample_rate=16000, num_channels=1, duration_sec=1.0)

    with patch.object(asr_service.settings, "asr_enabled", False):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/asr/transcribe",
                content=wav_bytes,
                headers={"Content-Type": "audio/wav"},
            )
            download = await client.post("/api/asr/download")

    assert resp.status_code == 503
    assert resp.json()["error"] == "asr_disabled"
    assert download.status_code == 503
    assert download.json()["error"] == "asr_disabled"


@pytest.mark.asyncio
async def test_api_asr_internal_error_does_not_leak_exception_text():
    """H-14: unexpected decoder failures return a stable code plus request ID."""
    asr_service = _asr_service()
    wav_bytes = make_test_wav(sample_rate=16000, num_channels=1, duration_sec=1.0)

    secret = "/home/ronak/.cache/needle/whistle.cact exploded"
    with patch.object(
        asr_service, "transcribe", new=AsyncMock(side_effect=Exception(secret))
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/asr/transcribe",
                content=wav_bytes,
                headers={"Content-Type": "audio/wav"},
            )

    assert resp.status_code == 500
    body = resp.json()
    assert body["error"] == "asr_failed"
    assert body["request_id"]
    assert ".cache/needle" not in resp.text


@pytest.mark.asyncio
async def test_api_asr_oversized_multipart_rejected_early():
    """KN-07: oversized ASR multipart payload is capped early without unbounded spooling."""
    oversized_data = b"RIFF" + b"\x00" * (10 * 1024 * 1024 + 100_000)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        files = {"file": ("large.wav", oversized_data, "audio/wav")}
        resp = await client.post("/api/asr/transcribe", files=files)
        assert resp.status_code == 413
        body = resp.json()
        assert body["error"] == "audio_too_large"

