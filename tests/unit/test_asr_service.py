import io
import os
import struct
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.asr.service import ASRDisabledError, ASRService
from app.config import Settings


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


@pytest.fixture
def asr_settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        asr_enabled=True,
        asr_max_audio_seconds=60,
    )


@pytest.fixture
def asr_service(asr_settings: Settings) -> ASRService:
    return ASRService(asr_settings)


def test_validate_wav_valid(asr_service: ASRService):
    wav_bytes = make_test_wav(sample_rate=16000, num_channels=1, duration_sec=2.0)
    frames, rate, duration = asr_service.validate_wav(wav_bytes)
    assert rate == 16000
    assert frames == 32000
    assert duration == pytest.approx(2.0, rel=1e-2)


def test_validate_wav_too_short(asr_service: ASRService):
    with pytest.raises(ValueError, match="too small"):
        asr_service.validate_wav(b"RIFF1234")


def test_validate_wav_invalid_header(asr_service: ASRService):
    with pytest.raises(ValueError, match="RIFF"):
        asr_service.validate_wav(b"NOT_A_WAV_FILE_HEADER_123456789012345678901234567890")


def test_validate_wav_invalid_channels(asr_service: ASRService):
    stereo_wav = make_test_wav(sample_rate=16000, num_channels=2, duration_sec=1.0)
    with pytest.raises(ValueError, match="mono"):
        asr_service.validate_wav(stereo_wav)


def test_validate_wav_invalid_rate(asr_service: ASRService):
    wav_44k = make_test_wav(sample_rate=44100, num_channels=1, duration_sec=1.0)
    with pytest.raises(ValueError, match="16000 Hz"):
        asr_service.validate_wav(wav_44k)


def test_validate_wav_exceeds_max_duration(asr_service: ASRService):
    wav_long = make_test_wav(sample_rate=16000, num_channels=1, duration_sec=65.0)
    with pytest.raises(ValueError, match="exceeds maximum"):
        asr_service.validate_wav(wav_long)


def test_get_status(asr_service: ASRService):
    status = asr_service.get_status()
    assert status["enabled"] is True
    assert status["model_repo"] == "Cactus-Compute/whistle"
    assert status["model_downloaded"] is False
    assert "en" in status["supported_languages"]
    assert "fr" in status["supported_languages"]


def test_status_reports_unknown_size_instead_of_fabricated_169(asr_service: ASRService):
    """H-11: no hardcoded 16.9 MB metadata when the model is not present."""
    status = asr_service.get_status()

    assert status["model_size_bytes"] is None
    assert status["model_size_mb"] is None
    assert "16.9" not in repr(status)


def test_status_when_model_file_exists(asr_service: ASRService):
    model_file = asr_service.model_path
    model_file.parent.mkdir(parents=True, exist_ok=True)
    # Write dummy file > 15 MB
    model_file.write_bytes(b"0" * (16 * 1024 * 1024))

    status = asr_service.get_status()
    assert status["model_downloaded"] is True
    assert status["model_size_bytes"] == 16 * 1024 * 1024
    assert status["model_size_mb"] == 16.0


@pytest.mark.asyncio
async def test_transcribe_missing_model_raises(asr_service: ASRService):
    wav_bytes = make_test_wav(sample_rate=16000, num_channels=1, duration_sec=1.0)
    with pytest.raises(RuntimeError, match="Whistle model not found"):
        await asr_service.transcribe(wav_bytes)


@pytest.mark.asyncio
async def test_transcribe_invalid_language_raises(asr_service: ASRService):
    wav_bytes = make_test_wav(sample_rate=16000, num_channels=1, duration_sec=1.0)
    with pytest.raises(ValueError, match="Unsupported language"):
        await asr_service.transcribe(wav_bytes, language="klingon")


@pytest.mark.asyncio
async def test_transcribe_success_with_mock_needle(asr_service: ASRService):
    # Fake model exists
    asr_service.model_path.parent.mkdir(parents=True, exist_ok=True)
    asr_service.model_path.write_bytes(b"0" * (16 * 1024 * 1024))

    wav_bytes = make_test_wav(sample_rate=16000, num_channels=1, duration_sec=1.5)

    mock_needle = MagicMock()
    mock_needle.transcribe.return_value = {
        "text": "Hello world from Cactus Whistle",
        "language": "en",
        "duration": 1.5,
    }

    with patch.object(asr_service, "_load_needle", return_value=mock_needle):
        result = await asr_service.transcribe(wav_bytes, language="en")
        assert result["transcript"] == "Hello world from Cactus Whistle"
        assert result["language"] == "en"
        assert result["duration_sec"] == pytest.approx(1.5, rel=1e-2)
        mock_needle.transcribe.assert_called_once()
        _, call_kwargs = mock_needle.transcribe.call_args
        assert "weights" in call_kwargs
        assert "model" not in call_kwargs


@pytest.mark.asyncio
async def test_disabled_asr_refuses_transcription_and_download(tmp_path: Path):
    """H-10: KNIGHT_ASR_ENABLED=false must actually disable the service."""
    settings = Settings(data_dir=tmp_path / "data", asr_enabled=False)
    service = ASRService(settings)
    wav_bytes = make_test_wav(sample_rate=16000, num_channels=1, duration_sec=1.0)

    with pytest.raises(ASRDisabledError):
        await service.transcribe(wav_bytes)

    with pytest.raises(ASRDisabledError):
        service.download_whistle_model()

    status = service.get_status()
    assert status["enabled"] is False
    assert status["available"] is False


@pytest.mark.asyncio
async def test_transcribe_does_not_mutate_process_environment(asr_service: ASRService):
    """H-13: weights are passed per call, never via os.environ."""
    asr_service.model_path.parent.mkdir(parents=True, exist_ok=True)
    asr_service.model_path.write_bytes(b"0" * (16 * 1024 * 1024))
    wav_bytes = make_test_wav(sample_rate=16000, num_channels=1, duration_sec=1.0)

    mock_needle = MagicMock()
    mock_needle.transcribe.return_value = {"text": "hello", "language": "en"}
    before = os.environ.copy()

    with patch.object(asr_service, "_load_needle", return_value=mock_needle):
        await asr_service.transcribe(wav_bytes, language="en")

    assert os.environ == before
    _, kwargs = mock_needle.transcribe.call_args
    assert kwargs["weights"] == str(asr_service.model_path)


def test_download_does_not_use_removed_symlink_argument(asr_service: ASRService, tmp_path: Path):
    """H-12: huggingface_hub no longer accepts local_dir_use_symlinks."""
    downloaded = tmp_path / "whistle.cact"
    downloaded.write_bytes(b"0" * 16)

    with patch("huggingface_hub.hf_hub_download", return_value=str(downloaded)) as fake:
        asr_service.download_whistle_model()

    _, kwargs = fake.call_args
    assert "local_dir_use_symlinks" not in kwargs

