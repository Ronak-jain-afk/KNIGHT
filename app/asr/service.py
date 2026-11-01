# SPDX-License-Identifier: Apache-2.0
import asyncio
import io
import logging
import struct
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

from app.config import Settings

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = ["en", "de", "fr", "es", "it", "nl", "pl"]

MODEL_FILENAME = "whistle.cact"
MODEL_MIN_BYTES = 15_000_000


class ASRDisabledError(RuntimeError):
    """Raised when ASR is turned off in settings."""


class ASRService:
    """ASR transcription service powered by the Whistle model by Cactus Compute."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._engine_available: bool | None = None

    @property
    def model_path(self) -> Path:
        """Default target path for whistle.cact model file."""
        return self.settings.asr_dir / MODEL_FILENAME

    def _load_needle(self):
        """Dynamic loader for needle module."""
        import needle

        return needle

    def _require_enabled(self) -> None:
        if not self.settings.asr_enabled:
            raise ASRDisabledError(
                "ASR is disabled. Set KNIGHT_ASR_ENABLED=true to enable it."
            )

    def is_engine_installed(self) -> bool:
        """Check if cactus-needle package is importable."""
        try:
            self._load_needle()
            return True
        except ImportError:
            return False

    def is_model_cached(self) -> bool:
        """Check if whistle.cact model file exists locally and has valid size."""
        local_model = self.settings.asr_dir / MODEL_FILENAME
        if local_model.is_file() and local_model.stat().st_size > MODEL_MIN_BYTES:
            return True
        # Check standard needle cache in ~/.cache/needle/
        home_cache = Path.home() / ".cache" / "needle" / MODEL_FILENAME
        return bool(home_cache.is_file() and home_cache.stat().st_size > MODEL_MIN_BYTES)

    def get_model_path(self) -> Path | None:
        """Get the local path to whistle.cact if cached."""
        local_model = self.settings.asr_dir / MODEL_FILENAME
        if local_model.is_file() and local_model.stat().st_size > MODEL_MIN_BYTES:
            return local_model
        home_cache = Path.home() / ".cache" / "needle" / MODEL_FILENAME
        if home_cache.is_file() and home_cache.stat().st_size > MODEL_MIN_BYTES:
            return home_cache
        return None

    def validate_wav(self, audio_bytes: bytes) -> tuple[int, int, float]:
        """Validate WAV audio bytes, ensuring proper RIFF header and duration bounds.

        Returns (num_frames, framerate, duration_sec) or raises ValueError on validation failure.
        """
        if not audio_bytes or len(audio_bytes) < 44:
            raise ValueError("Audio payload too small or empty")

        if not audio_bytes.startswith(b"RIFF"):
            raise ValueError("Invalid audio format: must be standard RIFF/WAV format")

        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
                channels = wav_file.getnchannels()
                framerate = wav_file.getframerate()
                nframes = wav_file.getnframes()
                if framerate != 16000:
                    raise ValueError(f"Unsupported sample rate: {framerate} Hz (expected 16000 Hz)")
                if channels != 1:
                    raise ValueError(f"Unsupported channel count: {channels} (expected 1 channel mono)")

                duration_sec = nframes / float(framerate)
                if duration_sec > self.settings.asr_max_audio_seconds:
                    raise ValueError(
                        f"Audio duration ({duration_sec:.1f}s) exceeds maximum allowed ({self.settings.asr_max_audio_seconds}s)"
                    )

                return nframes, framerate, duration_sec
        except wave.Error as e:
            raise ValueError(f"Failed to parse WAV header: {e}") from e
        except (struct.error, OSError) as e:
            raise ValueError(f"Corrupt WAV data: {e}") from e

    async def transcribe(
        self, audio_bytes: bytes, language: str | None = None
    ) -> dict[str, Any]:
        """Transcribe 16 kHz WAV audio bytes into text using Whistle."""
        self._require_enabled()
        _, _, duration_sec = self.validate_wav(audio_bytes)

        # Normalize and validate language
        lang = (language or self.settings.asr_language or "").strip().lower()
        if lang and lang not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{language}'. Supported: {', '.join(SUPPORTED_LANGUAGES)}"
            )

        if not self.is_model_cached():
            raise RuntimeError(
                f"Whistle model not found. Download it via settings or huggingface_hub: {self.settings.asr_model_repo}"
            )

        if not self.is_engine_installed():
            raise RuntimeError(
                "ASR engine 'cactus-needle' is not installed. Run 'uv pip install cactus-needle' to enable Whistle ASR."
            )

        t0 = time.perf_counter()

        def _do_transcribe() -> dict[str, Any]:
            needle = self._load_needle()

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            try:
                # Transcribe with Whistle via needle
                kwargs: dict[str, Any] = {}
                if lang:
                    kwargs["language"] = lang
                model_path = self.get_model_path()
                if model_path:
                    # Passed explicitly instead of mutating process-wide os.environ (H-13).
                    kwargs["weights"] = str(model_path)

                res = needle.transcribe(tmp_path, **kwargs)
                return res if isinstance(res, dict) else {"text": str(res)}
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        result = await asyncio.to_thread(_do_transcribe)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        raw_text = result.get("text", "")
        detected_lang = result.get("language", lang or "en")
        ttft_ms = result.get("ttft_ms", 0.0)

        return {
            "status": "ok",
            "transcript": raw_text.strip(),
            "text": raw_text.strip(),
            "language": detected_lang,
            "duration_sec": round(duration_sec, 2),
            "latency_ms": elapsed_ms,
            "ttft_ms": ttft_ms,
            "engine": "whistle",
        }

    def download_whistle_model(self) -> Path:
        """Download whistle.cact from Hugging Face into data/models/whistle/."""
        self._require_enabled()
        from huggingface_hub import hf_hub_download

        self.settings.asr_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Downloading Whistle model from %s into %s...",
            self.settings.asr_model_repo,
            self.settings.asr_dir,
        )
        downloaded_path = hf_hub_download(
            repo_id=self.settings.asr_model_repo,
            filename=MODEL_FILENAME,
            local_dir=str(self.settings.asr_dir),
        )
        return Path(downloaded_path)

    def get_status(self) -> dict[str, Any]:
        """Return diagnostic status of Whistle ASR."""
        engine_ok = self.is_engine_installed()
        cached = self.is_model_cached()
        model_path_obj = self.get_model_path()
        model_size = (
            model_path_obj.stat().st_size
            if model_path_obj and model_path_obj.exists()
            else None
        )
        return {
            "enabled": self.settings.asr_enabled,
            "engine_installed": engine_ok,
            "needle_available": engine_ok,
            "model_cached": cached,
            "model_downloaded": cached,
            "model_path": str(model_path_obj) if model_path_obj else None,
            "model_size_bytes": model_size,
            "model_size_mb": (
                round(model_size / (1024 * 1024), 2) if model_size is not None else None
            ),
            "model_repo": self.settings.asr_model_repo,
            "default_language": self.settings.asr_language or "auto",
            "supported_languages": SUPPORTED_LANGUAGES,
            "max_duration_seconds": self.settings.asr_max_audio_seconds,
            "available": self.settings.asr_enabled and engine_ok and cached,
        }
