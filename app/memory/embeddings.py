# SPDX-License-Identifier: Apache-2.0
import asyncio
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


def vector_to_blob(vec: Sequence[float] | np.ndarray) -> bytes:
    """Convert a 1D float vector to little-endian float32 bytes for SQLite BLOB storage."""
    return np.asarray(vec, dtype="<f4").tobytes()


def blob_to_vector(blob: bytes) -> np.ndarray:
    """Load a little-endian float32 BLOB from SQLite into a 1D numpy array."""
    return np.frombuffer(blob, dtype="<f4")


class EmbeddingService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.model_name = self.settings.embedding_model
        self.dims = self.settings.embedding_dims
        self.cache_dir: Path = self.settings.models_dir / "fastembed"
        self._model: TextEmbedding | None = None
        self._load_attempted: bool = False
        self._available: bool = False

    def is_available(self) -> bool:
        return self._available

    async def initialize(self) -> bool:
        if self._load_attempted:
            return self._available

        self._load_attempted = True
        return await asyncio.to_thread(self._sync_load)

    def _sync_load(self) -> bool:
        # Pre-check whether local weights exist in cache_dir before invoking FastEmbed.
        # Calling TextEmbedding(..., local_files_only=True) when cache is empty causes
        # FastEmbed's internal loguru logger to print alarming ERROR traces directly to stderr.
        has_local_weights = self.cache_dir.is_dir() and any(
            f.is_file() and (f.suffix in {".onnx", ".tar.gz", ".bin", ".json"})
            for f in self.cache_dir.rglob("*")
        )
        if not has_local_weights:
            logger.info(
                "FastEmbed embedding model '%s' not cached in %s. Using keyword search for memory retrieval.",
                self.model_name,
                self.cache_dir,
            )
            self._model = None
            self._available = False
            return False

        try:
            logger.info(
                "Loading FastEmbed embedding model '%s' from %s (local files only)...",
                self.model_name,
                self.cache_dir,
            )
            # Use local_files_only=True so it never attempts to download weights behind the user's back
            model = TextEmbedding(
                model_name=self.model_name,
                cache_dir=str(self.cache_dir),
                local_files_only=True,
            )
            self._model = model
            self._available = True
            logger.info("FastEmbed model '%s' loaded successfully.", self.model_name)
            return True
        except (RuntimeError, ValueError, OSError, TimeoutError) as exc:
            logger.info(
                "FastEmbed model '%s' could not be loaded (%s). Falling back to keyword search.",
                self.model_name,
                exc,
            )
            self._model = None
            self._available = False
            return False

    async def embed_text(self, text: str) -> np.ndarray | None:
        if not self._load_attempted:
            await self.initialize()

        if not self._available or self._model is None:
            return None

        clean_text = text.strip()
        if not clean_text:
            return None

        def _embed() -> np.ndarray:
            assert self._model is not None
            embeddings = list(self._model.embed([clean_text]))
            return np.asarray(embeddings[0], dtype="<f4")

        try:
            return await asyncio.to_thread(_embed)
        except (RuntimeError, ValueError, OSError, TimeoutError) as exc:
            logger.warning("Failed to embed text: %s", exc)
            return None


# Global singleton
_embedding_service: EmbeddingService | None = None


def get_embedding_service(settings: Settings | None = None) -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None or (settings is not None and _embedding_service.settings != settings):
        _embedding_service = EmbeddingService(settings or get_settings())
    return _embedding_service
