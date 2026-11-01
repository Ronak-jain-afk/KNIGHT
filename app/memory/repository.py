# SPDX-License-Identifier: Apache-2.0
import asyncio
import logging
import math
import re
import uuid
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

import numpy as np
from fastapi import Depends

from app.config import Settings, get_settings
from app.db import memories as db_memories
from app.db.memories import hash_memory_text
from app.memory.embeddings import (
    EmbeddingService,
    blob_to_vector,
    get_embedding_service,
    vector_to_blob,
)

logger = logging.getLogger(__name__)

# Anchored token shapes only: substring matches flagged ordinary prose such as
# "task-based" or "risk-averse" and silently dropped legitimate memories (H-15).
_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{36}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b"),
)
_PASSWORD_PATTERN = re.compile(r"\bpassword\b\s*(?:is|[:=])\s*\S+", re.IGNORECASE)
_HEX_SECRET_PATTERN = re.compile(r"\b[0-9a-fA-F]{32,}\b")
_BASE64_SECRET_PATTERN = re.compile(r"\b[A-Za-z0-9+/]{32,}={0,2}\b")


def looks_like_secret(text: str) -> bool:
    """Heuristic check to reject strings that look like API keys, tokens, or credentials."""
    if any(pattern.search(text) for pattern in _SECRET_PATTERNS):
        return True
    if _PASSWORD_PATTERN.search(text):
        return True
    # 32+ character hex strings
    if _HEX_SECRET_PATTERN.search(text):
        return True
    # 32+ character base64-like blobs
    return bool(_BASE64_SECRET_PATTERN.search(text))


class MemoryItem:
    def __init__(
        self,
        id: str,
        text: str,
        kind: str,
        importance: int,
        content_hash: str,
        created_at: str,
        updated_at: str,
        last_used_at: str | None,
        use_count: int,
        embedding: np.ndarray | None = None,
    ) -> None:
        self.id = id
        self.text = text
        self.kind = kind
        self.importance = importance
        self.content_hash = content_hash
        self.created_at = created_at
        self.updated_at = updated_at
        self.last_used_at = last_used_at
        self.use_count = use_count
        self.embedding = embedding

    @property
    def updated_datetime(self) -> datetime:
        try:
            return datetime.fromisoformat(self.updated_at)
        except (ValueError, TypeError):
            return datetime.now(UTC)

    @property
    def last_used_datetime(self) -> datetime | None:
        if not self.last_used_at:
            return None
        try:
            return datetime.fromisoformat(self.last_used_at)
        except (ValueError, TypeError):
            return None


class MemoryRepository:
    def __init__(
        self,
        settings: Settings | None = None,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedding_service = embedding_service or get_embedding_service(self.settings)
        self._items: dict[str, MemoryItem] = {}
        self._matrix: np.ndarray | None = None
        self._matrix_ids: list[str] = []
        self._initialized: bool = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        async with self._lock:
            if self._initialized:
                return

            # Initialize embedding service
            await self.embedding_service.initialize()

            rows = await db_memories.load_all_memories(self.settings.app_db_path)
            self._items.clear()
            for r in rows:
                vec = blob_to_vector(r["embedding"]) if r["embedding"] else None
                item = MemoryItem(
                    id=r["id"],
                    text=r["text"],
                    kind=r["kind"],
                    importance=r["importance"],
                    content_hash=r["content_hash"],
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                    last_used_at=r["last_used_at"],
                    use_count=r["use_count"],
                    embedding=vec,
                )
                self._items[item.id] = item

            self._rebuild_matrix()
            self._initialized = True
            logger.info("MemoryRepository loaded %d memories.", len(self._items))

    def _rebuild_matrix(self) -> None:
        items_with_emb = [item for item in self._items.values() if item.embedding is not None]
        if not items_with_emb:
            self._matrix = None
            self._matrix_ids = []
            return

        self._matrix_ids = [item.id for item in items_with_emb]
        self._matrix = np.vstack([item.embedding for item in items_with_emb])

    async def add_memory(
        self,
        text: str,
        kind: str = "fact",
        importance: int = 3,
        source_thread_id: str | None = None,
    ) -> str:
        if not self._initialized:
            await self.initialize()

        # 1. Secret check
        if looks_like_secret(text):
            return "Skipped: looks like a secret."

        clean_text = text.strip()
        if not clean_text:
            return "Skipped: empty text."

        importance = max(1, min(5, importance))
        content_hash = hash_memory_text(clean_text)

        async with self._lock:
            # 2. Exact duplicate check by content_hash
            existing = next(
                (item for item in self._items.values() if item.content_hash == content_hash),
                None,
            )
            if existing:
                higher_importance = max(existing.importance, importance)
                await db_memories.update_memory(
                    self.settings.app_db_path,
                    existing.id,
                    text=existing.text,
                    importance=higher_importance,
                )
                existing.importance = higher_importance
                existing.updated_at = datetime.now(UTC).isoformat()
                return "Already known."

            # 3. Semantic similarity check (dedup if cosine >= 0.90)
            emb = await self.embedding_service.embed_text(clean_text)

            if emb is not None:
                # Compare against same-kind items with embeddings
                same_kind = [
                    item
                    for item in self._items.values()
                    if item.kind == kind and item.embedding is not None
                ]
                if same_kind:
                    sub_matrix = np.vstack([item.embedding for item in same_kind])
                    dot = np.dot(sub_matrix, emb)
                    norms = np.linalg.norm(sub_matrix, axis=1) * np.linalg.norm(emb)
                    cosines = np.where(norms > 0, dot / norms, 0.0)

                    max_idx = int(np.argmax(cosines))
                    if cosines[max_idx] >= 0.90:
                        match = same_kind[max_idx]
                        merged_text = clean_text if len(clean_text) > len(match.text) else match.text
                        merged_importance = max(match.importance, importance)
                        merged_emb = emb if merged_text == clean_text else match.embedding
                        merged_blob = (
                            vector_to_blob(merged_emb) if merged_emb is not None else None
                        )

                        await db_memories.update_memory(
                            self.settings.app_db_path,
                            match.id,
                            text=merged_text,
                            importance=merged_importance,
                            embedding=merged_blob,
                            embedding_model=self.settings.embedding_model,
                        )
                        match.text = merged_text
                        match.importance = merged_importance
                        match.embedding = merged_emb
                        match.content_hash = hash_memory_text(merged_text)
                        match.updated_at = datetime.now(UTC).isoformat()
                        self._rebuild_matrix()
                        return "Already known."

            # 4. Insert new memory
            memory_id = uuid.uuid4().hex
            blob = vector_to_blob(emb) if emb is not None else None
            model_name = self.settings.embedding_model if emb is not None else None

            row = await db_memories.insert_memory(
                db_path=self.settings.app_db_path,
                memory_id=memory_id,
                text=clean_text,
                kind=kind,
                importance=importance,
                embedding=blob,
                embedding_model=model_name,
                source_thread_id=source_thread_id,
            )

            new_item = MemoryItem(
                id=memory_id,
                text=clean_text,
                kind=kind,
                importance=importance,
                content_hash=row["content_hash"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                last_used_at=None,
                use_count=0,
                embedding=emb,
            )
            self._items[memory_id] = new_item
            self._rebuild_matrix()
            return "Saved."

    async def delete_memory(self, memory_id: str) -> bool:
        if not self._initialized:
            await self.initialize()

        async with self._lock:
            deleted = await db_memories.delete_memory(self.settings.app_db_path, memory_id)
            if memory_id in self._items:
                del self._items[memory_id]
                self._rebuild_matrix()
            return deleted

    async def clear_memories(self) -> int:
        if not self._initialized:
            await self.initialize()

        async with self._lock:
            count = await db_memories.clear_all_memories(self.settings.app_db_path)
            self._items.clear()
            self._matrix = None
            self._matrix_ids = []
            return count

    async def update_memory(
        self,
        memory_id: str,
        text: str | None = None,
        importance: int | None = None,
    ) -> bool:
        if not self._initialized:
            await self.initialize()

        async with self._lock:
            item = self._items.get(memory_id)
            if not item:
                row = await db_memories.get_memory(self.settings.app_db_path, memory_id)
                if not row:
                    return False
                current_text = row["text"]
                current_importance = row["importance"]
            else:
                current_text = item.text
                current_importance = item.importance

            new_text = text.strip() if text is not None else current_text
            new_importance = importance if importance is not None else current_importance

            if looks_like_secret(new_text):
                return False

            emb = None
            if text is not None and self.embedding_service.is_available():
                emb = await self.embedding_service.embed_text(new_text)

            # A text change without a replacement vector must not keep the old
            # embedding, or semantic search returns the stale memory (H-17).
            clear_embedding = text is not None and emb is None

            blob = vector_to_blob(emb) if emb is not None else None
            model_name = self.settings.embedding_model if emb is not None else None

            success = await db_memories.update_memory(
                self.settings.app_db_path,
                memory_id,
                text=new_text,
                importance=new_importance,
                embedding=blob,
                embedding_model=model_name,
                clear_embedding=clear_embedding,
            )

            if success:
                if item:
                    item.text = new_text
                    item.importance = new_importance
                    if emb is not None:
                        item.embedding = emb
                    elif clear_embedding:
                        item.embedding = None
                    item.content_hash = hash_memory_text(new_text)
                    item.updated_at = datetime.now(UTC).isoformat()
                self._rebuild_matrix()
            return success

    async def search(
        self,
        query: str,
        top_k: int = 6,
        min_score: float = 0.35,
    ) -> list[dict[str, Any]]:
        if not self._initialized:
            await self.initialize()

        if not self._items:
            return []

        clean_query = query.strip()
        if not clean_query:
            return []

        now = datetime.now(UTC)

        # 1. Vector similarity search if embeddings are available
        if self.embedding_service.is_available() and self._matrix is not None and len(self._matrix_ids) > 0:
            query_emb = await self.embedding_service.embed_text(clean_query)
            if query_emb is not None:
                dot = np.dot(self._matrix, query_emb)
                norms = np.linalg.norm(self._matrix, axis=1) * np.linalg.norm(query_emb)
                cosines = np.where(norms > 0, dot / norms, 0.0)

                scored_candidates: list[tuple[float, MemoryItem]] = []
                for i, mid in enumerate(self._matrix_ids):
                    cosine = float(cosines[i])
                    item = self._items[mid]

                    # Filter floor on cosine
                    if cosine < min_score:
                        continue

                    days_since_update = max(
                        0.0, (now - item.updated_datetime).total_seconds() / 86400.0
                    )
                    recency = math.exp(-days_since_update / 30.0)
                    final_score = (
                        0.65 * cosine + 0.20 * recency + 0.15 * (item.importance / 5.0)
                    )
                    scored_candidates.append((final_score, item))

                # Sort by score descending
                scored_candidates.sort(key=lambda x: x[0], reverse=True)
                selected_ids = {c[1].id for c in scored_candidates[:top_k]}
                results = [
                    {
                        "id": item.id,
                        "text": item.text,
                        "kind": item.kind,
                        "importance": item.importance,
                        "score": round(score, 3),
                    }
                    for score, item in scored_candidates[:top_k]
                ]

                # Critical memories (importance == 5, older than 7 days) must not starve
                seven_days_ago = now - timedelta(days=7)
                for item in self._items.values():
                    if (
                        item.importance == 5
                        and item.id not in selected_ids
                        and (item.last_used_datetime is None or item.last_used_datetime < seven_days_ago)
                    ):
                        results.insert(
                            0,
                            {
                                "id": item.id,
                                "text": item.text,
                                "kind": item.kind,
                                "importance": item.importance,
                                "score": 1.0,
                            },
                        )
                        selected_ids.add(item.id)

                return results[:top_k]

        # 2. Keyword matching fallback
        keyword_rows = await db_memories.search_memories_keyword(
            self.settings.app_db_path, clean_query, limit=max(top_k * 3, top_k)
        )
        query_words = [w for w in re.findall(r"\w+", clean_query.lower()) if len(w) > 2]
        if not query_words:
            query_words = [w for w in re.findall(r"\w+", clean_query.lower()) if w]

        scored_keyword: list[tuple[float, dict[str, Any]]] = []
        for r in keyword_rows:
            text_lower = r["text"].lower()
            matched = sum(1 for word in query_words if word in text_lower)
            lexical_score = matched / len(query_words) if query_words else 0.0
            # Enforce the same relevance floor as the vector branch instead of
            # assigning a fake importance-derived similarity (H-16).
            if lexical_score < min_score:
                continue
            scored_keyword.append(
                (
                    lexical_score,
                    {
                        "id": r["id"],
                        "text": r["text"],
                        "kind": r["kind"],
                        "importance": r["importance"],
                        "score": round(lexical_score, 3),
                        "score_kind": "lexical",
                    },
                )
            )

        scored_keyword.sort(key=lambda pair: (pair[0], pair[1]["importance"]), reverse=True)
        return [item for _, item in scored_keyword[:top_k]]

    async def record_usage(self, memory_ids: list[str]) -> None:
        if not memory_ids:
            return
        await db_memories.record_memories_usage(self.settings.app_db_path, memory_ids)
        now_str = datetime.now(UTC).isoformat()
        for mid in memory_ids:
            if mid in self._items:
                self._items[mid].use_count += 1
                self._items[mid].last_used_at = now_str

    async def reindex(self) -> int:
        if not self._initialized:
            await self.initialize()

        if not self.embedding_service.is_available():
            await self.embedding_service.initialize()
            if not self.embedding_service.is_available():
                return 0

        count = 0
        async with self._lock:
            for item in self._items.values():
                emb = await self.embedding_service.embed_text(item.text)
                if emb is not None:
                    blob = vector_to_blob(emb)
                    await db_memories.update_memory_embedding(
                        self.settings.app_db_path,
                        item.id,
                        blob,
                        self.settings.embedding_model,
                    )
                    item.embedding = emb
                    count += 1

            self._rebuild_matrix()
        return count


# Global repository singleton
_memory_repository: MemoryRepository | None = None


def get_memory_repository(settings: Settings | None = None) -> MemoryRepository:
    global _memory_repository
    actual_settings = settings or get_settings()
    if _memory_repository is None or (
        settings is not None and _memory_repository.settings != actual_settings
    ):
        _memory_repository = MemoryRepository(actual_settings)
    return _memory_repository


def get_memory_repo_dep(
    settings: Annotated[Settings, Depends(get_settings)],
) -> MemoryRepository:
    return get_memory_repository(settings)
