import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.db.sqlite import get_db_connection


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def hash_memory_text(text: str) -> str:
    clean = re.sub(r"\s+", " ", text.strip().lower())
    clean = clean.rstrip(".!?,;: \t\n\r")
    return hashlib.sha256(clean.encode("utf-8")).hexdigest()


async def insert_memory(
    db_path: Path,
    memory_id: str,
    text: str,
    kind: str,
    importance: int = 3,
    embedding: bytes | None = None,
    embedding_model: str | None = None,
    source_thread_id: str | None = None,
) -> dict[str, Any]:
    now = utc_now_iso()
    content_hash = hash_memory_text(text)

    async with get_db_connection(db_path) as db:
        await db.execute(
            """
            INSERT INTO memories (
                id, text, kind, importance, embedding, embedding_model,
                content_hash, source_thread_id, created_at, updated_at, use_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                memory_id,
                text,
                kind,
                importance,
                embedding,
                embedding_model,
                content_hash,
                source_thread_id,
                now,
                now,
            ),
        )
        await db.commit()

    return {
        "id": memory_id,
        "text": text,
        "kind": kind,
        "importance": importance,
        "content_hash": content_hash,
        "source_thread_id": source_thread_id,
        "created_at": now,
        "updated_at": now,
        "last_used_at": None,
        "use_count": 0,
    }


async def get_memory(db_path: Path, memory_id: str) -> dict[str, Any] | None:
    async with get_db_connection(db_path) as db:
        cursor = await db.execute(
            """
            SELECT id, text, kind, importance, embedding, embedding_model,
                   content_hash, source_thread_id, created_at, updated_at,
                   last_used_at, use_count
            FROM memories
            WHERE id = ?
            """,
            (memory_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return dict(row)


async def list_memories(
    db_path: Path,
    kind: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    query = (
        "SELECT id, text, kind, importance, embedding_model, content_hash, "
        "source_thread_id, created_at, updated_at, last_used_at, use_count "
        "FROM memories "
    )
    params: list[Any] = []
    if kind:
        query += "WHERE kind = ? "
        params.append(kind)
    query += "ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    async with get_db_connection(db_path) as db:
        cursor = await db.execute(query, tuple(params))
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def delete_memory(db_path: Path, memory_id: str) -> bool:
    async with get_db_connection(db_path) as db:
        cursor = await db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        await db.commit()
        return cursor.rowcount > 0


async def update_memory(
    db_path: Path,
    memory_id: str,
    text: str,
    importance: int,
    embedding: bytes | None = None,
    embedding_model: str | None = None,
    clear_embedding: bool = False,
) -> bool:
    now = utc_now_iso()
    content_hash = hash_memory_text(text)
    async with get_db_connection(db_path) as db:
        if clear_embedding:
            cursor = await db.execute(
                """
                UPDATE memories
                SET text = ?, importance = ?, embedding = NULL, embedding_model = NULL,
                    content_hash = ?, updated_at = ?
                WHERE id = ?
                """,
                (text, importance, content_hash, now, memory_id),
            )
        elif embedding is not None:
            cursor = await db.execute(
                """
                UPDATE memories
                SET text = ?, importance = ?, embedding = ?, embedding_model = ?,
                    content_hash = ?, updated_at = ?
                WHERE id = ?
                """,
                (text, importance, embedding, embedding_model, content_hash, now, memory_id),
            )
        else:
            cursor = await db.execute(
                """
                UPDATE memories
                SET text = ?, importance = ?, content_hash = ?, updated_at = ?
                WHERE id = ?
                """,
                (text, importance, content_hash, now, memory_id),
            )
        await db.commit()
        return cursor.rowcount > 0


async def record_memories_usage(db_path: Path, memory_ids: list[str]) -> None:
    if not memory_ids:
        return
    now = utc_now_iso()
    async with get_db_connection(db_path) as db:
        params = [(now, mid) for mid in memory_ids]
        await db.executemany(
            "UPDATE memories SET use_count = use_count + 1, last_used_at = ? WHERE id = ?",
            params,
        )
        await db.commit()


async def load_all_memories(db_path: Path) -> list[dict[str, Any]]:
    async with get_db_connection(db_path) as db:
        cursor = await db.execute(
            """
            SELECT id, text, kind, importance, embedding, embedding_model,
                   content_hash, source_thread_id, created_at, updated_at,
                   last_used_at, use_count
            FROM memories
            ORDER BY created_at ASC
            """
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def update_memory_embedding(
    db_path: Path,
    memory_id: str,
    embedding: bytes,
    embedding_model: str,
) -> bool:
    async with get_db_connection(db_path) as db:
        cursor = await db.execute(
            "UPDATE memories SET embedding = ?, embedding_model = ? WHERE id = ?",
            (embedding, embedding_model, memory_id),
        )
        await db.commit()
        return cursor.rowcount > 0


async def search_memories_keyword(
    db_path: Path,
    query: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    clean = re.sub(r"[^\w\s]", " ", query.strip())
    if not clean.strip():
        return []
    words = [w.lower() for w in clean.split() if len(w) > 2]
    if not words:
        words = [w.lower() for w in clean.split() if w]
    if not words:
        return []

    where_clauses = ["LOWER(text) LIKE ?" for _ in words]
    where_sql = " OR ".join(where_clauses)
    params = [f"%{w}%" for w in words]
    params.append(limit)

    async with get_db_connection(db_path) as db:
        cursor = await db.execute(
            f"""
            SELECT id, text, kind, importance, embedding_model, content_hash,
                   source_thread_id, created_at, updated_at, last_used_at, use_count
            FROM memories
            WHERE {where_sql}
            ORDER BY importance DESC, updated_at DESC
            LIMIT ?
            """,
            tuple(params),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
