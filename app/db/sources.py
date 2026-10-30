from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.db.sqlite import get_db_connection


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


async def save_message_sources(
    db_path: Path, thread_id: str, message_id: str, sources: list[dict[str, Any]]
) -> None:
    now = utc_now_iso()
    records = [
        (
            thread_id,
            message_id,
            src["idx"],
            src["url"],
            src.get("title"),
            src.get("fetched_at") or now,
        )
        for src in sources
    ]
    if not records:
        return

    async with get_db_connection(db_path) as db:
        await db.executemany(
            """
            INSERT OR REPLACE INTO message_sources (thread_id, message_id, idx, url, title, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            records,
        )
        await db.commit()


async def get_message_sources(db_path: Path, thread_id: str, message_id: str) -> list[dict[str, Any]]:
    async with get_db_connection(db_path) as db:
        cursor = await db.execute(
            """
            SELECT idx, url, title, fetched_at
            FROM message_sources
            WHERE thread_id = ? AND message_id = ?
            ORDER BY idx ASC
            """,
            (thread_id, message_id),
        )
        rows = await cursor.fetchall()
        return [
            {
                "idx": r["idx"],
                "url": r["url"],
                "title": r["title"],
                "fetched_at": r["fetched_at"],
            }
            for r in rows
        ]


async def get_thread_sources(db_path: Path, thread_id: str) -> list[dict[str, Any]]:
    async with get_db_connection(db_path) as db:
        cursor = await db.execute(
            """
            SELECT message_id, idx, url, title, fetched_at
            FROM message_sources
            WHERE thread_id = ?
            ORDER BY fetched_at ASC, idx ASC
            """,
            (thread_id,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "message_id": r["message_id"],
                "idx": r["idx"],
                "url": r["url"],
                "title": r["title"],
                "fetched_at": r["fetched_at"],
            }
            for r in rows
        ]


async def copy_thread_sources(
    db_path: Path,
    source_thread_id: str,
    target_thread_id: str,
    message_ids: set[str],
) -> None:
    """Copy recorded citations from a source thread to a branched thread."""
    if not message_ids:
        return
    all_sources = await get_thread_sources(db_path, source_thread_id)
    matching = [s for s in all_sources if s.get("message_id") in message_ids]
    if not matching:
        return

    now = utc_now_iso()
    records = [
        (
            target_thread_id,
            s["message_id"],
            s["idx"],
            s["url"],
            s.get("title"),
            s.get("fetched_at") or now,
        )
        for s in matching
    ]
    async with get_db_connection(db_path) as db:
        await db.executemany(
            """
            INSERT OR REPLACE INTO message_sources (thread_id, message_id, idx, url, title, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            records,
        )
        await db.commit()
