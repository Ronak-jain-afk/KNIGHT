from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.db.sqlite import get_db_connection


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


async def create_conversation(
    db_path: Path,
    conversation_id: str,
    title: str = "New Chat",
    parent_conversation_id: str | None = None,
    forked_from_message_id: str | None = None,
    turn_count: int = 0,
) -> dict[str, Any]:
    now = utc_now_iso()
    async with get_db_connection(db_path) as db:
        await db.execute(
            """
            INSERT INTO conversations (
                id, title, created_at, updated_at, archived, turn_count,
                parent_conversation_id, forked_from_message_id
            )
            VALUES (?, ?, ?, ?, 0, ?, ?, ?)
            """,
            (
                conversation_id,
                title,
                now,
                now,
                turn_count,
                parent_conversation_id,
                forked_from_message_id,
            ),
        )
        await db.commit()
    return {
        "id": conversation_id,
        "title": title,
        "created_at": now,
        "updated_at": now,
        "archived": False,
        "pinned": False,
        "turn_count": turn_count,
        "parent_conversation_id": parent_conversation_id,
        "forked_from_message_id": forked_from_message_id,
    }


async def get_conversation(db_path: Path, conversation_id: str) -> dict[str, Any] | None:
    async with get_db_connection(db_path) as db:
        cursor = await db.execute(
            """
            SELECT id, title, created_at, updated_at, archived, pinned, turn_count,
                   parent_conversation_id, forked_from_message_id
            FROM conversations
            WHERE id = ?
            """,
            (conversation_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "title": row["title"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "archived": bool(row["archived"]),
            "pinned": bool(row["pinned"]),
            "turn_count": row["turn_count"],
            "parent_conversation_id": row["parent_conversation_id"],
            "forked_from_message_id": row["forked_from_message_id"],
        }


async def list_conversations(
    db_path: Path, limit: int = 50, offset: int = 0, include_archived: bool = False
) -> list[dict[str, Any]]:
    query = (
        "SELECT id, title, created_at, updated_at, archived, pinned, turn_count, "
        "parent_conversation_id, forked_from_message_id FROM conversations "
        + ("" if include_archived else "WHERE archived = 0 ")
        + "ORDER BY pinned DESC, updated_at DESC LIMIT ? OFFSET ?"
    )
    async with get_db_connection(db_path) as db:
        cursor = await db.execute(query, (limit, offset))
        rows = await cursor.fetchall()
        return [
            {
                "id": r["id"],
                "title": r["title"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "archived": bool(r["archived"]),
                "pinned": bool(r["pinned"]),
                "turn_count": r["turn_count"],
                "parent_conversation_id": r["parent_conversation_id"],
                "forked_from_message_id": r["forked_from_message_id"],
            }
            for r in rows
        ]


async def update_conversation_title(db_path: Path, conversation_id: str, title: str) -> bool:
    now = utc_now_iso()
    async with get_db_connection(db_path) as db:
        cursor = await db.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title, now, conversation_id),
        )
        await db.commit()
        return cursor.rowcount > 0


async def increment_turn_count(db_path: Path, conversation_id: str) -> bool:
    now = utc_now_iso()
    async with get_db_connection(db_path) as db:
        cursor = await db.execute(
            "UPDATE conversations SET turn_count = turn_count + 1, updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )
        await db.commit()
        return cursor.rowcount > 0


async def archive_conversation(db_path: Path, conversation_id: str, archived: bool = True) -> bool:
    now = utc_now_iso()
    async with get_db_connection(db_path) as db:
        cursor = await db.execute(
            "UPDATE conversations SET archived = ?, updated_at = ? WHERE id = ?",
            (1 if archived else 0, now, conversation_id),
        )
        await db.commit()
        return cursor.rowcount > 0


async def set_conversation_pinned(db_path: Path, conversation_id: str, pinned: bool = True) -> bool:
    now = utc_now_iso()
    async with get_db_connection(db_path) as db:
        cursor = await db.execute(
            "UPDATE conversations SET pinned = ?, updated_at = ? WHERE id = ?",
            (1 if pinned else 0, now, conversation_id),
        )
        await db.commit()
        return cursor.rowcount > 0


async def delete_conversation(db_path: Path, conversation_id: str) -> bool:
    async with get_db_connection(db_path) as db:
        await db.execute("DELETE FROM message_sources WHERE thread_id = ?", (conversation_id,))
        await db.execute("DELETE FROM runs WHERE thread_id = ?", (conversation_id,))
        cursor = await db.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        await db.commit()
        return cursor.rowcount > 0
