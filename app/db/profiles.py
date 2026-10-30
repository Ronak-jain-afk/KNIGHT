from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.db.sqlite import get_db_connection


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


async def list_profiles(db_path: Path) -> list[dict[str, Any]]:
    query = """
        SELECT id, name, icon, description, system_prompt, mode, is_builtin, created_at, updated_at
        FROM profiles
        ORDER BY is_builtin DESC, updated_at DESC
    """
    async with get_db_connection(db_path) as db:
        cursor = await db.execute(query)
        rows = await cursor.fetchall()
        return [
            {
                "id": r["id"],
                "name": r["name"],
                "icon": r["icon"],
                "description": r["description"],
                "system_prompt": r["system_prompt"],
                "mode": r["mode"],
                "is_builtin": bool(r["is_builtin"]),
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]


async def get_profile(db_path: Path, profile_id: str) -> dict[str, Any] | None:
    query = """
        SELECT id, name, icon, description, system_prompt, mode, is_builtin, created_at, updated_at
        FROM profiles
        WHERE id = ?
    """
    async with get_db_connection(db_path) as db:
        cursor = await db.execute(query, (profile_id,))
        r = await cursor.fetchone()
        if not r:
            return None
        return {
            "id": r["id"],
            "name": r["name"],
            "icon": r["icon"],
            "description": r["description"],
            "system_prompt": r["system_prompt"],
            "mode": r["mode"],
            "is_builtin": bool(r["is_builtin"]),
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
        }


async def create_profile(
    db_path: Path,
    profile_id: str,
    name: str,
    icon: str = "",
    description: str = "",
    system_prompt: str = "",
    mode: str = "agent",
) -> dict[str, Any]:
    now = utc_now_iso()
    async with get_db_connection(db_path) as db:
        await db.execute(
            """
            INSERT INTO profiles (id, name, icon, description, system_prompt, mode, is_builtin, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)
            """,
            (profile_id, name, icon, description, system_prompt, mode, now, now),
        )
        await db.commit()
    return {
        "id": profile_id,
        "name": name,
        "icon": icon,
        "description": description,
        "system_prompt": system_prompt,
        "mode": mode,
        "is_builtin": False,
        "created_at": now,
        "updated_at": now,
    }


async def update_profile(
    db_path: Path,
    profile_id: str,
    name: str | None = None,
    icon: str | None = None,
    description: str | None = None,
    system_prompt: str | None = None,
    mode: str | None = None,
) -> bool:
    existing = await get_profile(db_path, profile_id)
    if not existing:
        return False

    now = utc_now_iso()
    new_name = name if name is not None else existing["name"]
    new_icon = icon if icon is not None else existing["icon"]
    new_desc = description if description is not None else existing["description"]
    new_prompt = system_prompt if system_prompt is not None else existing["system_prompt"]
    new_mode = mode if mode is not None else existing["mode"]

    async with get_db_connection(db_path) as db:
        cursor = await db.execute(
            """
            UPDATE profiles
            SET name = ?, icon = ?, description = ?, system_prompt = ?, mode = ?, updated_at = ?
            WHERE id = ?
            """,
            (new_name, new_icon, new_desc, new_prompt, new_mode, now, profile_id),
        )
        await db.commit()
        return cursor.rowcount > 0


async def delete_profile(db_path: Path, profile_id: str) -> bool:
    existing = await get_profile(db_path, profile_id)
    if not existing:
        return False
    if existing["is_builtin"]:
        return False

    async with get_db_connection(db_path) as db:
        cursor = await db.execute("DELETE FROM profiles WHERE id = ?", (profile_id,))
        await db.commit()
        return cursor.rowcount > 0
