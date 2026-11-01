# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import logging
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from app.db.sqlite import get_db_connection

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(UTC)


def utc_now_iso() -> str:
    return utc_now().isoformat()


async def get_search_cache(db_path: Path, key: str) -> list[dict[str, Any]] | None:
    now = utc_now_iso()
    async with get_db_connection(db_path) as db:
        cursor = await db.execute(
            "SELECT payload FROM search_cache WHERE key = ? AND expires_at > ?",
            (key, now),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return json.loads(row["payload"])


async def set_search_cache(
    db_path: Path, key: str, payload: list[dict[str, Any]], ttl_hours: int = 24
) -> None:
    now_dt = utc_now()
    created_at = now_dt.isoformat()
    expires_at = (now_dt + timedelta(hours=ttl_hours)).isoformat()
    payload_str = json.dumps(payload)

    async with get_db_connection(db_path) as db:
        await db.execute(
            """
            INSERT OR REPLACE INTO search_cache (key, payload, created_at, expires_at)
            VALUES (?, ?, ?, ?)
            """,
            (key, payload_str, created_at, expires_at),
        )
        await db.commit()


async def get_page_cache(db_path: Path, url: str) -> dict[str, Any] | None:
    now = utc_now_iso()
    async with get_db_connection(db_path) as db:
        cursor = await db.execute(
            "SELECT url, final_url, title, markdown, fetched_at FROM page_cache WHERE url = ? AND expires_at > ?",
            (url, now),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return {
            "url": row["url"],
            "final_url": row["final_url"],
            "title": row["title"],
            "markdown": row["markdown"],
            "fetched_at": row["fetched_at"],
        }


async def set_page_cache(
    db_path: Path,
    url: str,
    final_url: str,
    title: str | None,
    markdown: str,
    ttl_hours: int = 168,
) -> None:
    now_dt = utc_now()
    fetched_at = now_dt.isoformat()
    expires_at = (now_dt + timedelta(hours=ttl_hours)).isoformat()

    async with get_db_connection(db_path) as db:
        await db.execute(
            """
            INSERT OR REPLACE INTO page_cache (url, final_url, title, markdown, fetched_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (url, final_url, title, markdown, fetched_at, expires_at),
        )
        await db.commit()


async def purge_expired_cache(db_path: Path) -> tuple[int, int]:
    now = utc_now_iso()
    async with get_db_connection(db_path) as db:
        c1 = await db.execute("DELETE FROM search_cache WHERE expires_at <= ?", (now,))
        deleted_search = c1.rowcount
        c2 = await db.execute("DELETE FROM page_cache WHERE expires_at <= ?", (now,))
        deleted_page = c2.rowcount
        await db.commit()
        return deleted_search, deleted_page


async def enforce_cache_row_cap(db_path: Path, max_rows: int) -> int:
    """Evict the oldest rows when a cache table grows past ``max_rows``."""
    if max_rows <= 0:
        return 0
    removed = 0
    async with get_db_connection(db_path) as db:
        for table, key_col, ts_col in (
            ("search_cache", "key", "created_at"),
            ("page_cache", "url", "fetched_at"),
        ):
            cursor = await db.execute(f"SELECT COUNT(*) AS n FROM {table}")
            row = await cursor.fetchone()
            count = int(row["n"]) if row else 0
            excess = count - max_rows
            if excess <= 0:
                continue
            await db.execute(
                f"""
                DELETE FROM {table}
                WHERE {key_col} IN (
                    SELECT {key_col} FROM {table} ORDER BY {ts_col} ASC LIMIT ?
                )
                """,
                (excess,),
            )
            removed += excess
        await db.commit()
    return removed


async def purge_expired_cache_loop(
    db_path: Path,
    interval_seconds: float = 3600.0,
    max_rows: int | None = None,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Periodically purge expired cache rows (and optionally enforce a size cap)."""
    while stop_event is None or not stop_event.is_set():
        try:
            deleted_search, deleted_page = await purge_expired_cache(db_path)
            evicted = (
                await enforce_cache_row_cap(db_path, max_rows)
                if max_rows is not None
                else 0
            )
            total = deleted_search + deleted_page + evicted
            if total:
                logger.info(
                    "Cache purge removed %d row(s): search=%d page=%d capped=%d",
                    total,
                    deleted_search,
                    deleted_page,
                    evicted,
                )
        except (sqlite3.Error, OSError, ValueError) as exc:
            logger.warning("Cache purge failed: %s", exc)

        if stop_event is not None:
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
                break
            except TimeoutError:
                continue
        await asyncio.sleep(interval_seconds)
