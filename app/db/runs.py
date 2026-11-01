# SPDX-License-Identifier: Apache-2.0
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.db.sqlite import get_db_connection


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


async def start_run(db_path: Path, run_id: str, thread_id: str, model: str | None = None) -> dict[str, Any]:
    now = utc_now_iso()
    async with get_db_connection(db_path) as db:
        await db.execute(
            """
            INSERT INTO runs (id, thread_id, started_at, status, model, tool_calls)
            VALUES (?, ?, ?, 'running', ?, 0)
            """,
            (run_id, thread_id, now, model),
        )
        await db.commit()
    return {
        "id": run_id,
        "thread_id": thread_id,
        "started_at": now,
        "status": "running",
        "model": model,
    }


async def finish_run(
    db_path: Path,
    run_id: str,
    status: str,
    *,
    error_code: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    tool_calls: int = 0,
    elapsed_ms: int | None = None,
) -> bool:
    now = utc_now_iso()
    async with get_db_connection(db_path) as db:
        cursor = await db.execute(
            """
            UPDATE runs
            SET finished_at = ?, status = ?, error_code = ?,
                prompt_tokens = ?, completion_tokens = ?,
                tool_calls = ?, elapsed_ms = ?
            WHERE id = ?
            """,
            (now, status, error_code, prompt_tokens, completion_tokens, tool_calls, elapsed_ms, run_id),
        )
        await db.commit()
        return cursor.rowcount > 0


async def get_run(db_path: Path, run_id: str) -> dict[str, Any] | None:
    async with get_db_connection(db_path) as db:
        cursor = await db.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
        row = await cursor.fetchone()
        if not row:
            return None
        return dict(row)
