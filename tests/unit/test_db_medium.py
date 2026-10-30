"""Regression tests for M-07 (transactional migrations), M-08 (connection reuse),
and M-15 (scheduled cache purge + row cap)."""

import asyncio
import sqlite3

import pytest

from app.db import cache
from app.db import sqlite as sqlite_module
from app.db.sqlite import (
    _pool_for,
    close_db_pools,
    get_db_connection,
    run_migrations,
)


@pytest.fixture(autouse=True)
async def _close_pools():
    yield
    await close_db_pools()


@pytest.mark.asyncio
async def test_failed_migration_rolls_back_whole_file(tmp_path, monkeypatch):
    """M-07: a bad statement must not leave a half-applied schema behind."""
    db_path = tmp_path / "rollback.sqlite"
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "001_broken.sql").write_text(
        "CREATE TABLE should_not_exist (id INTEGER);\n"
        "THIS IS NOT VALID SQL;\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sqlite_module, "MIGRATIONS_DIR", migrations)

    with pytest.raises(sqlite3.Error):
        await run_migrations(db_path)

    async with get_db_connection(db_path) as db:
        cursor = await db.execute("PRAGMA user_version;")
        assert (await cursor.fetchone())[0] == 0

        cursor = await db.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='should_not_exist';"
        )
        assert await cursor.fetchone() is None


@pytest.mark.asyncio
async def test_connection_is_reused_between_borrows(tmp_path):
    """M-08: sequential queries must not open a new SQLite connection each time."""
    db_path = tmp_path / "reuse.sqlite"
    pool = _pool_for(db_path)

    async with get_db_connection(db_path) as first:
        assert pool._size == 1
        assert pool._idle.qsize() == 0

    assert pool._idle.qsize() == 1

    async with get_db_connection(db_path) as second:
        assert second is first

    async with get_db_connection(db_path) as third:
        assert third is first


@pytest.mark.asyncio
async def test_purge_expired_cache_removes_only_expired_rows(tmp_path):
    """M-15: expired search and page rows are deleted, fresh rows survive."""
    db_path = tmp_path / "cache.sqlite"
    await run_migrations(db_path)

    await cache.set_search_cache(db_path, "old", [{"title": "old"}], ttl_hours=-1)
    await cache.set_search_cache(db_path, "fresh", [{"title": "fresh"}], ttl_hours=1)
    await cache.set_page_cache(
        db_path,
        "https://old.example",
        "https://old.example",
        "old",
        "# old",
        ttl_hours=-1,
    )

    deleted_search, deleted_page = await cache.purge_expired_cache(db_path)
    assert (deleted_search, deleted_page) == (1, 1)
    assert await cache.get_search_cache(db_path, "old") is None
    assert await cache.get_search_cache(db_path, "fresh") == [{"title": "fresh"}]


@pytest.mark.asyncio
async def test_cache_row_cap_evicts_oldest_rows(tmp_path):
    """M-15: the cache is bounded, oldest rows are evicted first."""
    db_path = tmp_path / "cap.sqlite"
    await run_migrations(db_path)

    future = "2999-01-01T00:00:00+00:00"
    async with get_db_connection(db_path) as db:
        for index in range(5):
            await db.execute(
                """
                INSERT INTO search_cache (key, payload, created_at, expires_at)
                VALUES (?, '[]', ?, ?)
                """,
                (f"k{index}", f"2026-01-0{index + 1}T00:00:00+00:00", future),
            )
        await db.commit()

    removed = await cache.enforce_cache_row_cap(db_path, max_rows=2)
    assert removed == 3

    async with get_db_connection(db_path) as db:
        cursor = await db.execute("SELECT key FROM search_cache ORDER BY key;")
        keys = [row["key"] for row in await cursor.fetchall()]
    assert keys == ["k3", "k4"]


@pytest.mark.asyncio
async def test_purge_loop_runs_on_interval_and_stops_cleanly(tmp_path):
    """M-15: the background loop purges on its timer and honours the stop event."""
    db_path = tmp_path / "loop.sqlite"
    await run_migrations(db_path)
    await cache.set_search_cache(db_path, "stale", [], ttl_hours=-1)

    stop = asyncio.Event()
    task = asyncio.create_task(
        cache.purge_expired_cache_loop(
            db_path, interval_seconds=0.02, max_rows=100, stop_event=stop
        )
    )
    try:
        for _ in range(50):
            await asyncio.sleep(0.01)
            if await cache.get_search_cache(db_path, "stale") is None:
                break
    finally:
        stop.set()
        await asyncio.wait_for(task, timeout=2)

    assert await cache.get_search_cache(db_path, "stale") is None
