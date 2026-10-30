"""SQLite helpers: a small per-loop connection pool and transactional migrations."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import weakref
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

# PRAGMAs that cannot run inside a transaction; they are applied before BEGIN.
_TXN_UNSAFE_PRAGMAS = (
    "PRAGMA JOURNAL_MODE",
    "PRAGMA SYNCHRONOUS",
    "PRAGMA FOREIGN_KEYS",
)


class _ConnectionPool:
    """A tiny bounded pool of WAL connections for one database file."""

    def __init__(self, db_path: Path, max_size: int = 4) -> None:
        self.db_path = db_path
        self.max_size = max_size
        self._idle: asyncio.Queue[aiosqlite.Connection] = asyncio.Queue()
        self._size = 0
        self._closed = False
        self._lock = asyncio.Lock()

    async def _new_connection(self) -> aiosqlite.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        db = await aiosqlite.connect(str(self.db_path))
        db.row_factory = aiosqlite.Row
        # PRAGMAs are applied exactly once per physical connection, not per query.
        await db.execute("PRAGMA journal_mode = WAL;")
        await db.execute("PRAGMA synchronous = NORMAL;")
        await db.execute("PRAGMA foreign_keys = ON;")
        await db.execute("PRAGMA busy_timeout = 5000;")
        await db.commit()
        return db

    async def acquire(self) -> aiosqlite.Connection:
        async with self._lock:
            if self._closed:
                raise RuntimeError("Database connection pool is closed")
            try:
                return self._idle.get_nowait()
            except asyncio.QueueEmpty:
                if self._size < self.max_size:
                    db = await self._new_connection()
                    self._size += 1
                    return db
        return await self._idle.get()

    async def release(self, db: aiosqlite.Connection) -> None:
        """Return a connection, rolling back any caller-left transaction."""
        try:
            if db.in_transaction:
                await db.rollback()
        except (sqlite3.Error, ValueError):
            logger.debug("Rollback failed while releasing pooled connection", exc_info=True)

        if self._closed:
            try:
                await db.close()
            except sqlite3.Error:  # pragma: no cover - best effort
                pass
            return
        self._idle.put_nowait(db)

    async def close(self) -> None:
        async with self._lock:
            self._closed = True
            while True:
                try:
                    db = self._idle.get_nowait()
                except asyncio.QueueEmpty:
                    break
                try:
                    await db.close()
                except sqlite3.Error:  # pragma: no cover - best effort
                    pass
            self._size = 0


# Keyed by event loop so aiosqlite connections are never shared across loops.
_pools: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[str, _ConnectionPool]
] = weakref.WeakKeyDictionary()


def _pool_for(db_path: Path) -> _ConnectionPool:
    loop = asyncio.get_running_loop()
    pools = _pools.get(loop)
    if pools is None:
        pools = {}
        _pools[loop] = pools
    key = str(Path(db_path).resolve())
    pool = pools.get(key)
    if pool is None:
        pool = _ConnectionPool(Path(db_path))
        pools[key] = pool
    return pool


@asynccontextmanager
async def get_db_connection(db_path: Path) -> AsyncIterator[aiosqlite.Connection]:
    """Borrow a pooled aiosqlite connection with WAL and busy_timeout configured."""
    pool = _pool_for(db_path)
    db = await pool.acquire()
    try:
        yield db
    finally:
        await pool.release(db)


async def close_db_pools() -> None:
    """Close every pooled connection for the current event loop (shutdown/tests)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    pools = _pools.pop(loop, None)
    if not pools:
        return
    for pool in pools.values():
        await pool.close()


async def init_checkpoint_db(checkpoint_db_path: Path) -> None:
    """Ensure the checkpoints database file exists and has WAL mode enabled."""
    checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(str(checkpoint_db_path)) as db:
        await db.execute("PRAGMA journal_mode = WAL;")
        await db.execute("PRAGMA synchronous = NORMAL;")
        await db.execute("PRAGMA busy_timeout = 5000;")
        await db.commit()


def split_sql_statements(script: str) -> list[str]:
    """Split a SQL script into complete statements without relying on executescript."""
    statements: list[str] = []
    buffer = ""
    for line in script.splitlines(keepends=True):
        buffer += line
        if sqlite3.complete_statement(buffer):
            statement = buffer.strip()
            if statement:
                statements.append(statement)
            buffer = ""
    leftover = buffer.strip()
    if leftover:
        statements.append(leftover)
    return statements


async def run_migrations(db_path: Path) -> None:
    """Apply SQL migrations transactionally, tracked by ``PRAGMA user_version``.

    Each file runs inside ``BEGIN IMMEDIATE … COMMIT`` together with its version
    bump, so a failing statement rolls back the whole migration instead of
    leaving a half-applied schema (``executescript`` commits implicitly).
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    async with get_db_connection(db_path) as db:
        cursor = await db.execute("PRAGMA user_version;")
        row = await cursor.fetchone()
        current_version = row[0] if row else 0

        migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))

        for sql_file in migration_files:
            # Parse prefix version number, e.g. 001_init.sql -> 1
            stem = sql_file.name.split("_")[0]
            try:
                target_version = int(stem)
            except ValueError:
                continue

            if target_version <= current_version:
                continue

            logger.info(
                "Applying migration %s (target version: %d)", sql_file.name, target_version
            )
            statements = split_sql_statements(sql_file.read_text(encoding="utf-8"))
            outside_tx = [
                s
                for s in statements
                if s.upper().startswith(_TXN_UNSAFE_PRAGMAS)
            ]
            body = [s for s in statements if s not in outside_tx]

            for statement in outside_tx:
                await db.execute(statement)

            await db.execute("BEGIN IMMEDIATE")
            try:
                for statement in body:
                    await db.execute(statement)
                await db.execute(f"PRAGMA user_version = {target_version};")
                await db.commit()
            except Exception:
                await db.rollback()
                logger.exception("Migration %s failed; rolled back", sql_file.name)
                raise

            current_version = target_version
