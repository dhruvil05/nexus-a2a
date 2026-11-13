"""
nexus_a2a/storage/sqlite_store.py

SqliteTaskStore — a durable TaskStore in a single local file.

For local development and single-machine deployments that need tasks to
survive a restart (a paused multi-turn task, for example) without running
Redis or PostgreSQL:

    store   = SqliteTaskStore("tasks.db")
    manager = TaskManager(store=store)

Uses only the standard library's sqlite3, run in a worker thread so the event
loop never blocks on disk I/O. No extra install is needed.

Like PostgresTaskStore, the whole Task (including its server-side ``owner``)
is stored as JSON in ``data``, with indexed scalar columns for filtering:

    CREATE TABLE IF NOT EXISTS nexus_a2a_tasks (
        id          TEXT PRIMARY KEY,
        context_id  TEXT NOT NULL,
        skill_id    TEXT,
        state       TEXT NOT NULL,
        data        TEXT NOT NULL,   -- full Task as JSON
        created_at  TEXT NOT NULL,   -- UTC, fixed-width ISO 8601
        updated_at  TEXT NOT NULL
    );

Limits:
  - One process. SQLite allows several readers but one writer at a time, and
    this store keeps tasks per file, not per network. Use RedisTaskStore or
    PostgresTaskStore when several replicas share task state.
  - No expiry: finished tasks stay until delete_older_than() or delete().
"""

from __future__ import annotations

import asyncio
import logging
import re
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar

from nexus_a2a.models.task import Task
from nexus_a2a.storage.task_store import AbstractTaskStore, _state_str

logger = logging.getLogger(__name__)

_DEFAULT_TABLE = "nexus_a2a_tasks"

# The table name is interpolated into SQL (identifiers cannot be bound as
# parameters), so it must be a plain identifier.
_TABLE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,62}")

# SQLite's special name for a private, in-process database.
_MEMORY = ":memory:"

_T = TypeVar("_T")


def _timestamp(value: datetime) -> str:
    """
    UTC, always with microseconds, so the text sorts in time order.

    datetime.isoformat() drops ".000000" when microsecond is 0, which would
    break ORDER BY and the cutoff comparison in delete_older_than().
    """
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class SqliteTaskStore(AbstractTaskStore):
    """
    SQLite-backed TaskStore: durable across restarts, no server needed.

    Usage::

        store = SqliteTaskStore("var/tasks.db")
        manager = TaskManager(store=store)
        ...
        await store.disconnect()

        # Or as an async context manager:
        async with SqliteTaskStore("var/tasks.db") as store:
            manager = TaskManager(store=store)

    The database is opened on first use, so calling connect() is optional;
    it is there so the store works anywhere PostgresTaskStore does (for
    example as a runtime resource that is connected and disconnected).

    Args:
        path:   Database file. Parent directories are created. ``":memory:"``
                gives a private in-memory database (tests only: it is lost on
                disconnect()).
        table:  Table name. Default "nexus_a2a_tasks". Must be a plain SQL
                identifier, otherwise ValueError is raised.
    """

    def __init__(self, path: str | Path, table: str = _DEFAULT_TABLE) -> None:
        if not isinstance(table, str) or not _TABLE_RE.fullmatch(table):
            raise ValueError(
                f"Invalid table name {table!r}: must match "
                "[A-Za-z_][A-Za-z0-9_]{0,62}."
            )
        self._path = str(path)
        self._table = table
        self._conn: sqlite3.Connection | None = None
        # One connection, used by one worker thread at a time.
        self._lock = asyncio.Lock()

    @property
    def path(self) -> str:
        """The database file (or ":memory:")."""
        return self._path

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Open the database and create the table if needed. Idempotent."""
        async with self._lock:
            if self._conn is None:
                self._conn = await asyncio.to_thread(self._open)
                logger.info(
                    "SqliteTaskStore: opened %s (table='%s')", self._path, self._table
                )

    async def disconnect(self) -> None:
        """Close the database. Safe to call more than once."""
        async with self._lock:
            if self._conn is not None:
                conn, self._conn = self._conn, None
                await asyncio.to_thread(conn.close)
                logger.info("SqliteTaskStore: closed %s", self._path)

    async def __aenter__(self) -> SqliteTaskStore:
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.disconnect()

    # ── AbstractTaskStore interface ───────────────────────────────────────────

    async def save(self, task: Task) -> None:
        """Insert the task, or replace the stored copy (keeps created_at)."""
        params = (
            task.id,
            task.context_id,
            task.skill_id,
            _state_str(task),
            task.model_dump_json(),
            _timestamp(task.created_at),
            _timestamp(task.updated_at),
        )

        def write(conn: sqlite3.Connection) -> None:
            with conn:
                conn.execute(
                    f"""
                    INSERT INTO {self._table}
                        (id, context_id, skill_id, state, data,
                         created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (id) DO UPDATE SET
                        state      = excluded.state,
                        data       = excluded.data,
                        updated_at = excluded.updated_at
                    """,
                    params,
                )

        await self._run(write)

    async def get(self, task_id: str) -> Task | None:
        """Return the task, or None if it is not stored."""
        rows = await self._fetch(
            f"SELECT data FROM {self._table} WHERE id = ?", (task_id,)
        )
        return Task.model_validate_json(rows[0][0]) if rows else None

    async def delete(self, task_id: str) -> None:
        """Remove a task. Does nothing if it is not stored."""
        await self._execute(f"DELETE FROM {self._table} WHERE id = ?", (task_id,))

    async def list_all(self) -> list[Task]:
        """Every stored task, most recently updated first."""
        return await self._tasks(
            f"SELECT data FROM {self._table} ORDER BY updated_at DESC", ()
        )

    # ── Extended query API (same as PostgresTaskStore) ────────────────────────

    async def list_by_state(self, state: str) -> list[Task]:
        """Tasks in one state ("working", "input_required", ...), newest first."""
        return await self._tasks(
            f"SELECT data FROM {self._table} WHERE state = ? ORDER BY updated_at DESC",
            (state,),
        )

    async def list_by_context(self, context_id: str) -> list[Task]:
        """Tasks in one context (conversation), oldest first."""
        return await self._tasks(
            f"SELECT data FROM {self._table} WHERE context_id = ? "
            "ORDER BY created_at ASC",
            (context_id,),
        )

    async def count(self) -> int:
        """Number of stored tasks."""
        rows = await self._fetch(f"SELECT COUNT(*) FROM {self._table}", ())
        return int(rows[0][0])

    async def count_by_state(self, state: str) -> int:
        """Number of stored tasks in one state."""
        rows = await self._fetch(
            f"SELECT COUNT(*) FROM {self._table} WHERE state = ?", (state,)
        )
        return int(rows[0][0])

    async def delete_older_than(self, days: float) -> int:
        """Delete tasks last updated more than ``days`` days ago; return how many."""
        cutoff = _timestamp(datetime.now(UTC) - timedelta(days=days))
        deleted = await self._execute(
            f"DELETE FROM {self._table} WHERE updated_at < ?", (cutoff,)
        )
        logger.info(
            "SqliteTaskStore: deleted %d tasks older than %s days.", deleted, days
        )
        return deleted

    async def clear(self) -> None:
        """Delete ALL tasks. Destructive; mainly for tests."""
        await self._execute(f"DELETE FROM {self._table}", ())
        logger.warning("SqliteTaskStore: all tasks cleared.")

    # ── Internals ─────────────────────────────────────────────────────────────

    def _open(self) -> sqlite3.Connection:
        """Open the connection and ensure the schema (runs in a thread)."""
        if self._path != _MEMORY:
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: calls arrive on asyncio.to_thread workers,
        # never two at once (self._lock).
        conn = sqlite3.connect(self._path, check_same_thread=False)
        try:
            if self._path != _MEMORY:
                # Readers do not block the writer, and a crash mid-write
                # leaves the last committed state intact.
                conn.execute("PRAGMA journal_mode=WAL")
            with conn:
                conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._table} (
                        id          TEXT PRIMARY KEY,
                        context_id  TEXT NOT NULL,
                        skill_id    TEXT,
                        state       TEXT NOT NULL,
                        data        TEXT NOT NULL,
                        created_at  TEXT NOT NULL,
                        updated_at  TEXT NOT NULL
                    )
                """)
                for column in ("state", "context_id", "updated_at"):
                    conn.execute(
                        f"CREATE INDEX IF NOT EXISTS {self._table}_{column}_idx "
                        f"ON {self._table} ({column})"
                    )
        except BaseException:
            conn.close()
            raise
        return conn

    async def _run(self, fn: Callable[[sqlite3.Connection], _T]) -> _T:
        """Run fn(connection) in a worker thread, opening the database first."""
        async with self._lock:
            if self._conn is None:
                self._conn = await asyncio.to_thread(self._open)
            return await asyncio.to_thread(fn, self._conn)

    async def _fetch(
        self, sql: str, params: tuple[object, ...]
    ) -> list[tuple[Any, ...]]:
        def query(conn: sqlite3.Connection) -> list[tuple[Any, ...]]:
            return list(conn.execute(sql, params).fetchall())

        return await self._run(query)

    async def _tasks(self, sql: str, params: tuple[object, ...]) -> list[Task]:
        rows = await self._fetch(sql, params)
        return [Task.model_validate_json(row[0]) for row in rows]

    async def _execute(self, sql: str, params: tuple[object, ...]) -> int:
        """Run one write statement in a transaction; return rows affected."""

        def write(conn: sqlite3.Connection) -> int:
            with conn:
                return conn.execute(sql, params).rowcount

        return await self._run(write)
