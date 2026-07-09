"""
nexus_a2a/storage/postgres_store.py

PostgresTaskStore — production TaskStore backed by PostgreSQL via asyncpg.

Drop-in replacement for InMemoryTaskStore and RedisTaskStore:

    store   = PostgresTaskStore(dsn="postgresql://user:pass@localhost/nexus")
    manager = TaskManager(store=store)
    await store.connect()

Features vs InMemoryTaskStore / RedisTaskStore:
  - ACID: full transaction guarantees — no partial writes.
  - Persistent: tasks survive process restarts AND server reboots.
  - Queryable: full SQL access to task history for analytics/debugging.
  - Distributed: any number of processes share the same task state.
  - No TTL pressure: old tasks stay until explicitly deleted or archived.
  - Audit-ready: created_at / updated_at columns indexed for time-range queries.

Schema (auto-created on first connect() if tables don't exist):

    CREATE TABLE IF NOT EXISTS nexus_a2a_tasks (
        id          TEXT        PRIMARY KEY,
        context_id  TEXT        NOT NULL,
        skill_id    TEXT,
        state       TEXT        NOT NULL,
        data        JSONB       NOT NULL,      -- full Task as JSON
        created_at  TIMESTAMPTZ NOT NULL,
        updated_at  TIMESTAMPTZ NOT NULL
    );

    CREATE INDEX IF NOT EXISTS nexus_a2a_tasks_state_idx
        ON nexus_a2a_tasks (state);

    CREATE INDEX IF NOT EXISTS nexus_a2a_tasks_context_idx
        ON nexus_a2a_tasks (context_id);

    CREATE INDEX IF NOT EXISTS nexus_a2a_tasks_updated_idx
        ON nexus_a2a_tasks (updated_at DESC);

Design decisions:
  - Full Task is stored as JSONB in `data` — Pydantic model_dump() / model_validate().
  - Scalar columns (id, state, context_id, created_at, updated_at) are indexed
    for efficient filtering without parsing JSONB.
  - asyncpg connection pool is used — no per-query connection overhead.
  - Pool is lazily created on connect() and torn down on disconnect().
  - Schema creation is idempotent — safe to run on every startup.
  - Table name is configurable to avoid conflicts in shared databases.

Requires: pip install nexus-a2a[postgres]
          (pulls in asyncpg)
"""

from __future__ import annotations

import logging
from typing import Any

from nexus_a2a.models.task import Task
from nexus_a2a.storage.task_store import AbstractTaskStore

logger = logging.getLogger(__name__)

# Default table name — configurable to avoid conflicts in shared databases
_DEFAULT_TABLE = "nexus_a2a_tasks"


class PostgresTaskStore(AbstractTaskStore):
    """
    PostgreSQL-backed TaskStore for production deployments.

    Uses asyncpg for fully asynchronous, high-performance database access.
    Stores each Task as a JSONB document with indexed scalar columns for
    efficient state-based and context-based queries.

    Usage::

        store = PostgresTaskStore(dsn="postgresql://user:pass@host/db")
        await store.connect()          # creates schema, opens pool
        try:
            manager = TaskManager(store=store)
            ...
        finally:
            await store.disconnect()   # closes pool cleanly

        # Or as an async context manager:
        async with PostgresTaskStore(dsn="postgresql://...") as store:
            manager = TaskManager(store=store)
            ...

    Args:
        dsn:        asyncpg connection string.
                    Format: postgresql://user:pass@host:port/dbname
                    Supports all asyncpg DSN options (SSL, connect_timeout, etc.)
        min_size:   Minimum pool connections. Default: 2.
        max_size:   Maximum pool connections. Default: 10.
        table:      Table name for task storage. Default: "nexus_a2a_tasks".
                    Override to isolate environments (e.g. "nexus_staging_tasks").
        create_schema: If True (default), run CREATE TABLE IF NOT EXISTS on
                    connect(). Set False if you manage schema via migrations.
    """

    def __init__(
        self,
        dsn: str,
        min_size: int = 2,
        max_size: int = 10,
        table: str = _DEFAULT_TABLE,
        create_schema: bool = True,
    ) -> None:
        self._dsn = dsn
        self._min_size = min_size
        self._max_size = max_size
        self._table = table
        self._create_schema = create_schema
        self._pool: Any = None  # asyncpg.Pool — set after connect()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """
        Open the asyncpg connection pool and optionally create the schema.

        Must be called before any store operations.

        Raises:
            ImportError:     asyncpg is not installed.
            OSError:         PostgreSQL server is unreachable.
            asyncpg.PostgresError: Authentication failure or invalid DSN.
        """
        try:
            import asyncpg
        except ImportError as exc:
            raise ImportError(
                "asyncpg is required for PostgresTaskStore. "
                "Install it with: pip install nexus-a2a[postgres]"
            ) from exc

        self._pool = await asyncpg.create_pool(
            dsn=self._dsn,
            min_size=self._min_size,
            max_size=self._max_size,
        )

        if self._create_schema:
            await self._ensure_schema()

        logger.info(
            "PostgresTaskStore: connected (pool %d-%d, table='%s')",
            self._min_size,
            self._max_size,
            self._table,
        )

    async def disconnect(self) -> None:
        """
        Close the asyncpg connection pool gracefully.

        Waits for all in-flight queries to complete before closing.
        Safe to call multiple times.
        """
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("PostgresTaskStore: disconnected.")

    # ── Context manager ───────────────────────────────────────────────────────

    async def __aenter__(self) -> PostgresTaskStore:
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.disconnect()

    # ── AbstractTaskStore interface ───────────────────────────────────────────

    async def save(self, task: Task) -> None:
        """
        Persist a task — insert if new, update if it already exists (upsert).

        Args:
            task: The Task object to persist.

        Raises:
            RuntimeError: Store not connected (connect() not called).
            asyncpg.PostgresError: Database write failure.
        """
        self._require_pool()
        data_json = task.model_dump_json()

        await self._pool.execute(
            f"""
            INSERT INTO {self._table}
                (id, context_id, skill_id, state, data, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7)
            ON CONFLICT (id) DO UPDATE SET
                state      = EXCLUDED.state,
                data       = EXCLUDED.data,
                updated_at = EXCLUDED.updated_at
            """,
            task.id,
            task.context_id,
            task.skill_id,
            task.state.value,
            data_json,
            task.created_at,
            task.updated_at,
        )
        logger.debug(
            "PostgresTaskStore: saved task %s (state=%s)", task.id, task.state.value
        )

    async def get(self, task_id: str) -> Task | None:
        """
        Retrieve a task by its ID.

        Args:
            task_id: The unique task identifier.

        Returns:
            The Task if found, otherwise None.

        Raises:
            RuntimeError: Store not connected.
            asyncpg.PostgresError: Database read failure.
        """
        self._require_pool()
        row = await self._pool.fetchrow(
            f"SELECT data FROM {self._table} WHERE id = $1",
            task_id,
        )
        if row is None:
            return None
        return Task.model_validate_json(row["data"])

    async def delete(self, task_id: str) -> None:
        """
        Remove a task permanently.
        Silently does nothing if the task does not exist.

        Args:
            task_id: The unique task identifier.

        Raises:
            RuntimeError: Store not connected.
            asyncpg.PostgresError: Database write failure.
        """
        self._require_pool()
        await self._pool.execute(
            f"DELETE FROM {self._table} WHERE id = $1",
            task_id,
        )
        logger.debug("PostgresTaskStore: deleted task %s", task_id)

    async def list_all(self) -> list[Task]:
        """
        Return all tasks currently in the store, ordered by updated_at DESC.

        Returns:
            List of Task objects. Empty list if no tasks exist.

        Raises:
            RuntimeError: Store not connected.
            asyncpg.PostgresError: Database read failure.
        """
        self._require_pool()
        rows = await self._pool.fetch(
            f"SELECT data FROM {self._table} ORDER BY updated_at DESC",
        )
        return [Task.model_validate_json(row["data"]) for row in rows]

    # ── Extended query API ────────────────────────────────────────────────────

    async def list_by_state(self, state: str) -> list[Task]:
        """
        Return all tasks with the given state.

        More efficient than list_all() + filtering in Python because it uses
        the state index in PostgreSQL.

        Args:
            state: TaskState value string, e.g. "working", "completed".

        Returns:
            List of matching Task objects, ordered by updated_at DESC.
        """
        self._require_pool()
        rows = await self._pool.fetch(
            f"""
            SELECT data FROM {self._table}
            WHERE state = $1
            ORDER BY updated_at DESC
            """,
            state,
        )
        return [Task.model_validate_json(row["data"]) for row in rows]

    async def list_by_context(self, context_id: str) -> list[Task]:
        """
        Return all tasks belonging to a context (conversation group).

        Args:
            context_id: The context identifier to filter by.

        Returns:
            List of matching Task objects, ordered by created_at ASC
            (chronological order within the conversation).
        """
        self._require_pool()
        rows = await self._pool.fetch(
            f"""
            SELECT data FROM {self._table}
            WHERE context_id = $1
            ORDER BY created_at ASC
            """,
            context_id,
        )
        return [Task.model_validate_json(row["data"]) for row in rows]

    async def count(self) -> int:
        """Return the total number of tasks in the store."""
        self._require_pool()
        result = await self._pool.fetchval(f"SELECT COUNT(*) FROM {self._table}")
        return int(result or 0)

    async def count_by_state(self, state: str) -> int:
        """Return the number of tasks in the given state."""
        self._require_pool()
        result = await self._pool.fetchval(
            f"SELECT COUNT(*) FROM {self._table} WHERE state = $1",
            state,
        )
        return int(result or 0)

    async def delete_older_than(self, days: int) -> int:
        """
        Delete tasks whose updated_at is older than `days` days.

        Useful for periodic cleanup of completed/failed tasks.

        Args:
            days: Tasks last updated more than this many days ago are deleted.

        Returns:
            Number of tasks deleted.
        """
        self._require_pool()
        result = await self._pool.execute(
            f"""
            DELETE FROM {self._table}
            WHERE updated_at < NOW() - INTERVAL '{days} days'
            """,
        )
        # asyncpg returns "DELETE N" as the status string
        deleted = int(result.split()[-1]) if result else 0
        logger.info(
            "PostgresTaskStore: deleted %d tasks older than %d days.",
            deleted,
            days,
        )
        return deleted

    async def clear(self) -> None:
        """
        Delete ALL tasks from the store.

        WARNING: This is destructive and irreversible. Primarily for testing.
        """
        self._require_pool()
        await self._pool.execute(f"TRUNCATE TABLE {self._table}")
        logger.warning("PostgresTaskStore: all tasks cleared (TRUNCATE).")

    # ── Schema management ─────────────────────────────────────────────────────

    async def _ensure_schema(self) -> None:
        """
        Create the tasks table and indexes if they don't already exist.

        Idempotent — safe to call on every startup. Uses IF NOT EXISTS so
        it never fails or modifies an existing schema.
        """
        async with self._pool.acquire() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    id          TEXT        PRIMARY KEY,
                    context_id  TEXT        NOT NULL,
                    skill_id    TEXT,
                    state       TEXT        NOT NULL,
                    data        JSONB       NOT NULL,
                    created_at  TIMESTAMPTZ NOT NULL,
                    updated_at  TIMESTAMPTZ NOT NULL
                )
            """)

            # State index — list_by_state() and ready check
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {self._table}_state_idx
                ON {self._table} (state)
            """)

            # Context index — list_by_context()
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {self._table}_context_idx
                ON {self._table} (context_id)
            """)

            # Updated index — list_all() ordered scan
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {self._table}_updated_idx
                ON {self._table} (updated_at DESC)
            """)

        logger.debug("PostgresTaskStore: schema verified (table='%s').", self._table)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _require_pool(self) -> None:
        """
        Raise RuntimeError if connect() has not been called.

        Called at the top of every public method to give a clear error
        instead of an AttributeError on self._pool.
        """
        if self._pool is None:
            raise RuntimeError(
                "PostgresTaskStore is not connected. "
                "Call 'await store.connect()' before using the store, "
                "or use it as an async context manager: "
                "'async with PostgresTaskStore(...) as store: ...'"
            )
