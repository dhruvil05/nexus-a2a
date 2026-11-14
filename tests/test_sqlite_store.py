"""
tests/test_sqlite_store.py

Tests for nexus_a2a/storage/sqlite_store.py — SqliteTaskStore.

Unlike the Redis and Postgres stores, SQLite needs no server, so these run
against real database files (pytest's tmp_path), mirroring the coverage of
tests/test_postgres_store.py:

  - __init__: defaults, table-name validation
  - connect()/disconnect(): idempotent, lazy open on first use, context manager
  - save(): insert and upsert (created_at kept), owner preserved
  - get()/delete()/list_all(): found, not found, ordering
  - list_by_state()/list_by_context()/count()/count_by_state()
  - delete_older_than()/clear()
  - Durability: a task survives closing and reopening the file
  - Custom table names stay isolated
  - Works as a TaskManager store and from [storage] backend = "sqlite"
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from nexus_a2a import SqliteTaskStore, TaskManager
from nexus_a2a.config import ConfigError, NexusConfig
from nexus_a2a.models.task import Message, Task, TaskState
from nexus_a2a.storage.sqlite_store import _DEFAULT_TABLE, _timestamp


def make_task(
    task_id: str = "task-1",
    state: TaskState = TaskState.SUBMITTED,
    context_id: str = "ctx-1",
) -> Task:
    return Task(id=task_id, context_id=context_id, skill_id="search", state=state)


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "tasks.db"


@pytest.fixture
async def store(db: Path):
    s = SqliteTaskStore(db)
    yield s
    await s.disconnect()


# ── __init__ ──────────────────────────────────────────────────────────────────


class TestInit:
    def test_default_table(self, db: Path) -> None:
        assert SqliteTaskStore(db)._table == _DEFAULT_TABLE

    def test_path_is_kept(self, db: Path) -> None:
        assert SqliteTaskStore(db).path == str(db)

    def test_nothing_opened_before_use(self, db: Path) -> None:
        SqliteTaskStore(db)
        assert not db.exists()

    @pytest.mark.parametrize(
        "table", ["tasks; DROP TABLE x", "1tasks", "a b", "", "x" * 64, "s.t"]
    )
    def test_invalid_table_rejected(self, db: Path, table: str) -> None:
        with pytest.raises(ValueError, match="Invalid table name"):
            SqliteTaskStore(db, table=table)


# ── Lifecycle ─────────────────────────────────────────────────────────────────


class TestLifecycle:
    async def test_connect_creates_file_and_parents(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "dir" / "tasks.db"
        store = SqliteTaskStore(path)
        await store.connect()
        await store.disconnect()
        assert path.exists()

    async def test_connect_twice_is_safe(self, store: SqliteTaskStore) -> None:
        await store.connect()
        await store.connect()
        assert await store.count() == 0

    async def test_disconnect_twice_is_safe(self, store: SqliteTaskStore) -> None:
        await store.connect()
        await store.disconnect()
        await store.disconnect()

    async def test_disconnect_without_connect_is_safe(self, db: Path) -> None:
        await SqliteTaskStore(db).disconnect()

    async def test_opens_lazily_on_first_use(self, store: SqliteTaskStore) -> None:
        await store.save(make_task())
        assert await store.get("task-1") is not None

    async def test_usable_again_after_disconnect(self, store: SqliteTaskStore) -> None:
        await store.save(make_task())
        await store.disconnect()
        assert await store.get("task-1") is not None

    async def test_context_manager(self, db: Path) -> None:
        async with SqliteTaskStore(db) as store:
            await store.save(make_task())
            assert store._conn is not None
        assert store._conn is None

    async def test_context_manager_closes_on_exception(self, db: Path) -> None:
        store = SqliteTaskStore(db)
        with pytest.raises(RuntimeError):
            async with store:
                raise RuntimeError("boom")
        assert store._conn is None

    async def test_schema_and_indexes(self, store: SqliteTaskStore, db: Path) -> None:
        await store.connect()
        with sqlite3.connect(db) as conn:
            cols = {r[1] for r in conn.execute(f"PRAGMA table_info({_DEFAULT_TABLE})")}
            idx = {r[1] for r in conn.execute(f"PRAGMA index_list({_DEFAULT_TABLE})")}
        conn.close()
        assert cols == {
            "id",
            "context_id",
            "skill_id",
            "state",
            "data",
            "created_at",
            "updated_at",
        }
        assert {
            f"{_DEFAULT_TABLE}_state_idx",
            f"{_DEFAULT_TABLE}_context_id_idx",
            f"{_DEFAULT_TABLE}_updated_at_idx",
        } <= idx

    async def test_memory_database(self) -> None:
        async with SqliteTaskStore(":memory:") as store:
            await store.save(make_task())
            assert await store.count() == 1


# ── save / get ────────────────────────────────────────────────────────────────


class TestSaveGet:
    async def test_round_trip(self, store: SqliteTaskStore) -> None:
        task = make_task()
        task.add_message(Message.user_text("hello"))
        await store.save(task)
        got = await store.get("task-1")
        assert got is not None
        assert got.model_dump() == task.model_dump()

    async def test_get_missing_returns_none(self, store: SqliteTaskStore) -> None:
        assert await store.get("nope") is None

    async def test_save_overwrites(self, store: SqliteTaskStore) -> None:
        task = make_task()
        await store.save(task)
        task.transition(TaskState.WORKING)
        await store.save(task)
        got = await store.get("task-1")
        assert got is not None and got.state == TaskState.WORKING
        assert await store.count() == 1

    async def test_upsert_keeps_created_at_column(
        self, store: SqliteTaskStore, db: Path
    ) -> None:
        task = make_task()
        await store.save(task)
        first = _timestamp(task.created_at)
        task.created_at = task.created_at + timedelta(days=1)
        await store.save(task)
        with sqlite3.connect(db) as conn:
            row = conn.execute(f"SELECT created_at FROM {_DEFAULT_TABLE}").fetchone()
        conn.close()
        assert row[0] == first

    async def test_owner_is_preserved(self, store: SqliteTaskStore) -> None:
        task = make_task()
        task.owner = "http://caller:9001"
        await store.save(task)
        got = await store.get("task-1")
        assert got is not None and got.owner == "http://caller:9001"

    async def test_concurrent_saves(self, store: SqliteTaskStore) -> None:
        await asyncio.gather(*(store.save(make_task(f"t-{i}")) for i in range(50)))
        assert await store.count() == 50


# ── Durability ────────────────────────────────────────────────────────────────


class TestDurability:
    async def test_task_survives_reopen(self, db: Path) -> None:
        task = make_task(state=TaskState.WORKING)
        task.transition(TaskState.INPUT_REQUIRED)
        async with SqliteTaskStore(db) as first:
            await first.save(task)

        async with SqliteTaskStore(db) as second:
            got = await second.get("task-1")
        assert got is not None
        assert got.state == TaskState.INPUT_REQUIRED

    async def test_paused_task_resumes_after_restart(self, db: Path) -> None:
        """The case that motivated this store: a multi-turn task paused for
        input, the process restarts, the user's answer still resumes it."""
        async with SqliteTaskStore(db) as store:
            manager = TaskManager(store=store)
            task = await manager.create(Message.user_text("hi"), skill_id="search")
            await manager.start(task.id)
            await manager.request_input(task.id, Message.agent_text("which one?"))

        async with SqliteTaskStore(db) as store:
            manager = TaskManager(store=store)
            resumed = await manager.provide_input(task.id, Message.user_text("B"))
        assert resumed.state == TaskState.WORKING
        assert [m.parts[0].content for m in resumed.history] == [
            "hi",
            "which one?",
            "B",
        ]


# ── delete / list / count ─────────────────────────────────────────────────────


class TestQueries:
    async def test_delete(self, store: SqliteTaskStore) -> None:
        await store.save(make_task())
        await store.delete("task-1")
        assert await store.get("task-1") is None

    async def test_delete_missing_is_silent(self, store: SqliteTaskStore) -> None:
        await store.delete("nope")

    async def test_list_all_empty(self, store: SqliteTaskStore) -> None:
        assert await store.list_all() == []

    async def test_list_all_newest_first(self, store: SqliteTaskStore) -> None:
        base = datetime(2026, 1, 1, tzinfo=UTC)
        for i, name in enumerate(["old", "new", "mid"]):
            task = make_task(name)
            # Whole seconds: isoformat() would drop the microseconds here.
            task.updated_at = base + timedelta(seconds={0: 0, 1: 2, 2: 1}[i])
            await store.save(task)
        assert [t.id for t in await store.list_all()] == ["new", "mid", "old"]

    async def test_list_by_state(self, store: SqliteTaskStore) -> None:
        await store.save(make_task("a", TaskState.WORKING))
        await store.save(make_task("b", TaskState.SUBMITTED))
        await store.save(make_task("c", TaskState.WORKING))
        working = await store.list_by_state("working")
        assert {t.id for t in working} == {"a", "c"}
        assert await store.list_by_state("failed") == []

    async def test_list_by_context_oldest_first(self, store: SqliteTaskStore) -> None:
        base = datetime(2026, 1, 1, tzinfo=UTC)
        for i, name in enumerate(["second", "first"]):
            task = make_task(name, context_id="conv")
            task.created_at = base - timedelta(minutes=i)
            await store.save(task)
        await store.save(make_task("other", context_id="elsewhere"))
        assert [t.id for t in await store.list_by_context("conv")] == [
            "first",
            "second",
        ]

    async def test_count_and_count_by_state(self, store: SqliteTaskStore) -> None:
        assert await store.count() == 0
        await store.save(make_task("a", TaskState.WORKING))
        await store.save(make_task("b", TaskState.SUBMITTED))
        assert await store.count() == 2
        assert await store.count_by_state("working") == 1
        assert await store.count_by_state("failed") == 0

    async def test_delete_older_than(self, store: SqliteTaskStore) -> None:
        old = make_task("old")
        old.updated_at = datetime.now(UTC) - timedelta(days=10)
        await store.save(old)
        await store.save(make_task("fresh"))
        assert await store.delete_older_than(7) == 1
        assert [t.id for t in await store.list_all()] == ["fresh"]

    async def test_clear(self, store: SqliteTaskStore) -> None:
        await store.save(make_task("a"))
        await store.save(make_task("b"))
        await store.clear()
        assert await store.count() == 0

    async def test_ping(self, store: SqliteTaskStore) -> None:
        await store.ping()

    async def test_custom_tables_are_isolated(self, db: Path) -> None:
        async with SqliteTaskStore(db, table="staging") as a:
            async with SqliteTaskStore(db, table="prod") as b:
                await a.save(make_task("only-staging"))
                assert await b.get("only-staging") is None
                assert await a.count() == 1


# ── Timestamps ────────────────────────────────────────────────────────────────


class TestTimestamp:
    def test_fixed_width_even_with_zero_microseconds(self) -> None:
        ts = _timestamp(datetime(2026, 1, 1, tzinfo=UTC))
        assert ts == "2026-01-01T00:00:00.000000Z"

    def test_naive_treated_as_utc(self) -> None:
        assert _timestamp(datetime(2026, 1, 1)) == "2026-01-01T00:00:00.000000Z"

    def test_converted_to_utc(self) -> None:
        from datetime import timezone

        plus2 = datetime(2026, 1, 1, 2, tzinfo=timezone(timedelta(hours=2)))
        assert _timestamp(plus2) == "2026-01-01T00:00:00.000000Z"


# ── Config ────────────────────────────────────────────────────────────────────


class TestConfig:
    def test_sqlite_backend_builds_store(self, db: Path) -> None:
        cfg = NexusConfig.from_dict(
            {"agent": {"name": "a"}, "storage": {"backend": "sqlite", "url": str(db)}}
        )
        store = cfg.build_task_store()
        assert isinstance(store, SqliteTaskStore)
        assert store.path == str(db)

    def test_sqlite_backend_requires_path(self) -> None:
        with pytest.raises(ConfigError, match="database file path"):
            NexusConfig.from_dict(
                {"agent": {"name": "a"}, "storage": {"backend": "sqlite"}}
            )
