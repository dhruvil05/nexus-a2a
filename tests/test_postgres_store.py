"""
tests/test_postgres_store.py

Tests for nexus_a2a/storage/postgres_store.py — PostgresTaskStore.

Since asyncpg requires a real PostgreSQL server, all tests use a fully
mocked asyncpg pool. This gives us 100% coverage of every code path
without needing a database in CI.

Integration tests that require a real PostgreSQL server are marked
@pytest.mark.integration and skipped unless NEXUS_TEST_PG_DSN is set.

Coverage:
  - __init__: default and custom arguments
  - connect(): pool created, schema created, ImportError for missing asyncpg
  - disconnect(): pool closed, safe to call twice
  - Async context manager (__aenter__ / __aexit__)
  - save(): upsert SQL, correct parameters
  - get(): found and not-found paths
  - delete(): correct SQL, not-found is silent
  - list_all(): empty and populated
  - list_by_state(): filters correctly
  - list_by_context(): filters and orders correctly
  - count() / count_by_state()
  - delete_older_than(): returns deleted count
  - clear(): truncates table
  - _require_pool(): raises RuntimeError before connect()
  - _ensure_schema(): correct table and index SQL
  - create_schema=False: schema creation skipped
  - Custom table name used in all queries
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from nexus_a2a.models.task import Task, TaskState
from nexus_a2a.storage.postgres_store import PostgresTaskStore, _DEFAULT_TABLE


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_task(state: TaskState = TaskState.SUBMITTED) -> Task:
    """Create a minimal Task for testing."""
    task = Task(
        id="task-test-001",
        context_id="ctx-001",
        skill_id="web_search",
        state=state,
    )
    return task


def make_row(task: Task) -> MagicMock:
    """Simulate an asyncpg Record row containing the task's JSON."""
    row = MagicMock()
    row.__getitem__ = lambda self, key: task.model_dump_json() if key == "data" else None
    return row


def make_pool(
    fetchrow_result=None,
    fetch_result=None,
    fetchval_result=0,
    execute_result="DELETE 0",
) -> MagicMock:
    """Build a mock asyncpg pool with configurable return values."""
    pool = AsyncMock()
    pool.execute   = AsyncMock(return_value=execute_result)
    pool.fetchrow  = AsyncMock(return_value=fetchrow_result)
    pool.fetch     = AsyncMock(return_value=fetch_result or [])
    pool.fetchval  = AsyncMock(return_value=fetchval_result)
    pool.close     = AsyncMock()

    # conn context manager for _ensure_schema
    conn = AsyncMock()
    conn.execute = AsyncMock()
    pool.acquire  = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=conn),
        __aexit__=AsyncMock(return_value=None),
    ))
    return pool


async def make_connected_store(
    table: str = _DEFAULT_TABLE,
    pool_kwargs: dict | None = None,
) -> PostgresTaskStore:
    """Return a PostgresTaskStore with a mocked pool already connected."""
    store = PostgresTaskStore(dsn="postgresql://test:test@localhost/test", table=table)
    store._pool = make_pool(**(pool_kwargs or {}))
    return store


# ── __init__ ──────────────────────────────────────────────────────────────────

class TestInit:

    def test_default_table(self):
        store = PostgresTaskStore(dsn="postgresql://x:x@localhost/db")
        assert store._table == _DEFAULT_TABLE

    def test_custom_table(self):
        store = PostgresTaskStore(dsn="...", table="my_custom_tasks")
        assert store._table == "my_custom_tasks"

    def test_default_pool_sizes(self):
        store = PostgresTaskStore(dsn="...")
        assert store._min_size == 2
        assert store._max_size == 10

    def test_custom_pool_sizes(self):
        store = PostgresTaskStore(dsn="...", min_size=5, max_size=20)
        assert store._min_size == 5
        assert store._max_size == 20

    def test_pool_is_none_before_connect(self):
        store = PostgresTaskStore(dsn="...")
        assert store._pool is None

    def test_create_schema_default_true(self):
        store = PostgresTaskStore(dsn="...")
        assert store._create_schema is True

    def test_create_schema_can_be_disabled(self):
        store = PostgresTaskStore(dsn="...", create_schema=False)
        assert store._create_schema is False


# ── connect() ─────────────────────────────────────────────────────────────────

class TestConnect:

    @pytest.mark.asyncio
    async def test_connect_creates_pool(self):
        store = PostgresTaskStore(dsn="postgresql://test@localhost/test")
        mock_pool = make_pool()
        mock_asyncpg = MagicMock()
        mock_asyncpg.create_pool = AsyncMock(return_value=mock_pool)

        with patch.dict("sys.modules", {"asyncpg": mock_asyncpg}):
            await store.connect()

        assert store._pool is mock_pool

    @pytest.mark.asyncio
    async def test_connect_creates_schema_by_default(self):
        store = PostgresTaskStore(dsn="postgresql://test@localhost/test")
        mock_pool = make_pool()
        mock_asyncpg = MagicMock()
        mock_asyncpg.create_pool = AsyncMock(return_value=mock_pool)

        with patch.dict("sys.modules", {"asyncpg": mock_asyncpg}):
            await store.connect()

        mock_pool.acquire().__aenter__.assert_called()

    @pytest.mark.asyncio
    async def test_connect_skips_schema_when_disabled(self):
        store = PostgresTaskStore(
            dsn="postgresql://test@localhost/test",
            create_schema=False,
        )
        mock_pool = make_pool()
        mock_asyncpg = MagicMock()
        mock_asyncpg.create_pool = AsyncMock(return_value=mock_pool)

        with patch.dict("sys.modules", {"asyncpg": mock_asyncpg}):
            await store.connect()

        mock_pool.acquire.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_raises_import_error_without_asyncpg(self):
        store = PostgresTaskStore(dsn="postgresql://test@localhost/test")
        with patch.dict("sys.modules", {"asyncpg": None}):
            with pytest.raises(ImportError, match="asyncpg"):
                await store.connect()

    @pytest.mark.asyncio
    async def test_connect_passes_pool_sizes(self):
        store = PostgresTaskStore(
            dsn="postgresql://test@localhost/test",
            min_size=3,
            max_size=15,
        )
        mock_pool = make_pool()
        mock_asyncpg = MagicMock()
        mock_asyncpg.create_pool = AsyncMock(return_value=mock_pool)

        with patch.dict("sys.modules", {"asyncpg": mock_asyncpg}):
            await store.connect()

        mock_asyncpg.create_pool.assert_awaited_once_with(
            dsn="postgresql://test@localhost/test",
            min_size=3,
            max_size=15,
        )


# ── disconnect() ──────────────────────────────────────────────────────────────

class TestDisconnect:

    @pytest.mark.asyncio
    async def test_disconnect_closes_pool(self):
        store = await make_connected_store()
        pool = store._pool
        await store.disconnect()
        pool.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disconnect_sets_pool_none(self):
        store = await make_connected_store()
        await store.disconnect()
        assert store._pool is None

    @pytest.mark.asyncio
    async def test_disconnect_twice_is_safe(self):
        store = await make_connected_store()
        await store.disconnect()
        await store.disconnect()   # should not raise

    @pytest.mark.asyncio
    async def test_disconnect_without_connect_is_safe(self):
        store = PostgresTaskStore(dsn="...")
        await store.disconnect()   # should not raise


# ── Async context manager ─────────────────────────────────────────────────────

class TestContextManager:

    @pytest.mark.asyncio
    async def test_aenter_connects(self):
        store = PostgresTaskStore(dsn="postgresql://test@localhost/test")
        mock_pool = make_pool()
        mock_asyncpg = MagicMock()
        mock_asyncpg.create_pool = AsyncMock(return_value=mock_pool)

        with patch.dict("sys.modules", {"asyncpg": mock_asyncpg}):
            async with store:
                assert store._pool is mock_pool

    @pytest.mark.asyncio
    async def test_aexit_disconnects(self):
        store = PostgresTaskStore(dsn="postgresql://test@localhost/test")
        mock_pool = make_pool()
        mock_asyncpg = MagicMock()
        mock_asyncpg.create_pool = AsyncMock(return_value=mock_pool)

        with patch.dict("sys.modules", {"asyncpg": mock_asyncpg}):
            async with store:
                pass

        assert store._pool is None
        mock_pool.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_aexit_disconnects_on_exception(self):
        store = PostgresTaskStore(dsn="postgresql://test@localhost/test")
        mock_pool = make_pool()
        mock_asyncpg = MagicMock()
        mock_asyncpg.create_pool = AsyncMock(return_value=mock_pool)

        with patch.dict("sys.modules", {"asyncpg": mock_asyncpg}):
            with pytest.raises(ValueError):
                async with store:
                    raise ValueError("intentional")

        mock_pool.close.assert_awaited_once()


# ── _require_pool() ───────────────────────────────────────────────────────────

class TestRequirePool:

    @pytest.mark.asyncio
    async def test_save_raises_before_connect(self):
        store = PostgresTaskStore(dsn="...")
        with pytest.raises(RuntimeError, match="not connected"):
            await store.save(make_task())

    @pytest.mark.asyncio
    async def test_get_raises_before_connect(self):
        store = PostgresTaskStore(dsn="...")
        with pytest.raises(RuntimeError, match="not connected"):
            await store.get("task-1")

    @pytest.mark.asyncio
    async def test_delete_raises_before_connect(self):
        store = PostgresTaskStore(dsn="...")
        with pytest.raises(RuntimeError, match="not connected"):
            await store.delete("task-1")

    @pytest.mark.asyncio
    async def test_list_all_raises_before_connect(self):
        store = PostgresTaskStore(dsn="...")
        with pytest.raises(RuntimeError, match="not connected"):
            await store.list_all()


# ── save() ────────────────────────────────────────────────────────────────────

class TestSave:

    @pytest.mark.asyncio
    async def test_save_calls_execute(self):
        store = await make_connected_store()
        task  = make_task()
        await store.save(task)
        store._pool.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_save_passes_correct_params(self):
        store = await make_connected_store()
        task  = make_task(state=TaskState.WORKING)
        await store.save(task)

        call_args = store._pool.execute.call_args
        # First positional arg after SQL is task.id
        params = call_args[0]
        assert params[1] == task.id
        assert params[2] == task.context_id
        assert params[3] == task.skill_id
        assert params[4] == task.state.value

    @pytest.mark.asyncio
    async def test_save_uses_upsert_sql(self):
        store = await make_connected_store()
        await store.save(make_task())
        sql = store._pool.execute.call_args[0][0]
        assert "ON CONFLICT" in sql
        assert "DO UPDATE" in sql

    @pytest.mark.asyncio
    async def test_save_uses_model_dump_json(self):
        store = await make_connected_store()
        task  = make_task()
        await store.save(task)
        sql_params = store._pool.execute.call_args[0]
        # data parameter should be valid JSON containing the task id
        data_json = sql_params[5]
        data = json.loads(data_json)
        assert data["id"] == task.id

    @pytest.mark.asyncio
    async def test_save_uses_correct_table(self):
        store = await make_connected_store(table="my_tasks")
        await store.save(make_task())
        sql = store._pool.execute.call_args[0][0]
        assert "my_tasks" in sql


# ── get() ─────────────────────────────────────────────────────────────────────

class TestGet:

    @pytest.mark.asyncio
    async def test_get_returns_task_when_found(self):
        task = make_task()
        row  = make_row(task)
        store = await make_connected_store(pool_kwargs={"fetchrow_result": row})
        result = await store.get("task-test-001")
        assert result is not None
        assert result.id == task.id

    @pytest.mark.asyncio
    async def test_get_returns_none_when_not_found(self):
        store = await make_connected_store(pool_kwargs={"fetchrow_result": None})
        result = await store.get("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_queries_correct_table(self):
        store = await make_connected_store(table="custom_table")
        store._pool.fetchrow = AsyncMock(return_value=None)
        await store.get("task-1")
        sql = store._pool.fetchrow.call_args[0][0]
        assert "custom_table" in sql

    @pytest.mark.asyncio
    async def test_get_passes_task_id(self):
        store = await make_connected_store()
        store._pool.fetchrow = AsyncMock(return_value=None)
        await store.get("specific-task-id")
        params = store._pool.fetchrow.call_args[0]
        assert params[1] == "specific-task-id"

    @pytest.mark.asyncio
    async def test_get_deserialises_task_correctly(self):
        original = make_task(state=TaskState.COMPLETED)
        row      = make_row(original)
        store    = await make_connected_store(pool_kwargs={"fetchrow_result": row})
        result   = await store.get(original.id)
        assert result.state == TaskState.COMPLETED
        assert result.context_id == original.context_id


# ── delete() ──────────────────────────────────────────────────────────────────

class TestDelete:

    @pytest.mark.asyncio
    async def test_delete_calls_execute(self):
        store = await make_connected_store()
        await store.delete("task-to-delete")
        store._pool.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete_passes_correct_id(self):
        store = await make_connected_store()
        await store.delete("my-task-id")
        params = store._pool.execute.call_args[0]
        assert params[1] == "my-task-id"

    @pytest.mark.asyncio
    async def test_delete_uses_correct_table(self):
        store = await make_connected_store(table="special_tasks")
        await store.delete("task-1")
        sql = store._pool.execute.call_args[0][0]
        assert "special_tasks" in sql

    @pytest.mark.asyncio
    async def test_delete_nonexistent_does_not_raise(self):
        store = await make_connected_store(pool_kwargs={"execute_result": "DELETE 0"})
        await store.delete("nonexistent-id")   # should not raise


# ── list_all() ────────────────────────────────────────────────────────────────

class TestListAll:

    @pytest.mark.asyncio
    async def test_list_all_returns_empty_when_no_tasks(self):
        store = await make_connected_store(pool_kwargs={"fetch_result": []})
        result = await store.list_all()
        assert result == []

    @pytest.mark.asyncio
    async def test_list_all_returns_all_tasks(self):
        t1, t2 = make_task(TaskState.SUBMITTED), make_task(TaskState.COMPLETED)
        t1.id = "task-001"; t2.id = "task-002"
        rows = [make_row(t1), make_row(t2)]
        store = await make_connected_store(pool_kwargs={"fetch_result": rows})
        result = await store.list_all()
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_list_all_deserialises_each_task(self):
        task = make_task(state=TaskState.FAILED)
        rows = [make_row(task)]
        store = await make_connected_store(pool_kwargs={"fetch_result": rows})
        result = await store.list_all()
        assert result[0].state == TaskState.FAILED

    @pytest.mark.asyncio
    async def test_list_all_uses_correct_table(self):
        store = await make_connected_store(table="prod_tasks")
        store._pool.fetch = AsyncMock(return_value=[])
        await store.list_all()
        sql = store._pool.fetch.call_args[0][0]
        assert "prod_tasks" in sql

    @pytest.mark.asyncio
    async def test_list_all_orders_by_updated_at(self):
        store = await make_connected_store()
        store._pool.fetch = AsyncMock(return_value=[])
        await store.list_all()
        sql = store._pool.fetch.call_args[0][0].upper()
        assert "ORDER BY" in sql and "UPDATED_AT" in sql


# ── list_by_state() ───────────────────────────────────────────────────────────

class TestListByState:

    @pytest.mark.asyncio
    async def test_list_by_state_passes_state_param(self):
        store = await make_connected_store(pool_kwargs={"fetch_result": []})
        await store.list_by_state("working")
        params = store._pool.fetch.call_args[0]
        assert params[1] == "working"

    @pytest.mark.asyncio
    async def test_list_by_state_returns_matching_tasks(self):
        task = make_task(TaskState.WORKING)
        rows = [make_row(task)]
        store = await make_connected_store(pool_kwargs={"fetch_result": rows})
        result = await store.list_by_state("working")
        assert len(result) == 1
        assert result[0].state == TaskState.WORKING

    @pytest.mark.asyncio
    async def test_list_by_state_uses_state_in_sql(self):
        store = await make_connected_store()
        store._pool.fetch = AsyncMock(return_value=[])
        await store.list_by_state("completed")
        sql = store._pool.fetch.call_args[0][0].upper()
        assert "WHERE" in sql and "STATE" in sql


# ── list_by_context() ─────────────────────────────────────────────────────────

class TestListByContext:

    @pytest.mark.asyncio
    async def test_list_by_context_passes_context_id(self):
        store = await make_connected_store(pool_kwargs={"fetch_result": []})
        await store.list_by_context("ctx-abc-123")
        params = store._pool.fetch.call_args[0]
        assert params[1] == "ctx-abc-123"

    @pytest.mark.asyncio
    async def test_list_by_context_returns_tasks(self):
        task = make_task()
        rows = [make_row(task)]
        store = await make_connected_store(pool_kwargs={"fetch_result": rows})
        result = await store.list_by_context("ctx-001")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_list_by_context_orders_by_created_at(self):
        store = await make_connected_store()
        store._pool.fetch = AsyncMock(return_value=[])
        await store.list_by_context("ctx-1")
        sql = store._pool.fetch.call_args[0][0].upper()
        assert "ORDER BY" in sql and "CREATED_AT" in sql


# ── count() / count_by_state() ────────────────────────────────────────────────

class TestCount:

    @pytest.mark.asyncio
    async def test_count_returns_integer(self):
        store = await make_connected_store(pool_kwargs={"fetchval_result": 5})
        result = await store.count()
        assert result == 5

    @pytest.mark.asyncio
    async def test_count_zero(self):
        store = await make_connected_store(pool_kwargs={"fetchval_result": 0})
        result = await store.count()
        assert result == 0

    @pytest.mark.asyncio
    async def test_count_uses_count_sql(self):
        store = await make_connected_store(pool_kwargs={"fetchval_result": 0})
        await store.count()
        sql = store._pool.fetchval.call_args[0][0].upper()
        assert "COUNT" in sql

    @pytest.mark.asyncio
    async def test_count_by_state_passes_state(self):
        store = await make_connected_store(pool_kwargs={"fetchval_result": 3})
        result = await store.count_by_state("submitted")
        assert result == 3
        params = store._pool.fetchval.call_args[0]
        assert params[1] == "submitted"

    @pytest.mark.asyncio
    async def test_count_by_state_zero(self):
        store = await make_connected_store(pool_kwargs={"fetchval_result": 0})
        result = await store.count_by_state("working")
        assert result == 0


# ── delete_older_than() ───────────────────────────────────────────────────────

class TestDeleteOlderThan:

    @pytest.mark.asyncio
    async def test_delete_older_than_returns_count(self):
        store = await make_connected_store(pool_kwargs={"execute_result": "DELETE 7"})
        result = await store.delete_older_than(days=30)
        assert result == 7

    @pytest.mark.asyncio
    async def test_delete_older_than_zero_deleted(self):
        store = await make_connected_store(pool_kwargs={"execute_result": "DELETE 0"})
        result = await store.delete_older_than(days=90)
        assert result == 0

    @pytest.mark.asyncio
    async def test_delete_older_than_uses_interval_sql(self):
        store = await make_connected_store(pool_kwargs={"execute_result": "DELETE 0"})
        await store.delete_older_than(days=14)
        sql = store._pool.execute.call_args[0][0]
        assert "14 days" in sql
        assert "updated_at" in sql


# ── clear() ───────────────────────────────────────────────────────────────────

class TestClear:

    @pytest.mark.asyncio
    async def test_clear_uses_truncate(self):
        store = await make_connected_store()
        await store.clear()
        sql = store._pool.execute.call_args[0][0].upper()
        assert "TRUNCATE" in sql

    @pytest.mark.asyncio
    async def test_clear_uses_correct_table(self):
        store = await make_connected_store(table="archive_tasks")
        await store.clear()
        sql = store._pool.execute.call_args[0][0]
        assert "archive_tasks" in sql


# ── _ensure_schema() ──────────────────────────────────────────────────────────

class TestEnsureSchema:

    @pytest.mark.asyncio
    async def test_schema_creates_table(self):
        store = await make_connected_store()
        conn  = store._pool.acquire().__aenter__.return_value
        await store._ensure_schema()
        sqls = [c[0][0].upper() for c in conn.execute.call_args_list]
        assert any("CREATE TABLE" in sql for sql in sqls)

    @pytest.mark.asyncio
    async def test_schema_uses_if_not_exists(self):
        store = await make_connected_store()
        conn  = store._pool.acquire().__aenter__.return_value
        await store._ensure_schema()
        sqls = [c[0][0].upper() for c in conn.execute.call_args_list]
        assert all("IF NOT EXISTS" in sql for sql in sqls)

    @pytest.mark.asyncio
    async def test_schema_creates_state_index(self):
        store = await make_connected_store()
        conn  = store._pool.acquire().__aenter__.return_value
        await store._ensure_schema()
        sqls = " ".join(c[0][0] for c in conn.execute.call_args_list)
        assert "state_idx" in sqls

    @pytest.mark.asyncio
    async def test_schema_creates_context_index(self):
        store = await make_connected_store()
        conn  = store._pool.acquire().__aenter__.return_value
        await store._ensure_schema()
        sqls = " ".join(c[0][0] for c in conn.execute.call_args_list)
        assert "context_idx" in sqls

    @pytest.mark.asyncio
    async def test_schema_creates_updated_index(self):
        store = await make_connected_store()
        conn  = store._pool.acquire().__aenter__.return_value
        await store._ensure_schema()
        sqls = " ".join(c[0][0] for c in conn.execute.call_args_list)
        assert "updated_idx" in sqls

    @pytest.mark.asyncio
    async def test_schema_uses_custom_table_name(self):
        store = await make_connected_store(table="my_agent_tasks")
        conn  = store._pool.acquire().__aenter__.return_value
        await store._ensure_schema()
        sqls = " ".join(c[0][0] for c in conn.execute.call_args_list)
        assert "my_agent_tasks" in sqls

    @pytest.mark.asyncio
    async def test_schema_includes_jsonb_column(self):
        store = await make_connected_store()
        conn  = store._pool.acquire().__aenter__.return_value
        await store._ensure_schema()
        sqls = " ".join(c[0][0].upper() for c in conn.execute.call_args_list)
        assert "JSONB" in sqls

    @pytest.mark.asyncio
    async def test_schema_includes_timestamptz_columns(self):
        store = await make_connected_store()
        conn  = store._pool.acquire().__aenter__.return_value
        await store._ensure_schema()
        sqls = " ".join(c[0][0].upper() for c in conn.execute.call_args_list)
        assert "TIMESTAMPTZ" in sqls


# ── Integration marker (skipped unless NEXUS_TEST_PG_DSN is set) ─────────────

@pytest.mark.integration
class TestPostgresIntegration:
    """
    Real database tests — skipped unless NEXUS_TEST_PG_DSN env var is set.

    Run with:
        NEXUS_TEST_PG_DSN="postgresql://user:pass@localhost/nexus_test" \
        pytest tests/test_postgres_store.py -m integration
    """

    @pytest.fixture(autouse=True)
    def skip_without_dsn(self):
        import os
        if not os.environ.get("NEXUS_TEST_PG_DSN"):
            pytest.skip("NEXUS_TEST_PG_DSN not set — skipping integration tests")

    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        import os
        dsn = os.environ["NEXUS_TEST_PG_DSN"]
        table = "nexus_a2a_tasks_test"

        async with PostgresTaskStore(dsn=dsn, table=table) as store:
            # Create
            task = make_task()
            await store.save(task)

            # Read
            fetched = await store.get(task.id)
            assert fetched is not None
            assert fetched.id == task.id

            # Update
            task.state = TaskState.COMPLETED
            await store.save(task)
            updated = await store.get(task.id)
            assert updated.state == TaskState.COMPLETED

            # List
            all_tasks = await store.list_all()
            assert any(t.id == task.id for t in all_tasks)

            # Count
            count = await store.count()
            assert count >= 1

            # Delete
            await store.delete(task.id)
            assert await store.get(task.id) is None

            # Cleanup
            await store.clear()
            assert await store.count() == 0