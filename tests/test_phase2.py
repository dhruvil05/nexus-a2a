"""
tests/test_phase2.py

Tests for Phase 2: InMemoryTaskStore, TaskManager, A2AHttpClient, AgentRegistry.
Run with:  uv run pytest tests/test_phase2.py -v
"""

from __future__ import annotations

import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

from nexus_a2a.models.agent import AgentCard, AgentSkill, AgentCapabilities
from nexus_a2a.models.task import Artifact, Message, Part, PartType, Task, TaskState
from nexus_a2a.storage.task_store import InMemoryTaskStore
from nexus_a2a.core.task_manager import (
    TaskManager,
    TaskNotFoundError,
    TaskAlreadyDoneError,
)
from nexus_a2a.transport.http_client import (
    A2AHttpClient,
    AgentUnreachableError,
    RemoteAgentError,
)
from nexus_a2a.core.registry import AgentRegistry


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def sample_card() -> AgentCard:
    return AgentCard(
        name="TestAgent",
        description="A test agent.",
        url="http://localhost:9001",
        skills=[AgentSkill(id="search", name="Search", description="Searches.")],
    )

@pytest.fixture
def user_msg() -> Message:
    return Message.user_text("Do something useful")

@pytest.fixture
def store() -> InMemoryTaskStore:
    return InMemoryTaskStore()

@pytest.fixture
def manager() -> TaskManager:
    return TaskManager()


# ══════════════════════════════════════════════════════════════════════════════
# InMemoryTaskStore
# ══════════════════════════════════════════════════════════════════════════════

class TestInMemoryTaskStore:
    async def test_save_and_get(self, store, user_msg):
        task = Task.create(initial_message=user_msg)
        await store.save(task)
        fetched = await store.get(task.id)
        assert fetched is not None
        assert fetched.id == task.id

    async def test_get_missing_returns_none(self, store):
        result = await store.get("nonexistent-id")
        assert result is None

    async def test_delete_removes_task(self, store, user_msg):
        task = Task.create(initial_message=user_msg)
        await store.save(task)
        await store.delete(task.id)
        assert await store.get(task.id) is None

    async def test_delete_missing_is_silent(self, store):
        await store.delete("ghost-id")  # should not raise

    async def test_list_all(self, store, user_msg):
        t1 = Task.create(initial_message=user_msg)
        t2 = Task.create(initial_message=user_msg)
        await store.save(t1)
        await store.save(t2)
        all_tasks = await store.list_all()
        assert len(all_tasks) == 2

    async def test_count(self, store, user_msg):
        assert await store.count() == 0
        await store.save(Task.create(initial_message=user_msg))
        assert await store.count() == 1

    async def test_clear(self, store, user_msg):
        await store.save(Task.create(initial_message=user_msg))
        await store.clear()
        assert await store.count() == 0

    async def test_save_overwrites(self, store, user_msg):
        task = Task.create(initial_message=user_msg)
        await store.save(task)
        task.transition(TaskState.WORKING)
        await store.save(task)
        fetched = await store.get(task.id)
        assert fetched.state == TaskState.WORKING


# ══════════════════════════════════════════════════════════════════════════════
# TaskManager
# ══════════════════════════════════════════════════════════════════════════════

class TestTaskManager:
    async def test_create_returns_submitted_task(self, manager, user_msg):
        task = await manager.create(user_msg, skill_id="search")
        assert task.state == TaskState.SUBMITTED
        assert task.skill_id == "search"
        assert len(task.history) == 1

    async def test_get_existing_task(self, manager, user_msg):
        task = await manager.create(user_msg)
        fetched = await manager.get(task.id)
        assert fetched.id == task.id

    async def test_get_missing_raises(self, manager):
        with pytest.raises(TaskNotFoundError):
            await manager.get("no-such-id")

    async def test_start_moves_to_working(self, manager, user_msg):
        task = await manager.create(user_msg)
        updated = await manager.start(task.id)
        assert updated.state == TaskState.WORKING

    async def test_complete_with_artifact(self, manager, user_msg):
        task = await manager.create(user_msg)
        await manager.start(task.id)
        art = Artifact(
            name="output",
            parts=[Part(type=PartType.TEXT, content="Done!")],
        )
        completed = await manager.complete(task.id, artifact=art)
        assert completed.state == TaskState.COMPLETED
        assert len(completed.artifacts) == 1

    async def test_complete_with_reply_message(self, manager, user_msg):
        task = await manager.create(user_msg)
        await manager.start(task.id)
        reply = Message.agent_text("Here is your result.")
        completed = await manager.complete(task.id, reply_message=reply)
        assert completed.history[-1].text() == "Here is your result."

    async def test_fail_requires_error(self, manager, user_msg):
        task = await manager.create(user_msg)
        await manager.start(task.id)
        failed = await manager.fail(task.id, error="Timeout occurred")
        assert failed.state == TaskState.FAILED
        assert failed.error == "Timeout occurred"

    async def test_cancel_from_submitted(self, manager, user_msg):
        task = await manager.create(user_msg)
        cancelled = await manager.cancel(task.id)
        assert cancelled.state == TaskState.CANCELLED

    async def test_cancel_from_working(self, manager, user_msg):
        task = await manager.create(user_msg)
        await manager.start(task.id)
        cancelled = await manager.cancel(task.id)
        assert cancelled.state == TaskState.CANCELLED

    async def test_operating_on_done_task_raises(self, manager, user_msg):
        task = await manager.create(user_msg)
        await manager.start(task.id)
        await manager.complete(task.id)
        with pytest.raises(TaskAlreadyDoneError):
            await manager.start(task.id)

    async def test_request_and_provide_input(self, manager, user_msg):
        task = await manager.create(user_msg)
        await manager.start(task.id)
        prompt = Message.agent_text("What is your budget?")
        await manager.request_input(task.id, prompt=prompt)

        reply = Message.user_text("$500")
        resumed = await manager.provide_input(task.id, message=reply)
        assert resumed.state == TaskState.WORKING
        assert resumed.history[-1].text() == "$500"

    async def test_provide_input_on_wrong_state_raises(self, manager, user_msg):
        task = await manager.create(user_msg)
        # Task is SUBMITTED, not INPUT_REQUIRED
        with pytest.raises(ValueError, match="input_required"):
            await manager.provide_input(task.id, message=Message.user_text("nope"))

    async def test_iter_by_state(self, manager, user_msg):
        t1 = await manager.create(user_msg)
        t2 = await manager.create(user_msg)
        await manager.start(t1.id)
        # t2 stays SUBMITTED

        working = [t async for t in manager.iter_by_state(TaskState.WORKING)]
        submitted = [t async for t in manager.iter_by_state(TaskState.SUBMITTED)]

        assert len(working) == 1
        assert working[0].id == t1.id
        assert len(submitted) == 1

    async def test_delete_task(self, manager, user_msg):
        task = await manager.create(user_msg)
        await manager.delete(task.id)
        with pytest.raises(TaskNotFoundError):
            await manager.get(task.id)

    async def test_list_all(self, manager, user_msg):
        await manager.create(user_msg)
        await manager.create(user_msg)
        all_tasks = await manager.list_all()
        assert len(all_tasks) == 2


# ══════════════════════════════════════════════════════════════════════════════
# A2AHttpClient
# ══════════════════════════════════════════════════════════════════════════════

# Helper: build a fake JSON-RPC 2.0 success response
def _rpc_response(result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": "test-id", "result": result}

# Helper: build a fake JSON-RPC 2.0 error response
def _rpc_error(code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": "test-id", "error": {"code": code, "message": message}}


class TestA2AHttpClient:
    """
    We mock httpx.AsyncClient so no real network calls are made.
    Tests focus on: envelope building, error translation, retry logic.
    """

    async def test_fetch_agent_card_success(self, sample_card):
        card_dict = sample_card.to_well_known_dict()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value=card_dict)

        with patch("nexus_a2a.transport.http_client.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get = AsyncMock(return_value=mock_response)
            instance.aclose = AsyncMock()
            MockClient.return_value.__aenter__ = AsyncMock(return_value=instance)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            async with A2AHttpClient("http://localhost:9001") as client:
                client._client = instance
                fetched = await client.fetch_agent_card()

        assert fetched.name == "TestAgent"

    async def test_rpc_error_raises_remote_agent_error(self):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value=_rpc_error(-32601, "Method not found"))

        with patch("nexus_a2a.transport.http_client.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post = AsyncMock(return_value=mock_response)
            instance.aclose = AsyncMock()
            MockClient.return_value.__aenter__ = AsyncMock(return_value=instance)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            async with A2AHttpClient("http://localhost:9001", max_retries=1) as client:
                client._client = instance
                with pytest.raises(RemoteAgentError, match="Method not found"):
                    await client._rpc("message/send", {})

    async def test_unreachable_raises_after_retries(self):
        with patch("nexus_a2a.transport.http_client.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
            instance.aclose = AsyncMock()
            MockClient.return_value.__aenter__ = AsyncMock(return_value=instance)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            async with A2AHttpClient("http://localhost:9001", max_retries=2) as client:
                client._client = instance
                with pytest.raises(AgentUnreachableError):
                    await client._rpc("message/send", {})

        # Confirm it retried max_retries times
        assert instance.post.call_count == 2

    async def test_require_client_outside_context_raises(self):
        client = A2AHttpClient("http://localhost:9001")
        with pytest.raises(RuntimeError, match="async context manager"):
            client._require_client()


# ══════════════════════════════════════════════════════════════════════════════
# AgentRegistry
# ══════════════════════════════════════════════════════════════════════════════

class TestAgentRegistry:
    """
    We mock A2AHttpClient so the registry tests don't need a live server.
    """

    def _patch_client(self, card: AgentCard):
        """Context manager that makes A2AHttpClient.fetch_agent_card() return card."""
        mock_client = AsyncMock()
        mock_client.fetch_agent_card = AsyncMock(return_value=card)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        return patch(
            "nexus_a2a.core.registry.A2AHttpClient",
            return_value=mock_client,
        )

    async def test_register_url_fetches_and_stores(self, sample_card):
        registry = AgentRegistry()
        with self._patch_client(sample_card):
            card = await registry.register_url("http://localhost:9001")
        assert card.name == "TestAgent"
        assert registry.get_by_name("TestAgent") is not None

    async def test_register_card_directly(self, sample_card):
        registry = AgentRegistry()
        await registry.register_card(sample_card)
        assert registry.get_by_url("http://localhost:9001") is not None

    async def test_unregister(self, sample_card):
        registry = AgentRegistry()
        await registry.register_card(sample_card)
        await registry.unregister("http://localhost:9001")
        assert registry.get_by_url("http://localhost:9001") is None

    async def test_find_by_skill_returns_match(self, sample_card):
        registry = AgentRegistry()
        await registry.register_card(sample_card)
        results = registry.find_by_skill("search")
        assert len(results) == 1
        assert results[0].name == "TestAgent"

    async def test_find_by_skill_no_match(self, sample_card):
        registry = AgentRegistry()
        await registry.register_card(sample_card)
        results = registry.find_by_skill("nonexistent_skill")
        assert results == []

    async def test_list_healthy(self, sample_card):
        registry = AgentRegistry()
        await registry.register_card(sample_card)
        healthy = registry.list_healthy()
        assert len(healthy) == 1

    async def test_check_health_success(self, sample_card):
        registry = AgentRegistry()
        await registry.register_card(sample_card)
        with self._patch_client(sample_card):
            result = await registry.check_health("http://localhost:9001")
        assert result is True
        assert registry.is_healthy("http://localhost:9001") is True

    async def test_check_health_failure_marks_unhealthy(self, sample_card):
        registry = AgentRegistry()
        await registry.register_card(sample_card)

        mock_client = AsyncMock()
        mock_client.fetch_agent_card = AsyncMock(
            side_effect=AgentUnreachableError("http://localhost:9001", "refused")
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("nexus_a2a.core.registry.A2AHttpClient", return_value=mock_client):
            result = await registry.check_health("http://localhost:9001")

        assert result is False
        assert registry.is_healthy("http://localhost:9001") is False

    async def test_find_by_skill_excludes_unhealthy(self, sample_card):
        registry = AgentRegistry()
        await registry.register_card(sample_card)
        # Manually mark unhealthy
        registry._entries["http://localhost:9001"].healthy = False
        results = registry.find_by_skill("search")
        assert results == []

    async def test_summary(self, sample_card):
        registry = AgentRegistry()
        await registry.register_card(sample_card)
        s = registry.summary()
        assert s["total"] == 1
        assert s["healthy"] == 1
        assert s["agents"][0]["name"] == "TestAgent"

    async def test_check_all_health(self, sample_card):
        registry = AgentRegistry()
        await registry.register_card(sample_card)
        with self._patch_client(sample_card):
            results = await registry.check_all_health()
        assert "http://localhost:9001" in results
        assert results["http://localhost:9001"] is True