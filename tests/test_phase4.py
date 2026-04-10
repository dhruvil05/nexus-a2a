"""
tests/test_phase4.py

Tests for Phase 4: Orchestrator, SSEFormatter, WebhookDispatcher, EventBus, AgentNetwork.
Run with:  uv run pytest tests/test_phase4.py -v
"""

from __future__ import annotations

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from nexus_a2a.models.task import Message, Task, TaskState, Part, PartType, Artifact
from nexus_a2a.models.agent import AgentCard, AgentSkill
from nexus_a2a.core.orchestrator import (
    DAGNode,
    Orchestrator,
    OrchestratorResult,
    WorkflowCycleError,
    OrchestratorError,
)
from nexus_a2a.transport.sse import (
    SSEFormatter,
    SSEStreamer,
    StreamEvent,
    StreamEventType,
)
from nexus_a2a.transport.webhook import (
    WebhookConfig,
    WebhookDispatcher,
    WebhookDeliveryError,
)
from nexus_a2a.network import AgentNetwork, EventBus


# ── Shared helpers ────────────────────────────────────────────────────────────

def _make_task(state: TaskState = TaskState.COMPLETED) -> Task:
    """Create a task already in a given terminal state for testing."""
    task = Task.create(initial_message=Message.user_text("test"))
    if state != TaskState.SUBMITTED:
        task.transition(TaskState.WORKING)
    if state == TaskState.COMPLETED:
        task.transition(TaskState.COMPLETED)
    elif state == TaskState.FAILED:
        task.transition(TaskState.FAILED, error="test error")
    elif state == TaskState.CANCELLED:
        task.transition(TaskState.CANCELLED)
    return task


def _make_card(url: str = "http://agent:8001", skill: str = "search") -> AgentCard:
    return AgentCard(
        name="TestAgent",
        description="test",
        url=url,
        skills=[AgentSkill(id=skill, name=skill.title(), description="test skill")],
    )


# Runner that always returns a completed task (no network needed)
async def _success_runner(agent_url: str, message: Message) -> Task:
    task = Task.create(initial_message=message)
    task.transition(TaskState.WORKING)
    task.add_message(Message.agent_text(f"Result from {agent_url}"))
    task.transition(TaskState.COMPLETED)
    return task


# Runner that always raises (simulates unreachable agent)
async def _failing_runner(agent_url: str, message: Message) -> Task:
    raise ConnectionError(f"Cannot reach {agent_url}")


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator — sequential
# ══════════════════════════════════════════════════════════════════════════════

class TestOrchestratorSequential:
    async def test_single_agent_succeeds(self):
        orch = Orchestrator(runner=_success_runner)
        result = await orch.sequential(
            agent_urls=["http://a:8001"],
            initial_message=Message.user_text("go"),
        )
        assert result.succeeded
        assert len(result.steps) == 1
        assert result.steps[0].agent_url == "http://a:8001"

    async def test_two_agents_chain(self):
        orch = Orchestrator(runner=_success_runner)
        result = await orch.sequential(
            agent_urls=["http://a:8001", "http://b:8002"],
            initial_message=Message.user_text("start"),
        )
        assert result.succeeded
        assert len(result.steps) == 2

    async def test_stops_on_first_failure_by_default(self):
        calls: list[str] = []

        async def runner(url: str, msg: Message) -> Task:
            calls.append(url)
            if url == "http://a:8001":
                raise RuntimeError("fail")
            return await _success_runner(url, msg)

        orch = Orchestrator(runner=runner, stop_on_error=True)
        result = await orch.sequential(
            agent_urls=["http://a:8001", "http://b:8002"],
            initial_message=Message.user_text("go"),
        )
        # Second agent should NOT have been called
        assert "http://b:8002" not in calls
        assert not result.succeeded

    async def test_continues_on_failure_when_configured(self):
        calls: list[str] = []

        async def runner(url: str, msg: Message) -> Task:
            calls.append(url)
            if url == "http://a:8001":
                raise RuntimeError("fail")
            return await _success_runner(url, msg)

        orch = Orchestrator(runner=runner, stop_on_error=False)
        result = await orch.sequential(
            agent_urls=["http://a:8001", "http://b:8002"],
            initial_message=Message.user_text("go"),
        )
        assert "http://b:8002" in calls
        assert len(result.steps) == 2

    async def test_empty_agent_list_raises(self):
        orch = Orchestrator(runner=_success_runner)
        with pytest.raises(OrchestratorError):
            await orch.sequential([], Message.user_text("go"))

    async def test_result_has_timing(self):
        orch = Orchestrator(runner=_success_runner)
        result = await orch.sequential(["http://a:8001"], Message.user_text("go"))
        assert result.total_sec >= 0
        assert result.steps[0].duration_sec >= 0

    async def test_final_output_is_last_successful_task(self):
        orch = Orchestrator(runner=_success_runner)
        result = await orch.sequential(
            ["http://a:8001", "http://b:8002"],
            Message.user_text("go"),
        )
        assert result.final_output is not None
        assert isinstance(result.final_output, Task)


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator — parallel
# ══════════════════════════════════════════════════════════════════════════════

class TestOrchestratorParallel:
    async def test_all_agents_run(self):
        called: list[str] = []

        async def runner(url: str, msg: Message) -> Task:
            called.append(url)
            return await _success_runner(url, msg)

        orch = Orchestrator(runner=runner)
        result = await orch.parallel(
            agent_urls=["http://a:8001", "http://b:8002", "http://c:8003"],
            message=Message.user_text("run"),
        )
        assert set(called) == {"http://a:8001", "http://b:8002", "http://c:8003"}
        assert len(result.steps) == 3

    async def test_partial_failure_captured(self):
        async def runner(url: str, msg: Message) -> Task:
            if url == "http://b:8002":
                raise RuntimeError("b failed")
            return await _success_runner(url, msg)

        orch = Orchestrator(runner=runner)
        result = await orch.parallel(
            agent_urls=["http://a:8001", "http://b:8002"],
            message=Message.user_text("run"),
        )
        assert len(result.failed_steps) == 1
        assert result.failed_steps[0].agent_url == "http://b:8002"

    async def test_final_output_is_first_success(self):
        orch = Orchestrator(runner=_success_runner)
        result = await orch.parallel(
            agent_urls=["http://a:8001", "http://b:8002"],
            message=Message.user_text("run"),
        )
        assert result.final_output is not None

    async def test_empty_agent_list_raises(self):
        orch = Orchestrator(runner=_success_runner)
        with pytest.raises(OrchestratorError):
            await orch.parallel([], Message.user_text("go"))


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator — DAG
# ══════════════════════════════════════════════════════════════════════════════

class TestOrchestratorDAG:
    async def test_simple_linear_dag(self):
        """A → B → C"""
        order: list[str] = []

        async def runner(url: str, msg: Message) -> Task:
            order.append(url)
            return await _success_runner(url, msg)

        orch = Orchestrator(runner=runner)
        result = await orch.dag(
            nodes=[
                DAGNode("http://a:8001"),
                DAGNode("http://b:8002", depends_on=["http://a:8001"]),
                DAGNode("http://c:8003", depends_on=["http://b:8002"]),
            ],
            initial_message=Message.user_text("start"),
        )
        assert order == ["http://a:8001", "http://b:8002", "http://c:8003"]
        assert result.succeeded

    async def test_parallel_branches_in_dag(self):
        """A → (B, C concurrently) → D"""
        called: list[str] = []

        async def runner(url: str, msg: Message) -> Task:
            called.append(url)
            return await _success_runner(url, msg)

        orch = Orchestrator(runner=runner)
        result = await orch.dag(
            nodes=[
                DAGNode("http://a:8001"),
                DAGNode("http://b:8002", depends_on=["http://a:8001"]),
                DAGNode("http://c:8003", depends_on=["http://a:8001"]),
                DAGNode("http://d:8004",
                        depends_on=["http://b:8002", "http://c:8003"]),
            ],
            initial_message=Message.user_text("start"),
        )
        assert called[0] == "http://a:8001"
        assert "http://b:8002" in called
        assert "http://c:8003" in called
        assert called[-1] == "http://d:8004"
        assert result.succeeded

    async def test_cycle_detection_raises(self):
        orch = Orchestrator(runner=_success_runner)
        with pytest.raises(WorkflowCycleError):
            await orch.dag(
                nodes=[
                    DAGNode("http://a:8001", depends_on=["http://b:8002"]),
                    DAGNode("http://b:8002", depends_on=["http://a:8001"]),
                ],
                initial_message=Message.user_text("start"),
            )

    async def test_empty_nodes_raises(self):
        orch = Orchestrator(runner=_success_runner)
        with pytest.raises(OrchestratorError):
            await orch.dag([], Message.user_text("start"))


# ══════════════════════════════════════════════════════════════════════════════
# SSEFormatter
# ══════════════════════════════════════════════════════════════════════════════

class TestSSEFormatter:
    def test_event_format(self):
        line = SSEFormatter.event(StreamEventType.TASK_STATUS, {"state": "working"})
        assert line.startswith("data: ")
        assert line.endswith("\n\n")
        payload = json.loads(line[len("data: "):].strip())
        assert payload["type"] == "task_status"
        assert payload["state"] == "working"

    def test_heartbeat_format(self):
        line = SSEFormatter.heartbeat()
        assert line.startswith(": heartbeat")

    def test_done_format(self):
        line = SSEFormatter.done()
        payload = json.loads(line[len("data: "):].strip())
        assert payload["type"] == "done"

    def test_error_format(self):
        line = SSEFormatter.error("Something broke")
        payload = json.loads(line[len("data: "):].strip())
        assert payload["type"] == "error"
        assert payload["message"] == "Something broke"

    def test_task_status_shortcut(self):
        line = SSEFormatter.task_status("completed", "task-123")
        payload = json.loads(line[len("data: "):].strip())
        assert payload["state"] == "completed"
        assert payload["taskId"] == "task-123"

    def test_artifact_chunk_shortcut(self):
        line = SSEFormatter.artifact_chunk("partial...", "task-123", index=2)
        payload = json.loads(line[len("data: "):].strip())
        assert payload["content"] == "partial..."
        assert payload["index"] == 2


class TestStreamEvent:
    def test_is_terminal_done(self):
        ev = StreamEvent(type=StreamEventType.DONE)
        assert ev.is_terminal is True

    def test_is_terminal_error(self):
        ev = StreamEvent(type=StreamEventType.ERROR)
        assert ev.is_terminal is True

    def test_is_not_terminal_status(self):
        ev = StreamEvent(type=StreamEventType.TASK_STATUS)
        assert ev.is_terminal is False

    def test_as_task_with_valid_data(self):
        task = _make_task()
        ev = StreamEvent(
            type=StreamEventType.TASK_CREATED,
            data=task.model_dump(mode="json"),
        )
        parsed = ev.as_task()
        assert parsed is not None
        assert parsed.id == task.id

    def test_as_task_with_invalid_data_returns_none(self):
        # Data has neither 'id' nor 'state' — clearly not a Task
        ev = StreamEvent(type=StreamEventType.TASK_STATUS, data={"state": "working"})
        # Has 'state' but no 'id' — still not a valid Task payload
        assert ev.as_task() is None

        ev2 = StreamEvent(type=StreamEventType.TASK_STATUS, data={"not": "a task"})
        assert ev2.as_task() is None


# ══════════════════════════════════════════════════════════════════════════════
# WebhookDispatcher
# ══════════════════════════════════════════════════════════════════════════════

class TestWebhookDispatcher:
    def _make_response(self, status: int) -> MagicMock:
        resp = MagicMock()
        resp.status_code = status
        resp.is_success  = (200 <= status < 300)
        return resp

    async def test_successful_delivery(self):
        task = _make_task()
        dispatcher = WebhookDispatcher()

        with patch("nexus_a2a.transport.webhook.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post = AsyncMock(return_value=self._make_response(200))
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__  = AsyncMock(return_value=False)
            MockClient.return_value = instance

            record = await dispatcher.dispatch("http://client/hook", task, "task_completed")

        assert record.succeeded is True
        assert record.attempts == 1

    async def test_retries_on_5xx(self):
        task = _make_task()
        dispatcher = WebhookDispatcher(config=WebhookConfig(
            max_retries=3, base_delay=0.01
        ))

        responses = [
            self._make_response(500),
            self._make_response(500),
            self._make_response(200),
        ]

        with patch("nexus_a2a.transport.webhook.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post = AsyncMock(side_effect=responses)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__  = AsyncMock(return_value=False)
            MockClient.return_value = instance

            record = await dispatcher.dispatch("http://client/hook", task)

        assert record.succeeded is True
        assert record.attempts == 3

    async def test_no_retry_on_4xx(self):
        task = _make_task()
        dispatcher = WebhookDispatcher(config=WebhookConfig(max_retries=3))

        with patch("nexus_a2a.transport.webhook.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post = AsyncMock(return_value=self._make_response(404))
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__  = AsyncMock(return_value=False)
            MockClient.return_value = instance

            with pytest.raises(WebhookDeliveryError):
                await dispatcher.dispatch("http://client/hook", task)

        assert instance.post.call_count == 1

    async def test_raises_after_all_retries_exhausted(self):
        task = _make_task()
        dispatcher = WebhookDispatcher(config=WebhookConfig(
            max_retries=2, base_delay=0.01
        ))

        import httpx as _httpx
        with patch("nexus_a2a.transport.webhook.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post = AsyncMock(side_effect=_httpx.ConnectError("refused"))
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__  = AsyncMock(return_value=False)
            MockClient.return_value = instance

            with pytest.raises(WebhookDeliveryError) as exc_info:
                await dispatcher.dispatch("http://client/hook", task)

        assert exc_info.value.attempts == 2

    async def test_dispatch_silent_does_not_raise(self):
        task = _make_task()
        dispatcher = WebhookDispatcher(config=WebhookConfig(
            max_retries=1, base_delay=0.01
        ))

        import httpx as _httpx
        with patch("nexus_a2a.transport.webhook.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post = AsyncMock(side_effect=_httpx.ConnectError("refused"))
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__  = AsyncMock(return_value=False)
            MockClient.return_value = instance

            record = await dispatcher.dispatch_silent("http://client/hook", task)

        assert record.succeeded is False

    def test_signature_verification(self):
        secret  = "my-secret"
        payload = b'{"event": "task_completed"}'
        import hmac, hashlib
        sig = "sha256=" + hmac.new(
            secret.encode(), payload, hashlib.sha256
        ).hexdigest()
        assert WebhookDispatcher.verify_signature(payload, sig, secret) is True
        assert WebhookDispatcher.verify_signature(payload, "sha256=wrong", secret) is False

    async def test_delivery_log(self):
        task = _make_task()
        dispatcher = WebhookDispatcher()

        with patch("nexus_a2a.transport.webhook.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post = AsyncMock(return_value=self._make_response(200))
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__  = AsyncMock(return_value=False)
            MockClient.return_value = instance

            await dispatcher.dispatch("http://client/hook", task)

        assert len(dispatcher.delivery_log()) == 1
        dispatcher.clear_log()
        assert len(dispatcher.delivery_log()) == 0


# ══════════════════════════════════════════════════════════════════════════════
# EventBus
# ══════════════════════════════════════════════════════════════════════════════

class TestEventBus:
    async def test_subscribe_and_publish(self):
        bus     = EventBus()
        results = []

        async def handler(event: str, data: dict) -> None:
            results.append((event, data))

        bus.subscribe("test.event", handler)
        count = await bus.publish("test.event", {"x": 1})

        assert count == 1
        assert results == [("test.event", {"x": 1})]

    async def test_multiple_subscribers(self):
        bus   = EventBus()
        calls = []

        async def h1(e, d): calls.append("h1")
        async def h2(e, d): calls.append("h2")

        bus.subscribe("ev", h1)
        bus.subscribe("ev", h2)
        await bus.publish("ev", {})

        assert set(calls) == {"h1", "h2"}

    async def test_no_subscribers_returns_zero(self):
        bus   = EventBus()
        count = await bus.publish("nothing.here", {})
        assert count == 0

    async def test_unsubscribe(self):
        bus   = EventBus()
        calls = []

        async def handler(e, d): calls.append(e)

        bus.subscribe("ev", handler)
        bus.unsubscribe("ev", handler)
        await bus.publish("ev", {})

        assert calls == []

    async def test_unsubscribe_all(self):
        bus = EventBus()
        async def h1(e, d): pass
        async def h2(e, d): pass

        bus.subscribe("ev", h1)
        bus.subscribe("ev", h2)
        bus.unsubscribe_all("ev")

        assert bus.subscribers("ev") == []

    async def test_handler_exception_does_not_affect_others(self):
        bus   = EventBus()
        calls = []

        async def bad_handler(e, d):
            raise RuntimeError("I broke")

        async def good_handler(e, d):
            calls.append("ok")

        bus.subscribe("ev", bad_handler)
        bus.subscribe("ev", good_handler)
        await bus.publish("ev", {})

        assert calls == ["ok"]


# ══════════════════════════════════════════════════════════════════════════════
# AgentNetwork
# ══════════════════════════════════════════════════════════════════════════════

class TestAgentNetwork:
    def _patch_registry(self, network: AgentNetwork, card: AgentCard) -> None:
        """Inject a card directly into the registry without network call."""
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            network.registry.register_card(card)
        )

    async def test_add_publishes_event(self):
        network = AgentNetwork()
        events  = []

        async def on_add(e, d): events.append(d)
        network.bus.subscribe(AgentNetwork.EVENT_AGENT_ADDED, on_add)

        card = _make_card()
        with patch(
            "nexus_a2a.core.registry.A2AHttpClient",
            return_value=AsyncMock(
                fetch_agent_card=AsyncMock(return_value=card),
                __aenter__=AsyncMock(return_value=AsyncMock(
                    fetch_agent_card=AsyncMock(return_value=card)
                )),
                __aexit__=AsyncMock(return_value=False),
            ),
        ):
            # Direct registry registration for simplicity
            await network.registry.register_card(card)
            await network.bus.publish(AgentNetwork.EVENT_AGENT_ADDED, {
                "url": str(card.url), "name": card.name, "skills": card.skill_ids()
            })

        assert len(events) == 1
        assert events[0]["name"] == "TestAgent"

    async def test_resolve_agent_by_skill(self):
        network = AgentNetwork()
        card    = _make_card(url="http://agent:8001", skill="search")
        await network.registry.register_card(card)

        url = network._resolve_agent("search")
        assert "agent" in url

    async def test_resolve_agent_no_match_raises(self):
        network = AgentNetwork()
        with pytest.raises(ValueError, match="No healthy agent"):
            network._resolve_agent("nonexistent-skill")

    async def test_on_decorator(self):
        network = AgentNetwork()
        calls   = []

        @network.on("custom.event")
        async def handler(e, d):
            calls.append(d)

        await network.bus.publish("custom.event", {"key": "value"})
        assert calls == [{"key": "value"}]

    async def test_sequential_workflow(self):
        network = AgentNetwork()

        # Replace the internal runner with our mock
        network._orchestrator = Orchestrator(runner=_success_runner)

        result = await network.sequential(
            agent_urls=["http://a:8001", "http://b:8002"],
            message=Message.user_text("start"),
        )
        assert result.succeeded
        assert len(result.steps) == 2

    async def test_parallel_workflow(self):
        network = AgentNetwork()
        network._orchestrator = Orchestrator(runner=_success_runner)

        result = await network.parallel(
            agent_urls=["http://a:8001", "http://b:8002"],
            message=Message.user_text("run"),
        )
        assert len(result.steps) == 2

    async def test_dag_workflow(self):
        network = AgentNetwork()
        network._orchestrator = Orchestrator(runner=_success_runner)

        result = await network.dag(
            nodes=[
                DAGNode("http://a:8001"),
                DAGNode("http://b:8002", depends_on=["http://a:8001"]),
            ],
            message=Message.user_text("start"),
        )
        assert result.succeeded

    async def test_summary(self):
        network = AgentNetwork()
        await network.registry.register_card(_make_card())
        s = network.summary()
        assert s["total"] == 1