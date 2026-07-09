"""
tests/test_v1_1.py

Tests for v1.1.0 additions:
  - Tracer + TraceStore (tracing.py)
  - CircuitBreaker + RetryConfig + 5xx retry (http_client.py)
  - TaskManager timeout watchdog (task_manager.py)
  - InputHandler pause/resume (input_handler.py)
  - DeadLetterQueue capture/replay/hooks (dead_letter.py)
  - CapabilityGuard validation (capability_guard.py)
  - AgentNetwork v1.1 wiring (network.py)

Run with: uv run pytest tests/test_v1_1.py -v
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from nexus_a2a.core.dead_letter import DeadLetterQueue, DLQEntry
from nexus_a2a.core.input_handler import (
    InputHandler,
    InputTimeoutError,
    NoInputWaiterError,
)
from nexus_a2a.core.task_manager import (
    TaskManager,
)
from nexus_a2a.models.agent import AgentCapabilities, AgentCard
from nexus_a2a.models.task import Message, Task, TaskState
from nexus_a2a.network import AgentNetwork
from nexus_a2a.security.capability_guard import (
    CapabilityGuard,
    CapabilityMismatchError,
    CapabilityNotSupportedError,
)
from nexus_a2a.transport.http_client import (
    A2AHttpClient,
    AgentUnreachableError,
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    RetryConfig,
)
from nexus_a2a.transport.tracing import TRACE_ID_HEADER, Span, Tracer, TraceStore

# ── Shared helpers ────────────────────────────────────────────────────────────


def _make_task(state: TaskState = TaskState.WORKING) -> Task:
    task = Task.create(initial_message=Message.user_text("test"))
    if state != TaskState.SUBMITTED:
        task.transition(TaskState.WORKING)
    if state == TaskState.COMPLETED:
        task.transition(TaskState.COMPLETED)
    elif state == TaskState.FAILED:
        task.transition(TaskState.FAILED, error="test error")
    return task


def _make_card(streaming: bool = False, push: bool = False) -> AgentCard:
    return AgentCard(
        name="TestAgent",
        description="test",
        url="http://agent:8001",
        capabilities=AgentCapabilities(
            streaming=streaming,
            push_notifications=push,
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Tracer + TraceStore
# ══════════════════════════════════════════════════════════════════════════════


class TestTracer:
    def test_new_trace_id_is_unique(self):
        ids = {Tracer.new_trace_id() for _ in range(100)}
        assert len(ids) == 100

    def test_inject_returns_header(self):
        headers = Tracer.inject("abc-123")
        assert headers[TRACE_ID_HEADER] == "abc-123"

    def test_extract_from_headers(self):
        headers = {TRACE_ID_HEADER: "abc-123"}
        assert Tracer.extract(headers) == "abc-123"

    def test_extract_lowercase_header(self):
        headers = {TRACE_ID_HEADER.lower(): "abc-123"}
        assert Tracer.extract(headers) == "abc-123"

    def test_extract_missing_returns_none(self):
        assert Tracer.extract({}) is None

    async def test_span_context_manager_records_span(self):
        store = TraceStore()
        async with Tracer.span("trace-1", "http://agent:8001", store=store) as span:
            span.set_status("completed")

        trace = store.get("trace-1")
        assert trace is not None
        assert len(trace.spans) == 1
        assert trace.spans[0].status == "completed"
        assert trace.spans[0].duration_ms is not None

    async def test_span_auto_marks_failed_on_exception(self):
        store = TraceStore()
        with pytest.raises(RuntimeError):
            async with Tracer.span("trace-2", "http://agent:8001", store=store):
                raise RuntimeError("boom")

        trace = store.get("trace-2")
        assert trace.spans[0].status == "failed"
        assert "boom" in trace.spans[0].error

    async def test_span_auto_completes_if_not_set(self):
        store = TraceStore()
        async with Tracer.span("trace-3", "http://agent:8001", store=store):
            pass  # never call set_status

        trace = store.get("trace-3")
        assert trace.spans[0].status == "completed"

    async def test_format_tree(self):
        store = TraceStore()
        async with Tracer.span("trace-4", "http://a:8001", store=store) as s1:
            s1.set_status("completed")
        async with Tracer.span("trace-4", "http://b:8002", store=store) as s2:
            s2.set_status("failed", error="timeout")

        trace = store.get("trace-4")
        output = trace.format_tree()
        assert "trace-4" in output
        assert "http://a:8001" in output
        assert "http://b:8002" in output
        assert "failed" in output


class TestTraceStore:
    async def test_record_and_get(self):
        store = TraceStore()
        span = Span(trace_id="t1", agent_url="http://a:8001")
        await store.record(span)
        trace = store.get("t1")
        assert trace is not None
        assert len(trace.spans) == 1

    async def test_max_traces_drops_oldest(self):
        store = TraceStore(max_traces=3)
        for i in range(5):
            await store.record(Span(trace_id=f"t{i}", agent_url="http://a:8001"))
        assert store.count() == 3
        # oldest t0, t1 should be gone
        assert store.get("t0") is None
        assert store.get("t4") is not None

    async def test_multiple_spans_per_trace(self):
        store = TraceStore()
        for i in range(3):
            await store.record(Span(trace_id="t1", agent_url=f"http://a{i}:800{i}"))
        assert len(store.get("t1").spans) == 3


# ══════════════════════════════════════════════════════════════════════════════
# CircuitBreaker
# ══════════════════════════════════════════════════════════════════════════════


class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED

    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.on_failure()
        assert cb.state == CircuitState.OPEN

    def test_before_call_raises_when_open(self):
        cb = CircuitBreaker(failure_threshold=1)
        cb.on_failure()
        assert cb.state == CircuitState.OPEN
        with pytest.raises(CircuitOpenError):
            cb.before_call("http://agent:8001")

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.on_failure()
        cb.on_failure()
        cb.on_success()  # reset
        cb.on_failure()
        cb.on_failure()
        assert cb.state == CircuitState.CLOSED  # need one more failure to open

    def test_half_open_after_recovery_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.on_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_success_closes(self):
        cb = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.01,
            success_threshold=1,
        )
        cb.on_failure()
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN
        cb.on_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.on_failure()
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN
        cb.on_failure()
        assert cb.state == CircuitState.OPEN


# ══════════════════════════════════════════════════════════════════════════════
# RetryConfig
# ══════════════════════════════════════════════════════════════════════════════


class TestRetryConfig:
    def test_delay_increases_exponentially(self):
        cfg = RetryConfig(base_delay=1.0, jitter=False)
        d1 = cfg.delay_for(1)
        d2 = cfg.delay_for(2)
        d3 = cfg.delay_for(3)
        assert d2 == pytest.approx(d1 * 2, rel=0.01)
        assert d3 == pytest.approx(d1 * 4, rel=0.01)

    def test_delay_capped_at_max(self):
        cfg = RetryConfig(base_delay=100.0, max_delay=5.0, jitter=False)
        assert cfg.delay_for(1) == 5.0

    def test_jitter_adds_variance(self):
        cfg = RetryConfig(base_delay=1.0, jitter=True)
        delays = {cfg.delay_for(1) for _ in range(20)}
        assert len(delays) > 1  # jitter should produce varying values


# ══════════════════════════════════════════════════════════════════════════════
# A2AHttpClient — 5xx retry + circuit breaker
# ══════════════════════════════════════════════════════════════════════════════


class TestA2AHttpClientV11:
    def _mock_response(self, status: int, body: dict | None = None) -> MagicMock:
        resp = MagicMock()
        resp.status_code = status
        resp.json = MagicMock(
            return_value=body
            or {
                "jsonrpc": "2.0",
                "id": "x",
                "result": {
                    "id": "task-1",
                    "context_id": "ctx-1",
                    "state": "submitted",
                    "history": [],
                    "artifacts": [],
                    "created_at": "2026-01-01T00:00:00Z",
                    "updated_at": "2026-01-01T00:00:00Z",
                },
            }
        )
        resp.is_success = 200 <= status < 300
        resp.raise_for_status = MagicMock(
            side_effect=None
            if resp.is_success
            else httpx.HTTPStatusError("err", request=MagicMock(), response=resp)
        )
        return resp

    def _patch_client(self, responses: list) -> patch:
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(side_effect=responses)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)
        return patch(
            "nexus_a2a.transport.http_client.httpx.AsyncClient",
            return_value=mock_http,
        ), mock_http

    async def test_retries_on_500(self):
        success_body = {
            "jsonrpc": "2.0",
            "id": "x",
            "result": {
                "id": "task-1",
                "context_id": "ctx-1",
                "state": "submitted",
                "history": [],
                "artifacts": [],
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
            },
        }
        responses = [
            self._mock_response(500),
            self._mock_response(500),
            self._mock_response(200, success_body),
        ]
        patch_ctx, mock_http = self._patch_client(responses)
        retry = RetryConfig(max_retries=3, base_delay=0.01, jitter=False)

        with patch_ctx:
            async with A2AHttpClient("http://agent:8001", retry=retry) as client:
                client._client = mock_http
                result = await client._rpc("message/send", {})

        assert mock_http.post.call_count == 3

    async def test_does_not_retry_on_400(self):
        responses = [self._mock_response(400)]
        patch_ctx, mock_http = self._patch_client(responses)
        retry = RetryConfig(max_retries=3, base_delay=0.01, jitter=False)

        with patch_ctx:
            async with A2AHttpClient("http://agent:8001", retry=retry) as client:
                client._client = mock_http
                with pytest.raises(AgentUnreachableError):
                    await client._rpc("message/send", {})

        assert mock_http.post.call_count == 1

    async def test_circuit_breaker_opens_and_blocks(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=999)
        responses = [
            self._mock_response(500),
            self._mock_response(500),
        ]
        patch_ctx, mock_http = self._patch_client(responses)
        retry = RetryConfig(max_retries=1, base_delay=0.01, jitter=False)

        with patch_ctx:
            async with A2AHttpClient(
                "http://agent:8001", retry=retry, circuit_breaker=cb
            ) as client:
                client._client = mock_http
                # Two failures open the circuit
                with pytest.raises(AgentUnreachableError):
                    await client._rpc("message/send", {})
                with pytest.raises(AgentUnreachableError):
                    await client._rpc("message/send", {})
                # Now circuit is OPEN — next call blocked immediately
                with pytest.raises(CircuitOpenError):
                    await client._rpc("message/send", {})

    async def test_trace_id_exposed_on_client(self):
        async with A2AHttpClient("http://agent:8001", trace_id="my-trace") as client:
            assert client.trace_id == "my-trace"

    async def test_trace_id_auto_generated(self):
        async with A2AHttpClient("http://agent:8001") as client:
            assert client.trace_id != ""
            assert len(client.trace_id) > 0


# ══════════════════════════════════════════════════════════════════════════════
# TaskManager — timeout watchdog
# ══════════════════════════════════════════════════════════════════════════════


class TestTaskManagerWatchdog:
    async def test_watchdog_starts_and_stops(self):
        async with TaskManager(timeout_sec=60) as manager:
            assert manager._watchdog_task is not None
            assert not manager._watchdog_task.done()
        # After context exit, watchdog should be stopped
        assert manager._watchdog_task is None

    async def test_watchdog_auto_fails_timed_out_task(self):
        timed_out: list[Task] = []

        async def on_timeout(task: Task) -> None:
            timed_out.append(task)

        manager = TaskManager(
            timeout_sec=0.05,  # 50ms timeout
            watchdog_interval=0.02,  # scan every 20ms
            on_timeout=on_timeout,
        )
        await manager.start_watchdog()

        task = await manager.create(Message.user_text("test"))
        await manager.start(task.id)  # WORKING

        # Wait for watchdog to fire
        await asyncio.sleep(0.2)
        await manager.stop_watchdog()

        updated = await manager.get(task.id)
        assert updated.state == TaskState.FAILED
        assert "watchdog" in updated.error.lower()
        assert len(timed_out) == 1

    async def test_completed_tasks_not_timed_out(self):
        timed_out: list[Task] = []

        async def on_timeout(task: Task) -> None:
            timed_out.append(task)

        manager = TaskManager(
            timeout_sec=0.01,
            watchdog_interval=0.02,
            on_timeout=on_timeout,
        )
        await manager.start_watchdog()

        task = await manager.create(Message.user_text("test"))
        await manager.start(task.id)
        await manager.complete(task.id)  # already done before watchdog fires

        await asyncio.sleep(0.1)
        await manager.stop_watchdog()

        assert timed_out == []


# ══════════════════════════════════════════════════════════════════════════════
# InputHandler — pause / resume
# ══════════════════════════════════════════════════════════════════════════════


class TestInputHandler:
    async def test_wait_and_submit_resumes_correctly(self):
        manager = TaskManager()
        task = await manager.create(Message.user_text("start"))
        await manager.start(task.id)

        handler = InputHandler(manager)
        prompt = Message.agent_text("What is your name?")
        reply = Message.user_text("Alice")

        async def agent_side() -> Message:
            return await handler.wait_for_input(task.id, prompt, timeout=2.0)

        async def client_side() -> None:
            await asyncio.sleep(0.05)
            await handler.submit_reply(task.id, reply)

        result, _ = await asyncio.gather(agent_side(), client_side())
        assert result.text() == "Alice"

    async def test_is_waiting_true_while_suspended(self):
        manager = TaskManager()
        task = await manager.create(Message.user_text("start"))
        await manager.start(task.id)

        handler = InputHandler(manager)

        async def agent() -> None:
            try:
                await handler.wait_for_input(
                    task.id,
                    Message.agent_text("?"),
                    timeout=0.5,
                )
            except InputTimeoutError:
                pass

        task_coro = asyncio.create_task(agent())
        await asyncio.sleep(0.05)
        assert handler.is_waiting(task.id) is True
        await task_coro
        assert handler.is_waiting(task.id) is False

    async def test_timeout_raises_and_fails_task(self):
        manager = TaskManager()
        task = await manager.create(Message.user_text("start"))
        await manager.start(task.id)

        handler = InputHandler(manager)
        with pytest.raises(InputTimeoutError):
            await handler.wait_for_input(
                task.id,
                Message.agent_text("?"),
                timeout=0.05,
            )

        updated = await manager.get(task.id)
        assert updated.state == TaskState.FAILED

    async def test_submit_without_waiter_raises(self):
        manager = TaskManager()
        handler = InputHandler(manager)
        with pytest.raises(NoInputWaiterError):
            await handler.submit_reply("nonexistent", Message.user_text("hi"))

    async def test_waiting_count(self):
        manager = TaskManager()
        handler = InputHandler(manager)

        t1 = await manager.create(Message.user_text("a"))
        t2 = await manager.create(Message.user_text("b"))
        await manager.start(t1.id)
        await manager.start(t2.id)

        async def wait(tid: str) -> None:
            try:
                await handler.wait_for_input(tid, Message.agent_text("?"), timeout=0.3)
            except InputTimeoutError:
                pass

        tasks = [asyncio.create_task(wait(t1.id)), asyncio.create_task(wait(t2.id))]
        await asyncio.sleep(0.05)
        assert handler.waiting_count() == 2
        await asyncio.gather(*tasks)


# ══════════════════════════════════════════════════════════════════════════════
# DeadLetterQueue
# ══════════════════════════════════════════════════════════════════════════════


class TestDeadLetterQueue:
    def _failed_task(self) -> Task:
        return _make_task(TaskState.FAILED)

    async def _success_runner(self, url: str, msg: Message) -> Task:
        return _make_task(TaskState.COMPLETED)

    async def _failing_runner(self, url: str, msg: Message) -> Task:
        raise ConnectionError("still down")

    async def test_capture_adds_entry(self):
        dlq = DeadLetterQueue()
        task = self._failed_task()
        await dlq.capture(task, agent_url="http://agent:8001")
        assert dlq.count() == 1
        assert dlq.pending_count() == 1

    async def test_failure_hook_fires_on_capture(self):
        dlq = DeadLetterQueue()
        fired: list[DLQEntry] = []

        @dlq.on_failure
        async def hook(entry: DLQEntry) -> None:
            fired.append(entry)

        task = self._failed_task()
        await dlq.capture(task)
        assert len(fired) == 1
        assert fired[0].task_id == task.id

    async def test_replay_succeeds(self):
        dlq = DeadLetterQueue(runner=self._success_runner)
        task = self._failed_task()
        await dlq.capture(task, agent_url="http://agent:8001")
        result = await dlq.replay(task.id)
        assert result.succeeded is True
        assert dlq.pending_count() == 0

    async def test_replay_all(self):
        dlq = DeadLetterQueue(runner=self._success_runner)
        for _ in range(3):
            t = self._failed_task()
            await dlq.capture(t, agent_url="http://agent:8001")

        results = await dlq.replay_all()
        assert all(r.succeeded for r in results)
        assert dlq.pending_count() == 0

    async def test_replay_failure_increments_retry_count(self):
        dlq = DeadLetterQueue(runner=self._failing_runner, max_retries=2)
        task = self._failed_task()
        await dlq.capture(task, agent_url="http://agent:8001")

        await dlq.replay(task.id)
        entry = dlq.get_entry(task.id)
        assert entry.retry_count == 1
        assert entry.replayed is False

    async def test_replay_where_filters_by_skill(self):
        dlq = DeadLetterQueue(runner=self._success_runner)
        t1 = self._failed_task()
        t2 = self._failed_task()
        await dlq.capture(t1, agent_url="http://a:8001", skill_id="search")
        await dlq.capture(t2, agent_url="http://b:8002", skill_id="summarise")

        results = await dlq.replay_where(skill_id="search")
        assert len(results) == 1
        assert results[0].succeeded is True

    async def test_replay_without_runner_raises(self):
        dlq = DeadLetterQueue()  # no runner
        task = self._failed_task()
        await dlq.capture(task, agent_url="http://agent:8001")
        with pytest.raises(RuntimeError, match="runner"):
            await dlq.replay(task.id)

    async def test_max_queue_size_drops_oldest(self):
        dlq = DeadLetterQueue(max_queue_size=3)
        ids = []
        for _ in range(5):
            t = self._failed_task()
            ids.append(t.id)
            await dlq.capture(t)
        assert dlq.count() == 3
        assert dlq.get_entry(ids[0]) is None  # oldest dropped
        assert dlq.get_entry(ids[4]) is not None

    async def test_clear_replayed(self):
        dlq = DeadLetterQueue(runner=self._success_runner)
        for _ in range(3):
            t = self._failed_task()
            await dlq.capture(t, agent_url="http://agent:8001")

        await dlq.replay_all()
        removed = dlq.clear_replayed()
        assert removed == 3
        assert dlq.count() == 0

    async def test_summary(self):
        dlq = DeadLetterQueue()
        task = self._failed_task()
        await dlq.capture(task)
        s = dlq.summary()
        assert s["total"] == 1
        assert s["pending"] == 1


# ══════════════════════════════════════════════════════════════════════════════
# CapabilityGuard
# ══════════════════════════════════════════════════════════════════════════════


class TestCapabilityGuard:
    # ── Agent class validation ────────────────────────────────────────────────

    def test_non_streaming_agent_with_streaming_flag_raises(self):
        guard = CapabilityGuard(mode="strict")
        card = _make_card(streaming=True)

        class SyncAgent:
            async def run(self, task):
                return "result"  # not a generator

        with pytest.raises(CapabilityMismatchError, match="streaming"):
            guard.validate_agent_class(SyncAgent, card)

    def test_async_generator_agent_passes_streaming_check(self):
        guard = CapabilityGuard(mode="strict")
        card = _make_card(streaming=True)

        class StreamAgent:
            async def run(self, task):
                yield "chunk1"
                yield "chunk2"

        warnings = guard.validate_agent_class(StreamAgent, card)
        assert warnings == []

    def test_stream_method_passes_streaming_check(self):
        guard = CapabilityGuard(mode="strict")
        card = _make_card(streaming=True)

        class StreamMethodAgent:
            async def run(self, task):
                return "result"

            async def stream(self, task):
                yield "chunk"

        warnings = guard.validate_agent_class(StreamMethodAgent, card)
        assert warnings == []

    def test_streaming_class_attr_bypasses_check(self):
        guard = CapabilityGuard(mode="strict")
        card = _make_card(streaming=True)

        class AdapterAgent:
            STREAMING = True  # handled by framework adapter

            async def run(self, task):
                return "result"

        warnings = guard.validate_agent_class(AdapterAgent, card)
        assert warnings == []

    def test_warn_mode_does_not_raise(self):
        guard = CapabilityGuard(mode="warn")
        card = _make_card(streaming=True)

        class SyncAgent:
            async def run(self, task):
                return "result"

        warnings = guard.validate_agent_class(SyncAgent, card)
        assert len(warnings) == 1  # warning returned, not raised

    def test_off_mode_skips_all_checks(self):
        guard = CapabilityGuard(mode="off")
        card = _make_card(streaming=True)

        class SyncAgent:
            async def run(self, task):
                return "result"

        warnings = guard.validate_agent_class(SyncAgent, card)
        assert warnings == []

    # ── Compatibility validation ──────────────────────────────────────────────

    def test_caller_wants_streaming_agent_has_it(self):
        guard = CapabilityGuard()
        card = _make_card(streaming=True)
        guard.validate_compatibility({"streaming": True}, card)  # no raise

    def test_caller_wants_streaming_agent_lacks_it(self):
        guard = CapabilityGuard()
        card = _make_card(streaming=False)
        with pytest.raises(CapabilityNotSupportedError, match="streaming"):
            guard.validate_compatibility({"streaming": True}, card)

    def test_assert_supports_streaming(self):
        guard = CapabilityGuard()
        assert guard.supports_streaming(_make_card(streaming=True)) is True
        assert guard.supports_streaming(_make_card(streaming=False)) is False

    def test_assert_supports_push(self):
        guard = CapabilityGuard()
        assert guard.supports_push(_make_card(push=True)) is True
        assert guard.supports_push(_make_card(push=False)) is False

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            CapabilityGuard(mode="invalid")


# ══════════════════════════════════════════════════════════════════════════════
# AgentNetwork v1.1 wiring
# ══════════════════════════════════════════════════════════════════════════════


class TestAgentNetworkV11:
    async def test_network_has_dlq(self):
        network = AgentNetwork()
        assert network.dead_letter_queue is not None
        assert isinstance(network.dead_letter_queue, DeadLetterQueue)

    async def test_network_has_input_handler(self):
        network = AgentNetwork()
        assert network.input_handler is not None
        assert isinstance(network.input_handler, InputHandler)

    async def test_summary_includes_dlq(self):
        network = AgentNetwork()
        s = network.summary()
        assert "dlq" in s
        assert s["dlq"]["pending"] == 0

    async def test_timeout_callback_fires_on_task_timeout(self):
        timeout_events: list = []

        # Use small watchdog_interval so scan fires quickly in test
        tm = TaskManager(timeout_sec=0.05, watchdog_interval=0.05)
        network = AgentNetwork(task_manager=tm)
        network.bus.subscribe(
            AgentNetwork.EVENT_TASK_TIMEOUT,
            lambda e, d: timeout_events.append(d) or asyncio.sleep(0),
        )

        await network.task_manager.start_watchdog()
        task = await network.task_manager.create(Message.user_text("test"))
        await network.task_manager.start(task.id)

        await asyncio.sleep(0.3)
        await network.task_manager.stop_watchdog()

        # Task should be in DLQ
        assert network.dead_letter_queue.pending_count() >= 1

    async def test_dlq_runner_wired_to_network(self):
        network = AgentNetwork()
        assert network.dead_letter_queue._runner is not None
