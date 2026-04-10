"""
tests/test_phase5.py

Tests for Phase 5: BaseAdapter, framework adapters, RedisTaskStore,
AuditLogger, MetricsCollector.

Framework adapters are tested with mock agent objects so no real
LangGraph/CrewAI/ADK/AutoGen installation is needed.

Run with:  uv run pytest tests/test_phase5.py -v
"""

from __future__ import annotations

import io
import json
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from nexus_a2a.models.task import Message, Task, TaskState
from nexus_a2a.adapters.base import (
    AdapterConfigError,
    AdapterExecutionError,
    AdapterResult,
    BaseAdapter,
)
from nexus_a2a.adapters.langgraph import LangGraphAdapter
from nexus_a2a.adapters.crewai import CrewAIAdapter
from nexus_a2a.adapters.autogen import AutoGenAdapter
from nexus_a2a.adapters.google_adk import GoogleADKAdapter
from nexus_a2a.storage.audit_logger import AuditEvent, AuditLogger
from nexus_a2a.storage.metrics import MetricsCollector


# ── Shared helpers ────────────────────────────────────────────────────────────

def _make_task(text: str = "Hello agent") -> Task:
    task = Task.create(initial_message=Message.user_text(text))
    task.transition(TaskState.WORKING)
    return task


# ══════════════════════════════════════════════════════════════════════════════
# BaseAdapter
# ══════════════════════════════════════════════════════════════════════════════

class ConcreteAdapter(BaseAdapter):
    """Minimal concrete adapter for testing the base class."""
    framework_name = "test"

    async def execute(self, task: Task) -> AdapterResult:
        text = self.extract_input(task)
        return self.make_result(output=f"echo: {text}")


class TestBaseAdapter:
    def test_none_agent_raises_at_init(self):
        with pytest.raises(AdapterConfigError, match="must not be None"):
            ConcreteAdapter(agent=None)

    def test_extract_input(self):
        adapter = ConcreteAdapter(agent=object())
        task    = _make_task("test input")
        assert adapter.extract_input(task) == "test input"

    def test_extract_input_no_messages(self):
        adapter = ConcreteAdapter(agent=object())
        task    = Task(
            id="x", context_id="y",
            state=TaskState.WORKING,
            history=[],
        )
        assert adapter.extract_input(task) == ""

    async def test_execute_returns_result(self):
        adapter = ConcreteAdapter(agent=object())
        task    = _make_task("hello")
        result  = await adapter.execute(task)
        assert result.succeeded
        assert result.output == "echo: hello"

    def test_make_result_builds_artifact(self):
        adapter  = ConcreteAdapter(agent=object())
        result   = adapter.make_result("output text", artifact_name="my_art")
        artifact = result.to_artifact()
        assert artifact.name == "my_art"
        assert artifact.parts[0].content == "output text"

    def test_make_error(self):
        adapter = ConcreteAdapter(agent=object())
        result  = adapter.make_error("something broke")
        assert not result.succeeded
        assert result.error == "something broke"

    def test_adapter_result_to_artifact_uses_existing(self):
        from nexus_a2a.models.task import Artifact, Part, PartType
        art    = Artifact(name="custom", parts=[Part(type=PartType.TEXT, content="x")])
        result = AdapterResult(output="y", artifact=art)
        assert result.to_artifact().name == "custom"

    def test_wrap_exception(self):
        adapter = ConcreteAdapter(agent=object())
        exc     = RuntimeError("boom")
        wrapped = adapter._wrap_exception(exc)
        assert isinstance(wrapped, AdapterExecutionError)
        assert "test" in str(wrapped)
        assert "boom" in str(wrapped)


# ══════════════════════════════════════════════════════════════════════════════
# LangGraphAdapter
# ══════════════════════════════════════════════════════════════════════════════

class TestLangGraphAdapter:
    def _mock_graph(self, reply: str = "LangGraph reply") -> MagicMock:
        graph = MagicMock()
        graph.ainvoke = AsyncMock(return_value={
            "messages": [MagicMock(content=reply)]
        })
        return graph

    def test_validate_requires_ainvoke(self):
        with pytest.raises(AdapterConfigError, match="ainvoke"):
            LangGraphAdapter(agent=object())   # plain object has no ainvoke

    async def test_execute_returns_reply(self):
        graph   = self._mock_graph("LangGraph reply")
        adapter = LangGraphAdapter(agent=graph)
        task    = _make_task("what is 2+2?")
        result  = await adapter.execute(task)
        assert result.succeeded
        assert result.output == "LangGraph reply"

    async def test_execute_empty_message_returns_error(self):
        graph   = self._mock_graph()
        adapter = LangGraphAdapter(agent=graph)
        task    = Task(
            id="x", context_id="y",
            state=TaskState.WORKING, history=[],
        )
        result = await adapter.execute(task)
        assert not result.succeeded

    async def test_execute_framework_exception_raises(self):
        graph         = MagicMock()
        graph.ainvoke = AsyncMock(side_effect=RuntimeError("graph broke"))
        adapter       = LangGraphAdapter(agent=graph)
        task          = _make_task("input")
        with pytest.raises(AdapterExecutionError, match="graph broke"):
            await adapter.execute(task)

    def test_extract_output_dict_message(self):
        graph   = MagicMock()
        graph.ainvoke = AsyncMock()
        adapter = LangGraphAdapter(agent=graph)
        state   = {"messages": [{"content": "hello from dict"}]}
        assert adapter._extract_output(state) == "hello from dict"

    def test_extract_output_empty_fallback(self):
        graph   = MagicMock()
        graph.ainvoke = AsyncMock()
        adapter = LangGraphAdapter(agent=graph)
        state   = {}
        assert adapter._extract_output(state) == str(state)

    async def test_metadata_contains_framework(self):
        graph   = self._mock_graph("ok")
        adapter = LangGraphAdapter(agent=graph)
        result  = await adapter.execute(_make_task("test"))
        assert result.metadata["framework"] == "langgraph"


# ══════════════════════════════════════════════════════════════════════════════
# CrewAIAdapter
# ══════════════════════════════════════════════════════════════════════════════

class TestCrewAIAdapter:
    def _mock_crew(self, reply: str = "Crew result") -> MagicMock:
        crew               = MagicMock()
        crew_output        = MagicMock()
        crew_output.raw    = reply
        crew.kickoff_async = AsyncMock(return_value=crew_output)
        return crew

    def test_validate_requires_kickoff(self):
        with pytest.raises(AdapterConfigError, match="kickoff"):
            CrewAIAdapter(agent=object())

    async def test_execute_returns_reply(self):
        crew    = self._mock_crew("Final crew answer")
        adapter = CrewAIAdapter(agent=crew, input_key="topic")
        result  = await adapter.execute(_make_task("AI trends"))
        assert result.succeeded
        assert result.output == "Final crew answer"

    async def test_execute_empty_message_returns_error(self):
        crew    = self._mock_crew()
        adapter = CrewAIAdapter(agent=crew)
        task    = Task(id="x", context_id="y", state=TaskState.WORKING, history=[])
        result  = await adapter.execute(task)
        assert not result.succeeded

    async def test_extract_output_final_output_fallback(self):
        crew               = MagicMock()
        crew_output        = MagicMock(spec=[])          # no .raw
        crew_output.final_output = "legacy output"
        crew.kickoff_async = AsyncMock(return_value=crew_output)
        adapter            = CrewAIAdapter(agent=crew)
        result             = await adapter.execute(_make_task("test"))
        assert result.output == "legacy output"

    async def test_framework_exception_raises(self):
        crew               = MagicMock()
        crew.kickoff_async = AsyncMock(side_effect=RuntimeError("crew failed"))
        adapter            = CrewAIAdapter(agent=crew)
        with pytest.raises(AdapterExecutionError, match="crew failed"):
            await adapter.execute(_make_task("test"))

    async def test_metadata_contains_framework(self):
        crew    = self._mock_crew()
        adapter = CrewAIAdapter(agent=crew)
        result  = await adapter.execute(_make_task("test"))
        assert result.metadata["framework"] == "crewai"


# ══════════════════════════════════════════════════════════════════════════════
# AutoGenAdapter
# ══════════════════════════════════════════════════════════════════════════════

class TestAutoGenAdapter:
    def _mock_agent(self, summary: str = "AutoGen reply") -> MagicMock:
        agent = MagicMock()
        agent.a_initiate_chat = AsyncMock(
            return_value=MagicMock(summary=summary)
        )
        return agent

    def test_validate_requires_initiate_chat(self):
        with pytest.raises(AdapterConfigError, match="initiate_chat"):
            AutoGenAdapter(agent=object())

    async def test_execute_returns_summary(self):
        agent   = self._mock_agent("AutoGen answer")
        adapter = AutoGenAdapter(agent=agent)

        # Mock the UserProxyAgent construction
        with patch("nexus_a2a.adapters.autogen.AutoGenAdapter._build_proxy") as mock_proxy:
            proxy               = MagicMock()
            chat_result         = MagicMock()
            chat_result.summary = "AutoGen answer"
            proxy.a_initiate_chat = AsyncMock(return_value=chat_result)
            mock_proxy.return_value = proxy

            result = await adapter.execute(_make_task("hello"))

        assert result.succeeded
        assert result.output == "AutoGen answer"

    async def test_empty_message_returns_error(self):
        agent   = self._mock_agent()
        adapter = AutoGenAdapter(agent=agent)
        task    = Task(id="x", context_id="y", state=TaskState.WORKING, history=[])
        result  = await adapter.execute(task)
        assert not result.succeeded

    def test_extract_last_reply_from_history(self):
        chat_result              = MagicMock(spec=[])
        chat_result.chat_history = [{"content": "from history"}]
        assert AutoGenAdapter._extract_last_reply(chat_result) == "from history"

    def test_extract_last_reply_fallback_str(self):
        chat_result = MagicMock(spec=[])
        result      = AutoGenAdapter._extract_last_reply(chat_result)
        assert isinstance(result, str)

    async def test_metadata_contains_framework(self):
        agent   = self._mock_agent()
        adapter = AutoGenAdapter(agent=agent)

        with patch("nexus_a2a.adapters.autogen.AutoGenAdapter._build_proxy") as mock_proxy:
            proxy                    = MagicMock()
            chat_result              = MagicMock()
            chat_result.summary      = "ok"
            proxy.a_initiate_chat    = AsyncMock(return_value=chat_result)
            mock_proxy.return_value  = proxy
            result = await adapter.execute(_make_task("test"))

        assert result.metadata["framework"] == "autogen"


# ══════════════════════════════════════════════════════════════════════════════
# GoogleADKAdapter
# ══════════════════════════════════════════════════════════════════════════════

class TestGoogleADKAdapter:
    def _mock_agent(self) -> MagicMock:
        agent      = MagicMock()
        agent.name = "mock_adk_agent"
        return agent

    def test_validate_requires_name(self):
        with pytest.raises(AdapterConfigError):
            GoogleADKAdapter(agent=object())   # no .name attribute

    async def test_execute_returns_output(self):
        agent   = self._mock_agent()
        adapter = GoogleADKAdapter(agent=agent)

        # Mock the internal runner and session building
        with patch.object(adapter, "_build_runner", new_callable=AsyncMock) as mock_runner, \
             patch.object(adapter, "_run_agent",   new_callable=AsyncMock) as mock_run:

            mock_runner.return_value = (MagicMock(), "session-123")
            mock_run.return_value    = "ADK reply text"

            result = await adapter.execute(_make_task("research AI"))

        assert result.succeeded
        assert result.output == "ADK reply text"

    async def test_empty_message_returns_error(self):
        agent   = self._mock_agent()
        adapter = GoogleADKAdapter(agent=agent)
        task    = Task(id="x", context_id="y", state=TaskState.WORKING, history=[])
        result  = await adapter.execute(task)
        assert not result.succeeded

    async def test_framework_exception_raises(self):
        agent   = self._mock_agent()
        adapter = GoogleADKAdapter(agent=agent)

        with patch.object(adapter, "_build_runner", new_callable=AsyncMock) as mock_runner:
            mock_runner.side_effect = RuntimeError("ADK failure")
            with pytest.raises(AdapterExecutionError, match="ADK failure"):
                await adapter.execute(_make_task("test"))

    async def test_metadata_contains_framework(self):
        agent   = self._mock_agent()
        adapter = GoogleADKAdapter(agent=agent)

        with patch.object(adapter, "_build_runner", new_callable=AsyncMock) as mock_runner, \
             patch.object(adapter, "_run_agent",   new_callable=AsyncMock) as mock_run:

            mock_runner.return_value = (MagicMock(), "session-abc")
            mock_run.return_value    = "ok"

            result = await adapter.execute(_make_task("test"))

        assert result.metadata["framework"] == "google_adk"


# ══════════════════════════════════════════════════════════════════════════════
# RedisTaskStore (mocked — no real Redis needed)
# ══════════════════════════════════════════════════════════════════════════════

class TestRedisTaskStore:
    """
    Tests use a mocked redis.asyncio client so no real Redis server is needed.
    """

    def _make_store(self):
        from nexus_a2a.storage.redis_store import RedisTaskStore
        store        = RedisTaskStore(url="redis://localhost:6379")
        mock_redis   = AsyncMock()
        store._redis = mock_redis   # inject mock directly
        return store, mock_redis

    async def test_save_calls_setex(self):
        store, redis = self._make_store()
        task         = _make_task()
        await store.save(task)
        redis.setex.assert_called_once()
        # First arg is the key, second is TTL, third is JSON
        key = redis.setex.call_args[0][0]
        assert task.id in key

    async def test_get_returns_task(self):
        store, redis = self._make_store()
        task         = _make_task()
        redis.get     = AsyncMock(return_value=task.model_dump_json())
        fetched      = await store.get(task.id)
        assert fetched is not None
        assert fetched.id == task.id

    async def test_get_missing_returns_none(self):
        store, redis = self._make_store()
        redis.get     = AsyncMock(return_value=None)
        result       = await store.get("nonexistent")
        assert result is None

    async def test_delete_calls_redis(self):
        store, redis = self._make_store()
        await store.delete("task-123")
        redis.delete.assert_called_once()

    async def test_require_connected_raises_when_not_connected(self):
        from nexus_a2a.storage.redis_store import RedisTaskStore
        store = RedisTaskStore()
        with pytest.raises(RuntimeError, match="not connected"):
            store._require_connected()

    async def test_list_all_uses_scan_iter(self):
        store, redis = self._make_store()
        task         = _make_task()

        async def fake_scan(*args, **kwargs):
            yield f"nexus_a2a:task:{task.id}"

        redis.scan_iter = fake_scan
        redis.get        = AsyncMock(return_value=task.model_dump_json())

        results = await store.list_all()
        assert len(results) == 1
        assert results[0].id == task.id


# ══════════════════════════════════════════════════════════════════════════════
# AuditLogger
# ══════════════════════════════════════════════════════════════════════════════

class TestAuditLogger:
    def _logger(self) -> AuditLogger:
        """AuditLogger writing to a string buffer for inspection."""
        return AuditLogger(stream=io.StringIO(), buffer_size=100)

    def test_task_created_logs_entry(self):
        audit = self._logger()
        task  = _make_task()
        audit.task_created(task)
        entries = audit.entries_by_event(AuditEvent.TASK_CREATED)
        assert len(entries) == 1
        assert entries[0].task_id == task.id

    def test_task_state_changed(self):
        audit = self._logger()
        task  = _make_task()
        audit.task_state_changed(task, old_state=TaskState.SUBMITTED)
        entry = audit.entries_by_event(AuditEvent.TASK_STATE_CHANGED)[0]
        assert entry.data["old_state"] == "submitted"
        assert entry.data["new_state"] == "working"

    def test_agent_called(self):
        audit = self._logger()
        audit.agent_called("http://agent:8001", "task-123", skill_id="search")
        entry = audit.entries_by_event(AuditEvent.AGENT_CALLED)[0]
        assert entry.agent_url == "http://agent:8001"
        assert entry.data["skill_id"] == "search"

    def test_agent_responded(self):
        audit = self._logger()
        audit.agent_responded("http://agent:8001", "task-123", 0.42, succeeded=True)
        entry = audit.entries_by_event(AuditEvent.AGENT_RESPONDED)[0]
        assert entry.data["duration_sec"] == 0.42
        assert entry.data["succeeded"] is True

    def test_auth_failure(self):
        audit = self._logger()
        audit.auth_failure("http://agent:8001", reason="bad token")
        entries = audit.entries_by_event(AuditEvent.AUTH_FAILURE)
        assert len(entries) == 1
        assert "bad token" in entries[0].data["reason"]

    def test_rate_limit_exceeded(self):
        audit = self._logger()
        audit.rate_limit_exceeded("http://agent:8001", retry_after=1.5)
        entry = audit.entries_by_event(AuditEvent.RATE_LIMIT_EXCEEDED)[0]
        assert entry.data["retry_after_sec"] == 1.5

    def test_workflow_completed(self):
        audit = self._logger()
        audit.workflow_completed("sequential", total_sec=2.1, steps=3, succeeded=True)
        entry = audit.entries_by_event(AuditEvent.WORKFLOW_COMPLETED)[0]
        assert entry.data["mode"] == "sequential"
        assert entry.data["steps"] == 3

    def test_custom_event(self):
        audit = self._logger()
        audit.custom("my.event", {"key": "value"}, task_id="t1")
        entry = audit.entries_by_event(AuditEvent.CUSTOM)[0]
        assert entry.data["custom_event"] == "my.event"
        assert entry.data["key"] == "value"

    def test_entries_for_task(self):
        audit = self._logger()
        task  = _make_task()
        audit.task_created(task)
        audit.agent_called("http://x:8001", "other-task")
        entries = audit.entries_for_task(task.id)
        assert all(e.task_id == task.id for e in entries)

    def test_json_output_is_valid(self):
        buf   = io.StringIO()
        audit = AuditLogger(stream=buf)
        task  = _make_task()
        audit.task_created(task)
        buf.seek(0)
        line = buf.readline()
        parsed = json.loads(line)
        assert parsed["event"] == "task_created"

    def test_buffer_size_limit(self):
        audit = AuditLogger(stream=io.StringIO(), buffer_size=3)
        for _ in range(5):
            audit.task_created(_make_task())
        assert audit.count() == 3   # oldest dropped

    def test_disabled_logs_nothing(self):
        audit = AuditLogger(stream=io.StringIO(), enabled=False)
        audit.task_created(_make_task())
        assert audit.count() == 0

    def test_clear(self):
        audit = self._logger()
        audit.task_created(_make_task())
        audit.clear()
        assert audit.count() == 0


# ══════════════════════════════════════════════════════════════════════════════
# MetricsCollector
# ══════════════════════════════════════════════════════════════════════════════

class TestMetricsCollector:
    def test_counters_increment(self):
        m = MetricsCollector()
        m.record_task_created()
        m.record_task_created()
        m.record_task_completed()
        m.record_task_failed()
        snap = m.snapshot()
        assert snap.tasks_created   == 2
        assert snap.tasks_completed == 1
        assert snap.tasks_failed    == 1

    def test_rate_limit_and_auth_counters(self):
        m = MetricsCollector()
        m.record_rate_limit_hit()
        m.record_auth_failure()
        snap = m.snapshot()
        assert snap.rate_limit_hits == 1
        assert snap.auth_failures   == 1

    def test_agent_error_per_agent(self):
        m = MetricsCollector()
        m.record_agent_error("http://a:8001")
        m.record_agent_error("http://a:8001")
        m.record_agent_error("http://b:8002")
        snap = m.snapshot()
        assert snap.agent_errors["http://a:8001"] == 2
        assert snap.agent_errors["http://b:8002"] == 1

    def test_call_duration_recording(self):
        m = MetricsCollector()
        m.record_call_duration("http://a:8001", 0.1)
        m.record_call_duration("http://a:8001", 0.3)
        snap = m.snapshot()
        assert len(snap.call_durations["http://a:8001"]) == 2
        avg  = snap.avg_latency("http://a:8001")
        assert abs(avg - 0.2) < 0.01

    def test_p99_latency(self):
        m = MetricsCollector()
        for i in range(100):
            m.record_call_duration("http://a:8001", float(i))
        snap = m.snapshot()
        p99  = snap.p99_latency("http://a:8001")
        assert p99 is not None
        assert p99 >= 97.0

    def test_avg_latency_no_data_returns_none(self):
        m    = MetricsCollector()
        snap = m.snapshot()
        assert snap.avg_latency("http://unknown:8001") is None

    def test_record_agent_call_context_manager(self):
        m = MetricsCollector()
        with m.record_agent_call("http://a:8001"):
            time.sleep(0.01)
        snap = m.snapshot()
        assert snap.total_calls() == 1
        assert snap.avg_latency("http://a:8001") >= 0.01

    def test_record_agent_call_records_error_on_exception(self):
        m = MetricsCollector()
        with pytest.raises(RuntimeError):
            with m.record_agent_call("http://a:8001"):
                raise RuntimeError("network error")
        snap = m.snapshot()
        assert snap.agent_errors.get("http://a:8001", 0) == 1

    def test_max_durations_drops_oldest(self):
        m = MetricsCollector(max_durations=3)
        for i in range(5):
            m.record_call_duration("http://a:8001", float(i))
        snap = m.snapshot()
        assert len(snap.call_durations["http://a:8001"]) == 3
        # Should contain the 3 most recent: 2.0, 3.0, 4.0
        assert snap.call_durations["http://a:8001"] == [2.0, 3.0, 4.0]

    def test_reset(self):
        m = MetricsCollector()
        m.record_task_created()
        m.record_agent_error("http://a:8001")
        m.reset()
        snap = m.snapshot()
        assert snap.tasks_created == 0
        assert snap.agent_errors  == {}

    def test_with_otel_graceful_on_no_meter(self):
        # Should not raise even with a mock meter that has no real instruments
        mock_meter = MagicMock()
        mock_meter.create_counter   = MagicMock(return_value=MagicMock())
        mock_meter.create_histogram = MagicMock(return_value=MagicMock())
        metrics = MetricsCollector.with_otel(mock_meter)
        metrics.record_task_created()   # should not raise
        snap = metrics.snapshot()
        assert snap.tasks_created == 1