"""
tests/test_v1_9.py

Library-level changes in 1.9.0:

  - InMemoryTaskStore retention (the default server no longer leaks tasks)
  - AuthManager default credentials (the nexus.toml secret is finally checked)
  - the agent card advertises the auth actually enforced
  - live-following GET /stream, and /metrics on A2AServer
  - the watchdog/finish race, Prometheus label escaping, trace lookup by task id
  - nexus.toml -> AgentRuntime wiring
  - 2.0 deprecation warnings
"""

from __future__ import annotations

import asyncio
import json
import warnings

import httpx
import pytest
from starlette.testclient import TestClient

from nexus_a2a import agent
from nexus_a2a.config import ConfigError, NexusConfig
from nexus_a2a.core.a2a_server import (
    A2AServer,
    _is_loopback,
    _is_terminal_line,
    _TaskWatchers,
)
from nexus_a2a.core.agent_server import _escape_label_value, _format_labels
from nexus_a2a.models.agent import AuthScheme
from nexus_a2a.models.task import Message, Task, TaskState
from nexus_a2a.runtime import AgentRuntime
from nexus_a2a.security.auth import (
    AgentCredentialConfig,
    AuthManager,
    InvalidCredentialsError,
    MissingCredentialsError,
    UnknownAgentError,
)
from nexus_a2a.security.middleware import (
    CALLER_HEADER,
    MissingCallerError,
    SecurityMiddleware,
)
from nexus_a2a.security.trust import TrustBoundary
from nexus_a2a.storage.task_store import InMemoryTaskStore
from nexus_a2a.transport.sse import SSEFormatter
from nexus_a2a.transport.tracing import Span, TraceStore

SERVER_URL = "http://agent-under-test:8001"
LONG_SECRET = "a-32-byte-or-longer-test-secret-value!"


@agent(name="Echo", description="Echoes.",
       skills=[{"id": "echo", "name": "Echo", "description": "Echo."}],
       url=SERVER_URL)
class EchoAgent:
    async def run(self, task: Task) -> str:
        return "echo: " + task.latest_message().text()


def send_params(text: str = "hi") -> dict:
    return {"message": Message.user_text(text).model_dump(mode="json")}


def rpc(client: TestClient, method: str, params: dict, headers: dict | None = None):
    return client.post(
        "/",
        json={"jsonrpc": "2.0", "id": "1", "method": method, "params": params},
        headers=headers or {},
    )


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def finished_task(text: str = "x") -> Task:
    task = Task.create(initial_message=Message.user_text(text))
    task.transition(TaskState.WORKING)
    task.transition(TaskState.COMPLETED)
    return task


def running_task(text: str = "x") -> Task:
    task = Task.create(initial_message=Message.user_text(text))
    task.transition(TaskState.WORKING)
    return task


# ── Task store retention ──────────────────────────────────────────────────────


class TestTaskStoreRetention:
    async def test_finished_task_expires_after_retention(self):
        clock = FakeClock()
        store = InMemoryTaskStore(retention_sec=60, clock=clock)
        task = finished_task()
        await store.save(task)
        clock.now += 59
        assert await store.get(task.id) is not None
        clock.now += 2
        assert await store.get(task.id) is None

    async def test_running_task_is_never_expired(self):
        clock = FakeClock()
        store = InMemoryTaskStore(retention_sec=1, clock=clock)
        task = running_task()
        await store.save(task)
        clock.now += 10_000
        assert await store.get(task.id) is not None

    async def test_retention_window_starts_when_the_task_finishes(self):
        clock = FakeClock()
        store = InMemoryTaskStore(retention_sec=60, clock=clock)
        task = running_task()
        await store.save(task)
        clock.now += 500  # long-running, still fine
        task.transition(TaskState.COMPLETED)
        await store.save(task)
        clock.now += 30
        assert await store.get(task.id) is not None

    async def test_resaving_a_finished_task_does_not_extend_retention(self):
        clock = FakeClock()
        store = InMemoryTaskStore(retention_sec=60, clock=clock)
        task = finished_task()
        await store.save(task)
        clock.now += 50
        await store.save(task)
        clock.now += 20
        assert await store.get(task.id) is None

    async def test_cap_evicts_oldest_finished_first(self):
        store = InMemoryTaskStore(retention_sec=None, max_tasks=2)
        first, second, third = finished_task("1"), finished_task("2"), finished_task("3")
        for task in (first, second, third):
            await store.save(task)
        assert await store.get(first.id) is None
        assert await store.get(second.id) is not None
        assert await store.get(third.id) is not None

    async def test_cap_never_evicts_active_tasks(self):
        store = InMemoryTaskStore(retention_sec=None, max_tasks=2)
        live = [running_task(str(i)) for i in range(4)]
        for task in live:
            await store.save(task)
        assert await store.count() == 4
        for task in live:
            assert await store.get(task.id) is not None

    async def test_cap_prefers_finished_over_active(self):
        store = InMemoryTaskStore(retention_sec=None, max_tasks=2)
        live_a, done, live_b = running_task("a"), finished_task("d"), running_task("b")
        for task in (live_a, done, live_b):
            await store.save(task)
        assert await store.get(done.id) is None
        assert await store.get(live_a.id) is not None
        assert await store.get(live_b.id) is not None

    async def test_listing_applies_retention(self):
        clock = FakeClock()
        store = InMemoryTaskStore(retention_sec=5, clock=clock)
        await store.save(finished_task())
        clock.now += 6
        assert await store.list_all() == []

    async def test_evict_expired_reports_count(self):
        clock = FakeClock()
        store = InMemoryTaskStore(retention_sec=5, clock=clock)
        for _ in range(3):
            await store.save(finished_task())
        clock.now += 6
        assert await store.evict_expired() == 3

    async def test_none_disables_both_limits(self):
        store = InMemoryTaskStore(retention_sec=None, max_tasks=None)
        for _ in range(50):
            await store.save(finished_task())
        assert await store.count() == 50

    async def test_delete_and_clear_forget_retention_state(self):
        store = InMemoryTaskStore(retention_sec=None, max_tasks=1)
        task = finished_task()
        await store.save(task)
        await store.delete(task.id)
        await store.clear()
        assert await store.count() == 0

    @pytest.mark.parametrize(
        "kwargs",
        [{"retention_sec": -1}, {"max_tasks": 0}, {"max_tasks": -5}],
    )
    def test_invalid_limits_rejected(self, kwargs):
        with pytest.raises(ValueError):
            InMemoryTaskStore(**kwargs)

    def test_default_is_bounded(self):
        store = InMemoryTaskStore()
        assert store._retention == 3600.0
        assert store._max_tasks == 10_000


# ── AuthManager default credentials ───────────────────────────────────────────


def api_key_config(key: str = "k") -> AgentCredentialConfig:
    return AgentCredentialConfig(scheme=AuthScheme.API_KEY, api_key=key)


class TestAuthDefaultCredential:
    async def test_default_covers_unregistered_callers(self):
        manager = AuthManager(default=api_key_config())
        claims = await manager.verify("http://anyone:1", {"X-API-Key": "k"})
        assert claims["scheme"] == "api_key"

    async def test_default_still_fails_closed_on_wrong_key(self):
        manager = AuthManager(default=api_key_config())
        with pytest.raises(InvalidCredentialsError):
            await manager.verify("http://anyone:1", {"X-API-Key": "wrong"})

    async def test_default_still_requires_a_key(self):
        manager = AuthManager(default=api_key_config())
        with pytest.raises(MissingCredentialsError):
            await manager.verify("http://anyone:1", {})

    async def test_registered_caller_takes_precedence(self):
        manager = AuthManager(default=api_key_config("shared"))
        manager.register_agent("http://vip:1", api_key_config("vip-only"))
        await manager.verify("http://vip:1", {"X-API-Key": "vip-only"})
        with pytest.raises(InvalidCredentialsError):
            await manager.verify("http://vip:1", {"X-API-Key": "shared"})

    async def test_wildcard_registration_sets_the_default(self):
        """Documented since 1.2, but "*" was stored as a literal URL."""
        manager = AuthManager()
        manager.register_agent("*", api_key_config())
        assert manager.has_default
        await manager.verify("http://anyone:1", {"X-API-Key": "k"})

    async def test_unregistering_wildcard_clears_the_default(self):
        manager = AuthManager(default=api_key_config())
        manager.unregister_agent("*")
        assert not manager.has_default
        with pytest.raises(UnknownAgentError):
            await manager.verify("http://anyone:1", {"X-API-Key": "k"})

    def test_invalid_default_is_rejected(self):
        with pytest.raises(ValueError):
            AuthManager(default=AgentCredentialConfig(scheme=AuthScheme.API_KEY))

    def test_unknown_agent_error_points_at_default(self):
        assert "default=" in str(UnknownAgentError("http://x:1"))


class TestAdvertisedConfig:
    def test_default_is_advertised(self):
        manager = AuthManager(default=api_key_config())
        assert manager.advertised_config().scheme == AuthScheme.API_KEY

    def test_uniform_registrations_are_advertised(self):
        manager = AuthManager()
        manager.register_agent("http://a:1", api_key_config("1"))
        manager.register_agent("http://b:1", api_key_config("2"))
        assert manager.advertised_config().scheme == AuthScheme.API_KEY

    def test_mixed_registrations_advertise_nothing(self):
        manager = AuthManager()
        manager.register_agent("http://a:1", api_key_config())
        manager.register_agent(
            "http://b:1",
            AgentCredentialConfig(scheme=AuthScheme.JWT, jwt_secret=LONG_SECRET),
        )
        assert manager.advertised_config() is None

    def test_empty_manager_advertises_nothing(self):
        assert AuthManager().advertised_config() is None


class TestBuildAuthManagerFixed:
    async def test_configured_secret_is_enforced(self):
        """Before 1.9.0 the correct key was refused with UnknownAgentError."""
        cfg = NexusConfig.from_dict({
            "agent": {"name": "a", "url": "http://localhost:8001"},
            "security": {"auth_scheme": "api_key", "auth_secret": "s3cret"},
        })
        manager = cfg.build_auth_manager()
        await manager.verify("http://any:1", {"X-API-Key": "s3cret"})
        with pytest.raises(InvalidCredentialsError):
            await manager.verify("http://any:1", {"X-API-Key": "nope"})

    async def test_jwt_secret_is_enforced(self):
        cfg = NexusConfig.from_dict({
            "agent": {"name": "a", "url": "http://localhost:8001"},
            "security": {"auth_scheme": "jwt", "auth_secret": LONG_SECRET},
        })
        manager = cfg.build_auth_manager()
        with pytest.raises(MissingCredentialsError):
            await manager.verify("http://any:1", {})

    def test_none_scheme_builds_empty_manager(self):
        cfg = NexusConfig.from_dict({"agent": {"name": "a"}})
        assert not cfg.build_auth_manager().has_default


# ── SecurityMiddleware with default credentials / warn-only trust ─────────────


class TestMiddlewareDefaults:
    async def test_anonymous_caller_authenticates_against_default(self):
        mw = SecurityMiddleware(auth=AuthManager(default=api_key_config()))
        identity = await mw.authorize({"X-API-Key": "k"})
        assert identity.anonymous
        assert identity.claims["scheme"] == "api_key"

    async def test_anonymous_caller_with_wrong_key_refused(self):
        mw = SecurityMiddleware(auth=AuthManager(default=api_key_config()))
        with pytest.raises(InvalidCredentialsError):
            await mw.authorize({"X-API-Key": "bad"})

    async def test_per_caller_auth_still_needs_identity(self):
        manager = AuthManager()
        manager.register_agent("http://a:1", api_key_config())
        mw = SecurityMiddleware(auth=manager)
        with pytest.raises(MissingCallerError):
            await mw.authorize({"X-API-Key": "k"})

    async def test_trust_still_needs_identity_with_a_default(self):
        mw = SecurityMiddleware(
            auth=AuthManager(default=api_key_config()),
            trust=TrustBoundary(),
            server_url=SERVER_URL,
        )
        with pytest.raises(MissingCallerError):
            await mw.authorize({"X-API-Key": "k"})

    async def test_warn_only_trust_allows_and_logs(self, caplog):
        mw = SecurityMiddleware(
            trust=TrustBoundary(), server_url=SERVER_URL, trust_warn_only=True
        )
        with caplog.at_level("WARNING"):
            identity = await mw.authorize({CALLER_HEADER: "http://stranger:1"})
        assert identity.url == "http://stranger:1"
        assert "warn-only" in caplog.text

    async def test_enforced_trust_still_refuses(self):
        from nexus_a2a.security.trust import AgentNotAllowedError

        mw = SecurityMiddleware(trust=TrustBoundary(), server_url=SERVER_URL)
        with pytest.raises(AgentNotAllowedError):
            await mw.authorize({CALLER_HEADER: "http://stranger:1"})

    def test_summary_reports_warn_only(self):
        mw = SecurityMiddleware(
            trust=TrustBoundary(), server_url=SERVER_URL, trust_warn_only=True
        )
        assert mw.summary()["trust_warn_only"] is True


# ── Agent card advertises enforced auth ───────────────────────────────────────


class TestCardAdvertisesEnforcedAuth:
    def card(self, security: SecurityMiddleware | None = None) -> dict:
        server = A2AServer(EchoAgent, security=security)
        return TestClient(server.app).get("/.well-known/agent-card.json").json()

    def test_unsecured_card_says_none(self):
        assert self.card()["authentication"]["scheme"] == "none"

    def test_default_api_key_is_advertised(self):
        card = self.card(SecurityMiddleware(auth=AuthManager(default=api_key_config())))
        assert card["authentication"] == {"scheme": "api_key", "header_name": "X-API-Key"}

    def test_custom_header_is_advertised(self):
        config = AgentCredentialConfig(
            scheme=AuthScheme.API_KEY, api_key="k", header_name="X-Token"
        )
        card = self.card(SecurityMiddleware(auth=AuthManager(default=config)))
        assert card["authentication"]["header_name"] == "X-Token"

    def test_secret_never_appears_on_the_card(self):
        card = self.card(
            SecurityMiddleware(auth=AuthManager(default=api_key_config("TOPSECRET")))
        )
        assert "TOPSECRET" not in json.dumps(card)

    def test_jwt_is_advertised(self):
        config = AgentCredentialConfig(scheme=AuthScheme.JWT, jwt_secret=LONG_SECRET)
        card = self.card(SecurityMiddleware(auth=AuthManager(default=config)))
        assert card["authentication"] == {"scheme": "jwt"}

    def test_mixed_registrations_leave_card_alone(self):
        manager = AuthManager()
        manager.register_agent("http://a:1", api_key_config())
        manager.register_agent(
            "http://b:1",
            AgentCredentialConfig(scheme=AuthScheme.JWT, jwt_secret=LONG_SECRET),
        )
        card = self.card(SecurityMiddleware(auth=manager))
        assert card["authentication"]["scheme"] == "none"


# ── /metrics on A2AServer ─────────────────────────────────────────────────────


class TestA2AServerMetrics:
    def test_counts_tasks_by_state(self):
        client = TestClient(A2AServer(EchoAgent).app)
        rpc(client, "message/send", send_params())
        body = client.get("/metrics").text
        assert 'nexus_a2a_tasks{state="completed"} 1' in body
        assert "nexus_a2a_tasks_active 0" in body

    def test_is_public_and_text(self):
        auth = SecurityMiddleware(auth=AuthManager(default=api_key_config()))
        resp = TestClient(A2AServer(EchoAgent, security=auth).app).get("/metrics")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/plain")

    def test_reports_no_task_content(self):
        client = TestClient(A2AServer(EchoAgent).app)
        rpc(client, "message/send", send_params("very-private-text"))
        assert "very-private-text" not in client.get("/metrics").text

    def test_every_state_has_a_series(self):
        body = TestClient(A2AServer(EchoAgent).app).get("/metrics").text
        for state in TaskState:
            assert f'state="{state.value}"' in body


# ── Live-following GET /stream ────────────────────────────────────────────────


class TestTaskWatchers:
    def test_publish_reaches_subscribers(self):
        hub = _TaskWatchers()
        watcher, backlog = hub.subscribe("t")
        hub.publish("t", "line-1")
        assert backlog == []
        assert watcher.queue.get_nowait() == ("line-1", False)

    def test_late_subscriber_gets_backlog(self):
        hub = _TaskWatchers()
        hub.publish("t", "a")
        hub.publish("t", "b")
        _, backlog = hub.subscribe("t")
        assert backlog == ["a", "b"]

    def test_terminal_line_clears_backlog(self):
        hub = _TaskWatchers()
        hub.publish("t", "a")
        hub.publish("t", "done", terminal=True)
        _, backlog = hub.subscribe("t")
        assert backlog == []

    def test_end_clears_backlog(self):
        hub = _TaskWatchers()
        hub.publish("t", "a")
        hub.end("t")
        assert hub.subscribe("t")[1] == []

    def test_slow_watcher_is_cut_loose(self):
        hub = _TaskWatchers()
        watcher, _ = hub.subscribe("t")
        for i in range(watcher.queue.maxsize + 5):
            hub.publish("t", f"line-{i}")
        assert watcher.overflowed
        assert hub.count() == 0

    def test_backlog_is_bounded(self):
        hub = _TaskWatchers()
        for i in range(10_000):
            hub.publish("t", f"line-{i}")
        assert len(hub.subscribe("t")[1]) <= 256

    def test_unsubscribe_forgets_empty_tasks(self):
        hub = _TaskWatchers()
        watcher, _ = hub.subscribe("t")
        hub.unsubscribe("t", watcher)
        assert hub.count() == 0
        hub.unsubscribe("t", watcher)  # idempotent

    def test_watchers_are_hashable(self):
        """Regression: a plain @dataclass is unhashable and they live in a set."""
        hub = _TaskWatchers()
        hub.subscribe("t")
        hub.subscribe("t")
        assert hub.count() == 2

    def test_terminal_line_detection(self):
        assert _is_terminal_line(SSEFormatter.done())
        assert _is_terminal_line(SSEFormatter.error("x"))
        assert not _is_terminal_line(SSEFormatter.task_status("completed", "t"))


class TestStreamFollowing:
    async def test_follower_sees_a_running_task_to_completion(self):
        gate = asyncio.Event()

        @agent(name="Gated", description="Waits.", streaming=True, url=SERVER_URL)
        class Gated:
            async def run(self, task):
                yield "before"
                await gate.wait()
                yield "after"

        server = A2AServer(Gated, stream_heartbeat=0.05)
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            producer = asyncio.create_task(
                c.post("/", json={"jsonrpc": "2.0", "id": "1",
                                  "method": "message/stream",
                                  "params": send_params()})
            )
            task_id = None
            for _ in range(200):
                await asyncio.sleep(0.01)
                tasks = await server.tasks.list_all()
                if tasks:
                    task_id = tasks[0].id
                    break
            assert task_id is not None
            await asyncio.sleep(0.05)  # let "before" be published

            follower = asyncio.create_task(
                c.get("/stream", params={"taskId": task_id})
            )
            await asyncio.sleep(0.15)  # follower is waiting; heartbeats flow
            gate.set()
            resp = await follower
            await producer

        chunks = [
            json.loads(line[5:])["content"]
            for line in resp.text.splitlines()
            if line.startswith("data:") and '"artifact_chunk"' in line
        ]
        assert chunks == ["before", "after"]  # "before" arrived via backlog
        assert ": heartbeat" in resp.text
        assert resp.text.strip().endswith('"type": "done"}')

    async def test_follower_of_message_send_run_gets_outcome(self):
        gate = asyncio.Event()

        @agent(name="Slow", description="Waits.", url=SERVER_URL)
        class Slow:
            async def run(self, task):
                await gate.wait()
                return "finished"

        server = A2AServer(Slow, stream_heartbeat=0.05)
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            producer = asyncio.create_task(
                c.post("/", json={"jsonrpc": "2.0", "id": "1",
                                  "method": "message/send", "params": send_params()})
            )
            task_id = None
            for _ in range(200):
                await asyncio.sleep(0.01)
                tasks = await server.tasks.list_all()
                if tasks:
                    task_id = tasks[0].id
                    break
            follower = asyncio.create_task(c.get("/stream", params={"taskId": task_id}))
            await asyncio.sleep(0.1)
            gate.set()
            resp = await follower
            await producer

        assert '"content": "finished"' in resp.text
        assert '"state": "completed"' in resp.text
        assert '"type": "done"' in resp.text

    def test_finished_task_is_reported_not_followed(self):
        client = TestClient(A2AServer(EchoAgent).app)
        task_id = rpc(client, "message/send", send_params()).json()["result"]["id"]
        body = client.get("/stream", params={"taskId": task_id}).text
        assert "heartbeat" not in body
        assert body.strip().endswith('"type": "done"}')

    def test_unknown_task_leaves_no_watcher(self):
        server = A2AServer(EchoAgent)
        assert TestClient(server.app).get("/stream", params={"taskId": "x"}).status_code == 404
        assert server._watchers.count() == 0


# ── Watchdog / finish race ────────────────────────────────────────────────────


class TestLateResultAfterWatchdog:
    async def test_late_result_is_discarded(self, caplog):
        server = A2AServer(EchoAgent)
        task = await server.tasks.create(initial_message=Message.user_text("x"))
        task = await server.tasks.start(task.id)
        await server.tasks.fail(task.id, "watchdog timeout")

        with caplog.at_level("WARNING"):
            result = await server._finish(task, "too late")
        assert result.state == TaskState.FAILED
        assert result.error == "watchdog timeout"
        assert "late result" in caplog.text


# ── Prometheus label escaping ─────────────────────────────────────────────────


class TestLabelEscaping:
    def test_quote_backslash_newline_are_escaped(self):
        assert _escape_label_value('a"b\\c\nd') == 'a\\"b\\\\c\\nd'

    def test_injected_metric_line_is_neutralised(self):
        evil = 'x"} 999\nnexus_a2a_dlq_pending{a="'
        rendered = _format_labels({"agent_url": evil})
        assert "\n" not in rendered
        assert rendered.count('"') - rendered.count('\\"') == 2

    def test_plain_values_unchanged(self):
        assert _format_labels({"agent_url": "http://a:1"}) == '{agent_url="http://a:1"}'


# ── Trace lookup by task id ───────────────────────────────────────────────────


class TestTraceByTaskId:
    async def test_find_by_task_id(self):
        store = TraceStore()
        span = Span(trace_id="trace-1", agent_url="http://a:1")
        span.metadata["task_id"] = "task-9"
        await store.record(span)
        assert store.find_by_task_id("task-9").trace_id == "trace-1"
        assert store.find_by_task_id("nope") is None

    async def test_resolve_prefers_trace_id(self):
        store = TraceStore()
        await store.record(Span(trace_id="trace-1"))
        assert store.resolve("trace-1").trace_id == "trace-1"

    async def test_resolve_falls_back_to_task_id(self):
        store = TraceStore()
        span = Span(trace_id="trace-2")
        span.metadata["task_id"] = "task-2"
        await store.record(span)
        assert store.resolve("task-2").trace_id == "trace-2"


# ── nexus.toml: new keys ──────────────────────────────────────────────────────


def cfg_from(**sections) -> NexusConfig:
    raw = {"agent": {"name": "a", "url": "http://127.0.0.1:8123"}}
    raw.update(sections)
    return NexusConfig.from_dict(raw)


class TestConfigNewKeys:
    def test_defaults(self):
        cfg = cfg_from()
        assert cfg.security.rate_limit == 0
        assert cfg.security.allow_insecure is False
        assert cfg.storage.task_retention_sec == 3600
        assert cfg.ops.port == 0
        assert cfg.push.allow_private_urls is False

    def test_parses_every_new_key(self):
        cfg = cfg_from(
            security={"rate_limit": 5, "rate_burst": 7, "max_payload_bytes": 2048,
                      "allow_insecure": True},
            storage={"task_retention_sec": 30, "max_tasks": 50},
            push={"signing_secret": "p", "allow_private_urls": True, "max_retries": 1},
            ops={"port": 9100, "host": "0.0.0.0", "url": "http://o:9100",
                 "admin_token": "t"},
        )
        assert (cfg.security.rate_limit, cfg.security.rate_burst) == (5, 7)
        assert cfg.security.max_payload_bytes == 2048
        assert cfg.security.allow_insecure is True
        assert (cfg.storage.task_retention_sec, cfg.storage.max_tasks) == (30, 50)
        assert cfg.push.signing_secret == "p"
        assert cfg.ops.port == 9100 and cfg.ops.admin_token == "t"

    @pytest.mark.parametrize(
        ("sections", "key"),
        [
            ({"security": {"rate_limit": -1}}, "security.rate_limit"),
            ({"security": {"rate_limit": 1, "rate_burst": 0}}, "security.rate_burst"),
            ({"security": {"max_payload_bytes": -1}}, "security.max_payload_bytes"),
            ({"storage": {"task_retention_sec": -1}}, "storage.task_retention_sec"),
            ({"storage": {"max_tasks": -1}}, "storage.max_tasks"),
            ({"ops": {"port": 70000}}, "ops.port"),
            ({"security": {"rate_limit": "fast"}}, "security.rate_limit"),
            ({"ops": {"port": True}}, "ops.port"),
        ],
    )
    def test_invalid_values_name_their_key(self, sections, key):
        with pytest.raises(ConfigError) as exc:
            cfg_from(**sections)
        assert exc.value.key == key

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("NEXUS_RATE_LIMIT", "9")
        monkeypatch.setenv("NEXUS_PUSH_SECRET", "env-push")
        monkeypatch.setenv("NEXUS_ADMIN_TOKEN", "env-admin")
        monkeypatch.setenv("NEXUS_OPS_PORT", "9200")
        monkeypatch.setenv("NEXUS_OPS_URL", "http://ops:9200")
        cfg = cfg_from()
        assert cfg.security.rate_limit == 9
        assert cfg.push.signing_secret == "env-push"
        assert cfg.ops.admin_token == "env-admin"
        assert cfg.ops.port == 9200
        assert cfg.ops.url == "http://ops:9200"

    @pytest.mark.parametrize("var", ["NEXUS_RATE_LIMIT", "NEXUS_OPS_PORT"])
    def test_bad_env_values_rejected(self, monkeypatch, var):
        monkeypatch.setenv(var, "not-a-number")
        with pytest.raises(ConfigError):
            cfg_from()

    def test_to_dict_redacts_new_secrets(self):
        d = cfg_from(push={"signing_secret": "PUSHSECRET"},
                     ops={"admin_token": "ADMINTOKEN"}).to_dict()
        assert "PUSHSECRET" not in json.dumps(d)
        assert "ADMINTOKEN" not in json.dumps(d)
        assert d["push"]["signing_secret"] == "***REDACTED***"

    def test_memory_store_gets_retention(self):
        store = cfg_from(storage={"task_retention_sec": 42, "max_tasks": 7}).build_task_store()
        assert store._retention == 42
        assert store._max_tasks == 7

    def test_zero_means_unlimited(self):
        store = cfg_from(storage={"task_retention_sec": 0, "max_tasks": 0}).build_task_store()
        assert store._retention is None
        assert store._max_tasks is None


class TestBuildSecurity:
    def test_nothing_configured_is_a_no_op(self):
        assert cfg_from().build_security("http://x:1").enabled is False

    def test_everything_configured(self):
        cfg = cfg_from(
            security={"auth_scheme": "api_key", "auth_secret": "k",
                      "trust_mode": "strict", "rate_limit": 3, "max_payload_bytes": 100},
            network={"agents": ["http://peer:1"]},
        )
        summary = cfg.build_security("http://me:1").summary()
        assert summary["auth"] and summary["trust"]
        assert summary["rate_limit"] and summary["validation"]
        assert summary["trust_warn_only"] is False

    def test_warn_mode(self):
        cfg = cfg_from(security={"trust_mode": "warn"})
        assert cfg.build_security("http://me:1").trust_warn_only is True

    async def test_strict_trust_allows_listed_peers_only(self):
        from nexus_a2a.security.trust import AgentNotAllowedError

        mw = cfg_from(
            security={"trust_mode": "strict"},
            network={"agents": ["http://peer:1"]},
        ).build_security("http://me:1")
        await mw.authorize({CALLER_HEADER: "http://peer:1"})
        with pytest.raises(AgentNotAllowedError):
            await mw.authorize({CALLER_HEADER: "http://other:1"})

    def test_strict_with_no_peers_warns(self, caplog):
        with caplog.at_level("WARNING"):
            cfg_from(security={"trust_mode": "strict"}).build_security("http://me:1")
        assert "refuses every caller" in caplog.text

    def test_redis_backend_uses_shared_rate_limiter(self):
        from nexus_a2a.security.redis_rate_limiter import RedisRateLimiter

        cfg = cfg_from(
            security={"rate_limit": 3},
            storage={"backend": "redis", "url": "redis://localhost:6379"},
        )
        assert isinstance(cfg.build_security("http://me:1").rate_limiter, RedisRateLimiter)


class TestBuildRuntime:
    def test_minimal_runtime(self):
        runtime = cfg_from().build_runtime(EchoAgent)
        assert isinstance(runtime, AgentRuntime)
        assert runtime.ops is None
        assert runtime.resources == []
        assert (runtime.server.host, runtime.server.port) == ("127.0.0.1", 8123)

    def test_host_port_overrides(self):
        runtime = cfg_from().build_runtime(EchoAgent, host="0.0.0.0", port=9999)
        assert (runtime.server.host, runtime.server.port) == ("0.0.0.0", 9999)

    def test_card_url_comes_from_config(self):
        runtime = cfg_from().build_runtime(EchoAgent)
        card = TestClient(runtime.server.app).get("/.well-known/agent-card.json").json()
        assert card["url"] == "http://127.0.0.1:8123"

    def test_ops_server_built_when_port_set(self):
        runtime = cfg_from(ops={"port": 9555, "admin_token": "t"}).build_runtime(EchoAgent)
        assert runtime.ops is not None
        assert runtime.ops.port == 9555
        assert runtime.ops._admin_token == "t"
        assert runtime.ops.host == "127.0.0.1"

    def test_ops_server_shares_the_task_manager(self):
        runtime = cfg_from(ops={"port": 9555}).build_runtime(EchoAgent)
        assert runtime.ops.network.task_manager is runtime.server.tasks

    def test_redis_backend_registers_resources(self):
        from nexus_a2a.storage.push_store import RedisPushStore
        from nexus_a2a.storage.redis_store import RedisTaskStore

        runtime = cfg_from(
            storage={"backend": "redis", "url": "redis://localhost:6379"},
            ops={"port": 9555},
        ).build_runtime(EchoAgent)
        kinds = {type(r).__name__ for r in runtime.resources}
        assert {"RedisTaskStore", "RedisPushStore", "RedisDLQStore"} <= kinds
        assert isinstance(runtime.server._push_targets, RedisPushStore)
        assert isinstance(runtime.server.tasks._store, RedisTaskStore)

    def test_push_settings_applied(self):
        runtime = cfg_from(
            push={"signing_secret": "p", "allow_private_urls": True, "max_retries": 1}
        ).build_runtime(EchoAgent)
        assert runtime.server._push_config.signing_secret == "p"
        assert runtime.server._push_config.allow_private_urls is True

    def test_watchdog_uses_configured_timeout(self):
        runtime = cfg_from(reliability={"task_timeout_sec": 7}).build_runtime(EchoAgent)
        assert runtime.watchdog._timeout_sec == 7

    async def test_configured_server_enforces_the_secret(self):
        runtime = cfg_from(
            security={"auth_scheme": "api_key", "auth_secret": "k"}
        ).build_runtime(EchoAgent)
        client = TestClient(runtime.server.app)
        assert rpc(client, "message/send", send_params()).status_code == 401
        ok = rpc(client, "message/send", send_params(), headers={"X-API-Key": "k"})
        assert ok.json()["result"]["state"] == "completed"


# ── AgentRuntime lifecycle ────────────────────────────────────────────────────


class Recorder:
    """Stands in for a server, store or watchdog and logs every call."""

    host = "127.0.0.1"
    port = 0

    def __init__(self, log: list[str], name: str, fail_on: str | None = None):
        self.log, self.name, self.fail_on = log, name, fail_on

    async def _step(self, what: str) -> None:
        self.log.append(f"{self.name}.{what}")
        if self.fail_on == what:
            raise RuntimeError(f"{self.name} {what} failed")

    async def connect(self):
        await self._step("connect")

    async def disconnect(self):
        await self._step("disconnect")

    async def start(self):
        await self._step("start")

    async def stop(self):
        await self._step("stop")

    async def start_watchdog(self):
        await self._step("start")

    async def stop_watchdog(self):
        await self._step("stop")


class TestAgentRuntime:
    def make(self, log, **fail):
        return AgentRuntime(
            server=Recorder(log, "server", fail.get("server")),
            ops=Recorder(log, "ops", fail.get("ops")),
            resources=[Recorder(log, "db", fail.get("db")),
                       Recorder(log, "cache", fail.get("cache"))],
            watchdog=Recorder(log, "watchdog", fail.get("watchdog")),
        )

    async def test_start_and_stop_order(self):
        log: list[str] = []
        runtime = self.make(log)
        async with runtime:
            pass
        assert log == [
            "db.connect", "cache.connect", "watchdog.start",
            "server.start", "ops.start",
            "ops.stop", "server.stop", "watchdog.stop",
            "cache.disconnect", "db.disconnect",
        ]

    async def test_failed_start_tears_down_what_started(self):
        log: list[str] = []
        runtime = self.make(log, server="start")
        with pytest.raises(RuntimeError, match="server start failed"):
            await runtime.start()
        assert "cache.disconnect" in log and "db.disconnect" in log
        assert "ops.start" not in log

    async def test_failed_connect_disconnects_only_what_connected(self):
        log: list[str] = []
        runtime = self.make(log, cache="connect")
        with pytest.raises(RuntimeError):
            await runtime.start()
        assert "db.disconnect" in log
        assert "cache.disconnect" not in log

    async def test_teardown_continues_past_a_failing_step(self):
        log: list[str] = []
        runtime = self.make(log, server="stop")
        await runtime.start()
        await runtime.stop()
        assert "db.disconnect" in log

    async def test_stop_is_idempotent(self):
        log: list[str] = []
        runtime = self.make(log)
        await runtime.stop()
        assert log == []
        await runtime.start()
        await runtime.stop()
        await runtime.stop()
        assert log.count("server.stop") == 1

    async def test_double_start_rejected(self):
        runtime = self.make([])
        await runtime.start()
        try:
            with pytest.raises(RuntimeError, match="already started"):
                await runtime.start()
        finally:
            await runtime.stop()


# ── 2.0 groundwork ────────────────────────────────────────────────────────────


class TestFutureWarnings:
    async def test_public_unsecured_bind_warns(self):
        server = A2AServer(EchoAgent, host="0.0.0.0", port=0)
        with pytest.warns(FutureWarning, match="2.0 will refuse"):
            server._warn_if_insecure()

    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
    def test_loopback_does_not_warn(self, host):
        server = A2AServer(EchoAgent, host=host)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            server._warn_if_insecure()

    def test_security_silences_the_warning(self):
        server = A2AServer(
            EchoAgent, host="0.0.0.0",
            security=SecurityMiddleware(auth=AuthManager(default=api_key_config())),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            server._warn_if_insecure()

    def test_allow_insecure_silences_the_warning(self):
        server = A2AServer(EchoAgent, host="0.0.0.0", allow_insecure=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            server._warn_if_insecure()

    def test_config_allow_insecure_reaches_the_server(self):
        runtime = cfg_from(security={"allow_insecure": True}).build_runtime(EchoAgent)
        assert runtime.server.allow_insecure is True

    def test_allow_unregistered_is_deprecated(self):
        with pytest.warns(DeprecationWarning, match="removed in nexus-a2a 2.0"):
            AuthManager(allow_unregistered=True)

    def test_short_hs256_secret_warns(self):
        with pytest.warns(FutureWarning, match="RFC 7518"):
            AuthManager().register_agent(
                "http://a:1",
                AgentCredentialConfig(scheme=AuthScheme.JWT, jwt_secret="short"),
            )

    def test_long_hs256_secret_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            AuthManager().register_agent(
                "http://a:1",
                AgentCredentialConfig(scheme=AuthScheme.JWT, jwt_secret=LONG_SECRET),
            )


class TestLoopbackDetection:
    @pytest.mark.parametrize(
        ("host", "expected"),
        [
            ("127.0.0.1", True), ("127.5.5.5", True), ("::1", True),
            ("localhost", True), ("0.0.0.0", False), ("10.0.0.1", False),
            ("example.com", False), ("", False),
        ],
    )
    def test_cases(self, host, expected):
        assert _is_loopback(host) is expected


# ── Unknown nexus.toml keys ───────────────────────────────────────────────────


class TestUnknownConfigKeys:
    def test_typo_is_reported_with_a_suggestion(self):
        from nexus_a2a.config import ConfigWarning

        with pytest.warns(ConfigWarning, match="did you mean 'auth_scheme'"):
            NexusConfig.from_dict({"agent": {"name": "a"},
                                   "security": {"auth_schem": "api_key"}})

    def test_unknown_section_is_reported(self):
        from nexus_a2a.config import ConfigWarning

        with pytest.warns(ConfigWarning, match=r"\[reliabilty\].*'reliability'"):
            NexusConfig.from_dict({"agent": {"name": "a"}, "reliabilty": {}})

    def test_formerly_documented_phantom_keys_are_reported(self):
        """The README documented these; nothing ever read them."""
        from nexus_a2a.config import _unknown_key_messages

        messages = _unknown_key_messages({
            "security": {"mtls_cert_file": "x"},
            "reliability": {"dlq_max_size": 5},
        })
        assert any("security.mtls_cert_file" in m for m in messages)
        assert any("reliability.dlq_max_size" in m for m in messages)

    def test_unknown_skill_key_is_reported(self):
        from nexus_a2a.config import _unknown_key_messages

        messages = _unknown_key_messages(
            {"agent": {"skills": [{"id": "a", "name": "A", "tag": ["x"]}]}}
        )
        assert messages == [
            "unknown key 'agent.skills[0].tag' (did you mean 'tags'?)"
        ]

    def test_valid_config_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cfg_from(
                security={"auth_scheme": "none", "rate_limit": 1},
                storage={"backend": "memory", "max_tasks": 5},
                push={"max_retries": 1},
                ops={"port": 0},
                observability={"log_level": "INFO"},
                network={"agents": []},
                dev={"agents": []},
            )

    def test_every_parsed_field_is_a_known_key(self):
        """Guards KNOWN_KEYS against drifting from the dataclasses."""
        import dataclasses

        from nexus_a2a import config as config_module

        sections = {
            "agent": config_module.AgentConfig,
            "network": config_module.NetworkConfig,
            "reliability": config_module.ReliabilityConfig,
            "security": config_module.SecurityConfig,
            "storage": config_module.StorageConfig,
            "observability": config_module.ObservabilityConfig,
            "push": config_module.PushConfig,
            "ops": config_module.OpsConfig,
        }
        for section, cls in sections.items():
            fields = {f.name for f in dataclasses.fields(cls)}
            assert fields <= config_module.KNOWN_KEYS[section], section
