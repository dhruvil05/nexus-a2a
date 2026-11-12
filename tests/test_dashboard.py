"""
tests/test_dashboard.py

The 2.1 dashboard: event hub, TaskManager listeners, the /ui API (access
control, login, CSRF, every endpoint), the live stream over a real server,
config and CLI wiring.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from starlette.testclient import TestClient

from nexus_a2a import agent
from nexus_a2a.config import ConfigError, NexusConfig
from nexus_a2a.core.a2a_server import A2AServer
from nexus_a2a.core.agent_server import AgentServer
from nexus_a2a.core.task_manager import TaskManager
from nexus_a2a.dashboard import Dashboard, DashboardConfig, EventHub
from nexus_a2a.models.agent import AuthScheme
from nexus_a2a.models.task import Message, NeedsInput, Task
from nexus_a2a.network import AgentNetwork
from nexus_a2a.security.auth import AgentCredentialConfig, AuthManager
from nexus_a2a.security.middleware import CALLER_HEADER, SecurityMiddleware

TOKEN = "dashboard-test-token-0123456789"
# The dashboard serves only a loopback peer that names a loopback Host.
UI_BASE = "http://127.0.0.1:9100"
LOOPBACK = ("127.0.0.1", 50000)


def ui_client(app: Any, **kwargs: Any) -> TestClient:
    kwargs.setdefault("base_url", UI_BASE)
    kwargs.setdefault("client", LOOPBACK)
    return TestClient(app, **kwargs)


AGENT_URL = "http://127.0.0.1:8765"
SKILLS = [{"id": "echo", "name": "Echo", "description": "Echoes."}]


@agent(
    name="DashEcho",
    description="Echo for dashboard tests.",
    skills=SKILLS,
    url=AGENT_URL,
    push_notifications=True,
)
class DashEcho:
    calls = 0

    async def run(self, task: Task):
        text = task.latest_message().text() if task.latest_message() else ""
        if text == "ask":
            return NeedsInput("need more")
        if text == "boom":
            raise RuntimeError("agent exploded")
        if text == "flaky":
            DashEcho.calls += 1
            if DashEcho.calls % 2 == 1:
                raise RuntimeError("first try fails")
        return f"echo: {text}"


# ── Fixtures ──────────────────────────────────────────────────────────────────


class Setup:
    def __init__(
        self,
        config: DashboardConfig | None = None,
        token: str | None = TOKEN,
        security: SecurityMiddleware | None = None,
    ) -> None:
        self.server = A2AServer(DashEcho, task_manager=TaskManager(), security=security)
        self.network = AgentNetwork(task_manager=self.server.tasks)
        self.dashboard = Dashboard(
            admin_token=token, server=self.server, network=self.network, config=config
        )
        self.ops = AgentServer(
            network=self.network, admin_token=token, dashboard=self.dashboard
        )
        self.agent = TestClient(self.server.app)
        self.ui = ui_client(self.ops._app)

    def send(self, text: str = "hi", headers: dict[str, str] | None = None) -> dict:
        msg = Message.user_text(text).model_dump(mode="json")
        return self.agent.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": "1",
                "method": "message/send",
                "params": {"message": msg},
            },
            headers=headers or {},
        ).json()

    def login(self) -> str:
        resp = self.ui.post("/ui/api/login", json={"token": TOKEN})
        assert resp.status_code == 200, resp.text
        return resp.json()["csrf"]

    def post(self, path: str, csrf: str, body: Any = None) -> httpx.Response:
        return self.ui.post(
            path,
            json=body or {},
            headers={"X-CSRF-Token": csrf, "Origin": UI_BASE},
        )


@pytest.fixture
def setup() -> Iterator[Setup]:
    s = Setup()
    yield s
    s.dashboard.detach()


@pytest.fixture
def logged_in(setup: Setup) -> tuple[Setup, str]:
    return setup, setup.login()


# ── EventHub and listeners ────────────────────────────────────────────────────


class TestTaskManagerListeners:
    async def test_lifecycle_events_in_order(self):
        manager = TaskManager()
        seen: list[str] = []
        manager.add_listener(lambda task, event: seen.append(event))
        task = await manager.create(Message.user_text("x"))
        await manager.start(task.id)
        await manager.request_input(task.id, Message.agent_text("?"))
        await manager.provide_input(task.id, Message.user_text("y"))
        await manager.complete(task.id)
        assert seen == ["created", "started", "input_required", "resumed", "completed"]

    async def test_broken_listener_never_breaks_a_task(self):
        manager = TaskManager()

        def broken(task: Task, event: str) -> None:
            raise RuntimeError("observer bug")

        manager.add_listener(broken)
        task = await manager.create(Message.user_text("x"))
        assert (await manager.start(task.id)).state.value == "working"

    async def test_remove_listener(self):
        manager = TaskManager()
        seen: list[str] = []
        listener = lambda task, event: seen.append(event)  # noqa: E731
        manager.add_listener(listener)
        manager.remove_listener(listener)
        await manager.create(Message.user_text("x"))
        assert seen == []


class TestEventHub:
    async def test_counts_follow_transitions(self):
        manager, hub = TaskManager(), EventHub()
        manager.add_listener(hub.on_task_event)
        a = await manager.create(Message.user_text("a"))
        b = await manager.create(Message.user_text("b"))
        await manager.start(a.id)
        await manager.complete(a.id)
        await manager.start(b.id)
        await manager.fail(b.id, "x")
        counts = hub.counts()
        assert counts["completed"] == 1 and counts["failed"] == 1
        assert counts["submitted"] == counts["working"] == 0
        assert hub.total_seen == 2
        assert hub.window(300) == (1, 1)
        assert hub.latency()["samples"] == 1

    async def test_seed_then_events(self):
        manager = TaskManager()
        old = await manager.create(Message.user_text("old"))
        hub = EventHub()
        hub.seed(await manager.list_all())
        manager.add_listener(hub.on_task_event)
        await manager.start(old.id)
        counts = hub.counts()
        assert counts["working"] == 1 and counts["submitted"] == 0
        assert hub.total_seen == 1

    async def test_recent_index_is_bounded(self):
        manager, hub = TaskManager(), EventHub(max_tasks=5)
        manager.add_listener(hub.on_task_event)
        for i in range(12):
            await manager.create(Message.user_text(str(i)))
        assert len(hub.recent_tasks()) == 5

    async def test_throughput_has_thirty_minutes(self):
        assert len(EventHub().throughput()) == 30

    async def test_slow_subscriber_is_dropped(self):
        hub = EventHub(queue_size=2)
        queue = hub.subscribe()
        assert queue is not None
        for i in range(5):
            hub.publish("log", {"n": i})
        assert hub.subscriber_count == 0

    async def test_subscriber_cap(self):
        hub = EventHub(max_subscribers=1)
        assert hub.subscribe() is not None
        assert hub.subscribe() is None


# ── Access control ────────────────────────────────────────────────────────────


class TestAccess:
    def test_off_without_admin_token(self):
        s = Setup(token=None)
        assert s.ui.get("/ui/").status_code == 404
        assert s.ui.get("/ui/api/session").status_code == 404

    def test_off_when_disabled(self):
        s = Setup(config=DashboardConfig(enabled=False))
        assert s.ui.get("/ui/api/session").status_code == 404

    def test_remote_clients_refused_by_default(self, setup: Setup):
        remote = ui_client(setup.ops._app, client=("10.1.2.3", 5000))
        assert remote.get("/ui/api/session").status_code == 403

    def test_remote_clients_allowed_when_configured(self):
        s = Setup(config=DashboardConfig(allow_remote=True))
        remote = ui_client(s.ops._app, client=("10.1.2.3", 5000))
        assert remote.get("/ui/api/session").status_code == 200
        s.dashboard.detach()

    def test_api_needs_login(self, setup: Setup):
        assert setup.ui.get("/ui/api/overview").status_code == 401

    def test_header_token_works_for_scripts(self, setup: Setup):
        resp = setup.ui.get("/ui/api/overview", headers={"X-Admin-Token": TOKEN})
        assert resp.status_code == 200

    def test_wrong_header_token_rejected(self, setup: Setup):
        resp = setup.ui.get("/ui/api/overview", headers={"X-Admin-Token": "nope"})
        assert resp.status_code == 401

    def test_security_headers(self, setup: Setup):
        resp = setup.ui.get("/ui/api/session")
        assert "frame-ancestors 'none'" in resp.headers["content-security-policy"]
        assert resp.headers["x-frame-options"] == "DENY"
        assert resp.headers["cache-control"] == "no-store"


class TestLogin:
    def test_wrong_token(self, setup: Setup):
        resp = setup.ui.post("/ui/api/login", json={"token": "wrong"})
        assert resp.status_code == 401
        kinds = [e["kind"] for e in setup.dashboard.hub.security_events()]
        assert "login_failed" in kinds

    def test_login_sets_strict_httponly_cookie(self, setup: Setup):
        resp = setup.ui.post("/ui/api/login", json={"token": TOKEN})
        cookie = resp.headers["set-cookie"].lower()
        assert "httponly" in cookie and "samesite=strict" in cookie
        assert "path=/ui" in cookie
        session = setup.ui.get("/ui/api/session").json()
        assert session["authenticated"] is True
        assert session["csrf"] == resp.json()["csrf"]

    def test_login_is_throttled(self, setup: Setup):
        codes = [
            setup.ui.post("/ui/api/login", json={"token": "x"}).status_code
            for _ in range(7)
        ]
        assert codes[-1] == 429

    def test_logout(self, logged_in):
        s, csrf = logged_in
        assert s.post("/ui/api/logout", csrf).status_code == 204
        assert s.ui.get("/ui/api/overview").status_code == 401

    def test_post_without_csrf_refused(self, logged_in):
        s, _ = logged_in
        assert s.ui.post("/ui/api/agents/check", json={}).status_code == 403

    def test_cross_origin_post_refused(self, logged_in):
        s, csrf = logged_in
        resp = s.ui.post(
            "/ui/api/agents/check",
            json={},
            headers={"X-CSRF-Token": csrf, "Origin": "http://evil.example"},
        )
        assert resp.status_code == 403

    def test_post_with_csrf_allowed(self, logged_in):
        s, csrf = logged_in
        assert s.post("/ui/api/agents/check", csrf).status_code == 200


# ── Endpoints ─────────────────────────────────────────────────────────────────


class TestOverviewAndTasks:
    def test_overview_counts(self, logged_in):
        s, _ = logged_in
        s.send("one")
        s.send("boom")
        s.send("ask")
        data = s.ui.get("/ui/api/overview").json()
        by_state = data["tasks"]["by_state"]
        assert by_state["completed"] == 1
        assert by_state["failed"] == 1
        assert by_state["input_required"] == 1
        assert data["tasks"]["in_flight"] == 1
        assert data["throughput"]["completed_last_5m"] == 1
        assert data["error_rate"] == 0.5
        assert data["health"]["store"] == "ok"
        assert data["agent"]["name"] == "DashEcho"
        assert len(data["throughput"]["per_minute"]) == 30

    def test_tasks_created_before_dashboard_are_seeded(self):
        server = A2AServer(DashEcho, task_manager=TaskManager())
        TestClient(server.app).post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": "1",
                "method": "message/send",
                "params": {
                    "message": Message.user_text("early").model_dump(mode="json")
                },
            },
        )
        dashboard = Dashboard(admin_token=TOKEN, server=server)
        dashboard.attach()
        ops = AgentServer(
            network=AgentNetwork(task_manager=server.tasks),
            admin_token=TOKEN,
            dashboard=dashboard,
        )
        data = (
            ui_client(ops._app)
            .get("/ui/api/tasks", headers={"X-Admin-Token": TOKEN})
            .json()
        )
        dashboard.detach()
        assert data["total"] == 1

    def test_task_list_filters(self, logged_in):
        s, _ = logged_in
        s.send("a")
        s.send("boom")
        failed = s.ui.get("/ui/api/tasks?state=failed").json()
        assert failed["total"] == 1 and failed["tasks"][0]["state"] == "failed"
        # Search sees only what is shown: hidden error text cannot be probed.
        assert s.ui.get("/ui/api/tasks?q=exploded").json()["total"] == 0
        assert s.ui.get("/ui/api/tasks?q=RuntimeError").json()["total"] == 1

    def test_content_hidden_by_default(self, logged_in):
        s, _ = logged_in
        task_id = s.send("secret prompt")["result"]["id"]
        summary = s.ui.get("/ui/api/tasks").json()["tasks"][0]
        assert summary["preview"] is None
        detail = s.ui.get(f"/ui/api/tasks/{task_id}").json()
        part = detail["history"][0]["parts"][0]
        assert part == {"type": "text", "content": None, "length": 13, "redacted": True}
        assert "secret prompt" not in json.dumps(detail)

    def test_content_shown_when_enabled(self):
        s = Setup(config=DashboardConfig(show_content=True))
        s.login()
        task_id = s.send("visible text")["result"]["id"]
        detail = s.ui.get(f"/ui/api/tasks/{task_id}").json()
        assert detail["history"][0]["parts"][0]["content"] == "visible text"
        assert detail["task"]["preview"] == "visible text"
        s.dashboard.detach()

    def test_detail_timeline_and_actions(self, logged_in):
        s, _ = logged_in
        task_id = s.send("hello")["result"]["id"]
        detail = s.ui.get(f"/ui/api/tasks/{task_id}").json()
        events = [e["event"] for e in detail["timeline"]]
        assert events == ["created", "started", "completed"]
        assert detail["actions"] == {"can_cancel": False, "can_retry": True}

    def test_unknown_task_404(self, logged_in):
        s, _ = logged_in
        assert s.ui.get("/ui/api/tasks/nope").status_code == 404


class TestActions:
    def test_cancel_waiting_task(self, logged_in):
        s, csrf = logged_in
        task_id = s.send("ask")["result"]["id"]
        resp = s.post(f"/ui/api/tasks/{task_id}/cancel", csrf)
        assert resp.status_code == 200
        assert resp.json()["task"]["state"] == "cancelled"
        assert s.post(f"/ui/api/tasks/{task_id}/cancel", csrf).status_code == 409

    async def test_retry_failed_task(self, logged_in):
        s, csrf = logged_in
        DashEcho.calls = 0
        task_id = s.send("flaky")["result"]["id"]  # first call fails
        resp = s.post(f"/ui/api/tasks/{task_id}/retry", csrf)
        assert resp.status_code == 202
        new_id = resp.json()["task"]["id"]
        assert new_id != task_id
        for _ in range(50):
            new = s.ui.get(f"/ui/api/tasks/{new_id}").json()["task"]
            if new["state"] == "completed":
                break
            await asyncio.sleep(0.02)
        assert new["state"] == "completed"
        timeline = s.ui.get(f"/ui/api/tasks/{task_id}").json()["timeline"]
        assert "retried" in [e["event"] for e in timeline]

    def test_retry_disabled(self):
        s = Setup(config=DashboardConfig(allow_replay=False))
        csrf = s.login()
        task_id = s.send("boom")["result"]["id"]
        assert s.post(f"/ui/api/tasks/{task_id}/retry", csrf).status_code == 403
        assert s.post("/ui/api/dlq/replay", csrf, {"all": True}).status_code == 403
        s.dashboard.detach()

    def test_retry_running_task_refused(self, logged_in):
        s, csrf = logged_in
        task_id = s.send("ask")["result"]["id"]
        assert s.post(f"/ui/api/tasks/{task_id}/retry", csrf).status_code == 409

    def test_dlq_replay_validates_body(self, logged_in):
        s, csrf = logged_in
        assert s.post("/ui/api/dlq/replay", csrf, {"task_ids": "x"}).status_code == 400
        resp = s.post("/ui/api/dlq/replay", csrf, {"task_ids": ["missing"]})
        assert resp.json()["results"][0]["success"] is False


class TestIssuesAndLogs:
    def test_security_events_from_agent_server(self):
        auth = AuthManager()
        auth.register_agent(
            "http://a:1", AgentCredentialConfig(scheme=AuthScheme.API_KEY, api_key="k")
        )
        s = Setup(security=SecurityMiddleware(auth=auth))
        s.login()
        s.send("x", headers={CALLER_HEADER: "http://a:1", "X-API-Key": "bad"})
        events = s.ui.get("/ui/api/issues").json()["security_events"]
        assert events[0]["kind"] == "auth_failed"
        assert events[0]["status"] == 401
        assert events[0]["caller"] == "http://a:1"
        s.dashboard.detach()

    def test_webhook_refusal_recorded(self, logged_in):
        s, _ = logged_in
        msg = Message.user_text("x").model_dump(mode="json")
        s.agent.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": "1",
                "method": "message/send",
                "params": {
                    "message": msg,
                    "pushNotification": {"url": "http://169.254.169.254/x"},
                },
            },
        )
        kinds = [
            e["kind"] for e in s.ui.get("/ui/api/issues").json()["security_events"]
        ]
        assert "webhook_refused" in kinds

    def test_failed_tasks_listed(self, logged_in):
        s, _ = logged_in
        s.send("boom")
        failed = s.ui.get("/ui/api/issues").json()["failed_tasks"]
        assert failed[0]["error"] == (
            "RuntimeError (details hidden: show_content is off)"
        )

    def test_admin_denied_recorded(self, setup: Setup):
        setup.ui.get("/dlq", headers={"X-Admin-Token": "wrong"})
        kinds = [e["kind"] for e in setup.dashboard.hub.security_events()]
        assert "admin_denied" in kinds

    def test_logs_capture_and_filter(self, logged_in):
        s, _ = logged_in
        task_id = s.send("boom")["result"]["id"]
        logs = s.ui.get(f"/ui/api/logs?q={task_id}").json()["logs"]
        assert logs and all(task_id in line["message"] for line in logs)
        errors = s.ui.get("/ui/api/logs?level=ERROR").json()["logs"]
        assert all(line["level"] in ("ERROR", "CRITICAL") for line in errors)

    def test_detach_removes_log_handler(self):
        s = Setup()
        s.dashboard.detach()
        handlers = logging.getLogger("nexus_a2a").handlers
        assert s.dashboard.log_handler not in handlers


class TestStatic:
    def test_index_served_with_csp(self, setup: Setup):
        resp = setup.ui.get("/ui/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "script-src 'self'" in resp.headers["content-security-policy"]

    def test_assets_served(self, setup: Setup):
        assert setup.ui.get("/ui/static/app.js").status_code == 200
        assert setup.ui.get("/ui/static/app.css").status_code == 200

    @pytest.mark.parametrize("name", ["..%2Fapp.py", ".hidden", "missing.js"])
    def test_no_traversal_or_unknown_files(self, setup: Setup, name: str):
        assert setup.ui.get(f"/ui/static/{name}").status_code == 404


# ── Live stream over a real server ────────────────────────────────────────────


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class TestLiveStream:
    async def test_stream_delivers_task_events(self):
        server = A2AServer(
            DashEcho, host="127.0.0.1", port=_free_port(), task_manager=TaskManager()
        )
        network = AgentNetwork(task_manager=server.tasks)
        dashboard = Dashboard(admin_token=TOKEN, server=server, network=network)
        ops = AgentServer(
            network=network,
            host="127.0.0.1",
            port=_free_port(),
            admin_token=TOKEN,
            dashboard=dashboard,
        )
        await server.start()
        await ops.start()
        try:
            base = f"http://127.0.0.1:{ops.port}"
            async with httpx.AsyncClient(base_url=base, timeout=10) as client:
                login = await client.post("/ui/api/login", json={"token": TOKEN})
                assert login.status_code == 200
                got: list[dict[str, Any]] = []

                async def read() -> None:
                    async with client.stream("GET", "/ui/api/stream") as resp:
                        assert resp.status_code == 200
                        event = None
                        async for line in resp.aiter_lines():
                            if line.startswith("event: "):
                                event = line[7:]
                            elif line.startswith("data: ") and event == "task":
                                got.append(json.loads(line[6:]))
                                if got[-1]["event"] == "completed":
                                    return

                reader = asyncio.create_task(read())
                await asyncio.sleep(0.2)
                msg = Message.user_text("live").model_dump(mode="json")
                async with httpx.AsyncClient() as agent_client:
                    await agent_client.post(
                        f"http://127.0.0.1:{server.port}/",
                        json={
                            "jsonrpc": "2.0",
                            "id": "1",
                            "method": "message/send",
                            "params": {"message": msg},
                        },
                    )
                await asyncio.wait_for(reader, timeout=5)
            assert [e["event"] for e in got] == ["created", "started", "completed"]
            assert got[0]["task"]["preview"] is None  # content hidden by default
        finally:
            await ops.stop()
            await server.stop()


# ── Config and CLI wiring ─────────────────────────────────────────────────────


def _cfg(**sections: Any) -> NexusConfig:
    raw: dict[str, Any] = {"agent": {"name": "a", "url": "http://127.0.0.1:8001"}}
    raw.update(sections)
    return NexusConfig.from_dict(raw)


class TestConfig:
    def test_disabled_by_default(self):
        assert _cfg().dashboard.enabled is False

    def test_needs_ops_port(self):
        with pytest.raises(ConfigError, match="ops"):
            _cfg(dashboard={"enabled": True})

    def test_needs_admin_token(self):
        with pytest.raises(ConfigError, match="admin_token"):
            _cfg(dashboard={"enabled": True}, ops={"port": 9100})

    def test_unknown_dashboard_key_rejected(self):
        with pytest.raises(ConfigError, match="show_contnet"):
            _cfg(dashboard={"show_contnet": True})

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("NEXUS_DASHBOARD", "true")
        monkeypatch.setenv("NEXUS_ADMIN_TOKEN", TOKEN)
        cfg = _cfg(ops={"port": 9100})
        assert cfg.dashboard.enabled is True

    def test_build_runtime_wires_dashboard(self):
        cfg = _cfg(
            ops={"port": 9100, "admin_token": TOKEN},
            dashboard={"enabled": True, "show_content": True, "allow_replay": False},
        )
        runtime = cfg.build_runtime(DashEcho)
        dashboard = runtime.ops.dashboard
        assert dashboard is not None and dashboard.server is runtime.server
        assert dashboard.config.show_content is True
        assert dashboard.config.allow_replay is False
        dashboard.detach()

    def test_dev_config_adds_private_ops(self):
        from nexus_a2a.cli.commands.dev import DevAgent, dev_config_for

        raw = {"dashboard": {"enabled": True}, "ops": {"port": 1}}
        plain = dev_config_for(raw, DevAgent("m:C", 8001), [])
        assert "dashboard" not in plain and "ops" not in plain
        with_ui = dev_config_for(raw, DevAgent("m:C", 8001), [], 9200, TOKEN)
        assert with_ui["ops"] == {
            "port": 9200,
            "host": "127.0.0.1",
            "admin_token": TOKEN,
        }
        assert with_ui["dashboard"]["enabled"] is True
        assert with_ui["dashboard"]["show_content"] is True

    def test_dashboard_command_prints_url(self):
        from click.testing import CliRunner

        from nexus_a2a.cli.main import cli

        result = CliRunner().invoke(
            cli, ["dashboard", "--agent", "http://127.0.0.1:9100", "--no-browser"]
        )
        assert result.exit_code == 0
        assert "http://127.0.0.1:9100/ui/" in result.output


# ── Regression tests from the 2.1 pre-release security review ───────────────

from nexus_a2a.dashboard.app import hide_error  # noqa: E402


class TestReviewStaticFiles:
    @pytest.mark.parametrize(
        "name", ["D:nexus.toml", "C:x", "c:..", "app.js.", "APP.JS", "%2e%2e"]
    )
    def test_only_shipped_file_names(self, setup: Setup, name: str):
        """Finding 1: a Windows drive-relative name escaped static/."""
        assert setup.ui.get(f"/ui/static/{name}").status_code == 404


class TestReviewHostAndProxy:
    def test_rebinding_host_refused(self, setup: Setup):
        """Finding 2: a DNS-rebinding page is a loopback peer with a foreign Host."""
        for path in ("/ui/", "/ui/api/session", "/ui/static/app.js"):
            resp = setup.ui.get(path, headers={"Host": "evil.example:9100"})
            assert resp.status_code == 403

    @pytest.mark.parametrize("host", ["localhost:9100", "127.0.0.1", "[::1]:9100"])
    def test_loopback_hosts_accepted(self, setup: Setup, host: str):
        assert (
            setup.ui.get("/ui/api/session", headers={"Host": host}).status_code == 200
        )

    def test_allowed_hosts_with_remote(self):
        s = Setup(
            config=DashboardConfig(
                allow_remote=True, allowed_hosts=("ops.example.com",)
            )
        )
        remote = ui_client(
            s.ops._app, client=("10.1.2.3", 5000), base_url="https://ops.example.com"
        )
        assert remote.get("/ui/api/session").status_code == 200
        bad = remote.get("/ui/api/session", headers={"Host": "evil.example"})
        assert bad.status_code == 403
        s.dashboard.detach()

    def test_testclient_peer_name_is_not_loopback(self, setup: Setup):
        """Finding 3: the 'testclient' special case is gone."""
        client = ui_client(setup.ops._app, client=("testclient", 50000))
        assert client.get("/ui/api/session").status_code == 403

    def test_ops_server_ignores_forwarded_headers(self):
        """Finding 3: X-Forwarded-For must never rewrite the client address."""
        import uvicorn

        captured: dict[str, Any] = {}
        real_config = uvicorn.Config

        def spy(*args: Any, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return real_config(*args, **kwargs)

        s = Setup()
        import nexus_a2a.core.agent_server as agent_server_module

        original = agent_server_module.uvicorn.Config
        agent_server_module.uvicorn.Config = spy
        try:
            asyncio.run(_start_stop(s.ops))
        finally:
            agent_server_module.uvicorn.Config = original
            s.dashboard.detach()
        assert captured["proxy_headers"] is False


async def _start_stop(ops: AgentServer) -> None:
    ops.port = _free_port()
    ops.host = "127.0.0.1"
    await ops.start()
    await ops.stop()


class TestReviewThrottle:
    def test_header_token_guesses_are_throttled_and_recorded(self, setup: Setup):
        """Finding 4."""
        codes = [
            setup.ui.get(
                "/ui/api/overview", headers={"X-Admin-Token": f"guess-{i}"}
            ).status_code
            for i in range(7)
        ]
        assert codes[:5] == [401] * 5 and codes[-1] == 429
        kinds = [e["kind"] for e in setup.dashboard.hub.security_events()]
        assert kinds.count("auth_failed") == 5
        # Even the right token waits out the lockout.
        ok = setup.ui.get("/ui/api/overview", headers={"X-Admin-Token": TOKEN})
        assert ok.status_code == 429

    def test_successful_logins_never_lock_out(self, setup: Setup):
        """Finding 7: only failures count."""
        for _ in range(10):
            assert (
                setup.ui.post("/ui/api/login", json={"token": TOKEN}).status_code == 200
            )

    def test_cross_origin_login_refused(self, setup: Setup):
        resp = setup.ui.post(
            "/ui/api/login",
            json={"token": TOKEN},
            headers={"Origin": "http://evil.example"},
        )
        assert resp.status_code == 403


class TestReviewRedaction:
    def test_hide_error(self):
        assert hide_error("RuntimeError: patient SSN 123") == (
            "RuntimeError (details hidden: show_content is off)"
        )
        assert hide_error("free text with secrets") == (
            "(details hidden: show_content is off)"
        )
        assert hide_error(None) is None
        timeout = "Task auto-failed by watchdog: exceeded timeout of 5s"
        assert hide_error(timeout) == timeout

    def test_errors_and_logs_hidden_everywhere(self, logged_in):
        """Finding 5: task errors and log lines quoted agent input."""
        s, _ = logged_in
        task_id = s.send("boom")["result"]["id"]  # raises "agent exploded"
        blobs = [
            s.ui.get("/ui/api/tasks").text,
            s.ui.get(f"/ui/api/tasks/{task_id}").text,
            s.ui.get("/ui/api/issues").text,
            s.ui.get("/ui/api/logs?limit=1000").text,
        ]
        for blob in blobs:
            assert "agent exploded" not in blob
        logs = s.ui.get("/ui/api/logs?limit=1000").json()["logs"]
        assert any(line["message"].startswith("Task failed: id=") for line in logs)
        assert any("[RuntimeError]" in line["message"] for line in logs)

    def test_errors_shown_with_show_content(self):
        s = Setup(config=DashboardConfig(show_content=True))
        s.login()
        s.send("boom")
        failed = s.ui.get("/ui/api/issues").json()["failed_tasks"]
        assert failed[0]["error"] == "RuntimeError: agent exploded"
        logs = s.ui.get("/ui/api/logs?limit=1000").text
        assert "agent exploded" in logs
        s.dashboard.detach()


class TestReviewDlqReplay:
    async def _dlq_setup(self) -> tuple[Setup, str, list[str]]:
        s = Setup()
        csrf = s.login()
        calls: list[str] = []

        async def runner(url: str, msg: Message) -> Task:
            calls.append(url)
            task = await s.server.tasks.create(msg)
            task = await s.server.tasks.start(task.id)
            return await s.server.tasks.complete(task.id)

        dlq = s.network.dead_letter_queue
        dlq.set_runner(runner)
        failed = await s.server.tasks.create(Message.user_text("x"))
        failed = await s.server.tasks.start(failed.id)
        failed = await s.server.tasks.fail(failed.id, "boom")
        await dlq.capture(failed, agent_url="http://peer:1")
        return s, csrf, calls

    async def test_duplicate_ids_replayed_once_and_replayed_skipped(self):
        """Finding 8."""
        s, csrf, calls = await self._dlq_setup()
        entry_id = s.network.dead_letter_queue.all_entries()[0].task_id
        resp = s.post("/ui/api/dlq/replay", csrf, {"task_ids": [entry_id] * 20})
        results = resp.json()["results"]
        assert len(results) == 1 and results[0]["success"] is True
        assert len(calls) == 1
        again = s.post("/ui/api/dlq/replay", csrf, {"task_ids": [entry_id]})
        assert again.json()["results"][0]["error"] == "already replayed"
        assert len(calls) == 1
        s.dashboard.detach()


class TestReviewCookie:
    def test_cookie_secure_when_remote_allowed(self):
        s = Setup(config=DashboardConfig(allow_remote=True))
        resp = s.ui.post("/ui/api/login", json={"token": TOKEN})
        assert "secure" in resp.headers["set-cookie"].lower()
        s.dashboard.detach()

    def test_cookie_not_secure_on_local_http(self, setup: Setup):
        resp = setup.ui.post("/ui/api/login", json={"token": TOKEN})
        assert "secure" not in resp.headers["set-cookie"].lower()


class TestReviewStreamEndsOnLogout:
    async def test_stream_closes_after_logout(self):
        """Finding 6: a stream outlived its session."""
        server = A2AServer(
            DashEcho, host="127.0.0.1", port=_free_port(), task_manager=TaskManager()
        )
        network = AgentNetwork(task_manager=server.tasks)
        dashboard = Dashboard(admin_token=TOKEN, server=server, network=network)
        ops = AgentServer(
            network=network,
            host="127.0.0.1",
            port=_free_port(),
            admin_token=TOKEN,
            dashboard=dashboard,
        )
        await server.start()
        await ops.start()
        try:
            base = f"http://127.0.0.1:{ops.port}"
            async with httpx.AsyncClient(base_url=base, timeout=10) as client:
                csrf = (
                    await client.post("/ui/api/login", json={"token": TOKEN})
                ).json()["csrf"]
                events: list[str] = []
                ended = asyncio.Event()

                async def read() -> None:
                    async with client.stream("GET", "/ui/api/stream") as resp:
                        async for line in resp.aiter_lines():
                            if line.startswith("event: "):
                                events.append(line[7:])
                    ended.set()

                reader = asyncio.create_task(read())
                await asyncio.sleep(0.2)
                logout = await client.post(
                    "/ui/api/logout", headers={"X-CSRF-Token": csrf, "Origin": base}
                )
                assert logout.status_code == 204
                msg = Message.user_text("after logout").model_dump(mode="json")
                async with httpx.AsyncClient() as agent_client:
                    await agent_client.post(
                        f"http://127.0.0.1:{server.port}/",
                        json={
                            "jsonrpc": "2.0",
                            "id": "1",
                            "method": "message/send",
                            "params": {"message": msg},
                        },
                    )
                await asyncio.wait_for(ended.wait(), timeout=5)
                await reader
            assert "task" not in events
        finally:
            await ops.stop()
            await server.stop()
