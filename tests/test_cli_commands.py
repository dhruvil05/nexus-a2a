"""
tests/test_cli_commands.py

Behaviour of each CLI command. Before 1.9.0 the commands were 17-44% covered,
which is how several of them shipped broken against real servers:

  - `nexus status` read metric names only the test mock emitted, so DLQ
    pending always showed 0;
  - `nexus trace --agent` and `nexus replay` never sent the admin token the
    endpoints have required since 1.5.0, so both always got 403;
  - `nexus trace <task_id>` looked the task id up as a trace id;
  - `nexus inspect` read `authentication.schemes`, so it always said "none";
  - any command printing a status icon crashed on a cp1252 console.
"""

from __future__ import annotations

import io
import json
import socket
from typing import Any

import httpx
import pytest
import respx
from click.testing import CliRunner
from starlette.applications import Starlette
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route

from nexus_a2a import (
    A2AServer,
    AgentCredentialConfig,
    AuthManager,
    AuthScheme,
    SecurityMiddleware,
    agent,
)
from nexus_a2a.cli import admin, output
from nexus_a2a.cli.commands import dev as dev_cmd
from nexus_a2a.cli.commands import replay as replay_cmd
from nexus_a2a.cli.commands import run as run_cmd
from nexus_a2a.cli.commands import status as status_cmd
from nexus_a2a.cli.main import cli
from nexus_a2a.verify import Status, verify_agent

AGENT = "http://agent.test:8001"
OPS = "http://ops.test:8080"
CARD = {
    "name": "Tested",
    "description": "An agent.",
    "version": "2.1.0",
    "url": AGENT,
    "skills": [{"id": "s", "name": "S", "description": "s"}],
    "authentication": {"scheme": "api_key", "header_name": "X-API-Key"},
    "capabilities": {"streaming": False, "push_notifications": False},
}


@agent(name="Echo", description="Echoes.", streaming=True, push_notifications=True,
       skills=[{"id": "echo", "name": "Echo", "description": "Echo."}],
       url="http://verify.test:8001")
class EchoAgent:
    async def run(self, task):
        yield "echo"


def invoke(*args: str, env: dict[str, str] | None = None) -> Any:
    return CliRunner().invoke(cli, list(args), env=env or {}, catch_exceptions=False)


@pytest.fixture(autouse=True)
def _no_ambient_config(tmp_path, monkeypatch):
    """Commands must not pick up a nexus.toml or tokens from the dev machine."""
    monkeypatch.chdir(tmp_path)
    for var in ("NEXUS_ADMIN_TOKEN", "NEXUS_OPS_URL", "NEXUS_OPS_PORT"):
        monkeypatch.delenv(var, raising=False)


def write_config(tmp_path, text: str) -> str:
    path = tmp_path / "nexus.toml"
    path.write_text(text, encoding="utf-8")
    return str(path)


# ── admin helpers ─────────────────────────────────────────────────────────────


class TestAdminResolution:
    def test_token_precedence(self, monkeypatch):
        cfg = {"ops": {"admin_token": "from-file"}}
        assert admin.resolve_admin_token(None, cfg) == "from-file"
        monkeypatch.setenv("NEXUS_ADMIN_TOKEN", "from-env")
        assert admin.resolve_admin_token(None, cfg) == "from-env"
        assert admin.resolve_admin_token("from-flag", cfg) == "from-flag"

    def test_no_token_is_none(self):
        assert admin.resolve_admin_token(None, {}) is None

    def test_url_precedence(self, monkeypatch):
        cfg = {"ops": {"url": "http://file-ops"}, "agent": {"url": "http://agent"}}
        assert admin.resolve_ops_url(None, {"agent": {"url": "http://agent"}}) == "http://agent"
        assert admin.resolve_ops_url(None, cfg) == "http://file-ops"
        monkeypatch.setenv("NEXUS_OPS_URL", "http://env-ops")
        assert admin.resolve_ops_url(None, cfg) == "http://env-ops"
        assert admin.resolve_ops_url("http://flag", cfg) == "http://flag"

    def test_headers(self):
        assert admin.admin_headers(None) == {}
        assert admin.admin_headers("t") == {"X-Admin-Token": "t"}

    def test_forbidden_hint_differs_by_cause(self):
        assert "--admin-token" in admin.forbidden_hint(OPS, None)
        assert "rejected the admin token" in admin.forbidden_hint(OPS, "bad")


# ── status ────────────────────────────────────────────────────────────────────


class TestPrometheusParsing:
    def test_parses_plain_gauges(self):
        text = "# HELP x y\n# TYPE x gauge\nnexus_a2a_tasks_active 3\nfoo 1.5\n"
        assert status_cmd.parse_prometheus(text) == {
            "nexus_a2a_tasks_active": 3.0, "foo": 1.5,
        }

    def test_skips_labelled_and_junk_lines(self):
        text = 'a{x="1"} 2\nbroken\nb notanumber\n\nc 4'
        assert status_cmd.parse_prometheus(text) == {"c": 4.0}


def metrics_body(active: int | None = None, dlq: int | None = None) -> str:
    lines = []
    if active is not None:
        lines.append(f"nexus_a2a_tasks_active {active}")
    if dlq is not None:
        lines.append(f"nexus_a2a_dlq_pending {dlq}")
    return "\n".join(lines) + "\n"


def mock_agent(router, url=AGENT, card=CARD, metrics: str | None = "", health=200):
    router.get(f"{url}/.well-known/agent-card.json").respond(200, json=card)
    router.get(f"{url}/health").respond(health, json={"status": "ok"})
    if metrics is None:
        router.get(f"{url}/metrics").respond(404)
    else:
        router.get(f"{url}/metrics").respond(200, text=metrics)


class TestStatus:
    def test_reads_real_metric_names(self):
        """Regression: the probe used names only the test mock emitted."""
        with respx.mock(assert_all_called=False) as router:
            mock_agent(router, metrics=metrics_body(active=4, dlq=2))
            result = invoke("--format", "json", "status", "--agents", AGENT)
        data = json.loads(result.output)
        assert data["agents"][0]["queue_depth"] == 4
        assert data["agents"][0]["dlq_pending"] == 2
        assert data["summary"]["total_dlq"] == 2
        assert result.exit_code == 0

    def test_unreported_values_are_null_not_zero(self):
        with respx.mock(assert_all_called=False) as router:
            mock_agent(router, metrics=None)
            result = invoke("--format", "json", "status", "--agents", AGENT)
        agent_row = json.loads(result.output)["agents"][0]
        assert agent_row["queue_depth"] is None
        assert agent_row["dlq_pending"] is None

    def test_table_shows_dash_for_unreported(self):
        with respx.mock(assert_all_called=False) as router:
            mock_agent(router, metrics=None)
            result = invoke("status", "--agents", AGENT)
        assert "—" in result.output or "-" in result.output
        assert "None" not in result.output

    def test_unhealthy_agent_exits_non_zero(self):
        with respx.mock(assert_all_called=False) as router:
            mock_agent(router, health=503)
            result = invoke("status", "--agents", AGENT)
        assert result.exit_code == 1

    def test_agents_from_config(self, tmp_path):
        path = write_config(tmp_path, f'[network]\nagents = ["{AGENT}"]\n')
        with respx.mock(assert_all_called=False) as router:
            mock_agent(router)
            result = invoke("--config", path, "--format", "json", "status")
        assert json.loads(result.output)["agents"][0]["name"] == "Tested"

    def test_no_agents_is_a_clean_exit(self):
        result = invoke("status")
        assert result.exit_code == 0
        assert "No agent URLs" in result.output


# ── trace ─────────────────────────────────────────────────────────────────────


TRACE = {"trace_id": "tr-1", "hops": [
    {"url": AGENT, "duration_ms": 12.5, "status": "completed", "error": None,
     "children": []},
]}


class TestTrace:
    def test_sends_the_admin_token(self):
        """Regression: /traces required a token the command never sent."""
        with respx.mock() as router:
            route = router.get(f"{OPS}/traces/tr-1").respond(200, json=TRACE)
            result = invoke("--format", "json", "trace", "tr-1",
                            "--agent", OPS, "--admin-token", "secret")
        assert route.calls.last.request.headers["X-Admin-Token"] == "secret"
        assert json.loads(result.output)["trace_id"] == "tr-1"

    def test_token_from_environment(self):
        with respx.mock() as router:
            route = router.get(f"{OPS}/traces/tr-1").respond(200, json=TRACE)
            invoke("trace", "tr-1", "--agent", OPS,
                   env={"NEXUS_ADMIN_TOKEN": "env-secret"})
        assert route.calls.last.request.headers["X-Admin-Token"] == "env-secret"

    def test_ops_url_and_token_from_config(self, tmp_path):
        path = write_config(
            tmp_path, f'[ops]\nurl = "{OPS}"\nadmin_token = "file-secret"\n'
        )
        with respx.mock() as router:
            route = router.get(f"{OPS}/traces/tr-1").respond(200, json=TRACE)
            invoke("--config", path, "trace", "tr-1")
        assert route.calls.last.request.headers["X-Admin-Token"] == "file-secret"

    def test_forbidden_gives_actionable_hint(self):
        with respx.mock() as router:
            router.get(f"{OPS}/traces/tr-1").respond(403, json={"error": "Forbidden"})
            result = CliRunner().invoke(cli, ["trace", "tr-1", "--agent", OPS])
        assert result.exit_code == 1
        assert "--admin-token" in result.output

    def test_not_found_exits_non_zero(self):
        with respx.mock() as router:
            router.get(f"{OPS}/traces/nope").respond(404)
            result = invoke("trace", "nope", "--agent", OPS)
        assert result.exit_code == 1
        assert "No trace found" in result.output

    async def test_local_store_resolves_a_task_id(self, monkeypatch):
        from nexus_a2a.cli.commands import trace as trace_cmd
        from nexus_a2a.transport.tracing import Span, TraceStore

        store = TraceStore()
        span = Span(trace_id="tr-local", agent_url=AGENT)
        span.metadata["task_id"] = "task-local"
        await store.record(span)
        monkeypatch.setattr("nexus_a2a.transport.tracing.default_store", store)

        found = trace_cmd._try_local_trace_store("task-local")
        assert found is not None and found["trace_id"] == "tr-local"


# ── replay ────────────────────────────────────────────────────────────────────


ENTRIES = {"entries": [
    {"task_id": "t-1", "error": "boom", "failed_at": 1_700_000_000.0,
     "agent_url": AGENT, "skill_id": "s", "retry_count": 0, "replayed": False},
]}


class TestReplay:
    @pytest.mark.parametrize(
        ("text", "seconds"),
        [("1h", 3600), ("30m", 1800), ("2h30m", 9000), ("7d", 604800), ("45s", 45)],
    )
    def test_parse_duration(self, text, seconds):
        assert replay_cmd._parse_duration(text).total_seconds() == seconds

    @pytest.mark.parametrize("text", ["", "soon", "1x", "h"])
    def test_parse_duration_rejects_junk(self, text):
        import click

        with pytest.raises(click.BadParameter):
            replay_cmd._parse_duration(text)

    def test_requires_failed_flag(self):
        result = invoke("replay")
        assert result.exit_code == 0
        assert "--failed" in result.output

    def test_needs_a_target(self):
        result = CliRunner().invoke(cli, ["replay", "--failed"])
        assert result.exit_code == 1

    def test_sends_token_on_list_and_replay(self):
        """Regression: /dlq and /dlq/replay required a token never sent."""
        with respx.mock() as router:
            listing = router.get(f"{OPS}/dlq").respond(200, json=ENTRIES)
            replaying = router.post(f"{OPS}/dlq/replay").respond(
                200, json={"succeeded": 1, "failed": 0, "results": []}
            )
            result = invoke("replay", "--failed", "--yes", "--agent", OPS,
                            "--admin-token", "secret")
        assert listing.calls.last.request.headers["X-Admin-Token"] == "secret"
        assert replaying.calls.last.request.headers["X-Admin-Token"] == "secret"
        assert json.loads(replaying.calls.last.request.content) == {"task_id": "t-1"}
        assert result.exit_code == 0

    def test_forbidden_gives_actionable_hint(self):
        with respx.mock() as router:
            router.get(f"{OPS}/dlq").respond(403)
            result = CliRunner().invoke(cli, ["replay", "--failed", "--agent", OPS])
        assert result.exit_code == 1
        assert "--admin-token" in result.output

    def test_dry_run_does_not_replay(self):
        with respx.mock(assert_all_called=False) as router:
            router.get(f"{OPS}/dlq").respond(200, json=ENTRIES)
            replaying = router.post(f"{OPS}/dlq/replay")
            result = invoke("replay", "--failed", "--dry-run", "--agent", OPS)
        assert not replaying.called
        assert "Dry run" in result.output
        assert result.exit_code == 0

    def test_empty_queue_is_a_clean_exit(self):
        with respx.mock() as router:
            router.get(f"{OPS}/dlq").respond(200, json={"entries": []})
            result = invoke("replay", "--failed", "--agent", OPS)
        assert result.exit_code == 0
        assert "No matching" in result.output

    def test_last_filter_drops_old_entries(self):
        with respx.mock() as router:
            router.get(f"{OPS}/dlq").respond(200, json=ENTRIES)  # from 2023
            result = invoke("replay", "--failed", "--last", "1h", "--agent", OPS)
        assert "No matching" in result.output

    def test_failed_replay_exits_non_zero(self):
        with respx.mock() as router:
            router.get(f"{OPS}/dlq").respond(200, json=ENTRIES)
            router.post(f"{OPS}/dlq/replay").respond(
                200, json={"succeeded": 0, "failed": 1,
                           "results": [{"error": "still broken"}]}
            )
            result = invoke("--verbose", "replay", "--failed", "--yes", "--agent", OPS)
        assert result.exit_code == 1

    def test_skill_filter_is_forwarded(self):
        with respx.mock() as router:
            route = router.get(f"{OPS}/dlq").respond(200, json={"entries": []})
            invoke("replay", "--failed", "--skill", "web", "--agent", OPS)
        assert route.calls.last.request.url.params["skill"] == "web"


# ── ping / inspect ────────────────────────────────────────────────────────────


class TestPing:
    def test_healthy(self):
        with respx.mock(assert_all_called=False) as router:
            mock_agent(router)
            result = invoke("--format", "json", "ping", AGENT)
        data = json.loads(result.output)
        assert data["healthy"] and data["name"] == "Tested"
        assert data["skills_count"] == 1
        assert result.exit_code == 0

    def test_card_failure(self):
        with respx.mock() as router:
            router.get(f"{AGENT}/.well-known/agent-card.json").respond(500)
            result = invoke("--format", "json", "ping", AGENT)
        assert "AgentCard HTTP 500" in result.output
        assert result.exit_code == 1

    def test_unhealthy(self):
        with respx.mock(assert_all_called=False) as router:
            mock_agent(router, health=503)
            result = invoke("ping", AGENT)
        assert result.exit_code == 1


class TestInspect:
    @pytest.mark.parametrize(
        ("auth", "expected"),
        [
            ({"scheme": "api_key"}, "api_key"),
            ({"schemes": ["bearer", "oauth2"]}, "bearer, oauth2"),
            ({}, "none"),
            ({"scheme": ""}, "none"),
        ],
    )
    def test_auth_scheme_of(self, auth, expected):
        assert output.auth_scheme_of(auth) == expected

    def test_shows_the_real_scheme(self):
        """Regression: read `schemes`, so every nexus-a2a agent showed none."""
        with respx.mock() as router:
            router.get(f"{AGENT}/.well-known/agent-card.json").respond(200, json=CARD)
            result = invoke("inspect", AGENT)
        assert "api_key" in result.output
        assert result.exit_code == 0

    def test_json_output(self):
        with respx.mock() as router:
            router.get(f"{AGENT}/.well-known/agent-card.json").respond(200, json=CARD)
            result = invoke("--format", "json", "inspect", AGENT)
        assert json.loads(result.output)["name"] == "Tested"

    def test_http_error(self):
        with respx.mock() as router:
            router.get(f"{AGENT}/.well-known/agent-card.json").respond(404)
            result = CliRunner().invoke(cli, ["inspect", AGENT])
        assert result.exit_code == 1


# ── run ───────────────────────────────────────────────────────────────────────


class TestRunHelpers:
    def test_card_fills_missing_agent_fields(self):
        cfg = run_cmd.build_config({}, EchoAgent)
        assert cfg.agent.name == "Echo"
        assert cfg.agent.url == "http://verify.test:8001"

    def test_file_values_win(self):
        cfg = run_cmd.build_config(
            {"agent": {"name": "Renamed", "url": "http://other:1"}}, EchoAgent
        )
        assert (cfg.agent.name, cfg.agent.url) == ("Renamed", "http://other:1")

    def test_does_not_mutate_the_raw_config(self):
        raw = {"agent": {}}
        run_cmd.build_config(raw, EchoAgent)
        assert raw == {"agent": {}}

    def test_describe_unsecured(self):
        cfg = run_cmd.build_config({}, EchoAgent)
        text = "\n".join(run_cmd.describe_runtime(cfg, cfg.build_runtime(EchoAgent)))
        assert "Serving agent 'Echo'" in text
        assert "none" in text and "Ops" not in text

    def test_describe_secured_with_ops(self):
        cfg = run_cmd.build_config(
            {"security": {"auth_scheme": "api_key", "auth_secret": "k"},
             "ops": {"port": 9555, "admin_token": "t"}},
            EchoAgent,
        )
        text = "\n".join(run_cmd.describe_runtime(cfg, cfg.build_runtime(EchoAgent)))
        assert "auth" in text
        assert "9555" in text and "admin enabled" in text

    def test_extract_host_port(self):
        assert run_cmd._extract_host_port("http://h:81") == ("h", 81)
        assert run_cmd._extract_host_port("http://h") == ("h", 8000)


class TestRunCommand:
    def test_unimportable_module(self):
        result = CliRunner().invoke(cli, ["run", "--module", "no.such.module:X"])
        assert result.exit_code == 1
        assert "Cannot import" in result.output

    def test_invalid_config_is_reported(self, tmp_path, monkeypatch):
        module = tmp_path / "bad_cfg_agent.py"
        module.write_text(
            "from nexus_a2a import agent\n"
            "@agent(name='B', description='b', url='http://127.0.0.1:1')\n"
            "class B:\n    async def run(self, task):\n        return 'x'\n",
            encoding="utf-8",
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        path = write_config(tmp_path, '[security]\nrate_limit = -5\n')
        result = CliRunner().invoke(
            cli, ["--config", path, "run", "--module", "bad_cfg_agent:B"]
        )
        assert result.exit_code == 1
        assert "rate_limit" in result.output

    def test_undecorated_class_is_reported(self, tmp_path, monkeypatch):
        module = tmp_path / "plain_mod.py"
        module.write_text("class P:\n    pass\n", encoding="utf-8")
        monkeypatch.syspath_prepend(str(tmp_path))
        result = CliRunner().invoke(cli, ["run", "--module", "plain_mod:P"])
        assert result.exit_code == 1


# ── verify ────────────────────────────────────────────────────────────────────


def asgi_client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app))


def by_name(report) -> dict[str, Any]:
    return {f"{c.group}.{c.name}": c for c in report.checks}


def fake_agent(card: dict, rpc_handler=None, health: bool = True) -> Starlette:
    async def card_ep(request):
        return JSONResponse(card)

    async def rpc(request):
        if rpc_handler is not None:
            return await rpc_handler(request)
        return JSONResponse({"jsonrpc": "2.0", "id": 1, "result": {"ok": True}})

    async def health_ep(request):
        return PlainTextResponse("ok")

    routes = [Route("/.well-known/agent-card.json", card_ep),
              Route("/", rpc, methods=["POST"])]
    if health:
        routes.append(Route("/health", health_ep))
    return Starlette(routes=routes)


class TestVerifyAgainstRealServer:
    URL = "http://verify.test:8001"

    async def test_conforming_agent_passes(self):
        server = A2AServer(EchoAgent)
        async with asgi_client(server.app) as client:
            report = await verify_agent(self.URL, client=client)
        failures = [c for c in report.checks if c.status == Status.FAIL]
        assert failures == [], failures
        assert report.passed

    async def test_ssrf_guard_is_checked(self):
        server = A2AServer(EchoAgent)
        async with asgi_client(server.app) as client:
            report = await verify_agent(self.URL, client=client)
        assert by_name(report)["push.ssrf_guard"].status == Status.PASS

    async def test_dev_mode_push_is_flagged(self):
        from nexus_a2a.transport.webhook import WebhookConfig

        server = A2AServer(EchoAgent, push_config=WebhookConfig(allow_private_urls=True))
        async with asgi_client(server.app) as client:
            report = await verify_agent(self.URL, client=client)
        assert by_name(report)["push.ssrf_guard"].status == Status.FAIL

    async def test_read_only_never_invokes_the_agent(self):
        calls = []

        @agent(name="Counted", description="c", url=self.URL)
        class Counted:
            async def run(self, task):
                calls.append(task.id)
                return "x"

        async with asgi_client(A2AServer(Counted).app) as client:
            report = await verify_agent(self.URL, client=client, read_only=True)
        assert calls == []
        assert by_name(report)["task.send"].status == Status.SKIP

    async def test_secured_agent_without_credentials_is_skipped_not_failed(self):
        security = SecurityMiddleware(auth=AuthManager(default=AgentCredentialConfig(
            scheme=AuthScheme.API_KEY, api_key="k")))
        server = A2AServer(EchoAgent, security=security)
        async with asgi_client(server.app) as client:
            report = await verify_agent(self.URL, client=client)
        checks = by_name(report)
        assert report.passed
        assert checks["auth.matches_card"].status == Status.PASS
        assert checks["task.send"].status == Status.SKIP

    async def test_secured_agent_with_credentials_is_fully_checked(self):
        security = SecurityMiddleware(auth=AuthManager(default=AgentCredentialConfig(
            scheme=AuthScheme.API_KEY, api_key="k")))
        server = A2AServer(EchoAgent, security=security)
        async with asgi_client(server.app) as client:
            report = await verify_agent(self.URL, client=client, api_key="k")
        checks = by_name(report)
        assert checks["auth.credentials"].status == Status.PASS
        assert checks["task.send"].status == Status.PASS
        assert report.passed

    async def test_wrong_credentials_fail_and_stop(self):
        security = SecurityMiddleware(auth=AuthManager(default=AgentCredentialConfig(
            scheme=AuthScheme.API_KEY, api_key="k")))
        server = A2AServer(EchoAgent, security=security)
        async with asgi_client(server.app) as client:
            report = await verify_agent(self.URL, client=client, api_key="wrong")
        checks = by_name(report)
        assert checks["auth.credentials"].status == Status.FAIL
        assert checks["task.send"].status == Status.SKIP


class TestVerifyCatchesBrokenAgents:
    URL = "http://broken.test:1"

    def card(self, **overrides):
        base = {"name": "B", "description": "b", "url": self.URL,
                "skills": [{"id": "x", "name": "X", "description": "x"}]}
        base.update(overrides)
        return base

    async def run(self, app, **kwargs):
        async with asgi_client(app) as client:
            return await verify_agent(self.URL, client=client, **kwargs)

    async def test_card_advertising_auth_that_is_not_enforced(self):
        report = await self.run(fake_agent(self.card(authentication={"scheme": "api_key"})))
        check = by_name(report)["auth.matches_card"]
        assert check.status == Status.FAIL
        assert "open to anyone" in check.detail

    async def test_card_hiding_auth_that_is_enforced(self):
        async def refuse(request):
            return JSONResponse({"error": "no"}, status_code=401)

        report = await self.run(fake_agent(self.card(), rpc_handler=refuse))
        check = by_name(report)["auth.matches_card"]
        assert check.status == Status.FAIL
        assert "trust the card" in check.detail

    async def test_no_error_handling(self):
        report = await self.run(fake_agent(self.card()))
        checks = by_name(report)
        for name in ("protocol.method_not_found", "protocol.parse_error",
                     "protocol.invalid_params", "protocol.unknown_task"):
            assert checks[name].status == Status.FAIL, name

    async def test_false_streaming_claim(self):
        report = await self.run(fake_agent(self.card(capabilities={"streaming": True})))
        assert by_name(report)["stream.message_stream"].status == Status.FAIL

    async def test_false_push_claim(self):
        async def handler(request):
            try:
                body = await request.json()
            except ValueError:
                return JSONResponse({"jsonrpc": "2.0", "id": None,
                                     "error": {"code": -32700, "message": "parse"}})
            if body["method"].startswith("tasks/pushNotificationConfig"):
                return JSONResponse({"jsonrpc": "2.0", "id": 1,
                                     "error": {"code": -32601, "message": "nope"}})
            if body["method"] == "message/send":
                return JSONResponse({"jsonrpc": "2.0", "id": 1, "result": {
                    "id": "t", "context_id": "c", "state": "completed",
                    "history": [], "artifacts": []}})
            return JSONResponse({"jsonrpc": "2.0", "id": 1,
                                 "error": {"code": -32001, "message": "x"}})

        report = await self.run(
            fake_agent(self.card(capabilities={"push_notifications": True}),
                       rpc_handler=handler)
        )
        assert by_name(report)["push.config_get"].status == Status.FAIL

    async def test_duplicate_skills(self):
        card = self.card(skills=[{"id": "a", "name": "A", "description": "a"},
                                 {"id": "a", "name": "B", "description": "b"}])
        report = await self.run(fake_agent(card))
        assert by_name(report)["card.skills"].status == Status.FAIL

    async def test_missing_skills_warn(self):
        report = await self.run(fake_agent(self.card(skills=[])))
        assert by_name(report)["card.skills"].status == Status.WARN

    async def test_mismatched_card_url_warns(self):
        report = await self.run(fake_agent(self.card(url="http://elsewhere:9")))
        assert by_name(report)["card.url"].status == Status.WARN

    async def test_relative_card_url_fails(self):
        report = await self.run(fake_agent(self.card(url="/relative")))
        assert by_name(report)["card.url"].status == Status.FAIL

    async def test_invalid_card_schema(self):
        report = await self.run(fake_agent({"url": self.URL}))
        assert by_name(report)["card.schema"].status == Status.FAIL

    async def test_missing_health_warns(self):
        report = await self.run(fake_agent(self.card(), health=False))
        assert by_name(report)["card.health"].status == Status.WARN

    async def test_unreachable_card_stops_early(self):
        report = await self.run(Starlette(routes=[]))
        assert [c.name for c in report.checks] == ["reachable"]
        assert not report.passed

    async def test_non_json_card(self):
        async def bad(request):
            return PlainTextResponse("<html>")

        app = Starlette(routes=[Route("/.well-known/agent-card.json", bad)])
        report = await self.run(app)
        assert by_name(report)["card.json"].status == Status.FAIL

    async def test_report_serialises(self):
        report = await self.run(fake_agent(self.card()))
        data = report.to_dict()
        assert set(data["summary"]) == {"pass", "warn", "fail", "skip"}
        json.dumps(data)


class TestVerifyReportVerdicts:
    def test_strict_fails_on_warn(self):
        from nexus_a2a.verify import CheckResult, VerifyReport

        report = VerifyReport(url="x", checks=[CheckResult("g", "n", Status.WARN)])
        assert report.passed
        assert not report.passed_strict()


class TestVerifyCommand:
    def fake_report(self, statuses):
        from nexus_a2a.verify import CheckResult, VerifyReport

        return VerifyReport(
            url=AGENT,
            checks=[CheckResult("g", f"c{i}", s, "detail") for i, s in enumerate(statuses)],
        )

    def patch(self, monkeypatch, report, seen=None):
        async def fake(url, **kwargs):
            if seen is not None:
                seen.update(kwargs, url=url)
            return report

        monkeypatch.setattr("nexus_a2a.verify.verify_agent", fake)

    def test_passing_exits_zero(self, monkeypatch):
        self.patch(monkeypatch, self.fake_report([Status.PASS, Status.SKIP]))
        result = invoke("verify", AGENT)
        assert result.exit_code == 0
        assert "PASSED" in result.output

    def test_failure_exits_one(self, monkeypatch):
        self.patch(monkeypatch, self.fake_report([Status.FAIL]))
        result = CliRunner().invoke(cli, ["verify", AGENT])
        assert result.exit_code == 1
        assert "FAILED" in result.output

    def test_warning_passes_unless_strict(self, monkeypatch):
        self.patch(monkeypatch, self.fake_report([Status.WARN]))
        assert invoke("verify", AGENT).exit_code == 0
        assert CliRunner().invoke(cli, ["verify", AGENT, "--strict"]).exit_code == 1

    def test_options_are_forwarded(self, monkeypatch):
        seen: dict = {}
        self.patch(monkeypatch, self.fake_report([Status.PASS]), seen)
        invoke("verify", AGENT, "--api-key", "k", "--api-key-header", "X-K",
               "--bearer", "b", "--caller-url", "http://me:1", "--skill", "s",
               "--message", "hello", "--read-only", "--timeout", "3")
        assert seen["api_key"] == "k" and seen["api_key_header"] == "X-K"
        assert seen["bearer"] == "b" and seen["caller_url"] == "http://me:1"
        assert seen["skill_id"] == "s" and seen["message"] == "hello"
        assert seen["read_only"] is True and seen["timeout"] == 3.0

    def test_json_output(self, monkeypatch):
        self.patch(monkeypatch, self.fake_report([Status.PASS]))
        result = invoke("--format", "json", "verify", AGENT)
        assert json.loads(result.output)["passed"] is True

    def test_crash_is_reported(self, monkeypatch):
        async def boom(url, **kwargs):
            raise RuntimeError("network on fire")

        monkeypatch.setattr("nexus_a2a.verify.verify_agent", boom)
        result = CliRunner().invoke(cli, ["verify", AGENT])
        assert result.exit_code == 1
        assert "network on fire" in result.output


# ── dev ───────────────────────────────────────────────────────────────────────


class TestDevPlanning:
    def test_flags_only_get_sequential_ports(self):
        planned = dev_cmd.plan_agents({}, ["a:A", "b:B"], 9000)
        assert [(p.module, p.port) for p in planned] == [("a:A", 9000), ("b:B", 9001)]

    def test_config_ports_are_honoured_and_skipped(self):
        raw = {"dev": {"agents": [{"module": "a:A", "port": 9001}]}}
        planned = dev_cmd.plan_agents(raw, ["b:B", "c:C"], 9000)
        assert [(p.module, p.port) for p in planned] == [
            ("a:A", 9001), ("b:B", 9000), ("c:C", 9002),
        ]

    def test_duplicate_ports_rejected(self):
        import click

        raw = {"dev": {"agents": [{"module": "a:A", "port": 1},
                                  {"module": "b:B", "port": 1}]}}
        with pytest.raises(click.UsageError, match="same port"):
            dev_cmd.plan_agents(raw, [], 9000)

    @pytest.mark.parametrize(
        "entry", [{"port": 1}, {"module": ""}, "a:A", {"module": "a:A", "port": "x"}]
    )
    def test_malformed_entries_rejected(self, entry):
        import click

        with pytest.raises(click.UsageError):
            dev_cmd.plan_agents({"dev": {"agents": [entry]}}, [], 9000)

    def test_url(self):
        assert dev_cmd.DevAgent("a:A", 9000).url == "http://127.0.0.1:9000"


class TestDevConfig:
    def test_per_agent_config(self):
        raw = {
            "agent": {"name": "Shared", "url": "http://shared:1"},
            "ops": {"port": 9999},
            "dev": {"agents": []},
            "security": {"auth_scheme": "api_key", "auth_secret": "k"},
            "network": {"agents": ["http://external:1"]},
        }
        agent_ = dev_cmd.DevAgent("a:A", 9000)
        cfg = dev_cmd.dev_config_for(raw, agent_, ["http://127.0.0.1:9000",
                                                   "http://127.0.0.1:9001"])
        assert cfg["agent"] == {"url": "http://127.0.0.1:9000"}
        assert "ops" not in cfg and "dev" not in cfg
        assert cfg["security"]["auth_secret"] == "k"
        assert cfg["network"]["agents"] == [
            "http://external:1", "http://127.0.0.1:9000", "http://127.0.0.1:9001",
        ]
        assert cfg["push"]["allow_private_urls"] is True
        assert raw["network"]["agents"] == ["http://external:1"]  # not mutated

    def test_explicit_push_setting_kept(self):
        cfg = dev_cmd.dev_config_for(
            {"push": {"allow_private_urls": False}}, dev_cmd.DevAgent("a:A", 1), []
        )
        assert cfg["push"]["allow_private_urls"] is False

    def test_runtimes_advertise_their_dev_address(self, monkeypatch):
        monkeypatch.setattr(run_cmd, "_load_agent_class", lambda module: EchoAgent)
        planned = [dev_cmd.DevAgent("x:Echo", 9100)]
        [(planned_agent, runtime)] = dev_cmd.build_runtimes({}, planned)
        assert runtime.server.port == 9100
        assert runtime.server.host == "127.0.0.1"
        assert runtime.server.public_url == "http://127.0.0.1:9100"
        assert runtime.ops is None


class TestDevCommand:
    def test_no_agents(self):
        result = CliRunner().invoke(cli, ["dev"])
        assert result.exit_code == 1
        assert "No agents" in result.output

    def test_busy_port(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            result = CliRunner().invoke(
                cli, ["dev", "--agent", "x:Y", "--base-port", str(port)]
            )
        assert result.exit_code == 1
        assert "already in use" in result.output


# ── output helpers ────────────────────────────────────────────────────────────


class TestEncodingSafety:
    def test_can_encode(self):
        class Stream:
            encoding = "cp1252"

        assert output._can_encode("plain", Stream())
        assert not output._can_encode("✓", Stream())
        assert output._can_encode("✓", None)  # unknown -> assume utf-8

    def test_unknown_encoding_is_unsafe(self):
        class Stream:
            encoding = "no-such-codec"

        assert not output._can_encode("x", Stream())

    def test_ensure_safe_output_sets_replace(self, monkeypatch):
        buffer = io.BytesIO()
        stream = io.TextIOWrapper(buffer, encoding="cp1252")
        monkeypatch.setattr("sys.stdout", stream)
        output.ensure_safe_output()
        stream.write("✓ ok")  # must not raise
        stream.flush()
        assert buffer.getvalue() == b"? ok"

    def test_ensure_safe_output_tolerates_odd_streams(self, monkeypatch):
        monkeypatch.setattr("sys.stdout", object())
        output.ensure_safe_output()

    def test_count_renders_dash_for_none(self):
        assert output._count(None) == "—"
        assert output._count(0) == "0"


class TestRenderVerify:
    REPORT = {
        "url": AGENT, "passed": False,
        "summary": {"pass": 1, "warn": 1, "fail": 1, "skip": 1},
        "checks": [
            {"group": "card", "name": "json", "status": "pass", "detail": ""},
            {"group": "card", "name": "url", "status": "warn", "detail": "mismatch"},
            {"group": "auth", "name": "matches_card", "status": "fail",
             "detail": "open to anyone"},
            {"group": "push", "name": "config_get", "status": "skip", "detail": "n/a"},
        ],
    }

    def test_table(self, capsys):
        output.render_verify(self.REPORT)
        out = capsys.readouterr().out
        assert "FAILED" in out and "open to anyone" in out

    def test_json(self, capsys):
        output.render_verify(self.REPORT, fmt="json")
        assert json.loads(capsys.readouterr().out)["summary"]["fail"] == 1
