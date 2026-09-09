"""
tests/test_agent_server.py

Tests for nexus_a2a/core/agent_server.py — AgentServer.

Coverage:
  - AgentServer lifecycle: start(), stop(), is_running, uptime_seconds
  - Async context manager (__aenter__ / __aexit__)
  - GET /health  → always 200 with uptime
  - GET /ready   → 200 when store ok, 503 when store fails
  - GET /ready   → registry checks (no agents, healthy, all unhealthy)
  - GET /metrics → Prometheus text format, correct metric names
  - GET /info    → JSON summary with version and uptime
  - _format_labels() helper
  - Double-start raises RuntimeError
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.testclient import TestClient

from nexus_a2a.core.agent_server import AgentServer, _format_labels

# ── Fixtures ──────────────────────────────────────────────────────────────────


def make_mock_network(
    *,
    store_ok: bool = True,
    agents: int = 0,
    healthy_agents: int = 0,
    dlq_pending: int = 0,
    dlq_total: int = 0,
) -> MagicMock:
    """Build a minimal mock AgentNetwork for testing AgentServer."""
    network = MagicMock()

    # task_manager._store.list_all()
    store = AsyncMock()
    if store_ok:
        store.list_all = AsyncMock(return_value=[])
    else:
        store.list_all = AsyncMock(side_effect=ConnectionError("Redis down"))
    network.task_manager._store = store

    # registry
    all_cards = [MagicMock() for _ in range(agents)]
    healthy_list = all_cards[:healthy_agents]
    network.registry.list_all = MagicMock(return_value=all_cards)
    network.registry.list_healthy = MagicMock(return_value=healthy_list)

    # DLQ
    network.dead_letter_queue.pending_count = MagicMock(return_value=dlq_pending)
    network.dead_letter_queue.count = MagicMock(return_value=dlq_total)

    # summary
    network.summary = MagicMock(
        return_value={
            "total_agents": agents,
            "healthy_agents": healthy_agents,
            "dlq": {"pending": dlq_pending, "total": dlq_total},
        }
    )

    # No metrics collector by default
    del network._metrics  # ensure getattr returns None via hasattr path

    return network


ADMIN_TOKEN = "test-admin-token"


def make_test_client(
    network: MagicMock | None = None,
    admin_token: str | None = ADMIN_TOKEN,
) -> TestClient:
    """
    Build a Starlette TestClient for AgentServer without starting uvicorn.

    Admin endpoints (/info, /traces, /dlq, /dlq/replay) require a token as of
    v1.5.0, so tests get one by default and send it via ADMIN_HEADERS. Pass
    admin_token=None to exercise the disabled-by-default path.
    """
    net = network or make_mock_network()
    server = AgentServer(network=net, port=9999, admin_token=admin_token)
    # Inject a fake start time so uptime is nonzero
    import time

    server._started_at = time.monotonic() - 10.0
    return TestClient(server._app)


ADMIN_HEADERS = {"X-Admin-Token": ADMIN_TOKEN}


# ── _format_labels() ──────────────────────────────────────────────────────────


class TestFormatLabels:
    def test_none_returns_empty_string(self):
        assert _format_labels(None) == ""

    def test_empty_dict_returns_empty_string(self):
        assert _format_labels({}) == ""

    def test_single_label(self):
        result = _format_labels({"agent_url": "http://agent:8001"})
        assert result == '{agent_url="http://agent:8001"}'

    def test_multiple_labels(self):
        result = _format_labels({"env": "prod", "region": "us-east"})
        assert 'env="prod"' in result
        assert 'region="us-east"' in result
        assert result.startswith("{")
        assert result.endswith("}")


# ── GET /health ───────────────────────────────────────────────────────────────


class TestHealthEndpoint:
    def test_returns_200(self):
        client = make_test_client()
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_response_has_status_ok(self):
        client = make_test_client()
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_response_has_uptime(self):
        client = make_test_client()
        data = client.get("/health").json()
        assert "uptime_seconds" in data
        assert isinstance(data["uptime_seconds"], float)
        assert data["uptime_seconds"] >= 0.0

    def test_uptime_is_nonzero_after_start(self):
        client = make_test_client()
        data = client.get("/health").json()
        # We set started_at to 10 seconds ago in make_test_client
        assert data["uptime_seconds"] >= 9.0

    def test_health_is_fast(self):
        """Health endpoint must not do any I/O."""
        import time

        client = make_test_client()
        t0 = time.monotonic()
        client.get("/health")
        elapsed = time.monotonic() - t0
        # Must respond in under 100ms (no I/O allowed)
        assert elapsed < 0.1


# ── GET /ready ────────────────────────────────────────────────────────────────


class TestReadyEndpoint:
    def test_returns_200_when_store_ok_no_agents(self):
        client = make_test_client(make_mock_network(store_ok=True, agents=0))
        resp = client.get("/ready")
        assert resp.status_code == 200

    def test_ready_status_in_body(self):
        client = make_test_client(make_mock_network(store_ok=True))
        data = client.get("/ready").json()
        assert data["status"] == "ready"

    def test_returns_503_when_store_fails(self):
        client = make_test_client(make_mock_network(store_ok=False))
        resp = client.get("/ready")
        assert resp.status_code == 503

    def test_not_ready_status_when_store_fails(self):
        client = make_test_client(make_mock_network(store_ok=False))
        data = client.get("/ready").json()
        assert data["status"] == "not_ready"

    def test_store_error_in_checks(self):
        client = make_test_client(make_mock_network(store_ok=False))
        data = client.get("/ready").json()
        assert "error" in data["checks"]["store"]

    def test_store_ok_in_checks(self):
        client = make_test_client(make_mock_network(store_ok=True))
        data = client.get("/ready").json()
        assert data["checks"]["store"] == "ok"

    def test_no_agents_registry_check(self):
        client = make_test_client(make_mock_network(agents=0))
        data = client.get("/ready").json()
        assert data["checks"]["registry"] == "no_agents"

    def test_healthy_agents_passes_registry_check(self):
        client = make_test_client(make_mock_network(agents=3, healthy_agents=2))
        data = client.get("/ready").json()
        assert "ok" in data["checks"]["registry"]
        assert "2/3" in data["checks"]["registry"]

    def test_all_unhealthy_agents_fails_check(self):
        client = make_test_client(make_mock_network(agents=2, healthy_agents=0))
        resp = client.get("/ready")
        data = resp.json()
        assert resp.status_code == 503
        assert "all_unhealthy" in data["checks"]["registry"]

    def test_checks_key_always_present(self):
        for store_ok in (True, False):
            client = make_test_client(make_mock_network(store_ok=store_ok))
            data = client.get("/ready").json()
            assert "checks" in data
            assert "store" in data["checks"]
            assert "registry" in data["checks"]


# ── GET /metrics ──────────────────────────────────────────────────────────────


class TestMetricsEndpoint:
    def test_returns_200(self):
        client = make_test_client()
        assert client.get("/metrics").status_code == 200

    def test_content_type_is_prometheus(self):
        client = make_test_client()
        resp = client.get("/metrics")
        assert "text/plain" in resp.headers["content-type"]

    def test_contains_task_counters(self):
        client = make_test_client()
        body = client.get("/metrics").text
        assert "nexus_a2a_tasks_created_total" in body
        assert "nexus_a2a_tasks_completed_total" in body
        assert "nexus_a2a_tasks_failed_total" in body

    def test_contains_dlq_metrics(self):
        client = make_test_client(make_mock_network(dlq_pending=3, dlq_total=5))
        body = client.get("/metrics").text
        assert "nexus_a2a_dlq_pending" in body
        assert "nexus_a2a_dlq_total" in body

    def test_dlq_values_correct(self):
        client = make_test_client(make_mock_network(dlq_pending=3, dlq_total=7))
        body = client.get("/metrics").text
        # Find the lines
        lines = {
            line.split(" ")[0]: line.split(" ")[-1]
            for line in body.splitlines()
            if not line.startswith("#") and " " in line
        }
        assert lines.get("nexus_a2a_dlq_pending") == "3"
        assert lines.get("nexus_a2a_dlq_total") == "7"

    def test_contains_registry_metrics(self):
        client = make_test_client(make_mock_network(agents=4, healthy_agents=3))
        body = client.get("/metrics").text
        assert "nexus_a2a_registry_agents_total" in body
        assert "nexus_a2a_registry_healthy_total" in body

    def test_registry_values_correct(self):
        client = make_test_client(make_mock_network(agents=4, healthy_agents=3))
        body = client.get("/metrics").text
        lines = {
            line.split(" ")[0]: line.split(" ")[-1]
            for line in body.splitlines()
            if not line.startswith("#") and " " in line
        }
        assert lines.get("nexus_a2a_registry_agents_total") == "4"
        assert lines.get("nexus_a2a_registry_healthy_total") == "3"

    def test_contains_uptime(self):
        client = make_test_client()
        body = client.get("/metrics").text
        assert "nexus_a2a_uptime_seconds" in body

    def test_uptime_nonzero(self):
        client = make_test_client()
        body = client.get("/metrics").text
        lines = {
            line.split(" ")[0]: line.split(" ")[-1]
            for line in body.splitlines()
            if not line.startswith("#") and " " in line
        }
        uptime = float(lines["nexus_a2a_uptime_seconds"])
        assert uptime >= 9.0

    def test_type_annotations_present(self):
        """Every metric must have a # TYPE line before its value."""
        client = make_test_client()
        body = client.get("/metrics").text
        lines = body.splitlines()
        type_lines = {line.split(" ")[2] for line in lines if line.startswith("# TYPE")}
        assert "nexus_a2a_tasks_created_total" in type_lines
        assert "nexus_a2a_dlq_pending" in type_lines

    def test_metrics_with_collector_wired(self):
        """When a MetricsCollector is attached, its data appears in output."""
        from nexus_a2a.storage.metrics import MetricsCollector, MetricsSnapshot

        network = make_mock_network()
        collector = MagicMock(spec=MetricsCollector)
        snap = MetricsSnapshot(
            tasks_created=10,
            tasks_completed=8,
            tasks_failed=2,
            tasks_cancelled=0,
            rate_limit_hits=1,
            auth_failures=0,
            agent_errors={"http://agent:8001": 2},
            call_durations={"http://agent:8001": [0.1, 0.2, 0.3]},
        )
        collector.snapshot.return_value = snap
        network._metrics = collector

        server = AgentServer(network=network, port=9999)
        import time

        server._started_at = time.monotonic() - 5.0
        client = TestClient(server._app)

        body = client.get("/metrics").text
        assert "10" in body  # tasks_created
        assert "http://agent:8001" in body


# ── GET /info ─────────────────────────────────────────────────────────────────


class TestInfoEndpoint:
    def test_returns_200(self):
        client = make_test_client()
        assert client.get("/info", headers=ADMIN_HEADERS).status_code == 200

    def test_response_is_json(self):
        client = make_test_client()
        resp = client.get("/info", headers=ADMIN_HEADERS)
        data = resp.json()
        assert isinstance(data, dict)

    def test_has_version(self):
        client = make_test_client()
        data = client.get("/info", headers=ADMIN_HEADERS).json()
        assert "version" in data
        assert isinstance(data["version"], str)

    def test_has_uptime(self):
        client = make_test_client()
        data = client.get("/info", headers=ADMIN_HEADERS).json()
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 9.0

    def test_has_network_summary(self):
        client = make_test_client()
        data = client.get("/info", headers=ADMIN_HEADERS).json()
        assert "network" in data
        assert isinstance(data["network"], dict)

    def test_network_summary_content(self):
        net = make_mock_network(agents=2, healthy_agents=1)
        client = make_test_client(net)
        data = client.get("/info", headers=ADMIN_HEADERS).json()
        summary = data["network"]
        assert summary["total_agents"] == 2
        assert summary["healthy_agents"] == 1


# ── AgentServer lifecycle ─────────────────────────────────────────────────────


class TestAgentServerLifecycle:
    def test_is_running_false_before_start(self):
        net = make_mock_network()
        server = AgentServer(network=net, port=19876)
        assert server.is_running is False

    def test_uptime_none_before_start(self):
        net = make_mock_network()
        server = AgentServer(network=net, port=19876)
        assert server.uptime_seconds is None

    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        net = make_mock_network()
        server = AgentServer(network=net, port=19877, log_level="critical")
        await server.start()
        assert server.is_running is True
        assert server.uptime_seconds is not None
        assert server.uptime_seconds >= 0.0
        await server.stop()
        assert server.is_running is False

    @pytest.mark.asyncio
    async def test_double_start_raises(self):
        net = make_mock_network()
        server = AgentServer(network=net, port=19878, log_level="critical")
        await server.start()
        try:
            with pytest.raises(RuntimeError, match="already running"):
                await server.start()
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        net = make_mock_network()
        async with AgentServer(network=net, port=19879, log_level="critical") as server:
            assert server.is_running is True
        assert server.is_running is False

    @pytest.mark.asyncio
    async def test_stop_when_not_running_is_safe(self):
        net = make_mock_network()
        server = AgentServer(network=net, port=19880)
        # Should not raise
        await server.stop()


# ── Admin endpoint authorisation (v1.5.0) ─────────────────────────────────────


class TestAdminEndpointsDisabledByDefault:
    """
    Without a token, /info, /traces, /dlq and /dlq/replay must be refused.
    An unauthenticated POST /dlq/replay would let any caller that can reach
    the port re-execute every failed task.
    """

    def _client(self, monkeypatch) -> TestClient:
        monkeypatch.delenv("NEXUS_ADMIN_TOKEN", raising=False)
        return make_test_client(admin_token=None)

    def test_info_forbidden(self, monkeypatch):
        assert self._client(monkeypatch).get("/info").status_code == 403

    def test_dlq_list_forbidden(self, monkeypatch):
        assert self._client(monkeypatch).get("/dlq").status_code == 403

    def test_dlq_replay_forbidden(self, monkeypatch):
        resp = self._client(monkeypatch).post("/dlq/replay", json={})
        assert resp.status_code == 403

    def test_trace_forbidden(self, monkeypatch):
        assert self._client(monkeypatch).get("/traces/abc").status_code == 403

    def test_replay_is_not_executed_when_forbidden(self, monkeypatch):
        """The 403 must short-circuit before any task is replayed."""
        net = make_mock_network()
        net.dead_letter_queue.replay_all = AsyncMock(return_value=[])
        monkeypatch.delenv("NEXUS_ADMIN_TOKEN", raising=False)
        client = make_test_client(net, admin_token=None)
        client.post("/dlq/replay", json={})
        net.dead_letter_queue.replay_all.assert_not_awaited()

    def test_probes_stay_public(self, monkeypatch):
        """Kubernetes probes and Prometheus must not need credentials."""
        client = self._client(monkeypatch)
        assert client.get("/health").status_code == 200
        assert client.get("/metrics").status_code == 200


class TestAdminEndpointsWithToken:
    def test_correct_token_allows_info(self):
        client = make_test_client()
        assert client.get("/info", headers=ADMIN_HEADERS).status_code == 200

    def test_bearer_scheme_also_accepted(self):
        client = make_test_client()
        resp = client.get("/info", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"})
        assert resp.status_code == 200

    def test_wrong_token_forbidden(self):
        client = make_test_client()
        resp = client.get("/info", headers={"X-Admin-Token": "wrong"})
        assert resp.status_code == 403

    def test_missing_token_forbidden(self):
        client = make_test_client()
        assert client.get("/info").status_code == 403

    def test_empty_token_forbidden(self):
        client = make_test_client()
        assert client.get("/info", headers={"X-Admin-Token": ""}).status_code == 403

    def test_token_read_from_environment(self, monkeypatch):
        monkeypatch.setenv("NEXUS_ADMIN_TOKEN", "from-env")
        client = make_test_client(admin_token=None)
        assert client.get("/info", headers={"X-Admin-Token": "from-env"}).status_code == 200
        assert client.get("/info", headers={"X-Admin-Token": "nope"}).status_code == 403
