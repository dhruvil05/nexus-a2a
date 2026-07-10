"""
nexus_a2a/core/agent_server.py

AgentServer — built-in Starlette HTTP server providing Kubernetes-compatible
health, readiness, and Prometheus metrics endpoints.

Endpoints:
    GET /health   — Liveness probe. Always 200 if the process is running.
                    Kubernetes restarts the pod if this returns non-200.

    GET /ready    — Readiness probe. Returns 200 only when the agent is fully
                    ready to serve requests (store reachable + registry healthy).
                    Kubernetes stops sending traffic if this returns non-200.

    GET /metrics  — Prometheus text format (exposition format v0.0.4).
                    Scraped by Prometheus, Grafana Agent, Datadog, etc.

    GET /info     — Human-readable JSON summary of network state.

Usage:
    server = AgentServer(network=network)
    await server.start(host="0.0.0.0", port=8080)
    # ... agent runs ...
    await server.stop()

    # Or as async context manager:
    async with AgentServer(network=network, port=8080) as server:
        await asyncio.Event().wait()   # run forever

Design:
    - Pure Starlette ASGI — no FastAPI overhead.
    - No extra deps beyond what nexus-a2a already requires.
    - Ready check pings the task store and registry concurrently.
    - Prometheus format is hand-built — no prometheus_client needed.
    - All endpoints are synchronous-safe (no blocking I/O in /health or /metrics).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

if TYPE_CHECKING:
    from nexus_a2a.network import AgentNetwork

logger = logging.getLogger(__name__)


# ── AgentServer ───────────────────────────────────────────────────────────────


class AgentServer:
    """
    Built-in HTTP server exposing liveness, readiness, and metrics endpoints.

    Designed to be Kubernetes-compatible out of the box:
    - /health  → livenessProbe
    - /ready   → readinessProbe
    - /metrics → Prometheus scrape target

    Args:
        network:    The AgentNetwork to monitor. Used for registry health
                    checks and metrics collection.
        host:       Host to bind the server to. Defaults to '0.0.0.0'.
        port:       Port to listen on. Defaults to 8080.
        log_level:  Uvicorn log level. Defaults to 'warning' (quiet).

    Example::

        server = AgentServer(network=network, port=8080)
        await server.start()

        # Kubernetes probes now work:
        # GET http://agent:8080/health  → 200 OK
        # GET http://agent:8080/ready   → 200 OK  (when ready)
        # GET http://agent:8080/metrics → Prometheus text

        await server.stop()
    """

    def __init__(
        self,
        network: AgentNetwork,
        host: str = "0.0.0.0",
        port: int = 8080,
        log_level: str = "warning",
    ) -> None:
        self.network = network
        self.host = host
        self.port = port
        self.log_level = log_level

        self._started_at: float | None = None
        self._server: uvicorn.Server | None = None
        self._serve_task: asyncio.Task[None] | None = None

        # Build the ASGI app
        self._app = self._build_app()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """
        Start the HTTP server in a background asyncio task.

        Returns immediately — the server runs in the background.
        Call stop() to shut it down cleanly.

        Raises:
            RuntimeError: If the server is already running.
        """
        if self._serve_task is not None and not self._serve_task.done():
            raise RuntimeError("AgentServer is already running.")

        config = uvicorn.Config(
            app=self._app,
            host=self.host,
            port=self.port,
            log_level=self.log_level,
            loop="asyncio",
        )
        self._server = uvicorn.Server(config)
        self._started_at = time.monotonic()

        self._serve_task = asyncio.create_task(
            self._server.serve(),
            name="nexus_a2a.agent_server",
        )

        # Give uvicorn a moment to bind the port before returning
        await asyncio.sleep(0.05)
        logger.info("AgentServer started on http://%s:%d", self.host, self.port)

    async def stop(self, timeout: float = 5.0) -> None:
        """
        Gracefully stop the HTTP server.

        Args:
            timeout: Seconds to wait for the server to shut down.
        """
        if self._server is not None:
            self._server.should_exit = True

        if self._serve_task is not None:
            try:
                await asyncio.wait_for(self._serve_task, timeout=timeout)
            except TimeoutError:
                logger.warning(
                    "AgentServer did not shut down within %.1fs; cancelling.", timeout
                )
                self._serve_task.cancel()
                try:
                    await self._serve_task
                except asyncio.CancelledError:
                    pass

        self._serve_task = None
        self._server = None
        logger.info("AgentServer stopped.")

    @property
    def is_running(self) -> bool:
        """True if the server background task is alive."""
        return self._serve_task is not None and not self._serve_task.done()

    @property
    def uptime_seconds(self) -> float | None:
        """Seconds since the server started, or None if not started."""
        if self._started_at is None:
            return None
        return time.monotonic() - self._started_at

    # ── Async context manager ─────────────────────────────────────────────────

    async def __aenter__(self) -> AgentServer:
        await self.start()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.stop()

    # ── ASGI app builder ──────────────────────────────────────────────────────

    def _build_app(self) -> Starlette:
        """Build and return the Starlette ASGI application."""

        async def health(request: Request) -> Response:
            return await self._handle_health(request)

        async def ready(request: Request) -> Response:
            return await self._handle_ready(request)

        async def metrics(request: Request) -> Response:
            return await self._handle_metrics(request)

        async def info(request: Request) -> Response:
            return await self._handle_info(request)

        async def trace_lookup(request: Request) -> Response:
            return await self._handle_trace(request)

        async def dlq_list(request: Request) -> Response:
            return await self._handle_dlq_list(request)

        async def dlq_replay(request: Request) -> Response:
            return await self._handle_dlq_replay(request)

        return Starlette(
            routes=[
                Route("/health", health),
                Route("/ready", ready),
                Route("/metrics", metrics),
                Route("/info", info),
                Route("/traces/{trace_id}", trace_lookup),
                Route("/dlq", dlq_list),
                Route("/dlq/replay", dlq_replay, methods=["POST"]),
            ],
        )

    # ── Endpoint handlers ─────────────────────────────────────────────────────

    async def _handle_health(self, request: Request) -> Response:
        """
        GET /health — Liveness probe.

        Always returns 200 as long as the process is alive and the event loop
        is processing requests. This is intentionally lightweight — Kubernetes
        uses this to decide whether to restart the container.

        Response body:
            {"status": "ok", "uptime_seconds": 42.3}
        """
        return JSONResponse(
            {
                "status": "ok",
                "uptime_seconds": round(self.uptime_seconds or 0.0, 2),
            }
        )

    async def _handle_ready(self, request: Request) -> Response:
        """
        GET /ready — Readiness probe.

        Returns 200 when the agent is fully ready to serve requests:
        - Task store is reachable (ping via list_all())
        - At least one agent in the registry is healthy (if any registered)

        Returns 503 if either check fails. Kubernetes stops routing traffic
        to the pod until this returns 200 again.

        Response body:
            {
                "status": "ready",           # or "not_ready"
                "checks": {
                    "store":    "ok",        # or "error: <message>"
                    "registry": "ok",        # or "no_agents" / "all_unhealthy"
                }
            }
        """
        checks: dict[str, str] = {}
        overall_ok = True

        # Check 1: task store reachable
        try:
            await self.network.task_manager._store.list_all()
            checks["store"] = "ok"
        except Exception as exc:
            checks["store"] = f"error: {exc}"
            overall_ok = False
            logger.warning("AgentServer /ready store check failed: %s", exc)

        # Check 2: registry health (only checked if agents are registered)
        agents = self.network.registry.list_all()
        if not agents:
            checks["registry"] = "no_agents"
        else:
            healthy = self.network.registry.list_healthy()
            if healthy:
                checks["registry"] = f"ok ({len(healthy)}/{len(agents)} healthy)"
            else:
                checks["registry"] = f"all_unhealthy ({len(agents)} registered)"
                overall_ok = False

        status_code = 200 if overall_ok else 503
        return JSONResponse(
            {
                "status": "ready" if overall_ok else "not_ready",
                "checks": checks,
            },
            status_code=status_code,
        )

    async def _handle_metrics(self, request: Request) -> Response:
        """
        GET /metrics — Prometheus text format exposition.

        Returns metrics in the standard Prometheus text format v0.0.4.
        Scraped by Prometheus, Grafana Agent, Datadog Agent, etc.

        Includes:
            nexus_a2a_tasks_created_total
            nexus_a2a_tasks_completed_total
            nexus_a2a_tasks_failed_total
            nexus_a2a_tasks_cancelled_total
            nexus_a2a_rate_limit_hits_total
            nexus_a2a_auth_failures_total
            nexus_a2a_agent_errors_total{agent_url="..."}
            nexus_a2a_agent_call_duration_seconds_avg{agent_url="..."}
            nexus_a2a_agent_call_duration_seconds_p99{agent_url="..."}
            nexus_a2a_dlq_pending
            nexus_a2a_dlq_total
            nexus_a2a_registry_agents_total
            nexus_a2a_registry_healthy_total
            nexus_a2a_uptime_seconds
        """
        lines: list[str] = []

        def gauge(
            name: str, value: float | int, labels: dict[str, str] | None = None
        ) -> None:
            label_str = _format_labels(labels)
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name}{label_str} {value}")

        def counter(
            name: str, value: float | int, labels: dict[str, str] | None = None
        ) -> None:
            label_str = _format_labels(labels)
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name}{label_str} {value}")

        # Get metrics snapshot
        try:
            # Try to get a metrics collector from the network if wired
            metrics_collector = getattr(self.network, "_metrics", None)
            if metrics_collector is not None:
                snap_data = metrics_collector.snapshot()
                counter("nexus_a2a_tasks_created_total", snap_data.tasks_created)
                counter("nexus_a2a_tasks_completed_total", snap_data.tasks_completed)
                counter("nexus_a2a_tasks_failed_total", snap_data.tasks_failed)
                counter("nexus_a2a_tasks_cancelled_total", snap_data.tasks_cancelled)
                counter("nexus_a2a_rate_limit_hits_total", snap_data.rate_limit_hits)
                counter("nexus_a2a_auth_failures_total", snap_data.auth_failures)

                for agent_url, error_count in snap_data.agent_errors.items():
                    counter(
                        "nexus_a2a_agent_errors_total",
                        error_count,
                        {"agent_url": agent_url},
                    )

                for agent_url in snap_data.call_durations:
                    avg = snap_data.avg_latency(agent_url)
                    p99 = snap_data.p99_latency(agent_url)
                    if avg is not None:
                        gauge(
                            "nexus_a2a_agent_call_duration_seconds_avg",
                            round(avg, 6),
                            {"agent_url": agent_url},
                        )
                    if p99 is not None:
                        gauge(
                            "nexus_a2a_agent_call_duration_seconds_p99",
                            round(p99, 6),
                            {"agent_url": agent_url},
                        )
            else:
                # No metrics collector wired — emit zero counters so
                # Prometheus targets don't disappear from the scrape list
                for name in (
                    "nexus_a2a_tasks_created_total",
                    "nexus_a2a_tasks_completed_total",
                    "nexus_a2a_tasks_failed_total",
                    "nexus_a2a_tasks_cancelled_total",
                    "nexus_a2a_rate_limit_hits_total",
                    "nexus_a2a_auth_failures_total",
                ):
                    counter(name, 0)
        except Exception as exc:
            logger.warning("AgentServer /metrics collection error: %s", exc)

        # DLQ metrics
        try:
            dlq = self.network.dead_letter_queue
            gauge("nexus_a2a_dlq_pending", dlq.pending_count())
            gauge("nexus_a2a_dlq_total", dlq.count())
        except Exception:
            gauge("nexus_a2a_dlq_pending", 0)
            gauge("nexus_a2a_dlq_total", 0)

        # Registry metrics
        try:
            all_agents = self.network.registry.list_all()
            healthy_agents = self.network.registry.list_healthy()
            gauge("nexus_a2a_registry_agents_total", len(all_agents))
            gauge("nexus_a2a_registry_healthy_total", len(healthy_agents))
        except Exception:
            gauge("nexus_a2a_registry_agents_total", 0)
            gauge("nexus_a2a_registry_healthy_total", 0)

        # Uptime
        gauge("nexus_a2a_uptime_seconds", round(self.uptime_seconds or 0.0, 2))

        # Prometheus format requires a trailing newline
        body = "\n".join(lines) + "\n"
        return PlainTextResponse(
            body,
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    async def _handle_info(self, request: Request) -> Response:
        """
        GET /info — Human-readable JSON summary of the network state.

        Not a Kubernetes probe — intended for operators checking agent status.

        Response body:
            {
                "version":        "1.2.0",
                "uptime_seconds": 42.3,
                "network":        { ... registry summary ... },
                "dlq":            { "pending": 0, "total": 0 }
            }
        """
        try:
            from nexus_a2a import __version__

            version = __version__
        except ImportError:
            version = "unknown"

        try:
            summary = self.network.summary()
        except Exception as exc:
            summary = {"error": str(exc)}

        return JSONResponse(
            {
                "version": version,
                "uptime_seconds": round(self.uptime_seconds or 0.0, 2),
                "network": summary,
            }
        )

    async def _handle_trace(self, request: Request) -> Response:
        """
        GET /traces/{trace_id} — Look up a distributed trace by ID.

        Reads from the process-wide TraceStore
        (nexus_a2a.transport.tracing.default_store), which Tracer.span()
        records into automatically for every outbound call made through
        this process. Returns 404 if the trace ID is unknown.

        Response body:
            {
                "trace_id": "...",
                "hops": [
                    {"url": "...", "duration_ms": 12.3, "status": "completed",
                     "error": null, "children": []},
                    ...
                ]
            }
        """
        trace_id = request.path_params["trace_id"]
        from nexus_a2a.transport.tracing import default_store

        trace = default_store.get(trace_id)
        if trace is None:
            return JSONResponse(
                {"error": f"trace '{trace_id}' not found"}, status_code=404
            )

        hops: list[dict[str, Any]] = [
            {
                "url": getattr(span, "agent_url", "unknown"),
                "duration_ms": getattr(span, "duration_ms", None),
                "status": getattr(span, "status", "unknown"),
                "error": getattr(span, "error", None),
                "children": [],
            }
            for span in getattr(trace, "spans", [])
        ]
        return JSONResponse({"trace_id": trace_id, "hops": hops})

    async def _handle_dlq_list(self, request: Request) -> Response:
        """
        GET /dlq — List Dead Letter Queue entries for this agent.

        Query params:
            skill:   optional skill_id filter
            pending: "true" to return only entries not yet successfully
                     replayed (default: return all entries)

        Response body:
            {"entries": [
                {"task_id": "...", "error": "...", "failed_at": 1720000000.0,
                 "agent_url": "...", "skill_id": "...", "retry_count": 0,
                 "last_retry_at": null, "replayed": false},
                ...
            ]}
        """
        dlq = self.network.dead_letter_queue
        pending_only = request.query_params.get("pending", "").lower() == "true"
        skill = request.query_params.get("skill")

        entries = dlq.pending_entries() if pending_only else dlq.all_entries()
        if skill:
            entries = [e for e in entries if e.skill_id == skill]

        return JSONResponse({"entries": [e.to_dict() for e in entries]})

    async def _handle_dlq_replay(self, request: Request) -> Response:
        """
        POST /dlq/replay — Replay one or more failed tasks from the DLQ.

        Request body (JSON), one of:
            {"task_id": "abc-123"}            — replay a single entry
            {"skill_id": "web_search"}        — replay all matching a skill
            {"agent_url": "http://agent:8001"} — replay all from one agent
            {}                                 — replay every pending entry

        Response body:
            {
                "succeeded": 2, "failed": 1,
                "results": [
                    {"task_id": "...", "succeeded": true, "error": null},
                    ...
                ]
            }
        """
        dlq = self.network.dead_letter_queue
        try:
            payload = await request.json()
        except Exception:
            payload = {}

        task_id = payload.get("task_id")
        skill_id = payload.get("skill_id")
        agent_url = payload.get("agent_url")

        try:
            if task_id:
                results = [await dlq.replay(task_id)]
            elif skill_id or agent_url:
                results = await dlq.replay_where(skill_id=skill_id, agent_url=agent_url)
            else:
                results = await dlq.replay_all()
        except KeyError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

        succeeded = sum(1 for r in results if getattr(r, "succeeded", False))
        return JSONResponse(
            {
                "succeeded": succeeded,
                "failed": len(results) - succeeded,
                "results": [
                    {
                        "task_id": getattr(r, "task_id", None),
                        "succeeded": getattr(r, "succeeded", False),
                        "error": getattr(r, "error", None),
                    }
                    for r in results
                ],
            }
        )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _format_labels(labels: dict[str, str] | None) -> str:
    """Format a label dict into Prometheus label syntax: {key="val",...}"""
    if not labels:
        return ""
    parts = [f'{k}="{v}"' for k, v in labels.items()]
    return "{" + ",".join(parts) + "}"
