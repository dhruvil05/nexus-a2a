"""
nexus_a2a/dashboard/app.py

Dashboard — a browser UI and JSON API for watching and operating one agent.

Served by the ops server (AgentServer) under /ui, never on the public agent
port. The HTTP API is documented in nexus_a2a/dashboard/API.md.

Security model:
  - Off unless enabled AND an admin token is configured.
  - Loopback clients only, unless allow_remote=True.
  - Log in with the admin token; the browser then holds a short-lived,
    HttpOnly, SameSite=Strict session cookie scoped to /ui.
  - Every state-changing request made with that cookie needs the session's
    CSRF token in X-CSRF-Token and a same-origin Origin header.
  - Scripts can skip the cookie and send X-Admin-Token instead.
  - Login attempts are throttled per client address.
  - Message content is hidden unless show_content=True.
  - Strict Content-Security-Policy; nothing loads from anywhere else.

Usage:
    dashboard = Dashboard(admin_token="...", server=a2a_server, network=network)
    ops = AgentServer(network=network, admin_token="...", dashboard=dashboard)
"""

from __future__ import annotations

import asyncio
import hmac
import ipaddress
import json
import logging
import re
import secrets
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from starlette.requests import Request
from starlette.responses import (
    FileResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from starlette.routing import BaseRoute, Route

from nexus_a2a.core.task_manager import TaskAlreadyDoneError, TaskNotFoundError
from nexus_a2a.dashboard.events import EventHub, state_of, task_summary
from nexus_a2a.dashboard.logbuffer import RingLogHandler

if TYPE_CHECKING:
    from nexus_a2a.core.a2a_server import A2AServer
    from nexus_a2a.core.task_manager import TaskManager
    from nexus_a2a.models.task import Part, Task
    from nexus_a2a.network import AgentNetwork

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
COOKIE_NAME = "nexus_ui_session"
CSRF_HEADER = "X-CSRF-Token"
ADMIN_HEADER = "X-Admin-Token"

_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; "
    "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; "
    "base-uri 'none'; form-action 'self'"
)
_SECURITY_HEADERS = {
    "Content-Security-Policy": _CSP,
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}
_TERMINAL = {"completed", "failed", "cancelled"}
_MAX_LOGIN_BODY = 4096
_MAX_JSON_BODY = 64 * 1024
_HEARTBEAT_SEC = 15.0
_MAX_REPLAY = 500


@dataclass
class DashboardConfig:
    """
    Dashboard settings ([dashboard] in nexus.toml).

    Fields:
        enabled:          Serve the dashboard at all.
        allow_replay:     Allow retrying tasks and replaying the DLQ from the UI.
        show_content:     Show message and artifact content. Off by default:
                          it often holds prompts or personal data.
        allow_remote:     Serve non-loopback clients. Off by default; behind a
                          proxy, the proxy is the client. Remote access should
                          be over HTTPS: the session cookie is then marked
                          Secure.
        allowed_hosts:    Host header values accepted when allow_remote is on
                          (e.g. "ops.example.com"). Empty = any. Without
                          allow_remote only localhost / 127.0.0.1 / [::1] are
                          accepted, which stops DNS-rebinding pages.
        session_ttl_sec:  Idle lifetime of a login session.
        login_attempts_per_minute: Per client address.
    """

    enabled: bool = True
    allow_replay: bool = True
    show_content: bool = False
    allow_remote: bool = False
    allowed_hosts: tuple[str, ...] = ()
    session_ttl_sec: float = 3600.0
    login_attempts_per_minute: int = 5


@dataclass
class _Session:
    csrf: str
    expires: float


class _ApiError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


Handler = Callable[[Request], Awaitable[Response]]


class Dashboard:
    """
    Browser dashboard for one agent process.

    Args:
        admin_token:  The ops server's admin token. Without one the dashboard
                      stays off (its routes answer 404).
        server:       The A2AServer being watched: agent card, security and
                      webhook events, retry and cancel.
        network:      AgentNetwork for registered peers, traces and the DLQ.
        task_manager: Defaults to server.tasks, then network.task_manager.
        config:       DashboardConfig.
    """

    def __init__(
        self,
        admin_token: str | None,
        server: A2AServer | None = None,
        network: AgentNetwork | None = None,
        task_manager: TaskManager | None = None,
        config: DashboardConfig | None = None,
        log_capacity: int = 5000,
    ) -> None:
        self.config = config or DashboardConfig()
        self._admin_token = admin_token or None
        self.server = server
        self.network = network
        manager = task_manager
        if manager is None and server is not None:
            manager = server.tasks
        if manager is None and network is not None:
            manager = network.task_manager
        if manager is None:
            raise ValueError("Dashboard needs a server, a network or a task_manager.")
        self.tasks: TaskManager = manager

        self.hub = EventHub()
        self.log_handler = RingLogHandler(
            capacity=log_capacity,
            hub=self.hub,
            hide_content=not self.config.show_content,
        )
        # Only the files shipped in static/, by exact name. Never build a
        # path from request input: on Windows "D:x" escapes any directory.
        self._static_files = (
            frozenset(p.name for p in STATIC_DIR.iterdir() if p.is_file())
            if STATIC_DIR.is_dir()
            else frozenset()
        )
        self._sessions: dict[str, _Session] = {}
        self._login_attempts: dict[str, deque[float]] = {}
        self._seed_lock = asyncio.Lock()
        self._attached = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    @property
    def active(self) -> bool:
        """True when the dashboard will serve requests."""
        return self.config.enabled and self._admin_token is not None

    def attach(self) -> None:
        """Start observing: task events, server events and nexus_a2a logs."""
        if self._attached or not self.active:
            return
        self.tasks.add_listener(self.hub.on_task_event)
        if self.server is not None:
            self.server.event_hub = self.hub
        logging.getLogger("nexus_a2a").addHandler(self.log_handler)
        self._attached = True
        logger.info("Dashboard enabled at /ui")

    def detach(self) -> None:
        """Stop observing. Safe to call twice."""
        if not self._attached:
            return
        self.tasks.remove_listener(self.hub.on_task_event)
        if self.server is not None and self.server.event_hub is self.hub:
            self.server.event_hub = None
        logging.getLogger("nexus_a2a").removeHandler(self.log_handler)
        self._attached = False

    def record_admin_denied(self, request: Request) -> None:
        """Called by AgentServer when an admin endpoint refuses a request."""
        if self.active:
            self.hub.record_security(
                "admin_denied",
                f"admin endpoint {request.url.path} refused",
                status=403,
                client=_client(request),
            )

    # ── Routes ────────────────────────────────────────────────────────────────

    def routes(self) -> list[BaseRoute]:
        """Starlette routes to add to the ops server's app."""
        api = [
            ("/ui/api/session", self._session, ["GET"], False, False),
            ("/ui/api/login", self._login, ["POST"], False, False),
            ("/ui/api/logout", self._logout, ["POST"], True, True),
            ("/ui/api/overview", self._overview, ["GET"], True, False),
            ("/ui/api/agents", self._agents, ["GET"], True, False),
            ("/ui/api/agents/check", self._agents_check, ["POST"], True, True),
            ("/ui/api/tasks", self._task_list, ["GET"], True, False),
            ("/ui/api/tasks/{task_id}", self._task_detail, ["GET"], True, False),
            ("/ui/api/tasks/{task_id}/cancel", self._task_cancel, ["POST"], True, True),
            ("/ui/api/tasks/{task_id}/retry", self._task_retry, ["POST"], True, True),
            ("/ui/api/issues", self._issues, ["GET"], True, False),
            ("/ui/api/dlq/replay", self._dlq_replay, ["POST"], True, True),
            ("/ui/api/logs", self._logs, ["GET"], True, False),
            ("/ui/api/stream", self._stream, ["GET"], True, False),
        ]
        routes: list[BaseRoute] = [
            Route(path, self._wrap(handler, auth, csrf), methods=methods)
            for path, handler, methods, auth, csrf in api
        ]
        routes += [
            Route("/ui", self._wrap(self._index, False, False), methods=["GET"]),
            Route("/ui/", self._wrap(self._index, False, False), methods=["GET"]),
            Route(
                "/ui/static/{name}",
                self._wrap(self._static, False, False),
                methods=["GET"],
            ),
        ]
        return routes

    def _wrap(self, handler: Handler, auth: bool, csrf: bool) -> Handler:
        async def endpoint(request: Request) -> Response:
            try:
                if not self.active:
                    raise _ApiError(404, "Not Found")
                self._check_client(request)
                if auth:
                    via_cookie = self._authenticate(request)
                    if csrf and via_cookie:
                        self._check_csrf(request)
                response = await handler(request)
            except _ApiError as exc:
                response = JSONResponse({"error": exc.message}, status_code=exc.status)
            except Exception:  # noqa: BLE001 - never leak internals to the UI
                logger.exception("Dashboard: unhandled error on %s", request.url.path)
                response = JSONResponse({"error": "Internal error"}, status_code=500)
            for key, value in _SECURITY_HEADERS.items():
                response.headers.setdefault(key, value)
            return response

        return endpoint

    # ── Access control ────────────────────────────────────────────────────────

    def _check_client(self, request: Request) -> None:
        host_name = _host_name(request)
        if self.config.allow_remote:
            allowed = {h.lower() for h in self.config.allowed_hosts}
            if allowed and host_name not in allowed:
                raise _ApiError(403, "Host not allowed.")
            return
        peer = _client(request)
        if peer is None or not _is_loopback_ip(peer):
            raise _ApiError(
                403,
                "The dashboard only serves this machine. Set "
                "[dashboard] allow_remote = true to serve other clients.",
            )
        # A DNS-rebinding page is a loopback client too, but its requests
        # carry the attacker's domain as Host.
        if not _is_loopback_name(host_name):
            raise _ApiError(403, "Host not allowed.")

    def _authenticate(self, request: Request) -> bool:
        """Return True for cookie auth, False for header-token auth."""
        expected = self._admin_token
        if not expected:
            raise _ApiError(404, "Not Found")
        provided = request.headers.get(ADMIN_HEADER)
        if provided is not None:
            client = _client(request) or "unknown"
            self._refuse_if_locked_out(client)
            if hmac.compare_digest(provided.encode(), expected.encode()):
                return False
            self._note_failed_login(client)
            self.hub.record_security(
                "auth_failed",
                "dashboard API call with a wrong X-Admin-Token",
                401,
                client=client,
            )
            raise _ApiError(401, "Invalid admin token.")
        if self._current_session(request) is None:
            raise _ApiError(401, "Not logged in.")
        return True

    def _current_session(self, request: Request) -> _Session | None:
        sid = request.cookies.get(COOKIE_NAME)
        if not sid:
            return None
        session = self._sessions.get(sid)
        now = time.time()
        if session is None or session.expires < now:
            self._sessions.pop(sid, None)
            return None
        session.expires = now + self.config.session_ttl_sec
        return session

    def _check_csrf(self, request: Request) -> None:
        session = self._current_session(request)
        sent = request.headers.get(CSRF_HEADER, "")
        if session is None or not hmac.compare_digest(
            sent.encode(), session.csrf.encode()
        ):
            raise _ApiError(403, "Missing or invalid CSRF token.")
        self._check_origin(request)

    @staticmethod
    def _check_origin(request: Request) -> None:
        origin = request.headers.get("origin")
        if origin is not None and not _same_origin(origin, request):
            raise _ApiError(403, "Cross-origin request refused.")

    def _session_alive(self, sid: str) -> bool:
        session = self._sessions.get(sid)
        return session is not None and session.expires >= time.time()

    def _recent_failures(self, client: str) -> deque[float]:
        now = time.time()
        failures = self._login_attempts.setdefault(client, deque())
        while failures and failures[0] < now - 60:
            failures.popleft()
        return failures

    def _refuse_if_locked_out(self, client: str) -> None:
        """Only FAILED attempts count, so a correct token is never locked out
        by its own successful logins."""
        if len(self._recent_failures(client)) >= self.config.login_attempts_per_minute:
            raise _ApiError(429, "Too many failed attempts. Try again in a minute.")

    def _note_failed_login(self, client: str) -> None:
        self._recent_failures(client).append(time.time())
        if len(self._login_attempts) > 1000:
            for key in list(self._login_attempts)[:500]:
                self._login_attempts.pop(key, None)

    # ── Session endpoints ─────────────────────────────────────────────────────

    async def _session(self, request: Request) -> Response:
        session = self._current_session(request)
        return JSONResponse(
            {
                "authenticated": session is not None,
                "csrf": session.csrf if session else None,
                "agent": self._agent_name(),
                "version": _nexus_version(),
                "allow_replay": self.config.allow_replay,
                "show_content": self.config.show_content,
            }
        )

    async def _login(self, request: Request) -> Response:
        client = _client(request) or "unknown"
        self._check_origin(request)
        self._refuse_if_locked_out(client)
        body = await _json_body(request, _MAX_LOGIN_BODY)
        token = body.get("token") if isinstance(body, dict) else None
        expected = self._admin_token or ""
        if (
            not expected
            or not isinstance(token, str)
            or not hmac.compare_digest(token.encode(), expected.encode())
        ):
            self._note_failed_login(client)
            self.hub.record_security(
                "login_failed", "dashboard login with a wrong token", 401, client=client
            )
            raise _ApiError(401, "Wrong token.")

        sid = secrets.token_urlsafe(32)
        csrf = secrets.token_urlsafe(32)
        self._sessions[sid] = _Session(
            csrf=csrf, expires=time.time() + self.config.session_ttl_sec
        )
        if len(self._sessions) > 100:
            oldest = sorted(self._sessions.items(), key=lambda kv: kv[1].expires)
            for old_sid, _ in oldest[: len(self._sessions) - 100]:
                self._sessions.pop(old_sid, None)
        logger.info("Dashboard: login from %s", client)
        response = JSONResponse({"csrf": csrf})
        response.set_cookie(
            COOKIE_NAME,
            sid,
            max_age=int(self.config.session_ttl_sec),
            path="/ui",
            httponly=True,
            samesite="strict",
            # Remote access must be HTTPS; a cookie that could travel in
            # clear text over the network is never issued.
            secure=request.url.scheme == "https" or self.config.allow_remote,
        )
        return response

    async def _logout(self, request: Request) -> Response:
        sid = request.cookies.get(COOKIE_NAME)
        if sid:
            self._sessions.pop(sid, None)
        response = Response(status_code=204)
        response.delete_cookie(COOKIE_NAME, path="/ui")
        return response

    # ── Overview ──────────────────────────────────────────────────────────────

    async def _overview(self, request: Request) -> Response:
        await self._ensure_seeded()
        counts = self.hub.counts()
        completed_5m, failed_5m = self.hub.window(300)
        finished = completed_5m + failed_5m
        dlq = self._dlq()
        store_ok = "ok"
        try:
            await asyncio.wait_for(self.tasks.ping(), timeout=2.0)
        except Exception:  # noqa: BLE001 - reported, not raised
            store_ok = "error"
        return JSONResponse(
            {
                "agent": self._agent_brief(),
                "nexus_version": _nexus_version(),
                "uptime_seconds": round(time.time() - self.hub.started_at, 1),
                "health": {"store": store_ok, "registry": self._registry_health()},
                "tasks": {
                    "by_state": counts,
                    "in_flight": sum(
                        counts.get(s, 0)
                        for s in ("submitted", "working", "input_required")
                    ),
                    "total_seen": self.hub.total_seen,
                },
                "throughput": {
                    "per_minute": self.hub.throughput(),
                    "completed_last_5m": completed_5m,
                    "failed_last_5m": failed_5m,
                },
                "latency_ms": self.hub.latency(),
                "error_rate": round(failed_5m / finished, 4) if finished else None,
                "dlq": {
                    "pending": dlq.pending_count() if dlq else 0,
                    "total": dlq.count() if dlq else 0,
                },
                "security_events_last_hour": self.hub.security_events_since(3600),
                "webhooks": {
                    "delivered": self.hub.webhooks_delivered,
                    "failed": self.hub.webhooks_failed,
                },
            }
        )

    # ── Agents ────────────────────────────────────────────────────────────────

    async def _agents(self, request: Request) -> Response:
        own: dict[str, Any] | None = None
        if self.server is not None:
            card = self.server.card
            caps = card.capabilities
            own = {
                "name": card.name,
                "description": card.description,
                "url": self.server.public_url,
                "version": card.version,
                "skills": [
                    {"id": s.id, "name": s.name, "description": s.description}
                    for s in card.skills
                ],
                "capabilities": {
                    "streaming": bool(caps.streaming),
                    "push_notifications": bool(caps.push_notifications),
                    "multi_turn": bool(caps.multi_turn),
                },
                "authentication": self.server._advertised_auth(),
            }
        peers = []
        if self.network is not None:
            now = time.monotonic()
            for entry in self.network.registry.entries():
                peers.append(
                    {
                        "name": entry.card.name,
                        "url": entry.url or str(entry.card.url),
                        "healthy": entry.healthy,
                        "last_seen_seconds_ago": round(now - entry.last_seen_at, 1),
                        "skills": [s.id for s in entry.card.skills],
                    }
                )
        return JSONResponse({"self": own, "peers": peers})

    async def _agents_check(self, request: Request) -> Response:
        if self.network is None:
            return JSONResponse({"results": {}})
        results = await self.network.registry.check_all_health()
        return JSONResponse({"results": results})

    # ── Tasks ─────────────────────────────────────────────────────────────────

    async def _task_list(self, request: Request) -> Response:
        await self._ensure_seeded()
        params = request.query_params
        state = params.get("state") or None
        skill = params.get("skill") or None
        query = (params.get("q") or "").strip().lower()
        limit = _int_param(params.get("limit"), 100, 1, 500)

        def matches(summary: dict[str, Any]) -> bool:
            if state and summary["state"] != state:
                return False
            if skill and summary["skill_id"] != skill:
                return False
            if query:
                haystack = " ".join(
                    str(summary.get(k) or "")
                    for k in ("id", "owner", "skill_id", "error")
                ).lower()
                if query not in haystack:
                    return False
            return True

        # Filter on what the viewer may see: searching hidden error text would
        # let a viewer probe content that show_content=false hides.
        rows = [
            s for s in map(self._public_summary, self.hub.recent_tasks()) if matches(s)
        ]
        return JSONResponse({"tasks": rows[:limit], "total": len(rows)})

    async def _task_detail(self, request: Request) -> Response:
        task = await self._get_task(request.path_params["task_id"])
        summary = self._public_summary(task_summary(task))
        timeline = self.hub.timeline(task.id) or [
            {"at": summary["created_at"], "event": "created"},
        ]
        return JSONResponse(
            {
                "task": summary,
                "history": [
                    {"role": str(m.role), "parts": [self._part(p) for p in m.parts]}
                    for m in task.history
                ],
                "artifacts": [
                    {
                        "name": getattr(a, "name", None),
                        "parts": [self._part(p) for p in a.parts],
                    }
                    for a in task.artifacts
                ],
                "timeline": timeline,
                "trace": _trace_for(task.id),
                "actions": {
                    "can_cancel": state_of(task) not in _TERMINAL,
                    "can_retry": self._can_retry(task),
                },
            }
        )

    async def _task_cancel(self, request: Request) -> Response:
        task_id = request.path_params["task_id"]
        try:
            if self.server is not None:
                task = await self.server.cancel_task(task_id)
            else:
                task = await self.tasks.cancel(task_id)
        except TaskNotFoundError as exc:
            raise _ApiError(404, f"Task not found: {task_id}") from exc
        except TaskAlreadyDoneError as exc:
            raise _ApiError(409, str(exc)) from exc
        logger.warning("Dashboard: task %s cancelled by operator", task_id)
        return JSONResponse({"task": self._public_summary(task_summary(task))})

    async def _task_retry(self, request: Request) -> Response:
        if not self.config.allow_replay:
            raise _ApiError(
                403, "Retry is disabled ([dashboard] allow_replay = false)."
            )
        if self.server is None:
            raise _ApiError(409, "Retry needs the agent's A2AServer.")
        task_id = request.path_params["task_id"]
        original = await self._get_task(task_id)
        if not self._can_retry(original):
            raise _ApiError(409, "This task cannot be retried.")
        try:
            new_task = await self.server.retry_task(task_id)
        except ValueError as exc:
            raise _ApiError(409, str(exc)) from exc
        self.hub.note_event(task_id, "retried")
        logger.warning(
            "Dashboard: task %s retried by operator as %s", task_id, new_task.id
        )
        return JSONResponse(
            {"task": self._public_summary(task_summary(new_task))}, status_code=202
        )

    # ── Issues ────────────────────────────────────────────────────────────────

    async def _issues(self, request: Request) -> Response:
        await self._ensure_seeded()
        failed = [
            self._public_summary(s)
            for s in self.hub.recent_tasks()
            if s["state"] == "failed"
        ][:100]
        dlq = self._dlq()
        entries = [e.to_dict() for e in dlq.all_entries()] if dlq else []
        entries.sort(key=lambda e: e.get("failed_at") or 0, reverse=True)
        if not self.config.show_content:
            entries = [{**e, "error": hide_error(e.get("error"))} for e in entries]
        return JSONResponse(
            {
                "failed_tasks": failed,
                "dlq": entries,
                "security_events": self.hub.security_events()[:200],
                "webhook_failures": self.hub.webhook_failures()[:100],
            }
        )

    async def _dlq_replay(self, request: Request) -> Response:
        if not self.config.allow_replay:
            raise _ApiError(
                403, "Replay is disabled ([dashboard] allow_replay = false)."
            )
        dlq = self._dlq()
        if dlq is None:
            raise _ApiError(409, "No dead letter queue is configured.")
        body = await _json_body(request, _MAX_JSON_BODY)
        if not isinstance(body, dict):
            raise _ApiError(400, "Body must be a JSON object.")
        if body.get("all") is True:
            ids = [e.task_id for e in dlq.pending_entries()][:_MAX_REPLAY]
        else:
            raw_ids = body.get("task_ids")
            if not isinstance(raw_ids, list) or not all(
                isinstance(i, str) for i in raw_ids
            ):
                raise _ApiError(400, "Send {'task_ids': [...]} or {'all': true}.")
            # Each entry at most once per request.
            ids = list(dict.fromkeys(raw_ids))[:_MAX_REPLAY]

        results = []
        for task_id in ids:
            entry = dlq.get_entry(task_id)
            if entry is not None and entry.replayed:
                results.append(
                    {
                        "task_id": task_id,
                        "success": False,
                        "error": "already replayed",
                        "new_task_id": None,
                    }
                )
                continue
            try:
                outcome = await dlq.replay(task_id)
                results.append(
                    {
                        "task_id": task_id,
                        "success": bool(outcome.succeeded),
                        "error": outcome.error,
                        "new_task_id": outcome.new_task.id
                        if outcome.new_task
                        else None,
                    }
                )
            except KeyError:
                results.append(
                    {
                        "task_id": task_id,
                        "success": False,
                        "error": "not in the dead letter queue",
                        "new_task_id": None,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - reported per item
                logger.exception("Dashboard: DLQ replay of %s failed", task_id)
                results.append(
                    {
                        "task_id": task_id,
                        "success": False,
                        "error": type(exc).__name__,
                        "new_task_id": None,
                    }
                )
        logger.warning("Dashboard: operator replayed %d DLQ task(s)", len(ids))
        return JSONResponse({"results": results})

    # ── Logs and live stream ──────────────────────────────────────────────────

    async def _logs(self, request: Request) -> Response:
        params = request.query_params
        level_name = (params.get("level") or "DEBUG").upper()
        level = logging.getLevelName(level_name)
        if not isinstance(level, int):
            level = logging.DEBUG
        lines = self.log_handler.lines(
            min_level=level,
            query=params.get("q") or "",
            limit=_int_param(params.get("limit"), 200, 1, 1000),
        )
        return JSONResponse({"logs": lines})

    async def _stream(self, request: Request) -> Response:
        queue = self.hub.subscribe()
        if queue is None:
            raise _ApiError(503, "Too many live streams open.")
        # A cookie-authenticated stream ends when its session does (logout or
        # expiry); one opened with X-Admin-Token is bound to no session.
        sid = (
            request.cookies.get(COOKIE_NAME)
            if request.headers.get(ADMIN_HEADER) is None
            else None
        )

        async def events() -> AsyncIterator[str]:
            try:
                yield ": connected\n\n"
                while True:
                    try:
                        item = await asyncio.wait_for(queue.get(), _HEARTBEAT_SEC)
                    except TimeoutError:
                        item = ("ping", {})
                    if sid is not None and not self._session_alive(sid):
                        return
                    if item is None:  # dropped as too slow
                        return
                    kind, data = item
                    if kind == "ping":
                        yield ": ping\n\n"
                        continue
                    if kind == "task":
                        data = {**data, "task": self._public_summary(data["task"])}
                    yield f"event: {kind}\ndata: {json.dumps(data)}\n\n"
            finally:
                self.hub.unsubscribe(queue)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )

    # ── Static files ──────────────────────────────────────────────────────────

    async def _index(self, request: Request) -> Response:
        return self._file("index.html")

    async def _static(self, request: Request) -> Response:
        return self._file(request.path_params["name"])

    def _file(self, name: str) -> Response:
        if name not in self._static_files:
            raise _ApiError(404, "Not Found")
        return FileResponse(STATIC_DIR / name)

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _ensure_seeded(self) -> None:
        if self.hub.seeded:
            return
        async with self._seed_lock:
            if self.hub.seeded:
                return
            try:
                tasks = await self.tasks.list_all()
            except Exception:  # noqa: BLE001 - live events still work
                logger.warning(
                    "Dashboard: could not load existing tasks", exc_info=True
                )
                tasks = []
            self.hub.seed(tasks)

    async def _get_task(self, task_id: str) -> Task:
        try:
            return await self.tasks.get(task_id)
        except TaskNotFoundError as exc:
            raise _ApiError(404, f"Task not found: {task_id}") from exc

    def _can_retry(self, task: Task) -> bool:
        return (
            self.server is not None
            and self.config.allow_replay
            and state_of(task) in _TERMINAL
            and any(m.role == "user" for m in task.history)
        )

    def _public_summary(self, summary: dict[str, Any]) -> dict[str, Any]:
        if self.config.show_content:
            return summary
        return {**summary, "preview": None, "error": hide_error(summary.get("error"))}

    def _part(self, part: Part) -> dict[str, Any]:
        content = part.content
        length = (
            len(content)
            if isinstance(content, str)
            else len(json.dumps(content, default=str))
        )
        kind = part.type.value if hasattr(part.type, "value") else str(part.type)
        if self.config.show_content:
            return {
                "type": kind,
                "content": content,
                "length": length,
                "redacted": False,
            }
        return {"type": kind, "content": None, "length": length, "redacted": True}

    def _dlq(self) -> Any:
        if self.network is None:
            return None
        return self.network.dead_letter_queue

    def _registry_health(self) -> str:
        if self.network is None:
            return "no_agents"
        entries = self.network.registry.entries()
        if not entries:
            return "no_agents"
        return "ok" if all(e.healthy for e in entries) else "degraded"

    def _agent_name(self) -> str:
        return self.server.card.name if self.server is not None else "nexus-a2a"

    def _agent_brief(self) -> dict[str, Any]:
        if self.server is None:
            return {"name": "nexus-a2a", "url": None, "version": None}
        return {
            "name": self.server.card.name,
            "url": self.server.public_url,
            "version": self.server.card.version,
        }


# ── Module helpers ────────────────────────────────────────────────────────────


def _client(request: Request) -> str | None:
    return request.client.host if request.client else None


def _is_loopback_ip(peer: str) -> bool:
    """The TCP peer is this machine. Only real IPs count."""
    try:
        return ipaddress.ip_address(peer).is_loopback
    except ValueError:
        return False


def _is_loopback_name(host: str) -> bool:
    """A Host header that names this machine."""
    if host == "localhost" or host.endswith(".localhost"):
        return True
    return _is_loopback_ip(host)


def _host_name(request: Request) -> str:
    """Hostname from the Host header, lower-case, without port or brackets."""
    raw = request.headers.get("host", "")
    try:
        return (urlsplit("//" + raw).hostname or "").lower()
    except ValueError:
        return ""


_ERROR_TYPE = re.compile(r"^([A-Za-z_][\w.]*)(?::|$)")
_PUBLIC_ERROR_PREFIXES = ("Task auto-failed by watchdog",)


def hide_error(error: str | None) -> str | None:
    """
    An error message with content removed: agent exceptions often quote the
    input. Keeps the exception type, and library messages that never carry
    content (watchdog timeouts).
    """
    if not error:
        return error
    if error.startswith(_PUBLIC_ERROR_PREFIXES):
        return error
    match = _ERROR_TYPE.match(error)
    if match:
        return f"{match.group(1)} (details hidden: show_content is off)"
    return "(details hidden: show_content is off)"


def _same_origin(origin: str, request: Request) -> bool:
    try:
        parts = urlsplit(origin)
    except ValueError:
        return False
    host = request.headers.get("host", "")
    return parts.netloc.lower() == host.lower()


async def _json_body(request: Request, limit: int) -> Any:
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise _ApiError(413, "Request body too large.")
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > limit:
            raise _ApiError(413, "Request body too large.")
    if not body:
        return {}
    try:
        return json.loads(body)
    except ValueError as exc:
        raise _ApiError(400, "Body is not valid JSON.") from exc


def _int_param(raw: str | None, default: int, low: int, high: int) -> int:
    try:
        value = int(raw) if raw is not None else default
    except ValueError:
        value = default
    return max(low, min(high, value))


def _trace_for(task_id: str) -> dict[str, Any] | None:
    from nexus_a2a.transport.tracing import default_store

    trace = default_store.find_by_task_id(task_id)
    if trace is None:
        return None
    return {
        "trace_id": getattr(trace, "trace_id", None),
        "hops": [
            {
                "url": getattr(span, "agent_url", "unknown"),
                "duration_ms": getattr(span, "duration_ms", None),
                "status": getattr(span, "status", "unknown"),
                "error": getattr(span, "error", None),
            }
            for span in getattr(trace, "spans", [])
        ],
    }


def _nexus_version() -> str:
    from nexus_a2a import __version__

    return __version__


__all__ = ["Dashboard", "DashboardConfig"]
