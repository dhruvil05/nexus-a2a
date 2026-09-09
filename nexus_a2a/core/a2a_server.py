"""
nexus_a2a/core/a2a_server.py

A2AServer — the inbound half of the A2A protocol.

Until now nexus-a2a could CALL other agents (A2AHttpClient) but could not BE
one: nothing served /.well-known/agent-card.json and nothing handled the
JSON-RPC methods, so two nexus-a2a agents could not actually talk to each
other. A2AServer closes that loop.

It takes a class decorated with @agent and serves it:

    GET  /.well-known/agent-card.json  — discovery; the card @agent built
    POST /                             — JSON-RPC 2.0 endpoint
    GET  /health                       — liveness probe
    GET  /ready                        — readiness probe

JSON-RPC methods (matching A2AHttpClient exactly):

    message/send   params: {message, skillId?, contextId?}  -> Task
    tasks/get      params: {taskId}                          -> Task
    tasks/cancel   params: {taskId}                          -> Task

Error model — the split is deliberate:

    HTTP status codes for TRANSPORT-level rejections that happen before
    dispatch (401 auth, 403 trust, 413 too large, 429 rate limit, 400
    malformed). Proxies, WAFs and dashboards can see these, and the client
    treats 4xx as non-retryable, which is correct — retrying a rejected
    credential never helps.

    JSON-RPC error objects for APPLICATION-level failures after dispatch
    (-32601 method not found, -32602 invalid params, -32001 task not found).
    These reach the caller as RemoteAgentError with the code intact.

An agent that raises is NOT an RPC error: the task is recorded as FAILED and
returned as a normal result, so the caller can inspect task.error and the
Dead Letter Queue can capture it.

Usage:
    from nexus_a2a import agent, A2AServer, Task

    @agent(name="Summariser", description="Summarises text.",
           skills=[{"id": "summarise", "name": "Summarise",
                    "description": "One-paragraph summary."}],
           url="http://localhost:8001")
    class SummaryAgent:
        async def run(self, task: Task) -> str:
            return f"Summary: {task.latest_message().text()[:100]}"

    server = A2AServer(SummaryAgent, port=8001)
    await server.start()

Ops endpoints (/metrics, /info, /dlq, traces) live in AgentServer, which is
designed to run on a separate port — the standard app-port / admin-port split.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from typing import TYPE_CHECKING, Any

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from nexus_a2a.core.task_manager import (
    TaskAlreadyDoneError,
    TaskManager,
    TaskNotFoundError,
)
from nexus_a2a.decorators import get_card
from nexus_a2a.models.agent import AgentCard
from nexus_a2a.models.task import Artifact, Message, Part, PartType, Task, TaskState
from nexus_a2a.security.middleware import SecurityMiddleware, http_status_for

if TYPE_CHECKING:
    from nexus_a2a.storage.task_store import AbstractTaskStore

logger = logging.getLogger(__name__)

AGENT_CARD_PATH = "/.well-known/agent-card.json"

# JSON-RPC method names — must match transport/http_client.py.
METHOD_SEND = "message/send"
METHOD_GET = "tasks/get"
METHOD_CANCEL = "tasks/cancel"

# JSON-RPC error codes. -32000..-32099 is the server-defined range.
ERR_PARSE = -32700
ERR_INVALID_REQUEST = -32600
ERR_METHOD_NOT_FOUND = -32601
ERR_INVALID_PARAMS = -32602
ERR_INTERNAL = -32603
ERR_TASK_NOT_FOUND = -32001


# ── Exceptions ────────────────────────────────────────────────────────────────


class A2AServerError(Exception):
    """Base class for A2AServer configuration errors."""


class InvalidAgentError(A2AServerError):
    """Raised when the object passed to A2AServer is not a usable agent."""


# ── JSON-RPC helpers ──────────────────────────────────────────────────────────


def _rpc_result(rpc_id: Any, result: dict[str, Any]) -> JSONResponse:
    """Build a JSON-RPC 2.0 success envelope."""
    return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "result": result})


def _rpc_error(
    rpc_id: Any,
    code: int,
    message: str,
    status_code: int = 200,
) -> JSONResponse:
    """
    Build a JSON-RPC 2.0 error envelope.

    Defaults to HTTP 200 because a JSON-RPC error is a well-formed protocol
    response, not a transport failure — the client unwraps it into
    RemoteAgentError with the code preserved.
    """
    return JSONResponse(
        {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": code, "message": message}},
        status_code=status_code,
    )


# ── A2AServer ─────────────────────────────────────────────────────────────────


class A2AServer:
    """
    Serves one @agent-decorated class over the A2A protocol.

    Args:
        agent:        A class decorated with @agent, or an instance of one.
                      A class is instantiated once with no arguments.
        host:         Bind host. Defaults to the agent card's URL host, then
                      '0.0.0.0'.
        port:         Bind port. Defaults to the agent card's URL port, then 8000.
        task_manager: Task lifecycle + persistence. Defaults to an in-memory
                      TaskManager.
        store:        Task store, used only when task_manager is not given.
        security:     SecurityMiddleware to enforce on every inbound RPC.
                      None = no enforcement (the default; harden explicitly).
        public_url:   URL advertised in the served agent card. Defaults to the
                      card's own url.
        log_level:    Uvicorn log level. Defaults to 'warning' (quiet).

    Raises:
        InvalidAgentError: agent is not @agent-decorated or has no async run().
    """

    def __init__(
        self,
        agent: Any,
        host: str | None = None,
        port: int | None = None,
        task_manager: TaskManager | None = None,
        store: AbstractTaskStore | None = None,
        security: SecurityMiddleware | None = None,
        public_url: str | None = None,
        log_level: str = "warning",
    ) -> None:
        self._agent_instance, self._card = self._resolve_agent(agent)

        card_host, card_port = _split_url(str(self._card.url))
        self.host = host or card_host or "0.0.0.0"
        self.port = port if port is not None else (card_port or 8000)

        self.tasks = task_manager or TaskManager(store=store)
        self.security = security or SecurityMiddleware()
        self.public_url = (public_url or str(self._card.url)).rstrip("/")
        self.log_level = log_level

        self._started_at: float | None = None
        self._server: uvicorn.Server | None = None
        self._serve_task: asyncio.Task[None] | None = None

        self._app = self._build_app()

    # ── Agent resolution ──────────────────────────────────────────────────────

    @staticmethod
    def _resolve_agent(agent: Any) -> tuple[Any, AgentCard]:
        """
        Accept either an @agent class or an instance of one, and return
        (instance, card).

        Raises:
            InvalidAgentError: Not decorated, or missing an async run().
        """
        target_cls = agent if inspect.isclass(agent) else type(agent)

        try:
            card = get_card(target_cls)
        except TypeError as exc:
            raise InvalidAgentError(
                f"'{target_cls.__name__}' is not decorated with @agent. "
                "Apply @agent to the class before serving it."
            ) from exc

        instance = agent() if inspect.isclass(agent) else agent

        run = getattr(instance, "run", None)
        if run is None or not inspect.iscoroutinefunction(run):
            raise InvalidAgentError(
                f"'{target_cls.__name__}' must define 'async def run(self, task)'."
            )

        return instance, card

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def card(self) -> AgentCard:
        """The AgentCard this server advertises."""
        return self._card

    @property
    def app(self) -> Starlette:
        """The underlying ASGI app — mount it yourself if you prefer."""
        return self._app

    @property
    def uptime_seconds(self) -> float | None:
        """Seconds since start(), or None if not running."""
        if self._started_at is None:
            return None
        return time.monotonic() - self._started_at

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """
        Start serving in a background asyncio task and return immediately.

        Raises:
            RuntimeError: If the server is already running.
        """
        if self._serve_task is not None and not self._serve_task.done():
            raise RuntimeError(f"A2AServer for '{self._card.name}' already running.")

        config = uvicorn.Config(
            self._app,
            host=self.host,
            port=self.port,
            log_level=self.log_level,
            lifespan="off",
        )
        self._server = uvicorn.Server(config)
        self._serve_task = asyncio.create_task(self._server.serve())
        self._started_at = time.monotonic()

        await self._wait_until_started()

        logger.info(
            "A2AServer '%s' listening on %s:%d (security: %s)",
            self._card.name,
            self.host,
            self.port,
            self.security.summary() if self.security.enabled else "disabled",
        )

    async def stop(self) -> None:
        """Shut the server down cleanly. Safe to call when not running."""
        if self._server is not None:
            self._server.should_exit = True

        if self._serve_task is not None:
            try:
                await asyncio.wait_for(self._serve_task, timeout=10.0)
            except TimeoutError:
                self._serve_task.cancel()
            except asyncio.CancelledError:
                pass

        self._server = None
        self._serve_task = None
        self._started_at = None
        logger.info("A2AServer '%s' stopped.", self._card.name)

    async def __aenter__(self) -> A2AServer:
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop()

    async def _wait_until_started(self, timeout: float = 5.0) -> None:
        """Poll uvicorn until it reports started, so start() is not racy."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._server is not None and self._server.started:
                return
            if self._serve_task is not None and self._serve_task.done():
                # Surface a bind failure instead of hanging.
                self._serve_task.result()
                return
            await asyncio.sleep(0.01)
        logger.warning(
            "A2AServer '%s' did not report started within %.1fs.",
            self._card.name,
            timeout,
        )

    # ── ASGI app ──────────────────────────────────────────────────────────────

    def _build_app(self) -> Starlette:
        """Build the Starlette application."""

        async def agent_card(request: Request) -> Response:
            return await self._handle_agent_card(request)

        async def rpc(request: Request) -> Response:
            return await self._handle_rpc(request)

        async def health(request: Request) -> Response:
            return JSONResponse({"status": "ok", "agent": self._card.name})

        async def ready(request: Request) -> Response:
            return await self._handle_ready(request)

        return Starlette(
            routes=[
                Route(AGENT_CARD_PATH, agent_card),
                Route("/health", health),
                Route("/ready", ready),
                Route("/", rpc, methods=["POST"]),
            ],
        )

    # ── Handlers ──────────────────────────────────────────────────────────────

    async def _handle_agent_card(self, request: Request) -> Response:
        """GET /.well-known/agent-card.json — the discovery document."""
        data = self._card.to_well_known_dict()
        data["url"] = self.public_url
        return JSONResponse(data)

    async def _handle_ready(self, request: Request) -> Response:
        """GET /ready — 200 only when the task store answers."""
        try:
            await self.tasks.list_all()
        except Exception as exc:
            return JSONResponse(
                {"ready": False, "reason": f"task store unreachable: {exc}"},
                status_code=503,
            )
        return JSONResponse({"ready": True, "agent": self._card.name})

    async def _handle_rpc(self, request: Request) -> Response:
        """POST / — the JSON-RPC 2.0 endpoint."""
        raw = await request.body()

        # ── Security stage 1: size, before any parsing ────────────────────────
        try:
            self.security.check_size(raw)
        except Exception as exc:
            return self._security_refusal(exc)

        # ── Parse the envelope ────────────────────────────────────────────────
        try:
            body = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as exc:
            return _rpc_error(None, ERR_PARSE, f"Parse error: {exc}")

        if not isinstance(body, dict):
            return _rpc_error(None, ERR_INVALID_REQUEST, "Request must be an object.")

        rpc_id = body.get("id")
        method = body.get("method")
        params = body.get("params") or {}

        if not isinstance(method, str):
            return _rpc_error(rpc_id, ERR_INVALID_REQUEST, "Missing 'method'.")
        if not isinstance(params, dict):
            return _rpc_error(rpc_id, ERR_INVALID_PARAMS, "'params' must be an object.")

        # ── Security stages 2-4: who is calling, may they ────────────────────
        skill_id = params.get("skillId")
        try:
            await self.security.authorize(
                dict(request.headers),
                skill_id=skill_id if isinstance(skill_id, str) else None,
            )
        except Exception as exc:
            return self._security_refusal(exc)

        # ── Dispatch ──────────────────────────────────────────────────────────
        try:
            if method == METHOD_SEND:
                return await self._rpc_send(rpc_id, params)
            if method == METHOD_GET:
                return await self._rpc_get(rpc_id, params)
            if method == METHOD_CANCEL:
                return await self._rpc_cancel(rpc_id, params)
            return _rpc_error(
                rpc_id, ERR_METHOD_NOT_FOUND, f"Method not found: {method}"
            )
        except Exception as exc:  # noqa: BLE001 - last-resort RPC boundary
            logger.exception("A2AServer: unhandled error in %s", method)
            return _rpc_error(rpc_id, ERR_INTERNAL, f"Internal error: {exc}")

    # ── RPC methods ───────────────────────────────────────────────────────────

    async def _rpc_send(self, rpc_id: Any, params: dict[str, Any]) -> Response:
        """
        message/send — create a task, run the agent, return the finished Task.

        Runs the agent inline and returns the terminal Task, which is what
        A2AHttpClient.send_message() expects.
        """
        raw_message = params.get("message")
        if raw_message is None:
            return _rpc_error(rpc_id, ERR_INVALID_PARAMS, "Missing 'message' param.")

        # ── Security stage 5: payload shape and limits ────────────────────────
        try:
            message = self.security.validate_message(raw_message)
        except Exception as exc:
            status = http_status_for(exc)
            if status == 500:
                return _rpc_error(rpc_id, ERR_INVALID_PARAMS, f"Invalid message: {exc}")
            return self._security_refusal(exc)

        skill_id = params.get("skillId")
        context_id = params.get("contextId")

        task = await self.tasks.create(
            initial_message=message,
            skill_id=skill_id if isinstance(skill_id, str) else None,
            context_id=context_id if isinstance(context_id, str) else None,
        )
        task = await self.tasks.start(task.id)

        # An agent that raises is a FAILED task, not an RPC error: the caller
        # still gets a Task back and can inspect .error, and the DLQ can
        # capture it.
        try:
            output = await self._agent_instance.run(task)
        except Exception as exc:  # noqa: BLE001 - agent code is untrusted
            logger.exception("Agent '%s' raised while running task %s",
                             self._card.name, task.id)
            failed = await self.tasks.fail(task.id, f"{type(exc).__name__}: {exc}")
            return _rpc_result(rpc_id, _task_dict(failed))

        return _rpc_result(rpc_id, _task_dict(await self._finish(task, output)))

    async def _rpc_get(self, rpc_id: Any, params: dict[str, Any]) -> Response:
        """tasks/get — look up a task by id."""
        task_id = params.get("taskId") or params.get("id")
        if not isinstance(task_id, str):
            return _rpc_error(rpc_id, ERR_INVALID_PARAMS, "Missing 'taskId' param.")
        try:
            task = await self.tasks.get(task_id)
        except TaskNotFoundError:
            return _rpc_error(rpc_id, ERR_TASK_NOT_FOUND, f"Task not found: {task_id}")
        return _rpc_result(rpc_id, _task_dict(task))

    async def _rpc_cancel(self, rpc_id: Any, params: dict[str, Any]) -> Response:
        """tasks/cancel — move a task to CANCELLED."""
        task_id = params.get("taskId") or params.get("id")
        if not isinstance(task_id, str):
            return _rpc_error(rpc_id, ERR_INVALID_PARAMS, "Missing 'taskId' param.")
        try:
            task = await self.tasks.cancel(task_id)
        except TaskNotFoundError:
            return _rpc_error(rpc_id, ERR_TASK_NOT_FOUND, f"Task not found: {task_id}")
        except TaskAlreadyDoneError as exc:
            # Already terminal — report the real state rather than inventing
            # a cancelled task the caller would then act on.
            return _rpc_error(rpc_id, ERR_INVALID_REQUEST, str(exc))
        return _rpc_result(rpc_id, _task_dict(task))

    # ── Result handling ───────────────────────────────────────────────────────

    async def _finish(self, task: Task, output: Any) -> Task:
        """
        Turn whatever run() returned into a completed Task.

        Accepted return values:
            None          — completed with no artifact
            str           — one text Artifact
            Artifact      — used as-is
            Message       — recorded as the agent's reply
            AdapterResult — .error fails the task, else .to_artifact()
            dict / list   — one JSON Artifact
            anything else — str()-ified into a text Artifact
        """
        artifact, reply, error = _interpret_output(output)

        if error is not None:
            return await self.tasks.fail(task.id, error)

        return await self.tasks.complete(
            task.id,
            artifact=artifact,
            reply_message=reply,
        )

    # ── Security refusals ─────────────────────────────────────────────────────

    def _security_refusal(self, exc: Exception) -> Response:
        """
        Turn a security failure into an HTTP response.

        Transport-level refusals carry a real status code and a bare JSON body
        — deliberately not a JSON-RPC envelope, because the request never
        reached dispatch. The reason is included; the credential never is.
        """
        status = http_status_for(exc)
        if status == 500:
            logger.exception("A2AServer: unexpected error in security chain")
            return _rpc_error(None, ERR_INTERNAL, "Internal error", status_code=500)

        logger.warning(
            "A2AServer '%s' refused a request: HTTP %d — %s",
            self._card.name,
            status,
            exc,
        )
        headers: dict[str, str] = {}
        retry_after = getattr(exc, "retry_after", None)
        if status == 429 and isinstance(retry_after, (int, float)):
            headers["Retry-After"] = str(max(1, int(retry_after + 0.5)))

        return JSONResponse(
            {"error": type(exc).__name__, "detail": str(exc)},
            status_code=status,
            headers=headers,
        )


# ── Module helpers ────────────────────────────────────────────────────────────


def _task_dict(task: Task) -> dict[str, Any]:
    """Serialise a Task for the wire, JSON-safe."""
    return task.model_dump(mode="json")


def _split_url(url: str) -> tuple[str | None, int | None]:
    """Extract (host, port) from a URL, tolerating a missing port."""
    from urllib.parse import urlparse

    try:
        parsed = urlparse(url)
        return parsed.hostname, parsed.port
    except ValueError:
        return None, None


def _interpret_output(
    output: Any,
) -> tuple[Artifact | None, Message | None, str | None]:
    """
    Normalise an agent's return value into (artifact, reply_message, error).

    Kept as a free function so the mapping can be tested without a server.
    """
    if output is None:
        return None, None, None

    if isinstance(output, Artifact):
        return output, None, None

    if isinstance(output, Message):
        return None, output, None

    # AdapterResult, duck-typed so importing the adapters package stays optional.
    if hasattr(output, "to_artifact") and hasattr(output, "error"):
        if output.error:
            return None, None, str(output.error)
        return output.to_artifact(), None, None

    if isinstance(output, str):
        return (
            Artifact(name="result", parts=[Part(type=PartType.TEXT, content=output)]),
            None,
            None,
        )

    if isinstance(output, (dict, list)):
        return (
            Artifact(name="result", parts=[Part(type=PartType.JSON, content=output)]),
            None,
            None,
        )

    return (
        Artifact(name="result", parts=[Part(type=PartType.TEXT, content=str(output))]),
        None,
        None,
    )


__all__ = [
    "A2AServer",
    "A2AServerError",
    "InvalidAgentError",
    "AGENT_CARD_PATH",
    "TaskState",
]
