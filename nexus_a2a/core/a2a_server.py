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
    GET  /stream?taskId=...            — observe a task as SSE
    GET  /health                       — liveness probe
    GET  /ready                        — readiness probe

JSON-RPC methods (matching A2AHttpClient exactly):

    message/send   params: {message, skillId?, contextId?}  -> Task
    message/stream params: {message, skillId?, contextId?}  -> SSE stream
    tasks/get      params: {taskId}                          -> Task
    tasks/cancel   params: {taskId}                          -> Task

    tasks/pushNotificationConfig/set  params: {taskId, pushNotificationConfig}
    tasks/pushNotificationConfig/get  params: {taskId}

Push notifications:
    A caller that will not wait registers a webhook, either with the message
    that creates the task ('pushNotification' in the params) or later with
    tasks/pushNotificationConfig/set. The agent then POSTs task_completed,
    task_failed, task_cancelled and task_input_required to it, HMAC-signed when
    a signing secret is configured.

    The URL comes from the caller, so registering one is an SSRF primitive:
    URLs resolving to private, loopback or link-local addresses are refused
    unless WebhookConfig(allow_private_urls=True) says otherwise.

Streaming:
    An agent streams by writing run() as an async generator, or by adding a
    stream() method that is one — the contract CapabilityGuard already checks:

        @agent(name="Writer", description="...", streaming=True, url=...)
        class Writer:
            async def run(self, task):
                for word in ["hello", " ", "world"]:
                    yield word

    message/stream emits task_created, one artifact_chunk per yield, then
    task_status and done. A non-streaming agent is still served there as a
    single chunk, and a streaming agent still works over message/send (its
    chunks are folded into one result), so neither side has to care how the
    other is written.

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
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from nexus_a2a.core.input_handler import InputHandler
from nexus_a2a.core.task_manager import (
    TaskAlreadyDoneError,
    TaskManager,
    TaskNotFoundError,
)
from nexus_a2a.decorators import get_card
from nexus_a2a.models.agent import AgentCard
from nexus_a2a.models.task import (
    Artifact,
    Message,
    NeedsInput,
    Part,
    PartType,
    PushNotificationConfig,
    Task,
    TaskState,
)
from nexus_a2a.security.middleware import SecurityMiddleware, http_status_for
from nexus_a2a.transport.sse import SSEFormatter, StreamEventType
from nexus_a2a.transport.webhook import (
    WebhookConfig,
    WebhookDispatcher,
    WebhookUrlError,
    validate_webhook_url,
)

if TYPE_CHECKING:
    from nexus_a2a.storage.task_store import AbstractTaskStore

logger = logging.getLogger(__name__)

AGENT_CARD_PATH = "/.well-known/agent-card.json"

# JSON-RPC method names — must match transport/http_client.py.
METHOD_SEND = "message/send"
METHOD_STREAM = "message/stream"
METHOD_GET = "tasks/get"
METHOD_CANCEL = "tasks/cancel"
METHOD_PUSH_SET = "tasks/pushNotificationConfig/set"
METHOD_PUSH_GET = "tasks/pushNotificationConfig/get"

# Path the SSEStreamer client GETs to observe an existing task.
STREAM_PATH = "/stream"

# JSON-RPC error codes. -32000..-32099 is the server-defined range.
ERR_PARSE = -32700
ERR_INVALID_REQUEST = -32600
ERR_METHOD_NOT_FOUND = -32601
ERR_INVALID_PARAMS = -32602
ERR_INTERNAL = -32603
ERR_TASK_NOT_FOUND = -32001
ERR_TASK_NOT_WAITING = -32002
ERR_PUSH_NOT_SUPPORTED = -32003
ERR_PUSH_INVALID_URL = -32004


# ── Exceptions ────────────────────────────────────────────────────────────────


class A2AServerError(Exception):
    """Base class for A2AServer configuration errors."""


class InvalidAgentError(A2AServerError):
    """Raised when the object passed to A2AServer is not a usable agent."""


# ── Agent invocation ──────────────────────────────────────────────────────────


def streaming_callable(agent: Any) -> Any | None:
    """
    Return the agent's async-generator entry point, or None if it does not
    stream.

    Matches the contract CapabilityGuard already checks for:
      1. a stream() method that is an async generator, or
      2. run() written as an async generator (uses `yield`).

    A class with STREAMING = True declares that its framework adapter handles
    streaming internally; there is no generator for the server to drive, so it
    is treated as non-streaming here and served through run().
    """
    stream = getattr(agent, "stream", None)
    if stream is not None and inspect.isasyncgenfunction(stream):
        return stream

    run = getattr(agent, "run", None)
    if run is not None and inspect.isasyncgenfunction(run):
        return run

    return None


async def _drain_stream(agent: Any, task: Task) -> Any:
    """
    Drain a streaming agent into a single run()-style output.

    A NeedsInput chunk ends the drain and becomes the output, exactly as it
    does in the SSE path — otherwise it would be folded into the chunk list
    and the task would complete instead of pausing.
    """
    gen = streaming_callable(agent)
    if gen is None:  # pragma: no cover - guarded by callers
        raise TypeError("Agent does not expose an async-generator entry point.")

    chunks: list[Any] = []
    async for chunk in gen(task):
        if isinstance(chunk, NeedsInput):
            return chunk
        chunks.append(chunk)
    return _join_chunks(chunks)


def _join_chunks(chunks: list[Any]) -> Any:
    """
    Fold streamed chunks into a single run()-style return value.

    All-strings join into one string, which is the common case (token-by-token
    text). Anything else is returned as a list so structured chunks survive.
    """
    if not chunks:
        return None
    if all(isinstance(c, str) for c in chunks):
        return "".join(chunks)
    if len(chunks) == 1:
        return chunks[0]
    return chunks


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
        input_handler: InputHandler | None = None,
        push_config: WebhookConfig | None = None,
    ) -> None:
        self._agent_instance, self._card = self._resolve_agent(agent)

        card_host, card_port = _split_url(str(self._card.url))
        self.host = host or card_host or "0.0.0.0"
        self.port = port if port is not None else (card_port or 8000)

        self.tasks = task_manager or TaskManager(store=store)
        # Lets agents that suspend via InputHandler.wait_for_input() be resumed
        # by the same wire call that resumes NeedsInput agents.
        self.input_handler = input_handler or InputHandler(self.tasks)
        self.push = WebhookDispatcher(push_config or WebhookConfig())
        self._push_config = push_config or WebhookConfig()
        # task_id -> where to POST its updates. In-memory, like the DLQ.
        self._push_targets: dict[str, PushNotificationConfig] = {}
        # Deliveries run detached; keep references so they are not GC'd
        # mid-flight, and so tests can await them.
        self._push_tasks: set[asyncio.Task[None]] = set()
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
        # An async generator is a valid run(): that is how a streaming agent is
        # written, and iscoroutinefunction() is False for those.
        if run is None or not (
            inspect.iscoroutinefunction(run) or inspect.isasyncgenfunction(run)
        ):
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

        async def stream_get(request: Request) -> Response:
            return await self._handle_stream_get(request)

        return Starlette(
            routes=[
                Route(AGENT_CARD_PATH, agent_card),
                Route("/health", health),
                Route("/ready", ready),
                Route(STREAM_PATH, stream_get),
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
            if method == METHOD_STREAM:
                return await self._rpc_stream(rpc_id, params)
            if method == METHOD_GET:
                return await self._rpc_get(rpc_id, params)
            if method == METHOD_CANCEL:
                return await self._rpc_cancel(rpc_id, params)
            if method == METHOD_PUSH_SET:
                return await self._rpc_push_set(rpc_id, params)
            if method == METHOD_PUSH_GET:
                return await self._rpc_push_get(rpc_id, params)
            return _rpc_error(
                rpc_id, ERR_METHOD_NOT_FOUND, f"Method not found: {method}"
            )
        except Exception as exc:  # noqa: BLE001 - last-resort RPC boundary
            logger.exception("A2AServer: unhandled error in %s", method)
            return _rpc_error(rpc_id, ERR_INTERNAL, f"Internal error: {exc}")

    # ── Push notifications ────────────────────────────────────────────────────

    def _register_push(
        self,
        task_id: str,
        raw: Any,
    ) -> tuple[int, str] | None:
        """
        Register where a task's updates should be POSTed.

        Returns None on success, or an (error_code, message) pair so the caller
        can build a response with the right RPC id.

        The URL comes from whoever called the agent, so it is validated here —
        this is the boundary where untrusted input enters.
        """
        if not self._card.capabilities.push_notifications:
            return (
                ERR_PUSH_NOT_SUPPORTED,
                f"Agent '{self._card.name}' declares push_notifications=False "
                "and does not deliver webhooks.",
            )

        try:
            config = PushNotificationConfig.model_validate(raw)
        except Exception as exc:
            return ERR_INVALID_PARAMS, f"Invalid pushNotification config: {exc}"

        try:
            validate_webhook_url(
                config.url, allow_private=self._push_config.allow_private_urls
            )
        except WebhookUrlError as exc:
            logger.warning("Refused webhook registration: %s", exc)
            return ERR_PUSH_INVALID_URL, str(exc)

        self._push_targets[task_id] = config
        logger.info("Push target registered for task %s", task_id)
        return None

    def _notify(self, task: Task, event: str) -> None:
        """
        Fire a push notification for a task, if one is registered.

        Delivery is detached: a slow or dead webhook must not stall the RPC
        response, and retry/backoff can take seconds. Failures are logged, never
        raised — a broken webhook is not the task's problem. Await
        drain_notifications() when you need delivery to have settled.
        """
        config = self._push_targets.get(task.id)
        if config is None:
            return

        async def deliver() -> None:
            payload_task = task
            try:
                await self.push.dispatch_silent(
                    config.url, payload_task, event=event
                )
            except Exception:  # noqa: BLE001 - detached, must never escape
                logger.exception("Push notification failed for task %s", task.id)

        job = asyncio.create_task(deliver())
        self._push_tasks.add(job)
        job.add_done_callback(self._push_tasks.discard)

        # A terminal task will get no further updates.
        if task.state in (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED):
            self._push_targets.pop(task.id, None)

    async def drain_notifications(self, timeout: float = 30.0) -> None:
        """Wait for in-flight push deliveries to finish. Mainly for tests."""
        if not self._push_tasks:
            return
        pending = list(self._push_tasks)
        await asyncio.wait(pending, timeout=timeout)

    async def _rpc_push_set(self, rpc_id: Any, params: dict[str, Any]) -> Response:
        """tasks/pushNotificationConfig/set — register or replace a target."""
        task_id = params.get("taskId") or params.get("id")
        if not isinstance(task_id, str):
            return _rpc_error(rpc_id, ERR_INVALID_PARAMS, "Missing 'taskId' param.")
        try:
            await self.tasks.get(task_id)
        except TaskNotFoundError:
            return _rpc_error(rpc_id, ERR_TASK_NOT_FOUND, f"Task not found: {task_id}")

        raw = params.get("pushNotificationConfig") or params.get("pushNotification")
        if raw is None:
            return _rpc_error(
                rpc_id, ERR_INVALID_PARAMS, "Missing 'pushNotificationConfig' param."
            )

        failure = self._register_push(task_id, raw)
        if failure is not None:
            return _rpc_error(rpc_id, failure[0], failure[1])

        return _rpc_result(
            rpc_id,
            {
                "taskId": task_id,
                "pushNotificationConfig": _push_dict(self._push_targets[task_id]),
            },
        )

    async def _rpc_push_get(self, rpc_id: Any, params: dict[str, Any]) -> Response:
        """tasks/pushNotificationConfig/get — read back a registered target."""
        task_id = params.get("taskId") or params.get("id")
        if not isinstance(task_id, str):
            return _rpc_error(rpc_id, ERR_INVALID_PARAMS, "Missing 'taskId' param.")

        config = self._push_targets.get(task_id)
        if config is None:
            return _rpc_result(rpc_id, {"taskId": task_id,
                                        "pushNotificationConfig": None})
        return _rpc_result(
            rpc_id, {"taskId": task_id, "pushNotificationConfig": _push_dict(config)}
        )

    # ── Task creation / continuation ──────────────────────────────────────────

    async def _begin_task(
        self,
        message: Message,
        params: dict[str, Any],
    ) -> tuple[Task | None, Response | None, bool]:
        """
        Resolve a send/stream request to the task it should run.

        Returns (task, error_response, resumed). Exactly one of task or
        error_response is set. `resumed` is True when a suspended coroutine was
        woken instead — the agent must NOT be invoked again in that case,
        because the original request is still holding it.

        With no 'taskId' this creates a task. With one it continues that task,
        which is how multi-turn works: COMPLETED is terminal, so an agent adds
        turns by returning NeedsInput rather than finishing.
        """
        rpc_id = params.get("__rpc_id")
        raw_task_id = params.get("taskId")

        if raw_task_id is None:
            skill_id = params.get("skillId")
            context_id = params.get("contextId")
            task = await self.tasks.create(
                initial_message=message,
                skill_id=skill_id if isinstance(skill_id, str) else None,
                context_id=context_id if isinstance(context_id, str) else None,
            )

            raw_push = params.get("pushNotification") or params.get(
                "pushNotificationConfig"
            )
            if raw_push is not None:
                failure = self._register_push(task.id, raw_push)
                if failure is not None:
                    # Refuse the whole call rather than silently running a task
                    # whose updates the caller believes they will receive.
                    return None, _rpc_error(rpc_id, failure[0], failure[1]), False

            return await self.tasks.start(task.id), None, False

        if not isinstance(raw_task_id, str):
            return None, _rpc_error(
                rpc_id, ERR_INVALID_PARAMS, "'taskId' must be a string."
            ), False

        # multi_turn is what continuation IS, so honour the card's own claim
        # rather than leaving the flag decorative.
        if not self._card.capabilities.multi_turn:
            return None, _rpc_error(
                rpc_id,
                ERR_TASK_NOT_WAITING,
                f"Agent '{self._card.name}' declares multi_turn=False and does "
                "not accept continuations. Omit 'taskId' to start a new task.",
            ), False

        try:
            task = await self.tasks.get(raw_task_id)
        except TaskNotFoundError:
            return None, _rpc_error(
                rpc_id, ERR_TASK_NOT_FOUND, f"Task not found: {raw_task_id}"
            ), False

        # An agent suspended inside InputHandler.wait_for_input() is resumed by
        # firing its event; the request that is still holding that coroutine
        # will finish the task.
        if self.input_handler.is_waiting(raw_task_id):
            await self.input_handler.submit_reply(raw_task_id, message)
            return await self.tasks.get(raw_task_id), None, True

        state = task.state
        if state != TaskState.INPUT_REQUIRED:
            return None, _rpc_error(
                rpc_id,
                ERR_TASK_NOT_WAITING,
                f"Task '{raw_task_id}' is '{_state_value(state)}', not "
                "'input_required'. Only a task awaiting input can be continued; "
                "omit 'taskId' to start a new one.",
            ), False

        # INPUT_REQUIRED -> WORKING, with the reply appended to history.
        return await self.tasks.provide_input(raw_task_id, message), None, False

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

        task, error, resumed = await self._begin_task(
            message, {**params, "__rpc_id": rpc_id}
        )
        if error is not None:
            return error
        assert task is not None  # _begin_task sets exactly one of the two

        if resumed:
            # A suspended coroutine was woken; the request holding it finishes
            # the task. Report the current state rather than running it twice.
            return _rpc_result(rpc_id, _task_dict(task))

        # An agent that raises is a FAILED task, not an RPC error: the caller
        # still gets a Task back and can inspect .error, and the DLQ can
        # capture it.
        try:
            # A streaming agent must still work over message/send — awaiting an
            # async generator raises TypeError, so drain it and fold the chunks.
            if streaming_callable(self._agent_instance) is not None:
                output = await _drain_stream(self._agent_instance, task)
            else:
                output = await self._agent_instance.run(task)
        except Exception as exc:  # noqa: BLE001 - agent code is untrusted
            logger.exception("Agent '%s' raised while running task %s",
                             self._card.name, task.id)
            failed = await self.tasks.fail(task.id, f"{type(exc).__name__}: {exc}")
            self._notify(failed, "task_failed")
            return _rpc_result(rpc_id, _task_dict(failed))

        return _rpc_result(rpc_id, _task_dict(await self._finish(task, output)))

    async def _rpc_stream(self, rpc_id: Any, params: dict[str, Any]) -> Response:
        """
        message/stream — send a message and stream the result back as SSE.

        Returns a text/event-stream instead of a JSON-RPC envelope. The event
        sequence is:

            task_created     the Task, so the caller learns its id immediately
            artifact_chunk   once per chunk the agent yields
            task_status      the terminal state
            done             stream closed normally

        A non-streaming agent is still served here: it runs to completion and
        its whole output is emitted as a single chunk, so callers can use one
        code path regardless of how the agent is written.

        Everything that can be rejected is rejected BEFORE the stream opens —
        once SSE starts, the status line is already sent and an HTTP error can
        no longer be signalled. Failures after that point arrive as a terminal
        'error' event.
        """
        raw_message = params.get("message")
        if raw_message is None:
            return _rpc_error(rpc_id, ERR_INVALID_PARAMS, "Missing 'message' param.")

        try:
            message = self.security.validate_message(raw_message)
        except Exception as exc:
            status = http_status_for(exc)
            if status == 500:
                return _rpc_error(rpc_id, ERR_INVALID_PARAMS, f"Invalid message: {exc}")
            return self._security_refusal(exc)

        task, error, resumed = await self._begin_task(
            message, {**params, "__rpc_id": rpc_id}
        )
        if error is not None:
            return error
        assert task is not None

        if resumed:
            return _rpc_result(rpc_id, _task_dict(task))

        return StreamingResponse(
            self._stream_task(task),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                # Stops nginx buffering the stream into one lump.
                "X-Accel-Buffering": "no",
            },
        )

    async def _stream_task(self, task: Task) -> AsyncIterator[str]:
        """Run the agent and yield SSE lines for each stage of the task."""
        yield SSEFormatter.event(StreamEventType.TASK_CREATED, _task_dict(task))

        gen = streaming_callable(self._agent_instance)
        chunks: list[Any] = []

        try:
            if gen is not None:
                index = 0
                async for chunk in gen(task):
                    # A NeedsInput yielded mid-stream pauses the task; the
                    # chunks before it stand, and nothing after it is consumed.
                    if isinstance(chunk, NeedsInput):
                        output = chunk
                        break
                    chunks.append(chunk)
                    yield SSEFormatter.artifact_chunk(
                        _chunk_text(chunk), task.id, index=index
                    )
                    index += 1
                else:
                    output = _join_chunks(chunks)
            else:
                # Non-streaming agent: one chunk carrying the whole output.
                output = await self._agent_instance.run(task)
                if output is not None and not isinstance(output, NeedsInput):
                    yield SSEFormatter.artifact_chunk(_chunk_text(output), task.id)

        except Exception as exc:  # noqa: BLE001 - agent code is untrusted
            logger.exception(
                "Agent '%s' raised while streaming task %s", self._card.name, task.id
            )
            reason = f"{type(exc).__name__}: {exc}"
            try:
                self._notify(await self.tasks.fail(task.id, reason), "task_failed")
            except Exception:  # pragma: no cover - store already unhappy
                logger.exception("Could not mark task %s failed", task.id)
            yield SSEFormatter.error(reason)
            return

        try:
            finished = await self._finish(task, output)
        except Exception as exc:  # noqa: BLE001 - store boundary
            logger.exception("Could not finalise task %s", task.id)
            yield SSEFormatter.error(f"{type(exc).__name__}: {exc}")
            return

        # A paused task ends the stream too, but with the prompt attached so
        # the caller knows what to answer and against which id.
        if isinstance(output, NeedsInput):
            yield SSEFormatter.event(
                StreamEventType.MESSAGE,
                {"taskId": finished.id, "content": output.as_message().text()},
            )

        yield SSEFormatter.task_status(_state_value(finished.state), finished.id)
        yield SSEFormatter.done()

    async def _handle_stream_get(self, request: Request) -> Response:
        """
        GET /stream?taskId=... — observe a task's current state as SSE.

        This is the endpoint SSEStreamer targets. Because message/send runs the
        agent inline, a task is already terminal by the time it can be looked
        up, so this reports the task's state and closes rather than following a
        run in progress. Use message/stream to watch work as it happens.
        """
        try:
            self.security.check_size(b"")
            await self.security.authorize(dict(request.headers))
        except Exception as exc:
            return self._security_refusal(exc)

        task_id = request.query_params.get("taskId") or request.query_params.get("id")
        if not task_id:
            return JSONResponse(
                {"error": "Missing 'taskId' query parameter."}, status_code=400
            )

        try:
            task = await self.tasks.get(task_id)
        except TaskNotFoundError:
            return JSONResponse(
                {"error": f"Task not found: {task_id}"}, status_code=404
            )

        async def emit() -> AsyncIterator[str]:
            yield SSEFormatter.event(StreamEventType.TASK_CREATED, _task_dict(task))
            for index, artifact in enumerate(task.artifacts):
                for part in artifact.parts:
                    yield SSEFormatter.artifact_chunk(
                        _chunk_text(part.content), task.id, index=index
                    )
            yield SSEFormatter.task_status(_state_value(task.state), task.id)
            yield SSEFormatter.done()

        return StreamingResponse(
            emit(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

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
        self._notify(task, "task_cancelled")
        return _rpc_result(rpc_id, _task_dict(task))

    # ── Result handling ───────────────────────────────────────────────────────

    async def _finish(self, task: Task, output: Any) -> Task:
        """
        Turn whatever run() returned into a completed Task.

        Accepted return values:
            NeedsInput    — pauses the task at INPUT_REQUIRED, prompt appended
            None          — completed with no artifact
            str           — one text Artifact
            Artifact      — used as-is
            Message       — recorded as the agent's reply
            AdapterResult — .error fails the task, else .to_artifact()
            dict / list   — one JSON Artifact
            anything else — str()-ified into a text Artifact
        """
        # NeedsInput is a pause, not a result: the task stays open so the
        # caller can answer against the same id.
        if isinstance(output, NeedsInput):
            paused = await self.tasks.request_input(task.id, output.as_message())
            self._notify(paused, "task_input_required")
            return paused

        artifact, reply, error = _interpret_output(output)

        if error is not None:
            failed = await self.tasks.fail(task.id, error)
            self._notify(failed, "task_failed")
            return failed

        completed = await self.tasks.complete(
            task.id,
            artifact=artifact,
            reply_message=reply,
        )
        self._notify(completed, "task_completed")
        return completed

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


def _push_dict(config: PushNotificationConfig) -> dict[str, Any]:
    """
    Serialise a push config for a response.

    The token is a shared secret the caller supplied; it is reported as present
    or absent but never echoed back, so reading a config cannot be used to
    recover one set by someone else.
    """
    return {"url": config.url, "hasToken": config.token is not None}


def _state_value(state: Any) -> str:
    """Return a task state as its wire string, whether enum or already str."""
    return state.value if isinstance(state, TaskState) else str(state)


def _chunk_text(chunk: Any) -> str:
    """
    Render one streamed chunk as text for an artifact_chunk event.

    Strings pass through untouched. Anything else is JSON-encoded when it can
    be, so structured chunks stay machine-readable on the wire rather than
    arriving as a Python repr.
    """
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, (dict, list, int, float, bool)) or chunk is None:
        try:
            return json.dumps(chunk)
        except (TypeError, ValueError):
            return str(chunk)
    if isinstance(chunk, Artifact):
        return " ".join(str(p.content) for p in chunk.parts)
    if isinstance(chunk, Message):
        return chunk.text()
    return str(chunk)


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
