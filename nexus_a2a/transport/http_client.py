"""
nexus_a2a/transport/http_client.py

A2AHttpClient — upgraded for v1.1 with:
  - 5xx HTTP status retry (was: only ConnectError retried)
  - Circuit breaker (CLOSED → OPEN → HALF_OPEN)
  - Configurable retry_on status codes
  - Exponential backoff with jitter
  - Automatic trace ID injection via Tracer

Circuit breaker states:
  CLOSED     — normal operation, all requests go through
  OPEN       — agent is unhealthy, requests fail immediately (no network call)
  HALF_OPEN  — one probe request allowed; success → CLOSED, failure → OPEN

This means a repeatedly failing agent stops receiving requests automatically,
protecting the rest of the pipeline from cascade failures.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from collections.abc import AsyncIterator
from enum import Enum
from typing import Any

import httpx

from nexus_a2a.models.agent import AgentCard
from nexus_a2a.models.task import Message, PushNotificationConfig, Task
from nexus_a2a.transport.sse import StreamEvent, iter_sse_events
from nexus_a2a.transport.tracing import Tracer, TraceStore

logger = logging.getLogger(__name__)

_AGENT_CARD_PATH = "/.well-known/agent-card.json"
# Header announcing the calling agent's own URL. Mirrors
# nexus_a2a.security.middleware.CALLER_HEADER — duplicated as a literal to
# keep the transport layer independent of the security package.
_CALLER_HEADER = "X-Nexus-Caller"
_METHOD_SEND = "message/send"
_METHOD_STREAM = "message/stream"
_METHOD_GET = "tasks/get"
_METHOD_CANCEL = "tasks/cancel"
_METHOD_PUSH_SET = "tasks/pushNotificationConfig/set"
_METHOD_PUSH_GET = "tasks/pushNotificationConfig/get"


def _build_send_params(
    message: Message,
    skill_id: str | None,
    context_id: str | None,
    task_id: str | None = None,
    push_notification: PushNotificationConfig | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the params object shared by message/send and message/stream."""
    params: dict[str, Any] = {"message": message.model_dump(mode="json")}
    if skill_id:
        params["skillId"] = skill_id
    if context_id:
        params["contextId"] = context_id
    if task_id:
        params["taskId"] = task_id
    if push_notification is not None:
        params["pushNotification"] = (
            push_notification.model_dump(mode="json")
            if isinstance(push_notification, PushNotificationConfig)
            else push_notification
        )
    return params

# Default HTTP status codes that should trigger a retry
DEFAULT_RETRY_ON = {500, 502, 503, 504}


# ── Exceptions ────────────────────────────────────────────────────────────────


class TransportError(Exception):
    """Base class for all HTTP transport errors."""


class AgentUnreachableError(TransportError):
    def __init__(self, url: str, reason: str) -> None:
        super().__init__(f"Agent at '{url}' is unreachable: {reason}")
        self.url = url
        self.reason = reason


class AgentCardFetchError(TransportError):
    def __init__(self, url: str, reason: str) -> None:
        super().__init__(f"Cannot fetch AgentCard from '{url}': {reason}")
        self.url = url


class RemoteAgentError(TransportError):
    def __init__(self, code: int, message: str, task_id: str | None = None) -> None:
        super().__init__(f"Remote agent error (code={code}): {message}")
        self.code = code
        self.message = message
        self.task_id = task_id


class CircuitOpenError(TransportError):
    """
    Raised when a request is rejected because the circuit breaker is OPEN.
    The agent has been failing repeatedly — no network call is made.
    """

    def __init__(self, url: str, retry_after: float) -> None:
        super().__init__(
            f"Circuit breaker OPEN for '{url}'. Retry after {retry_after:.1f}s."
        )
        self.url = url
        self.retry_after = retry_after


# ── Circuit breaker ───────────────────────────────────────────────────────────


class CircuitState(str, Enum):
    CLOSED = "closed"  # normal — requests go through
    OPEN = "open"  # unhealthy — requests blocked
    HALF_OPEN = "half_open"  # recovery probe — one request allowed


class CircuitBreaker:
    """
    Per-agent circuit breaker.

    Args:
        failure_threshold: Consecutive failures before opening. Default: 5.
        recovery_timeout:  Seconds to wait in OPEN before trying HALF_OPEN.
                           Default: 30s.
        success_threshold: Consecutive successes in HALF_OPEN to close.
                           Default: 2.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        success_threshold: int = 2,
    ) -> None:
        self._fail_thresh = failure_threshold
        self._recovery = recovery_timeout
        self._success_thresh = success_threshold

        self._state = CircuitState.CLOSED
        self._failures = 0
        self._successes = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> CircuitState:
        # Auto-transition OPEN → HALF_OPEN after recovery timeout
        if (
            self._state == CircuitState.OPEN
            and self._opened_at is not None
            and (time.monotonic() - self._opened_at) >= self._recovery
        ):
            self._state = CircuitState.HALF_OPEN
            self._successes = 0
            logger.info("Circuit breaker → HALF_OPEN (probe allowed)")
        return self._state

    def before_call(self, url: str) -> None:
        """
        Called before every outbound request.
        Raises CircuitOpenError if the circuit is OPEN.
        """
        state = self.state  # triggers auto-transition check

        if state == CircuitState.OPEN:
            elapsed = time.monotonic() - (self._opened_at or 0)
            retry_after = max(0.0, self._recovery - elapsed)
            raise CircuitOpenError(url, retry_after)

    def on_success(self) -> None:
        """Record a successful call."""
        if self._state == CircuitState.HALF_OPEN:
            self._successes += 1
            if self._successes >= self._success_thresh:
                self._state = CircuitState.CLOSED
                self._failures = 0
                logger.info("Circuit breaker → CLOSED (recovered)")
        else:
            self._failures = 0

    def on_failure(self) -> None:
        """Record a failed call."""
        self._failures += 1

        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()
            logger.warning("Circuit breaker → OPEN (probe failed)")
            return

        if self._failures >= self._fail_thresh:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()
            logger.warning(
                "Circuit breaker → OPEN after %d consecutive failures",
                self._failures,
            )


# ── Retry config ──────────────────────────────────────────────────────────────


class RetryConfig:
    """
    Configuration for retry behaviour.

    Args:
        max_retries:  Maximum number of attempts (including the first).
        retry_on:     Set of HTTP status codes that trigger a retry.
                      Default: {500, 502, 503, 504}.
        base_delay:   Initial retry delay in seconds. Doubles each attempt.
        max_delay:    Cap on retry delay. Default: 30s.
        jitter:       Add random jitter (±20%) to avoid thundering herd.
    """

    def __init__(
        self,
        max_retries: int = 3,
        retry_on: set[int] = DEFAULT_RETRY_ON,
        base_delay: float = 0.5,
        max_delay: float = 30.0,
        jitter: bool = True,
    ) -> None:
        self.max_retries = max_retries
        self.retry_on = retry_on
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter

    def delay_for(self, attempt: int) -> float:
        """Return the delay (seconds) before attempt N (1-indexed)."""
        delay: float = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
        if self.jitter:
            delay *= 0.8 + random.random() * 0.4  # ±20% jitter
        return delay


# ── A2AHttpClient ─────────────────────────────────────────────────────────────


class A2AHttpClient:
    """
    Async HTTP client for communicating with a remote A2A agent server.

    v1.1 upgrades over the original:
      - Retries on HTTP 5xx status codes (not just ConnectError)
      - Built-in circuit breaker per client instance
      - Exponential backoff with jitter between retries
      - Automatic trace ID injection on every request

    Usage:
        async with A2AHttpClient(
            "http://agent:8001",
            retry=RetryConfig(max_retries=3, retry_on={500, 502, 503}),
            circuit_breaker=CircuitBreaker(failure_threshold=5),
            trace_id="abc-123",
        ) as client:
            task = await client.send_message(message)

    Args:
        base_url:        Root URL of the remote A2A server.
        timeout:         Per-attempt HTTP timeout in seconds. Default: 30s.
        retry:           Retry configuration. Uses defaults if None.
        circuit_breaker: Circuit breaker instance. Disabled if None.
        headers:         Extra headers sent with every request (e.g. auth).
        trace_id:        Trace ID to propagate. Auto-generated if not provided.
        trace_store:     TraceStore for recording spans. Uses default if None.
        caller_url:      THIS agent's own base URL, announced to the remote
                         agent in the 'X-Nexus-Caller' header. A server running
                         SecurityMiddleware uses it to authenticate the caller
                         and to evaluate trust rules, which are 'caller ->
                         target'. Without it the call is anonymous, and a
                         server with auth or trust enabled will reject it.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        retry: RetryConfig | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        headers: dict[str, str] | None = None,
        trace_id: str | None = None,
        trace_store: TraceStore | None = None,
        caller_url: str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._retry = retry or RetryConfig()
        self._cb = circuit_breaker  # None = disabled
        self._caller_url = caller_url.rstrip("/") if caller_url else None
        self._extra_headers = dict(headers or {})
        # Explicit headers= win over caller_url, so a caller can always
        # override the announced identity.
        if self._caller_url and _CALLER_HEADER not in self._extra_headers:
            self._extra_headers[_CALLER_HEADER] = self._caller_url
        self._trace_id = trace_id or Tracer.new_trace_id()
        self._trace_store = trace_store
        self._client: httpx.AsyncClient | None = None

    # ── Context manager ───────────────────────────────────────────────────────

    async def __aenter__(self) -> A2AHttpClient:
        trace_headers = Tracer.inject(self._trace_id)
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                **trace_headers,
                **self._extra_headers,
            },
        )
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def trace_id(self) -> str:
        """The trace ID being propagated by this client."""
        return self._trace_id

    @property
    def circuit_state(self) -> CircuitState | None:
        """Current circuit breaker state, or None if disabled."""
        return self._cb.state if self._cb else None

    async def fetch_agent_card(self) -> AgentCard:
        """
        Fetch and parse the remote agent's AgentCard.

        Raises:
            AgentCardFetchError:   Card endpoint returned invalid response.
            AgentUnreachableError: Server did not respond after retries.
            CircuitOpenError:      Circuit breaker is open.
        """
        url = f"{self._base_url}{_AGENT_CARD_PATH}"
        try:
            response = await self._get_with_retry(url)
            return AgentCard.model_validate(response)
        except (KeyError, ValueError) as exc:
            raise AgentCardFetchError(self._base_url, str(exc)) from exc

    async def send_message(
        self,
        message: Message,
        skill_id: str | None = None,
        context_id: str | None = None,
        task_id: str | None = None,
        push_notification: PushNotificationConfig | dict[str, Any] | None = None,
    ) -> Task:
        """
        Send a message to the remote agent and return the Task.

        Args:
            message:    The message to send.
            skill_id:   Which skill to invoke.
            context_id: Groups related tasks into one conversation.
            task_id:    Continue an existing task that is awaiting input,
                        instead of starting a new one. This is how a multi-turn
                        conversation proceeds: the agent returns a task in
                        INPUT_REQUIRED with its question as the last message,
                        and you answer against the same id.

        Raises:
            AgentUnreachableError: Server unreachable after retries.
            RemoteAgentError:      Agent returned JSON-RPC error — including
                                   when task_id names a task that is not
                                   awaiting input.
            CircuitOpenError:      Circuit breaker is open.
        """
        params = _build_send_params(
            message, skill_id, context_id, task_id, push_notification
        )

        async with Tracer.span(
            self._trace_id,
            self._base_url,
            metadata={"skill_id": skill_id},
            store=self._trace_store,
        ) as span:
            result = await self._rpc(_METHOD_SEND, params)
            task = Task.model_validate(result)
            span.set_status("completed")
            span.metadata["task_id"] = task.id

        return task

    async def get_task(self, task_id: str) -> Task:
        """Poll the current state of a task from the remote agent."""
        result = await self._rpc(_METHOD_GET, {"taskId": task_id})
        return Task.model_validate(result)

    async def cancel_task(self, task_id: str) -> Task:
        """Request the remote agent cancel a running task."""
        result = await self._rpc(_METHOD_CANCEL, {"taskId": task_id})
        return Task.model_validate(result)

    async def set_push_config(
        self,
        task_id: str,
        config: PushNotificationConfig | dict[str, Any],
    ) -> dict[str, Any]:
        """
        Register or replace where the agent POSTs this task's updates.

        Use when the webhook was not supplied with the original message, or to
        point an in-flight task somewhere else.

        Raises:
            RemoteAgentError: The agent does not support push notifications, or
                              rejected the URL (private address, bad scheme).
        """
        payload = (
            config.model_dump(mode="json")
            if isinstance(config, PushNotificationConfig)
            else config
        )
        return await self._rpc(
            _METHOD_PUSH_SET,
            {"taskId": task_id, "pushNotificationConfig": payload},
        )

    async def get_push_config(self, task_id: str) -> dict[str, Any]:
        """
        Read back the push target registered for a task.

        The token is never echoed back — the response reports only whether one
        is set, so reading a config cannot recover a secret someone else chose.
        """
        return await self._rpc(_METHOD_PUSH_GET, {"taskId": task_id})

    async def stream_message(
        self,
        message: Message,
        skill_id: str | None = None,
        context_id: str | None = None,
        task_id: str | None = None,
        push_notification: PushNotificationConfig | dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """
        Send a message and yield StreamEvents as the agent produces them.

        The streaming counterpart to send_message(). Instead of blocking until
        the task finishes, this yields:

            task_created     the Task, so you have its id immediately
            artifact_chunk   once per chunk the agent yields
            task_status      the terminal state
            done             stream closed normally

        Iteration stops after the first terminal event (done or error).

        Usage:
            async with A2AHttpClient("http://agent:8001") as client:
                async for event in client.stream_message(Message.user_text("hi")):
                    if event.type == StreamEventType.ARTIFACT_CHUNK:
                        print(event.data["content"], end="", flush=True)

        A remote agent that does not stream is still served here — its whole
        output arrives as one artifact_chunk — so this works against any A2A
        agent, not only streaming ones.

        Retries and the circuit breaker do NOT apply: a half-consumed stream
        cannot be safely replayed, since the caller has already seen part of
        the output.

        Raises:
            AgentUnreachableError: Connection failed, or the server rejected
                                   the request before the stream opened.
        """
        client = self._require_client()
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": _METHOD_STREAM,
            "params": _build_send_params(
                message, skill_id, context_id, task_id, push_notification
            ),
        }

        try:
            async with client.stream(
                "POST",
                "/",
                json=payload,
                headers={"Accept": "text/event-stream"},
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise AgentUnreachableError(
                        self._base_url,
                        f"HTTP {response.status_code} opening stream: "
                        f"{response.text[:200]}",
                    )
                async for event in iter_sse_events(response):
                    yield event
                    if event.is_terminal:
                        return
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise AgentUnreachableError(self._base_url, str(exc)) from exc

    # ── Core retry + circuit breaker logic ───────────────────────────────────

    async def _rpc(
        self,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Send a JSON-RPC 2.0 request with retry + circuit breaker.

        Retry triggers on:
          - httpx.ConnectError / httpx.TimeoutException
          - HTTP status codes in retry.retry_on (default: 500,502,503,504)

        Does NOT retry on:
          - 4xx responses (client errors — retrying won't help)
          - JSON-RPC error responses (application-level errors)
        """
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": method,
            "params": params,
        }

        last_error: str = "unknown"

        for attempt in range(1, self._retry.max_retries + 1):
            # ── Circuit breaker check ──────────────────────────────────────
            if self._cb:
                self._cb.before_call(self._base_url)  # raises if OPEN

            try:
                response = await self._post("/", payload)

                # ── HTTP 5xx → retry ───────────────────────────────────────
                if response.status_code in self._retry.retry_on:
                    last_error = f"HTTP {response.status_code}"
                    logger.warning(
                        "RPC attempt %d/%d → %s for %s, will retry",
                        attempt,
                        self._retry.max_retries,
                        last_error,
                        method,
                    )
                    if self._cb:
                        self._cb.on_failure()
                    if attempt < self._retry.max_retries:
                        await asyncio.sleep(self._retry.delay_for(attempt))
                    continue

                # ── HTTP 4xx → do not retry ────────────────────────────────
                if 400 <= response.status_code < 500:
                    last_error = f"HTTP {response.status_code} (client error)"
                    if self._cb:
                        self._cb.on_failure()
                    raise AgentUnreachableError(self._base_url, last_error)

                # ── Success: unwrap JSON-RPC ───────────────────────────────
                body = response.json()
                result = self._unwrap_rpc(body)
                if self._cb:
                    self._cb.on_success()
                return result

            except CircuitOpenError:
                raise  # never retry when circuit is open

            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                last_error = str(exc)
                logger.warning(
                    "RPC attempt %d/%d connection error for %s: %s",
                    attempt,
                    self._retry.max_retries,
                    method,
                    exc,
                )
                if self._cb:
                    self._cb.on_failure()
                if attempt < self._retry.max_retries:
                    await asyncio.sleep(self._retry.delay_for(attempt))

        raise AgentUnreachableError(self._base_url, last_error)

    def _unwrap_rpc(self, body: dict[str, Any]) -> dict[str, Any]:
        """Extract 'result' from JSON-RPC response or raise RemoteAgentError."""
        if "error" in body:
            err = body["error"]
            raise RemoteAgentError(
                code=err.get("code", -1),
                message=err.get("message", "Unknown error"),
            )
        result: dict[str, Any] = body.get("result", {})
        return result

    async def _post(self, path: str, payload: dict[str, Any]) -> httpx.Response:
        """Send a POST and return the raw response (no raise_for_status)."""
        client = self._require_client()
        return await client.post(path, json=payload)

    async def _get_with_retry(self, url: str) -> dict[str, Any]:
        """
        GET an absolute URL with retry logic (used for agent card fetching).
        """
        last_error = "unknown"
        for attempt in range(1, self._retry.max_retries + 1):
            if self._cb:
                self._cb.before_call(self._base_url)
            try:
                client = self._require_client()
                response = await client.get(url)
                if response.status_code in self._retry.retry_on:
                    last_error = f"HTTP {response.status_code}"
                    if self._cb:
                        self._cb.on_failure()
                    if attempt < self._retry.max_retries:
                        await asyncio.sleep(self._retry.delay_for(attempt))
                    continue
                response.raise_for_status()
                if self._cb:
                    self._cb.on_success()
                data: dict[str, Any] = response.json()
                return data
            except CircuitOpenError:
                raise
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                last_error = str(exc)
                if self._cb:
                    self._cb.on_failure()
                if attempt < self._retry.max_retries:
                    await asyncio.sleep(self._retry.delay_for(attempt))
        raise AgentUnreachableError(self._base_url, last_error)

    def _require_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError(
                "A2AHttpClient must be used as an async context manager: "
                "async with A2AHttpClient(...) as client:"
            )
        return self._client
