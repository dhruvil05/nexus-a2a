"""
nexus_a2a/transport/http_client.py

A2AHttpClient — sends JSON-RPC 2.0 requests to a remote A2A agent server.

Responsibilities:
  - Fetch an agent's AgentCard from its well-known endpoint.
  - Send tasks (messages) to a remote agent.
  - Poll for task status.
  - Cancel a running task.

All network calls are async (httpx) with:
  - Configurable timeouts
  - Automatic retries on transient failures
  - Structured error types (no raw HTTP exceptions leaking out)
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, cast

import httpx

from nexus_a2a.models.agent import AgentCard
from nexus_a2a.models.task import Message, Task

logger = logging.getLogger(__name__)

# The well-known path where every A2A server exposes its AgentCard
_AGENT_CARD_PATH = "/.well-known/agent-card.json"

# JSON-RPC 2.0 method names defined by the A2A specification
_METHOD_SEND    = "message/send"
_METHOD_GET     = "tasks/get"
_METHOD_CANCEL  = "tasks/cancel"


# ── Exceptions ────────────────────────────────────────────────────────────────

class TransportError(Exception):
    """Base class for all HTTP transport errors."""


class AgentUnreachableError(TransportError):
    """Raised when the remote agent's server cannot be reached."""

    def __init__(self, url: str, reason: str) -> None:
        super().__init__(f"Agent at '{url}' is unreachable: {reason}")
        self.url = url
        self.reason = reason


class AgentCardFetchError(TransportError):
    """Raised when the agent card cannot be fetched or parsed."""

    def __init__(self, url: str, reason: str) -> None:
        super().__init__(f"Cannot fetch AgentCard from '{url}': {reason}")
        self.url = url


class RemoteAgentError(TransportError):
    """Raised when the remote agent returns a JSON-RPC error response."""

    def __init__(self, code: int, message: str, task_id: str | None = None) -> None:
        super().__init__(f"Remote agent error (code={code}): {message}")
        self.code = code
        self.message = message
        self.task_id = task_id


# ── Client ────────────────────────────────────────────────────────────────────

class A2AHttpClient:
    """
    Async HTTP client for communicating with a remote A2A agent server.

    Handles the JSON-RPC 2.0 envelope, retries, and error translation
    so the caller never deals with raw HTTP.

    Usage:
        async with A2AHttpClient("http://remote-agent:8001") as client:
            card = await client.fetch_agent_card()
            task = await client.send_message(
                message=Message.user_text("Summarise this text"),
                skill_id="summarise",
            )

    Args:
        base_url:    The root URL of the remote A2A server.
        timeout:     Seconds to wait before declaring a request timed out.
        max_retries: How many times to retry on connection errors (not on 4xx/5xx).
        headers:     Extra headers sent with every request (e.g. auth tokens).
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._base_url   = base_url.rstrip("/")
        self._timeout    = timeout
        self._max_retries = max_retries
        self._extra_headers: dict[str, str] = headers or {}
        self._client: httpx.AsyncClient | None = None

    # ── Context manager — ensures the connection pool is properly closed ───────

    async def __aenter__(self) -> A2AHttpClient:
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
            headers={
                "Content-Type": "application/json",
                "Accept":       "application/json",
                **self._extra_headers,
            },
        )
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    # ── Public API ────────────────────────────────────────────────────────────

    async def fetch_agent_card(self) -> AgentCard:
        """
        Fetch and parse the remote agent's AgentCard from its well-known URL.

        Returns:
            Parsed AgentCard.

        Raises:
            AgentCardFetchError: If the card cannot be fetched or parsed.
            AgentUnreachableError: If the server does not respond.
        """
        url = f"{self._base_url}{_AGENT_CARD_PATH}"
        try:
            response = await self._get(url)
            return AgentCard.model_validate(response)
        except (KeyError, ValueError) as exc:
            raise AgentCardFetchError(self._base_url, str(exc)) from exc

    async def send_message(
        self,
        message: Message,
        skill_id: str | None = None,
        context_id: str | None = None,
    ) -> Task:
        """
        Send a message to the remote agent and return the resulting Task.

        The remote agent creates a Task, begins processing, and returns the
        Task object (in SUBMITTED or WORKING state).

        Args:
            message:    The user message to send.
            skill_id:   Optional — which skill to invoke.
            context_id: Optional — group this task with related ones.

        Returns:
            The Task created by the remote agent.

        Raises:
            AgentUnreachableError: Server not reachable.
            RemoteAgentError:      Agent returned a JSON-RPC error.
        """
        params: dict[str, Any] = {
            "message": message.model_dump(mode="json"),
        }
        if skill_id:
            params["skillId"] = skill_id
        if context_id:
            params["contextId"] = context_id

        result = await self._rpc(_METHOD_SEND, params)
        return Task.model_validate(result)

    async def get_task(self, task_id: str) -> Task:
        """
        Retrieve the current state of a task from the remote agent.

        Useful for polling long-running tasks.

        Args:
            task_id: The ID of the task to poll.

        Returns:
            Latest Task snapshot.

        Raises:
            AgentUnreachableError: Server not reachable.
            RemoteAgentError:      Task not found or agent error.
        """
        result = await self._rpc(_METHOD_GET, {"taskId": task_id})
        return Task.model_validate(result)

    async def cancel_task(self, task_id: str) -> Task:
        """
        Request the remote agent to cancel a running task.

        Args:
            task_id: The ID of the task to cancel.

        Returns:
            Updated Task (should be in CANCELLED state).

        Raises:
            AgentUnreachableError: Server not reachable.
            RemoteAgentError:      Agent refused to cancel or task not found.
        """
        result = await self._rpc(_METHOD_CANCEL, {"taskId": task_id})
        return Task.model_validate(result)

    # ── JSON-RPC 2.0 plumbing ─────────────────────────────────────────────────

    async def _rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """
        Send a JSON-RPC 2.0 request and return the 'result' payload.

        Handles:
        - Building the JSON-RPC envelope
        - Retrying on connection errors
        - Translating JSON-RPC errors into RemoteAgentError

        Raises:
            AgentUnreachableError: After all retries are exhausted.
            RemoteAgentError:      If the agent returns an error response.
        """
        payload = {
            "jsonrpc": "2.0",
            "id":      str(uuid.uuid4()),
            "method":  method,
            "params":  params,
        }

        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                response = await self._post("/", payload)
                return self._unwrap_rpc(response)

            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                last_error = exc
                logger.warning(
                    "RPC attempt %d/%d failed for method '%s': %s",
                    attempt, self._max_retries, method, exc,
                )
                # Don't retry on the last attempt
                if attempt == self._max_retries:
                    break

        raise AgentUnreachableError(self._base_url, str(last_error))

    def _unwrap_rpc(self, body: dict[str, Any]) -> dict[str, Any]:
        """
        Extract the 'result' from a JSON-RPC 2.0 response body.

        Raises:
            RemoteAgentError: If the response contains an 'error' field.
        """
        if "error" in body:
            err = body["error"]
            raise RemoteAgentError(
                code=err.get("code", -1),
                message=err.get("message", "Unknown error"),
            )
        return cast(Dict[str, Any], body.get("result", {}))

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a POST request and return the parsed JSON body."""
        client = self._require_client()
        response = await client.post(path, json=payload)
        response.raise_for_status()
        return cast(Dict[str, Any], response.json())

    async def _get(self, url: str) -> dict[str, Any]:
        """Send a GET request to an absolute URL and return the parsed JSON body."""
        client = self._require_client()
        response = await client.get(url)
        response.raise_for_status()
        return cast(Dict[str, Any], response.json())

    def _require_client(self) -> httpx.AsyncClient:
        """Assert the client is open (used inside async with block)."""
        if self._client is None:
            raise RuntimeError(
                "A2AHttpClient must be used as an async context manager: "
                "'async with A2AHttpClient(...) as client:'"
            )
        return self._client
