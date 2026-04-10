"""
nexus_a2a/transport/sse.py

SSEStreamer — consumes a Server-Sent Events (SSE) stream from a remote A2A
agent and yields typed events as they arrive.

A2A streaming protocol:
  The remote agent sends a stream of newline-delimited SSE events:
    data: {"type": "task_status", ...}
    data: {"type": "artifact_chunk", ...}
    data: {"type": "done"}

  Each event is parsed and yielded as a typed StreamEvent so callers
  never deal with raw strings.

Also provides SSEFormatter — the server side utility that formats
Python objects into valid SSE lines to send to clients.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Dict, Any

import httpx

from nexus_a2a.models.task import Task

logger = logging.getLogger(__name__)


# ── Event types ───────────────────────────────────────────────────────────────

class StreamEventType(str, Enum):
    """All SSE event types emitted by an A2A streaming agent."""
    TASK_CREATED       = "task_created"       # Task object first appears
    TASK_STATUS        = "task_status"        # State transition (working→completed etc.)
    ARTIFACT_CHUNK     = "artifact_chunk"     # Partial artifact content
    ARTIFACT_COMPLETE  = "artifact_complete"  # Full artifact available
    MESSAGE            = "message"            # Intermediate agent message
    DONE               = "done"               # Stream closed normally
    ERROR              = "error"              # Stream closed with error
    HEARTBEAT          = "heartbeat"          # Keep-alive ping (no payload)


@dataclass
class StreamEvent:
    """
    One parsed event from an SSE stream.

    Fields:
        type:    What kind of event this is.
        data:    Parsed JSON payload (empty dict for heartbeat/done events).
        raw:     The original raw SSE line (useful for debugging).
    """
    type: StreamEventType
    data: Dict[str, Any]       = field(default_factory=dict)
    raw:  str        = ""

    @property
    def is_terminal(self) -> bool:
        """True if this event signals the stream has ended."""
        return self.type in {StreamEventType.DONE, StreamEventType.ERROR}

    def as_task(self) -> Task | None:
        """
        Try to parse the event's data as a Task.
        Returns None if the data does not look like a Task
        (missing required identifying fields: 'id' and 'state').
        """
        if "id" not in self.data or "state" not in self.data:
            return None
        try:
            return Task.model_validate(self.data)
        except Exception:
            return None


# ── Client-side: SSEStreamer ──────────────────────────────────────────────────

class SSEStreamer:
    """
    Connects to a remote A2A agent's streaming endpoint and yields
    StreamEvents as they arrive.

    Usage:
        streamer = SSEStreamer("http://agent:8001")
        async with streamer.stream(task_id="abc-123") as events:
            async for event in events:
                if event.type == StreamEventType.ARTIFACT_CHUNK:
                    print(event.data.get("content", ""))
                if event.is_terminal:
                    break

    Args:
        base_url:      The remote agent's base URL.
        timeout:       Max seconds to wait for the next event before giving up.
        headers:       Extra headers (e.g. auth) to include in the request.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 60.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout  = timeout
        self._headers  = headers or {}

    async def stream(
        self,
        task_id: str,
        path: str = "/stream",
    ) -> AsyncIterator[StreamEvent]:
        """
        Open an SSE connection and yield events until the stream ends.

        Args:
            task_id: The task ID to stream updates for.
            path:    SSE endpoint path on the remote server. Default: /stream.

        Yields:
            StreamEvent for each received SSE data line.
        """
        url = f"{self._base_url}{path}"
        params = {"taskId": task_id}

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream(
                "GET", url,
                params=params,
                headers={
                    "Accept": "text/event-stream",
                    "Cache-Control": "no-cache",
                    **self._headers,
                },
            ) as response:
                response.raise_for_status()
                async for event in self._parse_sse_lines(response):
                    yield event
                    if event.is_terminal:
                        return

    async def _parse_sse_lines(
        self,
        response: httpx.Response,
    ) -> AsyncIterator[StreamEvent]:
        """
        Parse raw SSE lines from the HTTP response body.

        SSE format:
            data: {"type": "task_status", "state": "working"}
            (blank line separates events)
        """
        async for line in response.aiter_lines():
            line = line.strip()
            if not line:
                continue   # blank line = event separator, skip

            if line.startswith("data:"):
                raw_data = line[len("data:"):].strip()
                event = self._parse_event(raw_data)
                if event:
                    yield event

            elif line.startswith(":"):
                # SSE comment line — used as heartbeat by some servers
                yield StreamEvent(type=StreamEventType.HEARTBEAT, raw=line)

    @staticmethod
    def _parse_event(raw: str) -> StreamEvent | None:
        """Parse a JSON SSE data payload into a StreamEvent."""
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("SSE: could not parse JSON from line: %r", raw)
            return None

        try:
            event_type = StreamEventType(payload.get("type", ""))
        except ValueError:
            logger.warning("SSE: unknown event type: %r", payload.get("type"))
            return None

        return StreamEvent(
            type=event_type,
            data=payload,
            raw=raw,
        )


# ── Server-side: SSEFormatter ─────────────────────────────────────────────────

class SSEFormatter:
    """
    Formats Python objects into SSE-compliant text lines for sending
    to clients from an agent server.

    Usage (inside a Starlette streaming response):
        async def stream_task(task_id: str):
            yield SSEFormatter.event(StreamEventType.TASK_STATUS,
                                     {"state": "working"})
            # ... do work ...
            yield SSEFormatter.event(StreamEventType.ARTIFACT_CHUNK,
                                     {"content": "partial result..."})
            yield SSEFormatter.done()
    """

    @staticmethod
    def event(event_type: StreamEventType, data: Dict[str, Any]) -> str:
        """
        Format a typed event as an SSE data line.

        Args:
            event_type: The type of event.
            data:       The payload to include.

        Returns:
            SSE-formatted string ending with double newline.

        Example output:
            data: {"type": "task_status", "state": "working"}\\n\\n
        """
        payload = json.dumps({"type": event_type.value, **data})
        return f"data: {payload}\n\n"

    @staticmethod
    def heartbeat() -> str:
        """
        Emit a keep-alive comment line.
        Prevents proxies and load balancers from closing idle connections.

        Returns:
            SSE comment line: ': heartbeat\\n\\n'
        """
        return ": heartbeat\n\n"

    @staticmethod
    def done() -> str:
        """
        Emit the terminal 'done' event, signalling the stream is complete.

        Returns:
            SSE data line with type=done.
        """
        return SSEFormatter.event(StreamEventType.DONE, {})

    @staticmethod
    def error(message: str) -> str:
        """
        Emit a terminal error event with a human-readable message.

        Args:
            message: Description of what went wrong.
        """
        return SSEFormatter.event(StreamEventType.ERROR, {"message": message})

    @staticmethod
    def task_status(state: str, task_id: str) -> str:
        """Shortcut: emit a task state transition event."""
        return SSEFormatter.event(
            StreamEventType.TASK_STATUS,
            {"state": state, "taskId": task_id},
        )

    @staticmethod
    def artifact_chunk(content: str, task_id: str, index: int = 0) -> str:
        """Shortcut: emit one chunk of an artifact being built incrementally."""
        return SSEFormatter.event(
            StreamEventType.ARTIFACT_CHUNK,
            {"content": content, "taskId": task_id, "index": index},
        )