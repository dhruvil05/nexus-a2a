"""
nexus_a2a/transport/tracing.py

Distributed tracing — propagates a trace ID across every agent hop
so you can reconstruct the full call tree for any pipeline run.

How it works:
  1. The first caller generates a unique trace_id.
  2. Every outbound request carries it in X-Nexus-Trace-ID header.
  3. Every agent reads the header and forwards it on its own outbound calls.
  4. Each hop records a Span (agent_url, duration, status).
  5. `nexus trace <trace_id>` reassembles the full tree from the store.

Two modes:
  Standalone  — spans stored in TraceStore (in-memory dict).
                Zero extra dependencies.
  OpenTelemetry — if otel SDK is installed, spans also exported to
                any OTEL backend (Jaeger, Tempo, Datadog, etc.).

Usage:
    # Outbound: inject header before sending
    headers = Tracer.inject(trace_id="abc-123")
    # → {"X-Nexus-Trace-ID": "abc-123"}

    # Inbound: extract trace_id from incoming request headers
    trace_id = Tracer.extract(request_headers)

    # Record a span
    async with Tracer.span(trace_id, agent_url="http://b:8002") as span:
        result = await client.send_message(message)
        span.set_status("completed")
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

# Header name used to propagate the trace ID
TRACE_ID_HEADER = "X-Nexus-Trace-ID"

# Context-local storage so trace_id flows through async call chains
# without being explicitly passed everywhere
_trace_ctx: dict[str, str] = {}


# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class Span:
    """
    One hop in a distributed trace — one agent call.

    Fields:
        trace_id:    The pipeline-level trace identifier.
        span_id:     Unique ID for this individual hop.
        agent_url:   The agent that was called.
        started_at:  Unix timestamp when the call began.
        ended_at:    Unix timestamp when the call finished (None if in progress).
        status:      "in_progress" | "completed" | "failed" | "cancelled"
        error:       Error message if status is "failed".
        metadata:    Any extra key-value pairs (skill_id, task_id, etc.)
    """

    trace_id: str
    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_url: str = ""
    started_at: float = field(default_factory=time.monotonic)
    ended_at: float | None = None
    status: str = "in_progress"
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float | None:
        """Wall-clock duration in milliseconds, or None if still in progress."""
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at) * 1000

    def set_status(self, status: str, error: str | None = None) -> None:
        """Mark this span as finished with the given status."""
        self.status = status
        self.error = error
        self.ended_at = time.monotonic()

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "agent_url": self.agent_url,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error": self.error,
            **self.metadata,
        }


@dataclass
class Trace:
    """
    The complete call tree for one pipeline run.

    Fields:
        trace_id: Unique identifier for this pipeline execution.
        spans:    Ordered list of all agent hops, in start order.
    """

    trace_id: str
    spans: list[Span] = field(default_factory=list)

    def add_span(self, span: Span) -> None:
        self.spans.append(span)

    def format_tree(self) -> str:
        """
        Render the trace as a human-readable tree.

        Example output:
            trace: abc-123
            ├── http://agent-a:8001   42ms   ✓ completed
            ├── http://agent-b:8002   891ms  ✓ completed
            └── http://agent-c:8003   12ms   ✗ failed: timeout
        """
        if not self.spans:
            return f"trace: {self.trace_id}\n  (no spans recorded)"

        lines = [f"trace: {self.trace_id}"]
        for i, span in enumerate(self.spans):
            is_last = i == len(self.spans) - 1
            prefix = "└──" if is_last else "├──"
            dur = f"{span.duration_ms:.0f}ms" if span.duration_ms else "..."
            icon = "✓" if span.status == "completed" else "✗"
            err = f": {span.error}" if span.error else ""
            lines.append(
                f"  {prefix} {span.agent_url:<35} {dur:<8} {icon} {span.status}{err}"
            )
        return "\n".join(lines)


# ── In-memory trace store ─────────────────────────────────────────────────────


class TraceStore:
    """
    Stores completed traces in memory.
    Keeps only the last `max_traces` traces to bound memory usage.

    Usage:
        store = TraceStore()
        store.record(span)
        trace = store.get("abc-123")
        print(trace.format_tree())
    """

    def __init__(self, max_traces: int = 1000) -> None:
        self._max = max_traces
        self._traces: dict[str, Trace] = {}
        self._lock = asyncio.Lock()

    async def record(self, span: Span) -> None:
        """Add a span to its trace. Creates the trace if it does not exist."""
        async with self._lock:
            if span.trace_id not in self._traces:
                # Drop oldest if at capacity
                if len(self._traces) >= self._max:
                    oldest = next(iter(self._traces))
                    del self._traces[oldest]
                self._traces[span.trace_id] = Trace(trace_id=span.trace_id)
            self._traces[span.trace_id].add_span(span)

    def get(self, trace_id: str) -> Trace | None:
        """Return the Trace for the given trace_id, or None."""
        return self._traces.get(trace_id)

    def list_ids(self) -> list[str]:
        """Return all stored trace IDs."""
        return list(self._traces.keys())

    def count(self) -> int:
        return len(self._traces)


# ── Global default store (used by Tracer) ─────────────────────────────────────
# Can be replaced: nexus_a2a.transport.tracing.default_store = MyStore()
default_store = TraceStore()


# ── Tracer ────────────────────────────────────────────────────────────────────


class Tracer:
    """
    Static utility class for injecting, extracting, and recording traces.

    All methods are static — no instance needed.

    Usage:
        # Generate a new trace ID (call once per pipeline entry point)
        trace_id = Tracer.new_trace_id()

        # Inject into outbound HTTP headers
        headers = Tracer.inject(trace_id)

        # Extract from inbound request headers
        trace_id = Tracer.extract(incoming_headers)

        # Record a span around an agent call
        async with Tracer.span(trace_id, "http://agent:8001",
                               metadata={"skill_id": "search"}) as span:
            result = await client.send_message(msg)
            span.set_status("completed")

        # Retrieve the full trace for display
        trace = Tracer.get_trace(trace_id)
        print(trace.format_tree())
    """

    @staticmethod
    def new_trace_id() -> str:
        """Generate a new unique trace ID."""
        return str(uuid.uuid4())

    @staticmethod
    def inject(trace_id: str) -> dict[str, str]:
        """
        Build the HTTP headers dict to propagate the trace ID.

        Args:
            trace_id: The trace ID to propagate.

        Returns:
            Dict with X-Nexus-Trace-ID header — merge into your request headers.
        """
        return {TRACE_ID_HEADER: trace_id}

    @staticmethod
    def extract(headers: dict[str, str]) -> str | None:
        """
        Extract the trace ID from incoming request headers.

        Checks both original and lowercase header names for compatibility
        with different HTTP frameworks.

        Args:
            headers: The incoming HTTP headers dict.

        Returns:
            The trace ID string, or None if not present.
        """
        return headers.get(TRACE_ID_HEADER) or headers.get(TRACE_ID_HEADER.lower())

    @staticmethod
    @asynccontextmanager
    async def span(
        trace_id: str,
        agent_url: str,
        metadata: dict[str, Any] | None = None,
        store: TraceStore | None = None,
    ) -> AsyncIterator[Span]:
        """
        Async context manager that records a span around a block of code.

        The span is automatically marked as 'failed' with the exception
        message if an exception escapes the block.

        Args:
            trace_id:  The trace ID this span belongs to.
            agent_url: The agent being called.
            metadata:  Optional key-value pairs (task_id, skill_id, etc.)
            store:     TraceStore to record into. Uses default_store if None.

        Yields:
            Span — call span.set_status("completed") when done.

        Example:
            async with Tracer.span("abc", "http://agent:8001") as span:
                result = await client.send_message(msg)
                span.set_status("completed")
        """
        _store = store or default_store
        span = Span(
            trace_id=trace_id,
            agent_url=agent_url,
            metadata=metadata or {},
        )

        try:
            yield span
        except Exception as exc:
            if span.status == "in_progress":
                span.set_status("failed", error=str(exc))
            raise
        finally:
            if span.status == "in_progress":
                span.set_status("completed")
            await _store.record(span)

        # Optional: export to OpenTelemetry if SDK is available
        Tracer._try_otel_export(span)

    @staticmethod
    def get_trace(
        trace_id: str,
        store: TraceStore | None = None,
    ) -> Trace | None:
        """
        Retrieve a complete Trace from the store.

        Args:
            trace_id: The trace to look up.
            store:    Store to query. Uses default_store if None.

        Returns:
            Trace if found, None otherwise.
        """
        return (store or default_store).get(trace_id)

    @staticmethod
    def _try_otel_export(span: Span) -> None:
        """
        Attempt to export the span to OpenTelemetry.
        Silently does nothing if the OTEL SDK is not installed.
        """
        try:
            from opentelemetry import trace as otel_trace  # type: ignore[import]

            tracer = otel_trace.get_tracer("nexus-a2a")
            with tracer.start_as_current_span(
                name=span.agent_url,
                attributes={
                    "nexus.trace_id": span.trace_id,
                    "nexus.agent_url": span.agent_url,
                    "nexus.status": span.status,
                    **({"nexus.error": span.error} if span.error else {}),
                },
            ):
                pass  # span already finished; attributes are what matter
        except ImportError:
            pass
        except Exception:
            pass
