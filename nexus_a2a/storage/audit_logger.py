"""
nexus_a2a/storage/audit_logger.py

AuditLogger — structured JSON logging of every significant event
in the agent network.

Every log entry is a single JSON object on one line (NDJSON format)
so it can be ingested by Datadog, Loki, CloudWatch, or any log
aggregator without extra parsing.

Events logged:
  - task_created        when a task is created
  - task_state_changed  on every state transition
  - agent_called        when an agent receives a task
  - agent_responded     when an agent completes a task
  - auth_failure        when authentication fails
  - rate_limit_exceeded when rate limit is hit
  - workflow_completed  when an Orchestrator run finishes
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TextIO

from nexus_a2a.models.task import Task, TaskState

logger = logging.getLogger(__name__)


# ── Event types ───────────────────────────────────────────────────────────────

class AuditEvent(str, Enum):
    TASK_CREATED        = "task_created"
    TASK_STATE_CHANGED  = "task_state_changed"
    AGENT_CALLED        = "agent_called"
    AGENT_RESPONDED     = "agent_responded"
    AUTH_FAILURE        = "auth_failure"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    WORKFLOW_COMPLETED  = "workflow_completed"
    CUSTOM              = "custom"


# ── Log entry ─────────────────────────────────────────────────────────────────

@dataclass
class AuditEntry:
    """
    One structured audit log entry.

    Fields:
        event:      The event type.
        timestamp:  Unix timestamp (float, UTC).
        data:       Event-specific payload.
        task_id:    Optional — the task this event relates to.
        agent_url:  Optional — the agent this event relates to.
    """
    event:     AuditEvent
    data:      dict[str, Any]   = field(default_factory=dict)
    timestamp: float            = field(default_factory=time.time)
    task_id:   str | None       = None
    agent_url: str | None       = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event":     self.event.value,
            "timestamp": self.timestamp,
            "task_id":   self.task_id,
            "agent_url": self.agent_url,
            **self.data,
        }

    def to_json(self) -> str:
        """Serialise to a single JSON line (NDJSON format)."""
        return json.dumps(self.to_dict(), default=str)


# ── AuditLogger ───────────────────────────────────────────────────────────────

class AuditLogger:
    """
    Writes structured audit entries to a stream (stdout by default)
    and keeps an in-memory buffer for programmatic inspection.

    Usage:
        audit = AuditLogger()                    # writes to stdout
        audit = AuditLogger(stream=open("audit.ndjson", "a"))  # writes to file

        # Log events
        audit.task_created(task)
        audit.task_state_changed(task, old_state=TaskState.WORKING)
        audit.agent_called(agent_url="http://agent:8001", task_id="abc")
        audit.auth_failure(agent_url="http://agent:8001", reason="bad token")

        # Inspect in-memory log (useful in tests)
        entries = audit.entries()
        failed  = audit.entries_by_event(AuditEvent.AUTH_FAILURE)

    Args:
        stream:      Where to write JSON lines. Default: sys.stdout.
        buffer_size: Max number of entries kept in memory. 0 = unlimited.
                     Oldest entries are dropped when the buffer is full.
        enabled:     Set to False to disable all logging (useful in tests).
    """

    def __init__(
        self,
        stream:      TextIO | None = None,
        buffer_size: int = 1000,
        enabled:     bool = True,
    ) -> None:
        self._stream      = stream or sys.stdout
        self._buffer_size = buffer_size
        self._enabled     = enabled
        self._buffer:     list[AuditEntry] = []

    # ── Public logging methods ────────────────────────────────────────────────

    def task_created(self, task: Task) -> None:
        """Log a new task being created."""
        self._log(AuditEntry(
            event=AuditEvent.TASK_CREATED,
            task_id=task.id,
            data={
                "skill_id":   task.skill_id,
                "context_id": task.context_id,
                "state":      task.state.value,
            },
        ))

    def task_state_changed(
        self,
        task: Task,
        old_state: TaskState,
    ) -> None:
        """Log a task transitioning from one state to another."""
        self._log(AuditEntry(
            event=AuditEvent.TASK_STATE_CHANGED,
            task_id=task.id,
            data={
                "old_state": old_state.value,
                "new_state": task.state.value,
                "error":     task.error,
            },
        ))

    def agent_called(
        self,
        agent_url: str,
        task_id: str,
        skill_id: str | None = None,
    ) -> None:
        """Log an outbound call to a remote agent."""
        self._log(AuditEntry(
            event=AuditEvent.AGENT_CALLED,
            task_id=task_id,
            agent_url=agent_url,
            data={"skill_id": skill_id},
        ))

    def agent_responded(
        self,
        agent_url: str,
        task_id: str,
        duration_sec: float,
        succeeded: bool,
    ) -> None:
        """Log a response received from a remote agent."""
        self._log(AuditEntry(
            event=AuditEvent.AGENT_RESPONDED,
            task_id=task_id,
            agent_url=agent_url,
            data={
                "duration_sec": round(duration_sec, 4),
                "succeeded":    succeeded,
            },
        ))

    def auth_failure(
        self,
        agent_url: str,
        reason: str,
        task_id: str | None = None,
    ) -> None:
        """Log an authentication failure."""
        self._log(AuditEntry(
            event=AuditEvent.AUTH_FAILURE,
            task_id=task_id,
            agent_url=agent_url,
            data={"reason": reason},
        ))

    def rate_limit_exceeded(
        self,
        agent_url: str,
        retry_after: float,
        task_id: str | None = None,
    ) -> None:
        """Log a rate limit being hit."""
        self._log(AuditEntry(
            event=AuditEvent.RATE_LIMIT_EXCEEDED,
            task_id=task_id,
            agent_url=agent_url,
            data={"retry_after_sec": round(retry_after, 2)},
        ))

    def workflow_completed(
        self,
        mode: str,
        total_sec: float,
        steps: int,
        succeeded: bool,
    ) -> None:
        """Log the completion of an Orchestrator workflow."""
        self._log(AuditEntry(
            event=AuditEvent.WORKFLOW_COMPLETED,
            data={
                "mode":        mode,
                "total_sec":   round(total_sec, 4),
                "steps":       steps,
                "succeeded":   succeeded,
            },
        ))

    def custom(
        self,
        event_name: str,
        data: dict[str, Any],
        task_id: str | None = None,
        agent_url: str | None = None,
    ) -> None:
        """Log a custom event with arbitrary data."""
        self._log(AuditEntry(
            event=AuditEvent.CUSTOM,
            task_id=task_id,
            agent_url=agent_url,
            data={"custom_event": event_name, **data},
        ))

    # ── Inspection ────────────────────────────────────────────────────────────

    def entries(self) -> list[AuditEntry]:
        """Return a copy of all buffered entries."""
        return list(self._buffer)

    def entries_by_event(self, event: AuditEvent) -> list[AuditEntry]:
        """Return all buffered entries of a specific event type."""
        return [e for e in self._buffer if e.event == event]

    def entries_for_task(self, task_id: str) -> list[AuditEntry]:
        """Return all buffered entries related to a specific task."""
        return [e for e in self._buffer if e.task_id == task_id]

    def clear(self) -> None:
        """Clear the in-memory buffer. Does not affect the stream."""
        self._buffer.clear()

    def count(self) -> int:
        """Return the number of entries currently in the buffer."""
        return len(self._buffer)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _log(self, entry: AuditEntry) -> None:
        """Write entry to stream and add to buffer."""
        if not self._enabled:
            return

        # Add to buffer (drop oldest if at capacity)
        if self._buffer_size > 0 and len(self._buffer) >= self._buffer_size:
            self._buffer.pop(0)
        self._buffer.append(entry)

        # Write JSON line to stream
        try:
            self._stream.write(entry.to_json() + "\n")
            self._stream.flush()
        except Exception as exc:
            logger.warning("AuditLogger: failed to write entry: %s", exc)
