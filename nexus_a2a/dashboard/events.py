"""
nexus_a2a/dashboard/events.py

EventHub — the in-process record of what an agent is doing, for the dashboard.

Everything here is fed by events as they happen (TaskManager listeners,
A2AServer security and webhook hooks, the log handler), so serving the
dashboard never scans the task store. All buffers are bounded.

Single-process by design (nexus-a2a 2.1): with several replicas, each one's
dashboard shows that replica only.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from collections import Counter, OrderedDict, deque
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from nexus_a2a.models.task import Task, TaskState

logger = logging.getLogger(__name__)

_TERMINAL = {
    TaskState.COMPLETED.value,
    TaskState.FAILED.value,
    TaskState.CANCELLED.value,
}
_IN_FLIGHT = (
    TaskState.SUBMITTED.value,
    TaskState.WORKING.value,
    TaskState.INPUT_REQUIRED.value,
)
_EVENT_STATE = {
    "created": TaskState.SUBMITTED.value,
    "started": TaskState.WORKING.value,
    "resumed": TaskState.WORKING.value,
    "input_required": TaskState.INPUT_REQUIRED.value,
    "completed": TaskState.COMPLETED.value,
    "failed": TaskState.FAILED.value,
    "timed_out": TaskState.FAILED.value,
    "cancelled": TaskState.CANCELLED.value,
}

THROUGHPUT_MINUTES = 30
PREVIEW_CHARS = 120


def iso(ts: float | datetime | None = None) -> str:
    """ISO-8601 UTC with a trailing Z."""
    if ts is None:
        dt = datetime.now(UTC)
    elif isinstance(ts, datetime):
        dt = ts if ts.tzinfo else ts.replace(tzinfo=UTC)
        dt = dt.astimezone(UTC)
    else:
        dt = datetime.fromtimestamp(ts, UTC)
    return dt.isoformat().replace("+00:00", "Z")


def state_of(task: Task) -> str:
    state = task.state
    return state.value if hasattr(state, "value") else str(state)


def task_preview(task: Task) -> str | None:
    """First characters of the task's first message text."""
    if not task.history:
        return None
    text = task.history[0].text() or ""
    text = " ".join(text.split())
    return text[:PREVIEW_CHARS] if text else None


def task_summary(task: Task) -> dict[str, Any]:
    """The TaskSummary shape from API.md (preview included; callers strip it)."""
    state = state_of(task)
    duration = None
    if state in _TERMINAL:
        duration = round(
            (task.updated_at - task.created_at).total_seconds() * 1000.0, 1
        )
    return {
        "id": task.id,
        "state": state,
        "skill_id": task.skill_id,
        "context_id": task.context_id,
        "owner": task.owner,
        "created_at": iso(task.created_at),
        "updated_at": iso(task.updated_at),
        "duration_ms": duration,
        "error": task.error,
        "preview": task_preview(task),
    }


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100.0 * (len(ordered) - 1))))
    return round(ordered[index], 1)


class EventHub:
    """
    Bounded, incrementally-maintained view of one process's activity.

    Args:
        max_tasks:        Tasks kept in the recent index (list views).
        max_events:       Security / webhook events kept.
        max_timeline:     Timeline entries kept per task.
        max_subscribers:  Concurrent live streams.
        queue_size:       Events buffered per live stream before it is dropped.
    """

    def __init__(
        self,
        max_tasks: int = 1000,
        max_events: int = 500,
        max_timeline: int = 50,
        max_subscribers: int = 20,
        queue_size: int = 1000,
    ) -> None:
        self.started_at = time.time()
        self._max_tasks = max_tasks
        self._max_timeline = max_timeline
        self._max_subscribers = max_subscribers
        self._queue_size = queue_size

        self._recent: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._timelines: OrderedDict[str, deque[dict[str, str]]] = OrderedDict()
        # Last known state of tasks not yet finished — bounded by in-flight count.
        self._open_state: dict[str, str] = {}
        self._counts: Counter[str] = Counter()
        self.total_seen = 0
        self._seeded = False

        # (minute epoch, completed, failed)
        self._minutes: deque[list[int]] = deque(maxlen=THROUGHPUT_MINUTES)
        self._latencies: deque[tuple[float, float]] = deque(maxlen=1000)
        self._terminal_times: deque[tuple[float, bool]] = deque(maxlen=5000)

        self._security: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._webhooks: deque[dict[str, Any]] = deque(maxlen=max_events)
        self.webhooks_delivered = 0
        self.webhooks_failed = 0

        self._subscribers: set[asyncio.Queue[tuple[str, dict[str, Any]] | None]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()

    # ── Seeding ───────────────────────────────────────────────────────────────

    @property
    def seeded(self) -> bool:
        return self._seeded

    def seed(self, tasks: Iterable[Task]) -> None:
        """
        Load existing tasks once (tasks created before the hub was attached),
        newest kept. Events after this keep everything current.
        """
        if self._seeded:
            return
        self._seeded = True
        ordered = sorted(tasks, key=lambda t: t.updated_at)
        for task in ordered:
            if task.id in self._recent:
                continue
            state = state_of(task)
            self._counts[state] += 1
            self.total_seen += 1
            if state not in _TERMINAL:
                self._open_state[task.id] = state
            self._remember(task)
        # Seeded tasks may have arrived out of order relative to live events.
        self._recent = OrderedDict(
            sorted(self._recent.items(), key=lambda kv: kv[1]["updated_at"])
        )
        while len(self._recent) > self._max_tasks:
            self._recent.popitem(last=False)

    # ── Inputs ────────────────────────────────────────────────────────────────

    def on_task_event(self, task: Task, event: str) -> None:
        """TaskManager listener: keep counts, index and timeline current."""
        new_state = _EVENT_STATE.get(event, state_of(task))
        previous = self._open_state.pop(task.id, None)
        if previous is not None:
            self._counts[previous] -= 1
            if self._counts[previous] <= 0:
                del self._counts[previous]
        elif event == "created":
            self.total_seen += 1
        elif task.id not in self._recent:
            # A task we never saw (created before attach, not seeded).
            self.total_seen += 1
        self._counts[new_state] += 1
        if new_state not in _TERMINAL:
            self._open_state[task.id] = new_state
        else:
            self._record_terminal(task, new_state)

        timeline = self._timelines.get(task.id)
        if timeline is None:
            timeline = deque(maxlen=self._max_timeline)
            self._timelines[task.id] = timeline
        timeline.append({"at": iso(), "event": event})
        self._timelines.move_to_end(task.id)

        summary = self._remember(task)
        self.publish("task", {"event": event, "task": summary})

    def note_event(self, task_id: str, event: str) -> None:
        """Add a timeline entry that is not a state change (e.g. 'retried')."""
        timeline = self._timelines.setdefault(task_id, deque(maxlen=self._max_timeline))
        timeline.append({"at": iso(), "event": event})

    def record_security(
        self,
        kind: str,
        detail: str,
        status: int | None = None,
        caller: str | None = None,
        client: str | None = None,
    ) -> None:
        event = {
            "at": iso(),
            "kind": kind,
            "status": status,
            "caller": caller,
            "client": client,
            "detail": detail[:500],
        }
        self._security.append(event)
        self.publish("security", event)

    def record_webhook(
        self,
        task_id: str,
        event: str,
        url: str,
        succeeded: bool,
        status: int,
        attempts: int,
        error: str | None,
    ) -> None:
        entry = {
            "at": iso(),
            "task_id": task_id,
            "event": event,
            "url": url,
            "succeeded": succeeded,
            "status": status,
            "attempts": attempts,
            "error": error,
        }
        if succeeded:
            self.webhooks_delivered += 1
        else:
            self.webhooks_failed += 1
        self._webhooks.append(entry)
        self.publish("webhook", entry)

    # ── Queries ───────────────────────────────────────────────────────────────

    def recent_tasks(self) -> list[dict[str, Any]]:
        """Newest updated first."""
        return [dict(v) for v in reversed(self._recent.values())]

    def timeline(self, task_id: str) -> list[dict[str, str]]:
        return list(self._timelines.get(task_id, ()))

    def security_events(self) -> list[dict[str, Any]]:
        return list(reversed(self._security))

    def webhook_failures(self) -> list[dict[str, Any]]:
        return [dict(w) for w in reversed(self._webhooks) if not w["succeeded"]]

    def security_events_since(self, seconds: float) -> int:
        cutoff = iso(time.time() - seconds)
        return sum(1 for e in self._security if e["at"] >= cutoff)

    def counts(self) -> dict[str, int]:
        result = {state.value: 0 for state in TaskState}
        result.update({k: v for k, v in self._counts.items() if v > 0})
        return result

    def throughput(self) -> list[dict[str, int]]:
        """Completed/failed per minute for the last 30 minutes, oldest first."""
        now_minute = int(time.time() // 60)
        by_minute = {m: (c, f) for m, c, f in self._minutes}
        return [
            {
                "completed": by_minute.get(minute, (0, 0))[0],
                "failed": by_minute.get(minute, (0, 0))[1],
            }
            for minute in range(now_minute - THROUGHPUT_MINUTES + 1, now_minute + 1)
        ]

    def window(self, seconds: float) -> tuple[int, int]:
        """(completed, failed) that finished within the last `seconds`."""
        cutoff = time.time() - seconds
        completed = failed = 0
        for at, ok in self._terminal_times:
            if at >= cutoff:
                if ok:
                    completed += 1
                else:
                    failed += 1
        return completed, failed

    def latency(self) -> dict[str, Any]:
        values = [ms for _, ms in self._latencies]
        return {
            "p50": _percentile(values, 50),
            "p95": _percentile(values, 95),
            "samples": len(values),
        }

    # ── Live subscribers ──────────────────────────────────────────────────────

    def subscribe(self) -> asyncio.Queue[tuple[str, dict[str, Any]] | None] | None:
        """A queue of (event_type, data); None when the subscriber cap is hit."""
        if len(self._subscribers) >= self._max_subscribers:
            return None
        self._loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, dict[str, Any]] | None] = asyncio.Queue(
            maxsize=self._queue_size
        )
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[Any]) -> None:
        self._subscribers.discard(queue)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def publish(self, kind: str, data: dict[str, Any]) -> None:
        """
        Fan an event out to live streams. Safe to call from any thread: off
        the event loop's thread it is handed over with call_soon_threadsafe.
        """
        if not self._subscribers:
            return
        loop = self._loop
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if loop is not None and running is not loop:
            with contextlib.suppress(RuntimeError):  # loop closed
                loop.call_soon_threadsafe(self._fan_out, kind, data)
            return
        self._fan_out(kind, data)

    def _fan_out(self, kind: str, data: dict[str, Any]) -> None:
        for queue in list(self._subscribers):
            try:
                queue.put_nowait((kind, data))
            except asyncio.QueueFull:
                # A consumer that cannot keep up is dropped rather than
                # letting its backlog grow; the UI reconnects and re-fetches.
                self._subscribers.discard(queue)
                with contextlib.suppress(asyncio.QueueFull):
                    queue.get_nowait()
                    queue.put_nowait(None)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _remember(self, task: Task) -> dict[str, Any]:
        summary = task_summary(task)
        self._recent[task.id] = summary
        self._recent.move_to_end(task.id)
        while len(self._recent) > self._max_tasks:
            old_id, _ = self._recent.popitem(last=False)
            self._timelines.pop(old_id, None)
        while len(self._timelines) > self._max_tasks:
            self._timelines.popitem(last=False)
        return dict(summary)

    def _record_terminal(self, task: Task, state: str) -> None:
        now = time.time()
        ok = state == TaskState.COMPLETED.value
        if state == TaskState.CANCELLED.value:
            return
        self._terminal_times.append((now, ok))
        minute = int(now // 60)
        if self._minutes and self._minutes[-1][0] == minute:
            bucket = self._minutes[-1]
        else:
            bucket = [minute, 0, 0]
            self._minutes.append(bucket)
        bucket[1 if ok else 2] += 1
        if ok:
            ms = (task.updated_at - task.created_at).total_seconds() * 1000.0
            if ms >= 0:
                self._latencies.append((now, ms))


__all__ = ["EventHub", "iso", "task_summary"]
