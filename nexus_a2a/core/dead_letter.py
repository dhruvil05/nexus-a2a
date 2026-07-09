"""
nexus_a2a/core/dead_letter.py

DeadLetterQueue — captures FAILED tasks and provides structured replay.

The problem this solves:
  Before v1.1, FAILED tasks just sat in the TaskStore forever.
  There was no built-in way to inspect them, retry them, or alert
  on them. Developers had to build this themselves.

What the DLQ provides:
  - Automatic capture of FAILED tasks (via on_failure hook in TaskManager).
  - Manual and automatic replay with configurable backoff.
  - Failure hooks — async callbacks fired when a task fails.
  - Filtering — replay by skill_id, agent_url, time range.
  - Full inspection — see what failed, why, and when.

Usage:
    dlq = DeadLetterQueue(network=network)

    # Register a failure alert
    @dlq.on_failure
    async def alert(entry: DLQEntry) -> None:
        await send_slack_alert(f"Task {entry.task_id} failed: {entry.error}")

    # Later — replay everything that failed
    results = await dlq.replay_all()

    # Or replay one specific task
    result = await dlq.replay(task_id="abc-123")

    # Or replay with a filter
    results = await dlq.replay_where(skill_id="web_search")
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from nexus_a2a.models.task import Message, Task

logger = logging.getLogger(__name__)

# Callback type for failure hooks
FailureHook = Callable[["DLQEntry"], Awaitable[None]]

# Callback type for the replay runner
# Takes (agent_url, message) and returns a Task — same as Orchestrator runner
ReplayRunner = Callable[[str, Message], Awaitable[Task]]


# ── DLQ entry ─────────────────────────────────────────────────────────────────


@dataclass
class DLQEntry:
    """
    One record in the dead letter queue.

    Fields:
        task:         The original failed Task (snapshot at time of failure).
        error:        The error message that caused the failure.
        failed_at:    Unix timestamp when the task failed.
        agent_url:    The agent that was processing the task (if known).
        skill_id:     The skill that was being invoked (if known).
        retry_count:  How many replay attempts have been made.
        last_retry_at: Unix timestamp of the most recent replay attempt.
        replayed:     True once a replay succeeds.
    """

    task: Task
    error: str
    failed_at: float = field(default_factory=time.time)
    agent_url: str | None = None
    skill_id: str | None = None
    retry_count: int = 0
    last_retry_at: float | None = None
    replayed: bool = False

    @property
    def task_id(self) -> str:
        return self.task.id

    @property
    def original_message(self) -> Message | None:
        """The first message in the task's history — the original input."""
        return self.task.history[0] if self.task.history else None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "error": self.error,
            "failed_at": self.failed_at,
            "agent_url": self.agent_url,
            "skill_id": self.skill_id,
            "retry_count": self.retry_count,
            "last_retry_at": self.last_retry_at,
            "replayed": self.replayed,
        }


# ── Replay result ─────────────────────────────────────────────────────────────


@dataclass
class ReplayResult:
    """Outcome of a single DLQ replay attempt."""

    task_id: str
    succeeded: bool
    new_task: Task | None = None
    error: str | None = None


# ── DeadLetterQueue ───────────────────────────────────────────────────────────


class DeadLetterQueue:
    """
    Captures FAILED tasks, fires failure hooks, and replays them.

    Usage with AgentNetwork (recommended):
        # AgentNetwork automatically passes dlq to task routing
        dlq = DeadLetterQueue(runner=network._run_agent)
        network.dead_letter_queue = dlq

        # Register hooks
        @dlq.on_failure
        async def on_fail(entry: DLQEntry) -> None:
            print(f"FAILED: {entry.task_id} — {entry.error}")

        # Capture a failed task
        await dlq.capture(task, agent_url="http://agent:8001")

        # Replay
        await dlq.replay_all()

    Standalone usage:
        dlq = DeadLetterQueue()

        async def my_runner(url: str, msg: Message) -> Task:
            async with A2AHttpClient(url) as c:
                return await c.send_message(msg)

        dlq.set_runner(my_runner)

    Args:
        runner:          Async callable(agent_url, message) → Task.
                         Used to re-send failed tasks. Can be set later
                         via set_runner().
        max_retries:     Max replay attempts per entry. Default: 3.
        retry_delay:     Base delay between replay attempts (exponential).
        max_queue_size:  Max DLQ entries kept. Oldest dropped when full.
    """

    def __init__(
        self,
        runner: ReplayRunner | None = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        max_queue_size: int = 500,
    ) -> None:
        self._runner = runner
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._max_size = max_queue_size
        # task_id → DLQEntry
        self._entries: dict[str, DLQEntry] = {}
        self._hooks: list[FailureHook] = []
        self._lock = asyncio.Lock()

    def set_runner(self, runner: ReplayRunner) -> None:
        """Set (or replace) the replay runner after construction."""
        self._runner = runner

    # ── Failure hook registration ─────────────────────────────────────────────

    def on_failure(self, hook: FailureHook) -> FailureHook:
        """
        Decorator to register a failure hook.
        Called immediately when a task is captured into the DLQ.

        Usage:
            @dlq.on_failure
            async def alert(entry: DLQEntry) -> None:
                await notify_team(entry.error)
        """
        self._hooks.append(hook)
        return hook

    def add_failure_hook(self, hook: FailureHook) -> None:
        """Add a failure hook programmatically (non-decorator form)."""
        self._hooks.append(hook)

    def remove_failure_hook(self, hook: FailureHook) -> None:
        """Remove a previously registered failure hook."""
        try:
            self._hooks.remove(hook)
        except ValueError:
            pass

    # ── Capture ───────────────────────────────────────────────────────────────

    async def capture(
        self,
        task: Task,
        agent_url: str | None = None,
        skill_id: str | None = None,
    ) -> DLQEntry:
        """
        Add a FAILED task to the DLQ and fire all failure hooks.

        Called automatically by AgentNetwork when a task fails.
        Can also be called manually.

        Args:
            task:      The failed Task.
            agent_url: The agent that was processing it (for routing replay).
            skill_id:  The skill that was invoked.

        Returns:
            The DLQEntry created.
        """
        entry = DLQEntry(
            task=task,
            error=task.error or "Unknown error",
            agent_url=agent_url,
            skill_id=skill_id,
        )

        async with self._lock:
            # Drop oldest if at capacity
            if len(self._entries) >= self._max_size:
                oldest = next(iter(self._entries))
                del self._entries[oldest]
            self._entries[task.id] = entry

        logger.warning(
            "DLQ captured task %s (error=%r skill=%s)",
            task.id,
            entry.error,
            skill_id,
        )

        # Fire failure hooks concurrently
        await self._fire_hooks(entry)
        return entry

    # ── Replay ────────────────────────────────────────────────────────────────

    async def replay(self, task_id: str) -> ReplayResult:
        """
        Replay one specific failed task.

        Re-sends the original message to the same agent (by agent_url)
        or routes by skill_id if agent_url is not available.

        Args:
            task_id: The task to replay.

        Returns:
            ReplayResult describing success or failure.

        Raises:
            KeyError:    task_id not in DLQ.
            RuntimeError: No runner configured.
        """
        entry = self._entries.get(task_id)
        if entry is None:
            raise KeyError(f"Task '{task_id}' is not in the dead letter queue.")

        return await self._attempt_replay(entry)

    async def replay_all(
        self,
        max_retries_filter: int | None = None,
    ) -> list[ReplayResult]:
        """
        Replay all pending (not yet successfully replayed) DLQ entries.

        Args:
            max_retries_filter: Only replay entries with fewer than N
                                previous retry attempts. None = replay all.

        Returns:
            List of ReplayResult, one per entry attempted.
        """
        entries = [
            e
            for e in self._entries.values()
            if not e.replayed
            and (max_retries_filter is None or e.retry_count < max_retries_filter)
        ]

        if not entries:
            logger.info("DLQ replay_all: nothing to replay")
            return []

        logger.info("DLQ replay_all: replaying %d entries", len(entries))
        results = []
        for entry in entries:
            result = await self._attempt_replay(entry)
            results.append(result)
            # Small gap between replays to avoid hammering agents
            if not result.succeeded:
                await asyncio.sleep(self._retry_delay)

        succeeded = sum(1 for r in results if r.succeeded)
        logger.info(
            "DLQ replay_all complete: %d/%d succeeded",
            succeeded,
            len(results),
        )
        return results

    async def replay_where(
        self,
        skill_id: str | None = None,
        agent_url: str | None = None,
    ) -> list[ReplayResult]:
        """
        Replay DLQ entries matching a filter.

        Args:
            skill_id:  Only replay entries with this skill_id.
            agent_url: Only replay entries from this agent.

        Returns:
            List of ReplayResult for matched entries.
        """
        entries = [
            e
            for e in self._entries.values()
            if not e.replayed
            and (skill_id is None or e.skill_id == skill_id)
            and (agent_url is None or e.agent_url == agent_url)
        ]
        results = []
        for entry in entries:
            results.append(await self._attempt_replay(entry))
        return results

    # ── Inspection ────────────────────────────────────────────────────────────

    def all_entries(self) -> list[DLQEntry]:
        """Return all DLQ entries (pending and replayed)."""
        return list(self._entries.values())

    def pending_entries(self) -> list[DLQEntry]:
        """Return entries not yet successfully replayed."""
        return [e for e in self._entries.values() if not e.replayed]

    def get_entry(self, task_id: str) -> DLQEntry | None:
        """Return the DLQ entry for a task_id, or None."""
        return self._entries.get(task_id)

    def count(self) -> int:
        """Total number of entries in the DLQ."""
        return len(self._entries)

    def pending_count(self) -> int:
        """Number of entries not yet successfully replayed."""
        return sum(1 for e in self._entries.values() if not e.replayed)

    def clear_replayed(self) -> int:
        """Remove successfully replayed entries. Returns count removed."""
        to_remove = [tid for tid, e in self._entries.items() if e.replayed]
        for tid in to_remove:
            del self._entries[tid]
        return len(to_remove)

    def summary(self) -> dict:
        """Return a human-readable summary of the DLQ state."""
        return {
            "total": self.count(),
            "pending": self.pending_count(),
            "entries": [e.to_dict() for e in self._entries.values()],
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _attempt_replay(self, entry: DLQEntry) -> ReplayResult:
        """Try to replay one DLQ entry with exponential backoff."""
        if self._runner is None:
            raise RuntimeError(
                "DeadLetterQueue has no runner configured. "
                "Call dlq.set_runner(runner) before replaying."
            )

        msg = entry.original_message
        if msg is None:
            return ReplayResult(
                task_id=entry.task_id,
                succeeded=False,
                error="Task has no history — cannot determine original message.",
            )

        url = entry.agent_url
        if url is None:
            return ReplayResult(
                task_id=entry.task_id,
                succeeded=False,
                error="No agent_url recorded — cannot route replay.",
            )

        entry.retry_count += 1
        entry.last_retry_at = time.time()
        delay = min(
            self._retry_delay * (2 ** (entry.retry_count - 1)),
            60.0,
        )

        logger.info(
            "DLQ replaying task %s (attempt %d, delay=%.1fs)",
            entry.task_id,
            entry.retry_count,
            delay,
        )

        try:
            new_task = await self._runner(url, msg)
            entry.replayed = True
            logger.info("DLQ replay succeeded for task %s", entry.task_id)
            return ReplayResult(
                task_id=entry.task_id,
                succeeded=True,
                new_task=new_task,
            )
        except Exception as exc:
            error = str(exc)
            logger.warning("DLQ replay failed for task %s: %s", entry.task_id, error)
            if entry.retry_count >= self._max_retries:
                logger.error(
                    "DLQ task %s exhausted %d retries — giving up",
                    entry.task_id,
                    self._max_retries,
                )
            return ReplayResult(
                task_id=entry.task_id,
                succeeded=False,
                error=error,
            )

    async def _fire_hooks(self, entry: DLQEntry) -> None:
        """Call all failure hooks concurrently. Swallows individual errors."""

        async def _safe(hook: FailureHook) -> None:
            try:
                await hook(entry)
            except Exception as exc:
                logger.error("DLQ failure hook error: %s", exc)

        if self._hooks:
            await asyncio.gather(*(_safe(h) for h in self._hooks))
