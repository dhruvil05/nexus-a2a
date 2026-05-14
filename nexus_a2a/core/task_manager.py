"""
nexus_a2a/core/task_manager.py

TaskManager — upgraded for v1.1 with:
  - Task timeout watchdog: auto-fails any task stuck in WORKING
    longer than timeout_sec.
  - On-timeout callback: optional hook called when a task times out,
    so you can trigger DLQ, send alerts, etc.

The watchdog runs as a background asyncio task. It wakes every
`watchdog_interval` seconds and scans for timed-out tasks.

Usage with timeout:
    manager = TaskManager(timeout_sec=120)
    await manager.start_watchdog()    # start background scan
    ...
    await manager.stop_watchdog()     # clean shutdown

Or as async context manager:
    async with TaskManager(timeout_sec=120) as manager:
        task = await manager.create(Message.user_text("hello"))
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable

from nexus_a2a.models.task import (
    Artifact,
    Message,
    Task,
    TaskState,
)
from nexus_a2a.storage.task_store import AbstractTaskStore, InMemoryTaskStore

logger = logging.getLogger(__name__)

# Callback type: called when a task times out
TimeoutCallback = Callable[[Task], Awaitable[None]]


# ── Exceptions ────────────────────────────────────────────────────────────────

class TaskNotFoundError(Exception):
    def __init__(self, task_id: str) -> None:
        super().__init__(f"Task '{task_id}' not found in the store.")
        self.task_id = task_id


class TaskAlreadyDoneError(Exception):
    def __init__(self, task_id: str, state: TaskState) -> None:
        super().__init__(
            f"Task '{task_id}' is already terminal ('{state.value}')."
        )
        self.task_id = task_id
        self.state   = state


class TaskTimeoutError(Exception):
    """Raised (or logged) when the watchdog auto-fails a timed-out task."""
    def __init__(self, task_id: str, timeout_sec: float) -> None:
        super().__init__(
            f"Task '{task_id}' timed out after {timeout_sec:.0f}s in WORKING state."
        )
        self.task_id    = task_id
        self.timeout_sec = timeout_sec


# ── TaskManager ───────────────────────────────────────────────────────────────

class TaskManager:
    """
    Creates, tracks, and drives Tasks through their lifecycle.

    v1.1 additions:
      - timeout_sec: auto-fail tasks stuck in WORKING beyond this duration.
      - on_timeout:  optional async callback called when timeout fires.
      - start_watchdog() / stop_watchdog(): manage the background scanner.
      - Async context manager support for clean startup/shutdown.

    Usage (basic — same as before):
        manager = TaskManager()
        task    = await manager.create(Message.user_text("hello"))
        await manager.start(task.id)
        await manager.complete(task.id)

    Usage (with timeout watchdog):
        async with TaskManager(timeout_sec=60) as manager:
            task = await manager.create(Message.user_text("hello"))
            await manager.start(task.id)
            # if agent takes >60s, task auto-fails

    Args:
        store:             TaskStore backend. Default: InMemoryTaskStore.
        timeout_sec:       Seconds before a WORKING task is auto-failed.
                           None = no timeout (default).
        watchdog_interval: How often the watchdog scans. Default: 10s.
        on_timeout:        Async callback(task) called on timeout.
    """

    def __init__(
        self,
        store:             AbstractTaskStore | None = None,
        timeout_sec:       float | None             = None,
        watchdog_interval: float                    = 10.0,
        on_timeout:        TimeoutCallback | None   = None,
    ) -> None:
        self._store             = store or InMemoryTaskStore()
        self._timeout_sec       = timeout_sec
        self._watchdog_interval = watchdog_interval
        self._on_timeout        = on_timeout
        self._watchdog_task: asyncio.Task | None = None

    # ── Async context manager ─────────────────────────────────────────────────

    async def __aenter__(self) -> TaskManager:
        if self._timeout_sec is not None:
            await self.start_watchdog()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop_watchdog()

    # ── Watchdog ──────────────────────────────────────────────────────────────

    async def start_watchdog(self) -> None:
        """
        Start the background timeout watchdog.
        Safe to call multiple times — only one watchdog runs at a time.
        """
        if self._watchdog_task and not self._watchdog_task.done():
            return
        self._watchdog_task = asyncio.create_task(
            self._watchdog_loop(),
            name="nexus-task-watchdog",
        )
        logger.info(
            "TaskManager watchdog started (timeout=%.0fs, interval=%.0fs)",
            self._timeout_sec, self._watchdog_interval,
        )

    async def stop_watchdog(self) -> None:
        """Stop the background watchdog gracefully."""
        if self._watchdog_task and not self._watchdog_task.done():
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass
        self._watchdog_task = None
        logger.info("TaskManager watchdog stopped")

    async def _watchdog_loop(self) -> None:
        """Periodically scan for timed-out WORKING tasks."""
        while True:
            try:
                await asyncio.sleep(self._watchdog_interval)
                await self._check_timeouts()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Watchdog scan error: %s", exc)

    async def _check_timeouts(self) -> None:
        """Find and auto-fail any WORKING tasks past their timeout."""
        if self._timeout_sec is None:
            return

        import time
        now    = time.time()
        failed = 0

        for task in await self._store.list_all():
            if task.state != TaskState.WORKING:
                continue
            age = now - task.updated_at.timestamp()
            if age < self._timeout_sec:
                continue

            reason = (
                f"Task auto-failed by watchdog: "
                f"exceeded timeout of {self._timeout_sec:.0f}s "
                f"(was WORKING for {age:.0f}s)"
            )
            try:
                task.transition(TaskState.FAILED, error=reason)
                await self._store.save(task)
                failed += 1
                logger.warning(
                    "Watchdog timed out task %s (age=%.0fs)", task.id, age
                )
                if self._on_timeout:
                    await self._on_timeout(task)
            except Exception as exc:
                logger.error(
                    "Watchdog failed to expire task %s: %s", task.id, exc
                )

        if failed:
            logger.info("Watchdog expired %d task(s)", failed)

    # ── CRUD ──────────────────────────────────────────────────────────────────

    async def create(
        self,
        initial_message: Message,
        skill_id:    str | None = None,
        context_id:  str | None = None,
    ) -> Task:
        """Create a new Task in SUBMITTED state and persist it."""
        task = Task.create(
            initial_message=initial_message,
            skill_id=skill_id,
            context_id=context_id,
        )
        await self._store.save(task)
        logger.info("Task created: id=%s skill=%s", task.id, skill_id)
        return task

    async def get(self, task_id: str) -> Task:
        """Retrieve a task. Raises TaskNotFoundError if missing."""
        task = await self._store.get(task_id)
        if task is None:
            raise TaskNotFoundError(task_id)
        return task

    async def list_all(self) -> list[Task]:
        """Return all tasks in the store."""
        return await self._store.list_all()

    async def delete(self, task_id: str) -> None:
        """Remove a task permanently."""
        await self.get(task_id)
        await self._store.delete(task_id)

    # ── Lifecycle transitions ─────────────────────────────────────────────────

    async def start(self, task_id: str) -> Task:
        """SUBMITTED → WORKING."""
        task = await self._get_active(task_id)
        task.transition(TaskState.WORKING)
        await self._store.save(task)
        logger.info("Task started: id=%s", task_id)
        return task

    async def complete(
        self,
        task_id:       str,
        artifact:      Artifact | None = None,
        reply_message: Message | None  = None,
    ) -> Task:
        """WORKING → COMPLETED. Optionally attach artifact and reply."""
        task = await self._get_active(task_id)
        if reply_message:
            task.add_message(reply_message)
        if artifact:
            task.add_artifact(artifact)
        task.transition(TaskState.COMPLETED)
        await self._store.save(task)
        logger.info("Task completed: id=%s", task_id)
        return task

    async def fail(self, task_id: str, error: str) -> Task:
        """WORKING → FAILED with error message."""
        task = await self._get_active(task_id)
        task.transition(TaskState.FAILED, error=error)
        await self._store.save(task)
        logger.warning("Task failed: id=%s error=%r", task_id, error)
        return task

    async def cancel(self, task_id: str) -> Task:
        """Move task to CANCELLED (from SUBMITTED, WORKING, or INPUT_REQUIRED)."""
        task = await self._get_active(task_id)
        task.transition(TaskState.CANCELLED)
        await self._store.save(task)
        logger.info("Task cancelled: id=%s", task_id)
        return task

    async def request_input(self, task_id: str, prompt: Message) -> Task:
        """WORKING → INPUT_REQUIRED with a prompt message."""
        task = await self._get_active(task_id)
        task.add_message(prompt)
        task.transition(TaskState.INPUT_REQUIRED)
        await self._store.save(task)
        return task

    async def provide_input(self, task_id: str, message: Message) -> Task:
        """INPUT_REQUIRED → WORKING after client provides a reply."""
        task = await self._get_active(task_id)
        if task.state != TaskState.INPUT_REQUIRED:
            raise ValueError(
                f"Task '{task_id}' is '{task.state.value}', not 'input_required'."
            )
        task.add_message(message)
        task.transition(TaskState.WORKING)
        await self._store.save(task)
        return task

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def add_message(self, task_id: str, message: Message) -> Task:
        """Append a message without changing state."""
        task = await self._get_active(task_id)
        task.add_message(message)
        await self._store.save(task)
        return task

    async def iter_by_state(self, state: TaskState) -> AsyncIterator[Task]:
        """Async-iterate over all tasks in a given state."""
        for task in await self._store.list_all():
            if task.state == state:
                yield task

    async def _get_active(self, task_id: str) -> Task:
        """Fetch task and assert it is not terminal."""
        task = await self.get(task_id)
        if task.is_done():
            raise TaskAlreadyDoneError(task_id, task.state)
        return task
