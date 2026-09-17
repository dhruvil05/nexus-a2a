"""
nexus_a2a/storage/task_store.py

TaskStore — persists Task objects during their lifecycle.

Phase 2 ships InMemoryTaskStore (zero config, ideal for dev/testing).
Phase 5 will add RedisTaskStore and PostgresTaskStore as drop-in replacements.

All stores implement the same AbstractTaskStore interface, so swapping
backends in production requires changing one line.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable

from nexus_a2a.models.task import Task

logger = logging.getLogger(__name__)

# A finished task stays retrievable this long. Matches RedisTaskStore's
# default TTL so the two backends forget on the same schedule.
_DEFAULT_RETENTION_SEC = 3600.0

# Upper bound on stored tasks; only finished tasks are evicted to meet it.
_DEFAULT_MAX_TASKS = 10_000

# ── Abstract interface ────────────────────────────────────────────────────────


class AbstractTaskStore(ABC):
    """
    Interface every TaskStore backend must implement.
    The TaskManager depends only on this interface, never on a concrete class.
    """

    @abstractmethod
    async def save(self, task: Task) -> None:
        """Persist a task (create or overwrite)."""

    @abstractmethod
    async def get(self, task_id: str) -> Task | None:
        """Return the task with the given ID, or None if not found."""

    @abstractmethod
    async def delete(self, task_id: str) -> None:
        """Remove a task permanently."""

    @abstractmethod
    async def list_all(self) -> list[Task]:
        """Return every task currently in the store."""


# ── In-memory implementation (Phase 2 default) ───────────────────────────────


class InMemoryTaskStore(AbstractTaskStore):
    """
    Stores tasks in a plain Python dict — no external dependencies.

    Characteristics:
    - Zero config: works out of the box.
    - Thread-safe via asyncio.Lock (safe for concurrent async code).
    - Not persistent: all tasks are lost when the process exits.
    - Not distributed: tasks are local to one process.
    - Bounded (since 1.9.0): finished tasks are evicted, see below.

    Retention:
        Before 1.9.0 this store never forgot anything. The task watchdog only
        moves timed-out tasks to FAILED, and A2AServer never deletes, so every
        request a default server handled kept its full task — history and
        artifacts included — until the process exited.

        Now a task is evicted once it has been in a terminal state (COMPLETED,
        FAILED, CANCELLED) for `retention_sec`, and the oldest finished tasks
        are evicted first when the store holds more than `max_tasks`. A task
        that is still running or waiting for input is NEVER evicted, whatever
        the cap — dropping live work would be worse than exceeding a limit.

        The default retention matches RedisTaskStore's default TTL (1 hour).
        Pass retention_sec=None and max_tasks=None for the old behaviour.

    Args:
        retention_sec: Seconds a finished task stays retrievable. None keeps
                       finished tasks until max_tasks forces them out.
        max_tasks:     Upper bound on stored tasks. Only finished tasks are
                       evicted to honour it. None means no cap.
        clock:         Time source, injectable for tests.

    Use this for: development, testing, single-process deployments.
    Switch to RedisTaskStore for multi-process / distributed setups.
    """

    def __init__(
        self,
        retention_sec: float | None = _DEFAULT_RETENTION_SEC,
        max_tasks: int | None = _DEFAULT_MAX_TASKS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if retention_sec is not None and retention_sec < 0:
            raise ValueError(f"retention_sec must be >= 0, got {retention_sec}")
        if max_tasks is not None and max_tasks <= 0:
            raise ValueError(f"max_tasks must be > 0, got {max_tasks}")

        # task_id → Task
        self._store: dict[str, Task] = {}
        # Finished task ids in the order they finished, with when. Oldest first,
        # so both expiry and cap enforcement pop from the front in O(1).
        self._finished: OrderedDict[str, float] = OrderedDict()
        self._retention = retention_sec
        self._max_tasks = max_tasks
        self._clock = clock
        # Prevents race conditions when multiple coroutines read/write concurrently
        self._lock = asyncio.Lock()

    async def save(self, task: Task) -> None:
        """
        Save or overwrite a task, then evict whatever retention says to.

        Args:
            task: The Task object to persist.
        """
        async with self._lock:
            self._store[task.id] = task
            if task.is_done():
                # First time we see it finished starts its retention window;
                # a later re-save of a finished task does not extend it.
                if task.id not in self._finished:
                    self._finished[task.id] = self._clock()
            else:
                self._finished.pop(task.id, None)
            self._evict_locked()

    def _evict_locked(self) -> int:
        """Drop expired and over-cap finished tasks. Caller holds the lock."""
        evicted = 0

        if self._retention is not None:
            cutoff = self._clock() - self._retention
            while self._finished:
                task_id, finished_at = next(iter(self._finished.items()))
                if finished_at > cutoff:
                    break
                self._finished.popitem(last=False)
                self._store.pop(task_id, None)
                evicted += 1

        if self._max_tasks is not None:
            while len(self._store) > self._max_tasks and self._finished:
                task_id, _ = self._finished.popitem(last=False)
                self._store.pop(task_id, None)
                evicted += 1
            if len(self._store) > self._max_tasks:
                logger.warning(
                    "InMemoryTaskStore holds %d tasks, above max_tasks=%d, but "
                    "all of them are still active and none can be evicted.",
                    len(self._store),
                    self._max_tasks,
                )

        if evicted:
            logger.debug("InMemoryTaskStore evicted %d finished task(s)", evicted)
        return evicted

    async def evict_expired(self) -> int:
        """
        Apply retention now and return how many tasks were dropped.

        Retention also runs on every save and listing, so this is only needed
        to reclaim memory on a server that has gone idle.
        """
        async with self._lock:
            return self._evict_locked()

    async def get(self, task_id: str) -> Task | None:
        """
        Retrieve a task by its ID.

        Args:
            task_id: The unique task identifier.

        Returns:
            The Task if found, otherwise None.
        """
        async with self._lock:
            # Expired tasks must stop being readable, not linger until the
            # next write happens to trigger eviction.
            self._evict_locked()
            return self._store.get(task_id)

    async def delete(self, task_id: str) -> None:
        """
        Remove a task from the store.
        Silently does nothing if the task does not exist.

        Args:
            task_id: The unique task identifier.
        """
        async with self._lock:
            self._store.pop(task_id, None)
            self._finished.pop(task_id, None)

    async def list_all(self) -> list[Task]:
        """
        Return a snapshot of all tasks currently in the store.
        The returned list is a copy — modifying it does not affect the store.
        """
        async with self._lock:
            self._evict_locked()
            return list(self._store.values())

    async def count(self) -> int:
        """Return the number of tasks currently stored."""
        async with self._lock:
            self._evict_locked()
            return len(self._store)

    async def clear(self) -> None:
        """
        Remove all tasks from the store.
        Mainly useful in tests to reset state between test cases.
        """
        async with self._lock:
            self._store.clear()
            self._finished.clear()
