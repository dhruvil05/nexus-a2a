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
from abc import ABC, abstractmethod

from nexus_a2a.models.task import Task

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

    Use this for: development, testing, single-process deployments.
    Switch to RedisTaskStore (Phase 5) for multi-process / distributed setups.
    """

    def __init__(self) -> None:
        # task_id → Task
        self._store: dict[str, Task] = {}
        # Prevents race conditions when multiple coroutines read/write concurrently
        self._lock = asyncio.Lock()

    async def save(self, task: Task) -> None:
        """
        Save or overwrite a task.

        Args:
            task: The Task object to persist.
        """
        async with self._lock:
            self._store[task.id] = task

    async def get(self, task_id: str) -> Task | None:
        """
        Retrieve a task by its ID.

        Args:
            task_id: The unique task identifier.

        Returns:
            The Task if found, otherwise None.
        """
        async with self._lock:
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

    async def list_all(self) -> list[Task]:
        """
        Return a snapshot of all tasks currently in the store.
        The returned list is a copy — modifying it does not affect the store.
        """
        async with self._lock:
            return list(self._store.values())

    async def count(self) -> int:
        """Return the number of tasks currently stored."""
        async with self._lock:
            return len(self._store)

    async def clear(self) -> None:
        """
        Remove all tasks from the store.
        Mainly useful in tests to reset state between test cases.
        """
        async with self._lock:
            self._store.clear()
