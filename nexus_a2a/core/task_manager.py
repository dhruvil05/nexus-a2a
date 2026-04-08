"""
nexus_a2a/core/task_manager.py

TaskManager — the single authority over task lifecycle.

Responsibilities:
  - Create tasks and persist them in the TaskStore.
  - Drive state transitions (submit → working → completed / failed).
  - Attach messages and artifacts to tasks.
  - Emit lifecycle events so other components can react (Phase 4: EventBus).

Every operation goes through TaskManager — nothing mutates a Task directly
from outside this module.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from nexus_a2a.models.task import (
    Artifact,
    Message,
    Task,
    TaskState,
)
from nexus_a2a.storage.task_store import AbstractTaskStore, InMemoryTaskStore

logger = logging.getLogger(__name__)


# ── Exceptions ────────────────────────────────────────────────────────────────

class TaskNotFoundError(Exception):
    """Raised when an operation references a task ID that does not exist."""

    def __init__(self, task_id: str) -> None:
        super().__init__(f"Task '{task_id}' not found in the store.")
        self.task_id = task_id


class TaskAlreadyDoneError(Exception):
    """Raised when trying to mutate a task that has already reached a terminal state."""

    def __init__(self, task_id: str, state: TaskState) -> None:
        super().__init__(
            f"Task '{task_id}' is already in terminal state '{state.value}' "
            "and cannot be modified."
        )
        self.task_id = task_id
        self.state = state


# ── TaskManager ───────────────────────────────────────────────────────────────

class TaskManager:
    """
    Creates, tracks, and drives Tasks through their lifecycle.

    Usage:
        manager = TaskManager()

        # Create a task
        task = await manager.create(
            initial_message=Message.user_text("Search for AI papers"),
            skill_id="web_search",
        )

        # Move it through its lifecycle
        await manager.start(task.id)
        await manager.complete(task.id, artifact=Artifact(...))

        # Retrieve it
        task = await manager.get(task.id)

    Args:
        store: The TaskStore backend to use.
               Defaults to InMemoryTaskStore (good for dev/tests).
    """

    def __init__(self, store: AbstractTaskStore | None = None) -> None:
        self._store: AbstractTaskStore = store or InMemoryTaskStore()

    # ── CRUD ─────────────────────────────────────────────────────────────────

    async def create(
        self,
        initial_message: Message,
        skill_id: str | None = None,
        context_id: str | None = None,
    ) -> Task:
        """
        Create a new Task in SUBMITTED state and persist it.

        Args:
            initial_message: The first Message from the client.
            skill_id:        Optional — which agent skill to target.
            context_id:      Optional — group related tasks under one context.

        Returns:
            The newly created Task.
        """
        task = Task.create(
            initial_message=initial_message,
            skill_id=skill_id,
            context_id=context_id,
        )
        await self._store.save(task)
        logger.info("Task created: id=%s skill=%s", task.id, skill_id)
        return task

    async def get(self, task_id: str) -> Task:
        """
        Retrieve a task by ID.

        Raises:
            TaskNotFoundError: If no task with the given ID exists.
        """
        task = await self._store.get(task_id)
        if task is None:
            raise TaskNotFoundError(task_id)
        return task

    async def list_all(self) -> list[Task]:
        """Return all tasks currently in the store."""
        return await self._store.list_all()

    async def delete(self, task_id: str) -> None:
        """
        Permanently remove a task from the store.

        Raises:
            TaskNotFoundError: If the task does not exist.
        """
        await self.get(task_id)   # raises if not found
        await self._store.delete(task_id)
        logger.info("Task deleted: id=%s", task_id)

    # ── Lifecycle transitions ─────────────────────────────────────────────────

    async def start(self, task_id: str) -> Task:
        """
        Move a task from SUBMITTED → WORKING.
        Call this when the agent begins processing.

        Raises:
            TaskNotFoundError:    Task does not exist.
            TaskAlreadyDoneError: Task is already in a terminal state.
        """
        task = await self._get_active(task_id)
        task.transition(TaskState.WORKING)
        await self._store.save(task)
        logger.info("Task started: id=%s", task_id)
        return task

    async def complete(
        self,
        task_id: str,
        artifact: Artifact | None = None,
        reply_message: Message | None = None,
    ) -> Task:
        """
        Move a task to COMPLETED and optionally attach an artifact or message.

        Args:
            task_id:       The task to complete.
            artifact:      Optional output artifact to attach.
            reply_message: Optional final agent message to append to history.

        Raises:
            TaskNotFoundError:    Task does not exist.
            TaskAlreadyDoneError: Task is already terminal.
        """
        task = await self._get_active(task_id)

        if reply_message:
            task.add_message(reply_message)
        if artifact:
            task.add_artifact(artifact)

        task.transition(TaskState.COMPLETED)
        await self._store.save(task)
        logger.info("Task completed: id=%s artifacts=%d", task_id, len(task.artifacts))
        return task

    async def fail(self, task_id: str, error: str) -> Task:
        """
        Move a task to FAILED with an error message.

        Args:
            task_id: The task to fail.
            error:   Human-readable description of what went wrong.

        Raises:
            TaskNotFoundError:    Task does not exist.
            TaskAlreadyDoneError: Task is already terminal.
        """
        task = await self._get_active(task_id)
        task.transition(TaskState.FAILED, error=error)
        await self._store.save(task)
        logger.warning("Task failed: id=%s error=%r", task_id, error)
        return task

    async def cancel(self, task_id: str) -> Task:
        """
        Move a task to CANCELLED.
        Valid from SUBMITTED, WORKING, or INPUT_REQUIRED states.

        Raises:
            TaskNotFoundError:    Task does not exist.
            TaskAlreadyDoneError: Task is already terminal.
        """
        task = await self._get_active(task_id)
        task.transition(TaskState.CANCELLED)
        await self._store.save(task)
        logger.info("Task cancelled: id=%s", task_id)
        return task

    async def request_input(self, task_id: str, prompt: Message) -> Task:
        """
        Move a task to INPUT_REQUIRED and append a prompt message.
        Use when the agent needs more information from the client before continuing.

        Args:
            task_id: The task awaiting input.
            prompt:  The agent's message asking for more information.

        Raises:
            TaskNotFoundError:    Task does not exist.
            TaskAlreadyDoneError: Task is already terminal.
        """
        task = await self._get_active(task_id)
        task.add_message(prompt)
        task.transition(TaskState.INPUT_REQUIRED)
        await self._store.save(task)
        logger.info("Task awaiting input: id=%s", task_id)
        return task

    async def provide_input(self, task_id: str, message: Message) -> Task:
        """
        Append a client reply to an INPUT_REQUIRED task and resume it (→ WORKING).

        Args:
            task_id: The task to resume.
            message: The client's reply message.

        Raises:
            TaskNotFoundError:    Task does not exist.
            TaskAlreadyDoneError: Task is already terminal.
            ValueError:           Task is not in INPUT_REQUIRED state.
        """
        task = await self._get_active(task_id)

        if task.state != TaskState.INPUT_REQUIRED:
            raise ValueError(
                f"Task '{task_id}' is in state '{task.state.value}', "
                "not 'input_required'. Cannot provide input."
            )

        task.add_message(message)
        task.transition(TaskState.WORKING)
        await self._store.save(task)
        logger.info("Task resumed with input: id=%s", task_id)
        return task

    # ── Convenience helpers ───────────────────────────────────────────────────

    async def add_message(self, task_id: str, message: Message) -> Task:
        """
        Append a message to a task's history without changing its state.
        Useful for logging intermediate agent thoughts or streaming chunks.
        """
        task = await self._get_active(task_id)
        task.add_message(message)
        await self._store.save(task)
        return task

    async def iter_by_state(self, state: TaskState) -> AsyncIterator[Task]:
        """
        Async-iterate over all tasks currently in a given state.

        Example:
            async for task in manager.iter_by_state(TaskState.WORKING):
                print(task.id)
        """
        for task in await self._store.list_all():
            if task.state == state:
                yield task

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _get_active(self, task_id: str) -> Task:
        """
        Fetch a task and assert it has not reached a terminal state.

        Raises:
            TaskNotFoundError:    Task does not exist in the store.
            TaskAlreadyDoneError: Task is already completed / failed / cancelled.
        """
        task = await self.get(task_id)
        if task.is_done():
            raise TaskAlreadyDoneError(task_id, task.state)
        return task
