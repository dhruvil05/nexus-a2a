"""
nexus_a2a/core/input_handler.py

InputHandler — true pause/resume for INPUT_REQUIRED tasks.

The problem this solves:
  Before v1.1, INPUT_REQUIRED existed as a TaskState but had no
  server-side mechanism to actually pause execution and wait for
  a client reply. Developers had to simulate it manually.

How it works now:
  1. Agent calls `await handler.wait_for_input(task_id, timeout)`.
     This SUSPENDS the coroutine — the agent stops running.
  2. Client POSTs a reply (via TaskManager.provide_input or
     InputHandler.submit_reply).
  3. The asyncio.Event fires, the agent coroutine resumes with
     the reply Message.

This is the correct production-grade pattern — the agent is not
polling, not sleeping in a loop, not blocking a thread. It is
properly suspended via asyncio cooperative multitasking.

Usage inside an agent's run() method:
    handler = InputHandler(task_manager)

    # Ask the user a question and pause
    prompt = Message.agent_text("What is your budget?")
    reply  = await handler.wait_for_input(
        task_id=task.id,
        prompt=prompt,
        timeout=300,   # 5 minutes
    )

    # resume here with the client's reply
    budget = reply.text()

Usage on the client side (HTTP endpoint calls this):
    await handler.submit_reply(
        task_id="abc-123",
        message=Message.user_text("My budget is $500"),
    )
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from nexus_a2a.core.task_manager import TaskManager
from nexus_a2a.models.task import Message

logger = logging.getLogger(__name__)


# ── Exceptions ────────────────────────────────────────────────────────────────

class InputTimeoutError(Exception):
    """Raised when wait_for_input() times out before a reply arrives."""

    def __init__(self, task_id: str, timeout: float) -> None:
        super().__init__(
            f"Task '{task_id}' timed out waiting for client input "
            f"after {timeout:.0f}s."
        )
        self.task_id = task_id
        self.timeout = timeout


class NoInputWaiterError(Exception):
    """Raised when submit_reply() is called for a task that is not waiting."""

    def __init__(self, task_id: str) -> None:
        super().__init__(
            f"Task '{task_id}' is not currently waiting for input. "
            "Either it was never paused or the wait already timed out."
        )
        self.task_id = task_id


# ── InputHandler ──────────────────────────────────────────────────────────────

class InputHandler:
    """
    Manages the pause/resume lifecycle for INPUT_REQUIRED tasks.

    One InputHandler instance should be shared across the application
    (e.g. attached to AgentNetwork) so both the agent side and the
    HTTP endpoint side can reference the same waiter registry.

    Usage:
        handler = InputHandler(task_manager)

        # Agent side — suspends until reply or timeout
        reply = await handler.wait_for_input(
            task_id=task.id,
            prompt=Message.agent_text("What is your budget?"),
            timeout=300,
        )

        # Client/HTTP side — fires the event and resumes the agent
        await handler.submit_reply(
            task_id=task.id,
            message=Message.user_text("$500"),
        )

    Args:
        task_manager: The shared TaskManager instance.
    """

    def __init__(self, task_manager: TaskManager) -> None:
        self._manager  = task_manager
        # task_id → (asyncio.Event, Message | None)
        # Event fires when a reply is submitted.
        # Message slot starts None; filled by submit_reply().
        self._waiters: dict[str, dict[str, Any]] = {}

    # ── Agent side ────────────────────────────────────────────────────────────

    async def wait_for_input(
        self,
        task_id: str,
        prompt:  Message,
        timeout: float = 300.0,
    ) -> Message:
        """
        Transition the task to INPUT_REQUIRED, send a prompt message,
        and SUSPEND until the client submits a reply or the timeout fires.

        This is a true suspension — the coroutine yields control and
        does not consume CPU while waiting.

        Args:
            task_id: The task ID to pause.
            prompt:  Message sent to the client asking for input.
            timeout: Max seconds to wait. Raises InputTimeoutError if exceeded.

        Returns:
            The Message submitted by the client.

        Raises:
            InputTimeoutError: Client did not reply within timeout seconds.
            TaskNotFoundError: task_id does not exist.
        """
        # Transition task state
        await self._manager.request_input(task_id, prompt)

        # Register the waiter slot BEFORE suspending
        event: asyncio.Event = asyncio.Event()
        self._waiters[task_id] = {"event": event, "reply": None}

        logger.info(
            "Task %s waiting for client input (timeout=%.0fs)", task_id, timeout
        )

        try:
            # Suspend — releases event loop to other coroutines
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except TimeoutError:
            self._waiters.pop(task_id, None)
            # Auto-fail the task so it does not stay in INPUT_REQUIRED forever
            try:
                await self._manager.fail(
                    task_id,
                    error=f"Input timeout: client did not reply within {timeout:.0f}s",
                )
            except Exception:
                pass
            raise InputTimeoutError(task_id, timeout)

        # Retrieve the reply that submit_reply() stored
        reply: Message | None = self._waiters.pop(task_id, {}).get("reply")

        if reply is None:
            # Should not happen in normal flow — defensive guard
            raise NoInputWaiterError(task_id)

        # Resume the task (INPUT_REQUIRED → WORKING)
        await self._manager.provide_input(task_id, reply)
        logger.info("Task %s resumed with client input", task_id)
        return reply

    # ── Client / HTTP side ────────────────────────────────────────────────────

    async def submit_reply(self, task_id: str, message: Message) -> None:
        """
        Accept the client's reply and resume the waiting agent coroutine.

        This is called from the HTTP endpoint that receives the client's
        POST to /tasks/{task_id}/reply.

        Args:
            task_id: The task to resume.
            message: The client's reply message.

        Raises:
            NoInputWaiterError: The task is not currently waiting for input.
        """
        waiter = self._waiters.get(task_id)
        if waiter is None:
            raise NoInputWaiterError(task_id)

        # Store the reply and fire the event — this wakes the suspended coroutine
        waiter["reply"] = message
        waiter["event"].set()
        logger.info("Input submitted for task %s", task_id)

    # ── Introspection ─────────────────────────────────────────────────────────

    def is_waiting(self, task_id: str) -> bool:
        """Return True if the task is currently suspended waiting for input."""
        return task_id in self._waiters

    def waiting_task_ids(self) -> list[str]:
        """Return all task IDs currently suspended waiting for input."""
        return list(self._waiters.keys())

    def waiting_count(self) -> int:
        """Return the number of tasks currently suspended."""
        return len(self._waiters)
