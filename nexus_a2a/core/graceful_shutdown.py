"""
nexus_a2a/core/graceful_shutdown.py

GracefulShutdown — SIGTERM/SIGINT drain handler for production deployments.

What it does:
    1. Traps SIGTERM (Kubernetes pod eviction, `docker stop`) and SIGINT (Ctrl-C).
    2. Marks the agent as "draining" — new tasks are rejected with 503.
    3. Waits up to `drain_timeout_sec` for all WORKING/SUBMITTED tasks to finish.
    4. Cancels any tasks still running after the drain window expires.
    5. Stops the AgentServer (health/ready endpoints) cleanly.
    6. Stops the TaskManager watchdog.
    7. Fires a final `network.shutdown` event so application code can clean up.

Usage (typical production entrypoint):

    async def main():
        network = AgentNetwork.from_config("nexus.toml")
        server  = AgentServer(network=network, port=8080)

        async with GracefulShutdown(network=network, server=server) as sd:
            await server.start()
            await network.add("http://summariser:8001")
            await sd.wait()          # blocks until SIGTERM/SIGINT received
                                     # then drains and cleans up automatically

    asyncio.run(main())

Or without context manager:

    sd = GracefulShutdown(network=network, server=server, drain_timeout_sec=30)
    sd.install()                     # register signal handlers
    await sd.wait()                  # run until signal
    await sd.drain()                 # finish in-flight tasks

Design:
    - Signal handling uses asyncio.loop.add_signal_handler (POSIX-only).
      On Windows, only SIGINT is supported; SIGTERM is not available.
      A warning is logged on Windows instead of crashing.
    - drain() is idempotent — safe to call multiple times.
    - All state is exposed via properties for observability.
    - Shared task stores (Redis, Postgres): several replicas see the same
      tasks, so a replica must never wait for or cancel another replica's
      work. With a shared store, only tasks this process created, started or
      resumed (through the network's TaskManager, after GracefulShutdown was
      constructed) are drained/cancelled, and paused INPUT_REQUIRED tasks are
      left for any replica to resume. See `cancel_scope`.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import signal
import sys
import time
from typing import TYPE_CHECKING, Any

from nexus_a2a.models.task import TaskState

if TYPE_CHECKING:
    from nexus_a2a.core.agent_server import AgentServer
    from nexus_a2a.network import AgentNetwork

logger = logging.getLogger(__name__)

# Active states that need to finish before we shut down
_ACTIVE_STATES = {TaskState.SUBMITTED, TaskState.WORKING, TaskState.INPUT_REQUIRED}

# With a shared store, a paused INPUT_REQUIRED task is not running anywhere and
# can be resumed by any replica, so only tasks actually in flight count.
_OWNED_ACTIVE_STATES = {TaskState.SUBMITTED, TaskState.WORKING}

# Store classes known to be shared between processes/replicas. Matched by
# class name (anywhere in the MRO) so redis/asyncpg need not be importable.
_SHARED_STORE_CLASSES = frozenset({"RedisTaskStore", "PostgresTaskStore"})

_CANCEL_SCOPES = ("auto", "owned", "all")

# TaskManager methods whose returned task is now being handled by this process.
_CLAIMING_METHODS = ("create", "start", "provide_input")
# TaskManager methods after which this process no longer handles the task.
# request_input pauses the task; whichever replica resumes it claims it.
_RELEASING_METHODS = ("request_input", "complete", "fail", "cancel", "delete")


def _is_shared_store(store: Any) -> bool:
    """True if *store* is known to be shared across processes."""
    if getattr(store, "shared", None) is True:
        return True
    return any(cls.__name__ in _SHARED_STORE_CLASSES for cls in type(store).__mro__)


# ── GracefulShutdown ──────────────────────────────────────────────────────────


class GracefulShutdown:
    """
    SIGTERM/SIGINT drain handler — zero-task-loss shutdown for production.

    Lifecycle:
        install()  → register signal handlers (or call via `async with`)
        wait()     → suspend until a shutdown signal arrives
        drain()    → finish in-flight tasks, then tear down the network

    Args:
        network:           The AgentNetwork to drain and shut down.
        server:            Optional AgentServer to stop before final teardown.
        drain_timeout_sec: Seconds to wait for active tasks to complete
                           before forcefully cancelling them. Default: 30.
        poll_interval_sec: How often to check for active tasks during drain.
                           Default: 0.5.
        cancel_scope:      Which tasks drain waits for and force-cancels.
                           "auto" (default): "owned" when the TaskManager's
                           store is shared (RedisTaskStore, PostgresTaskStore,
                           or any store with `shared = True`), else "all".
                           "all": every SUBMITTED/WORKING/INPUT_REQUIRED task
                           in the store (single-process behaviour).
                           "owned": only SUBMITTED/WORKING tasks this process
                           created, started or resumed via the network's
                           TaskManager after this object was constructed (or
                           registered with track_task()). Paused
                           INPUT_REQUIRED tasks are left for any replica to
                           resume.

    Example::

        async with GracefulShutdown(network=network, server=server) as sd:
            await server.start()
            await sd.wait()   # runs forever until SIGTERM/SIGINT
        # drain() called automatically on exit
    """

    def __init__(
        self,
        network: AgentNetwork,
        server: AgentServer | None = None,
        drain_timeout_sec: float = 30.0,
        poll_interval_sec: float = 0.5,
        cancel_scope: str = "auto",
    ) -> None:
        if cancel_scope not in _CANCEL_SCOPES:
            raise ValueError(
                f"cancel_scope must be one of {_CANCEL_SCOPES}, got {cancel_scope!r}"
            )
        self.network = network
        self.server = server
        self.drain_timeout_sec = drain_timeout_sec
        self.poll_interval_sec = poll_interval_sec

        self._owned_task_ids: set[str] = set()
        if cancel_scope == "auto":
            store = getattr(network.task_manager, "_store", None)
            cancel_scope = "owned" if _is_shared_store(store) else "all"
        self._owned_only: bool = cancel_scope == "owned"
        if self._owned_only:
            self._install_ownership_tracking()
            logger.debug(
                "GracefulShutdown: shared task store — drain only covers tasks "
                "handled by this process."
            )

        self._shutdown_event: asyncio.Event = asyncio.Event()
        self._draining: bool = False
        self._drained: bool = False
        self._received_signal: str | None = None
        self._drain_started_at: float | None = None
        self._drain_ended_at: float | None = None

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_draining(self) -> bool:
        """True from the moment a shutdown signal is received until drain completes."""
        return self._draining

    @property
    def is_drained(self) -> bool:
        """True once drain() has fully completed."""
        return self._drained

    @property
    def received_signal(self) -> str | None:
        """Name of the signal that triggered shutdown, or None if not yet received."""
        return self._received_signal

    @property
    def drain_duration_seconds(self) -> float | None:
        """How long drain() took, in seconds. None if not yet completed."""
        if self._drain_started_at is None or self._drain_ended_at is None:
            return None
        return self._drain_ended_at - self._drain_started_at

    @property
    def owned_task_ids(self) -> frozenset[str]:
        """Task ids this process is currently handling (only tracked in "owned" scope)."""
        return frozenset(self._owned_task_ids)

    def track_task(self, task_id: str) -> None:
        """
        Mark a task as handled by this process, so drain covers it in the
        "owned" scope. Only needed for tasks moved into SUBMITTED/WORKING
        without going through the network's TaskManager (create/start/
        provide_input are tracked automatically).
        """
        self._owned_task_ids.add(task_id)

    # ── Signal handling ───────────────────────────────────────────────────────

    def install(self) -> None:
        """
        Register SIGTERM and SIGINT handlers on the running event loop.

        Safe to call multiple times — handlers are idempotent.
        On Windows, only SIGINT is registered (SIGTERM is unavailable).

        Raises:
            RuntimeError: If called outside a running asyncio event loop.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            raise RuntimeError(
                "GracefulShutdown.install() must be called from inside a running "
                "asyncio event loop (i.e. inside an async function)."
            ) from exc

        if sys.platform == "win32":
            # Windows: only SIGINT available via asyncio
            try:
                loop.add_signal_handler(signal.SIGINT, self._trigger, "SIGINT")
            except (NotImplementedError, OSError):
                logger.warning(
                    "GracefulShutdown: could not register SIGINT handler "
                    "(loop does not support signal handling)."
                )
            logger.warning(
                "GracefulShutdown: SIGTERM is not supported on Windows. "
                "Only SIGINT (Ctrl-C) will trigger graceful shutdown."
            )
        else:
            try:
                loop.add_signal_handler(signal.SIGTERM, self._trigger, "SIGTERM")
                loop.add_signal_handler(signal.SIGINT, self._trigger, "SIGINT")
                logger.debug("GracefulShutdown: SIGTERM + SIGINT handlers installed.")
            except (NotImplementedError, OSError) as exc:
                logger.warning(
                    "GracefulShutdown: could not register signal handlers (%s). "
                    "Graceful shutdown via signals will not work in this environment "
                    "(e.g. test runners, Windows Subsystem for Linux). "
                    "Call drain() manually to shut down.",
                    exc,
                )

    def uninstall(self) -> None:
        """
        Remove signal handlers. Called automatically by __aexit__.
        Safe to call even if install() was never called.

        Note: Some event loop implementations (e.g. pytest-asyncio test loops
        on certain platforms) raise NotImplementedError for remove_signal_handler.
        Those are silently ignored — the process is shutting down anyway.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        if sys.platform != "win32":
            try:
                loop.remove_signal_handler(signal.SIGTERM)
            except (NotImplementedError, OSError):
                pass
        try:
            loop.remove_signal_handler(signal.SIGINT)
        except (NotImplementedError, OSError):
            pass

    def _trigger(self, sig_name: str) -> None:
        """Called by the event loop when a signal is received."""
        if self._shutdown_event.is_set():
            logger.warning(
                "GracefulShutdown: received %s again — already shutting down.", sig_name
            )
            return
        logger.info("GracefulShutdown: received %s — beginning shutdown.", sig_name)
        self._received_signal = sig_name
        self._shutdown_event.set()

    # ── Wait / drain ──────────────────────────────────────────────────────────

    async def wait(self) -> None:
        """
        Suspend until a shutdown signal is received.

        Typical usage::

            sd.install()
            await asyncio.gather(
                server.start(),
                sd.wait(),      # this blocks
            )
            await sd.drain()
        """
        await self._shutdown_event.wait()

    async def drain(self) -> None:
        """
        Drain in-flight tasks and shut down the network.

        Steps:
            1. Mark as draining (new tasks will be rejected).
            2. Wait for WORKING/SUBMITTED tasks to finish (up to drain_timeout_sec).
            3. Cancel any remaining active tasks.
            4. Stop AgentServer (if provided).
            5. Stop TaskManager watchdog.
            6. Publish 'network.shutdown' event.

        Idempotent — safe to call multiple times; only runs once.
        """
        if self._drained:
            logger.debug("GracefulShutdown.drain() called again — already drained.")
            return

        self._draining = True
        self._drain_started_at = time.monotonic()

        logger.info(
            "GracefulShutdown: draining (timeout=%.1fs, poll=%.2fs)...",
            self.drain_timeout_sec,
            self.poll_interval_sec,
        )

        # ── Step 1: wait for active tasks ─────────────────────────────────────
        deadline = time.monotonic() + self.drain_timeout_sec

        while time.monotonic() < deadline:
            active = await self._count_active_tasks()
            if active == 0:
                logger.info("GracefulShutdown: all tasks finished — clean drain.")
                break
            remaining = deadline - time.monotonic()
            logger.debug(
                "GracefulShutdown: %d active task(s) — %.1fs remaining in drain window.",
                active,
                remaining,
            )
            await asyncio.sleep(self.poll_interval_sec)
        else:
            # Drain window expired — cancel remaining tasks
            cancelled = await self._cancel_active_tasks()
            logger.warning(
                "GracefulShutdown: drain timeout (%.1fs) — forcefully cancelled %d task(s).",
                self.drain_timeout_sec,
                cancelled,
            )

        # ── Step 2: stop AgentServer ──────────────────────────────────────────
        if self.server is not None and self.server.is_running:
            logger.info("GracefulShutdown: stopping AgentServer...")
            try:
                await self.server.stop()
            except Exception as exc:
                logger.warning("GracefulShutdown: error stopping AgentServer: %s", exc)

        # ── Step 3: stop TaskManager watchdog ─────────────────────────────────
        try:
            await self.network.task_manager.stop_watchdog()
            logger.debug("GracefulShutdown: TaskManager watchdog stopped.")
        except Exception as exc:
            logger.warning("GracefulShutdown: error stopping watchdog: %s", exc)

        # ── Step 4: publish shutdown event ────────────────────────────────────
        try:
            await self.network.bus.publish(
                "network.shutdown",
                {
                    "signal": self._received_signal,
                    "drain_duration": time.monotonic() - (self._drain_started_at or 0),
                },
            )
        except Exception as exc:
            logger.warning("GracefulShutdown: error publishing shutdown event: %s", exc)

        self._drain_ended_at = time.monotonic()
        self._drained = True
        self._draining = False

        logger.info(
            "GracefulShutdown: complete in %.2fs.",
            self.drain_duration_seconds or 0.0,
        )

    # ── Async context manager ─────────────────────────────────────────────────

    async def __aenter__(self) -> GracefulShutdown:
        self.install()
        return self

    async def __aexit__(self, *_: Any) -> None:
        self.uninstall()
        if not self._drained:
            await self.drain()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _install_ownership_tracking(self) -> None:
        """
        Wrap the network TaskManager's lifecycle methods (on this instance
        only) so the task ids this process handles are recorded.
        """
        manager = self.network.task_manager
        owned = self._owned_task_ids

        def wrap(name: str, claim: bool) -> None:
            original = getattr(manager, name, None)
            if original is None or not callable(original):
                return

            @functools.wraps(original)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                result = await original(*args, **kwargs)
                task_id = getattr(result, "id", None)
                if not isinstance(task_id, str):
                    task_id = kwargs.get("task_id", args[0] if args else None)
                if isinstance(task_id, str):
                    if claim:
                        owned.add(task_id)
                    else:
                        owned.discard(task_id)
                return result

            setattr(manager, name, wrapper)

        for name in _CLAIMING_METHODS:
            wrap(name, claim=True)
        for name in _RELEASING_METHODS:
            wrap(name, claim=False)

    async def _active_tasks(self) -> list[Any]:
        """Tasks this drain must wait for / cancel, according to the scope."""
        manager = self.network.task_manager
        if not self._owned_only:
            tasks = await manager.list_all()
            return [t for t in tasks if t.state in _ACTIVE_STATES]

        # Owned scope: look up only our own tasks — never scan the shared
        # store, which also holds every other replica's tasks.
        active: list[Any] = []
        for task_id in list(self._owned_task_ids):
            try:
                task = await manager.get(task_id)
            except Exception:
                # Deleted or expired — nothing left to drain.
                self._owned_task_ids.discard(task_id)
                continue
            if task.state in _OWNED_ACTIVE_STATES:
                active.append(task)
            else:
                self._owned_task_ids.discard(task_id)
        return active

    async def _count_active_tasks(self) -> int:
        """
        Return the number of active tasks this drain must wait for.

        Scope "all": SUBMITTED, WORKING or INPUT_REQUIRED tasks in the store.
        Scope "owned": this process's SUBMITTED or WORKING tasks.
        """
        try:
            return len(await self._active_tasks())
        except Exception as exc:
            logger.warning("GracefulShutdown: error counting active tasks: %s", exc)
            return 0

    async def _cancel_active_tasks(self) -> int:
        """
        Forcefully cancel the active tasks in this drain's scope.

        Returns:
            Number of tasks cancelled.
        """
        cancelled = 0
        try:
            tasks = await self._active_tasks()
            for task in tasks:
                if task.state in _ACTIVE_STATES:
                    try:
                        await self.network.task_manager.cancel(task.id)
                        cancelled += 1
                        logger.debug(
                            "GracefulShutdown: force-cancelled task %s (was %s).",
                            task.id,
                            task.state.value,
                        )
                    except Exception as exc:
                        logger.warning(
                            "GracefulShutdown: error cancelling task %s: %s",
                            task.id,
                            exc,
                        )
        except Exception as exc:
            logger.warning(
                "GracefulShutdown: error listing tasks for cancellation: %s", exc
            )
        return cancelled


# ── Context helper ────────────────────────────────────────────────────────────


class _ignore_errors:
    """Context manager that silently suppresses all exceptions."""

    def __enter__(self) -> _ignore_errors:
        return self

    def __exit__(self, *_: Any) -> bool:
        return True
