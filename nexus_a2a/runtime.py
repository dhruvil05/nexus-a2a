"""
nexus_a2a/runtime.py

AgentRuntime — everything a served agent needs, started and stopped together.

A production agent is more than an A2AServer: its stores may need connecting,
its task watchdog needs running, and it usually wants an ops server on a second
port. Through 1.8.0 each of those was a separate manual step, and `nexus run`
did none of them — so `nexus.toml` could not configure auth, rate limits,
storage or push delivery at all, and a Redis backend was never even connected.

Build one from config:

    cfg = NexusConfig.from_file("nexus.toml")
    runtime = cfg.build_runtime(MyAgent)

    async with runtime:
        await runtime.wait()          # until cancelled

Or assemble one by hand when you need something config cannot express:

    runtime = AgentRuntime(server=A2AServer(MyAgent), resources=[redis_store])

Startup order is resources -> watchdog -> agent server -> ops server, and
shutdown runs in reverse, so nothing serves traffic against a store that is not
yet connected, and nothing is disconnected while traffic can still reach it.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nexus_a2a.core.a2a_server import A2AServer
    from nexus_a2a.core.agent_server import AgentServer
    from nexus_a2a.core.task_manager import TaskManager

logger = logging.getLogger(__name__)


@dataclass
class AgentRuntime:
    """
    An A2AServer plus the things it depends on, with one lifecycle.

    Fields:
        server:    The protocol server.
        ops:       Optional AgentServer on a separate port (health, metrics,
                   admin endpoints).
        resources: Objects with async connect()/disconnect() — stores, a Redis
                   rate limiter — connected before serving and disconnected
                   after.
        watchdog:  TaskManager whose timeout watchdog should run while serving.
    """

    server: A2AServer
    ops: AgentServer | None = None
    resources: list[Any] = field(default_factory=list)
    watchdog: TaskManager | None = None

    _connected: list[Any] = field(default_factory=list, init=False, repr=False)
    _started: bool = field(default=False, init=False, repr=False)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """
        Connect resources, then start serving.

        If any step fails, whatever already started is torn down before the
        error propagates, so a failed start never leaves a half-running agent.

        Raises:
            RuntimeError: If already started.
        """
        if self._started:
            raise RuntimeError("AgentRuntime is already started.")
        self._started = True

        try:
            for resource in self.resources:
                await resource.connect()
                self._connected.append(resource)

            if self.watchdog is not None:
                await self.watchdog.start_watchdog()

            await self.server.start()

            if self.ops is not None:
                await self.ops.start()
        except BaseException:
            await self._teardown()
            raise

        logger.info(
            "AgentRuntime started: agent on %s:%d%s",
            self.server.host,
            self.server.port,
            f", ops on {self.ops.host}:{self.ops.port}" if self.ops else "",
        )

    async def stop(self) -> None:
        """Stop serving, then disconnect resources. Safe to call twice."""
        if not self._started:
            return
        await self._teardown()
        logger.info("AgentRuntime stopped.")

    async def wait(self) -> None:
        """Block until the surrounding task is cancelled (Ctrl+C, SIGTERM)."""
        await asyncio.Event().wait()

    async def __aenter__(self) -> AgentRuntime:
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop()

    # ── Internals ─────────────────────────────────────────────────────────────

    async def _teardown(self) -> None:
        """Reverse of start(). Each step is isolated so one failure cannot
        leave the rest running."""
        if self.ops is not None:
            await _quietly("ops server", self.ops.stop())
        await _quietly("agent server", self.server.stop())
        if self.watchdog is not None:
            await _quietly("task watchdog", self.watchdog.stop_watchdog())
        while self._connected:
            resource = self._connected.pop()
            await _quietly(type(resource).__name__, resource.disconnect())
        self._started = False


async def _quietly(what: str, awaitable: Any) -> None:
    """Await a shutdown step, logging rather than raising on failure."""
    try:
        await awaitable
    except Exception:
        logger.exception("AgentRuntime: failed to stop %s cleanly", what)


__all__ = ["AgentRuntime"]
