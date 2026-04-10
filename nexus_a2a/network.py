"""
nexus_a2a/network.py

Two public classes that tie Phase 2-4 together into a clean developer API:

  EventBus     — async pub/sub between agents within a network.
  AgentNetwork — the top-level object developers use to build a
                 multi-agent network (register agents, send tasks,
                 run workflows).

This is the file that makes the package feel like a coherent product
rather than a collection of components.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from nexus_a2a.core.orchestrator import DAGNode, Orchestrator, OrchestratorResult
from nexus_a2a.core.registry import AgentRegistry
from nexus_a2a.core.task_manager import TaskManager
from nexus_a2a.models.task import Message, Task
from nexus_a2a.transport.http_client import A2AHttpClient

logger = logging.getLogger(__name__)

# Type alias: subscriber callback receives event_name and payload
EventHandler = Callable[[str, dict[str, Any]], Awaitable[None]]


# ── EventBus ──────────────────────────────────────────────────────────────────

class EventBus:
    """
    Lightweight async pub/sub bus for broadcasting events between agents
    within the same process.

    Events are identified by a string name (e.g. "task.completed",
    "agent.registered"). Subscribers register a coroutine that gets
    called whenever an event with a matching name is published.

    This is an in-process bus — it does NOT send events over the network.
    For cross-process events, use webhooks (WebhookDispatcher).

    Usage:
        bus = EventBus()

        # Subscribe
        async def on_task_done(event: str, data: dict) -> None:
            print(f"Task {data['task_id']} completed!")

        bus.subscribe("task.completed", on_task_done)

        # Publish — calls all subscribers concurrently
        await bus.publish("task.completed", {"task_id": "abc-123"})

        # Unsubscribe
        bus.unsubscribe("task.completed", on_task_done)
    """

    def __init__(self) -> None:
        # event_name → list of handler coroutines
        self._subscribers: dict[str, list[EventHandler]] = {}

    # ── Subscription management ───────────────────────────────────────────────

    def subscribe(self, event: str, handler: EventHandler) -> None:
        """
        Register a coroutine to be called when event is published.

        Args:
            event:   Event name to listen for. Supports exact match only.
            handler: Async callable(event_name, data) → None.
        """
        self._subscribers.setdefault(event, []).append(handler)
        logger.debug("EventBus: subscribed to '%s'", event)

    def unsubscribe(self, event: str, handler: EventHandler) -> None:
        """
        Remove a previously registered handler.
        Silently does nothing if the handler was not registered.
        """
        handlers = self._subscribers.get(event, [])
        try:
            handlers.remove(handler)
        except ValueError:
            pass

    def unsubscribe_all(self, event: str) -> None:
        """Remove all handlers for a given event."""
        self._subscribers.pop(event, None)

    def subscribers(self, event: str) -> list[EventHandler]:
        """Return all handlers registered for an event."""
        return list(self._subscribers.get(event, []))

    # ── Publishing ────────────────────────────────────────────────────────────

    async def publish(self, event: str, data: dict[str, Any] | None = None) -> int:
        """
        Publish an event to all subscribers concurrently.

        Handlers are called concurrently via asyncio.gather.
        Exceptions in individual handlers are logged but do not affect others.

        Args:
            event: The event name to publish.
            data:  Optional payload dict passed to every handler.

        Returns:
            Number of handlers that were called.
        """
        handlers = self._subscribers.get(event, [])
        if not handlers:
            return 0

        payload = data or {}

        async def _safe_call(handler: EventHandler) -> None:
            try:
                await handler(event, payload)
            except Exception as exc:
                logger.error(
                    "EventBus handler error for event '%s': %s", event, exc
                )

        await asyncio.gather(*(_safe_call(h) for h in handlers))
        logger.debug("EventBus: published '%s' to %d handler(s)", event, len(handlers))
        return len(handlers)

    async def publish_nowait(self, event: str, data: dict[str, Any] | None = None) -> None:
        """
        Fire-and-forget version of publish().
        Schedules delivery as a background task without awaiting it.
        Use when you don't want to block the caller.
        """
        asyncio.create_task(self.publish(event, data))  # noqa: RUF006


# ── AgentNetwork ──────────────────────────────────────────────────────────────

class AgentNetwork:
    """
    The top-level object for building a multi-agent network.

    Combines AgentRegistry, TaskManager, Orchestrator, and EventBus
    into a single coherent interface.

    Usage:
        network = AgentNetwork()

        # Register agents
        await network.add("http://research-agent:8001")
        await network.add("http://summary-agent:8002")

        # Send a task to the best agent for a skill
        task = await network.send(
            message=Message.user_text("Find AI papers"),
            skill_id="web_search",
        )

        # Run a sequential workflow
        result = await network.sequential(
            agent_urls=["http://research-agent:8001", "http://summary-agent:8002"],
            message=Message.user_text("Research and summarise AI papers"),
        )

        # Run a parallel workflow
        result = await network.parallel(
            agent_urls=["http://agent-a:8001", "http://agent-b:8002"],
            message=Message.user_text("Analyse this dataset"),
        )

        # Listen for events
        @network.on("task.completed")
        async def handle_done(event: str, data: dict) -> None:
            print(f"Done: {data['task_id']}")

    Args:
        task_manager: Custom TaskManager (uses InMemoryTaskStore by default).
        bus:          Custom EventBus (creates a new one by default).
    """

    # Built-in event names published by AgentNetwork
    EVENT_AGENT_ADDED      = "agent.added"
    EVENT_AGENT_REMOVED    = "agent.removed"
    EVENT_TASK_SENT        = "task.sent"
    EVENT_TASK_COMPLETED   = "task.completed"
    EVENT_TASK_FAILED      = "task.failed"
    EVENT_WORKFLOW_DONE    = "workflow.done"

    def __init__(
        self,
        task_manager: TaskManager | None = None,
        bus: EventBus | None = None,
    ) -> None:
        self.registry     = AgentRegistry()
        self.task_manager = task_manager or TaskManager()
        self.bus          = bus or EventBus()
        self._orchestrator: Orchestrator | None = None   # created lazily

    # ── Agent management ──────────────────────────────────────────────────────

    async def add(self, url: str) -> None:
        """
        Register a remote agent by URL.
        Fetches its AgentCard automatically and publishes 'agent.added'.

        Args:
            url: Base URL of the remote A2A server.
        """
        card = await self.registry.register_url(url)
        await self.bus.publish(self.EVENT_AGENT_ADDED, {
            "url":    url,
            "name":  card.name,
            "skills": card.skill_ids(),
        })
        logger.info("AgentNetwork: added '%s' at %s", card.name, url)

    async def remove(self, url: str) -> None:
        """
        Unregister a remote agent and publish 'agent.removed'.

        Args:
            url: The agent's base URL.
        """
        await self.registry.unregister(url)
        await self.bus.publish(self.EVENT_AGENT_REMOVED, {"url": url})
        logger.info("AgentNetwork: removed agent at %s", url)

    # ── Sending tasks ─────────────────────────────────────────────────────────

    async def send(
        self,
        message: Message,
        skill_id: str | None = None,
        agent_url: str | None = None,
    ) -> Task:
        """
        Send a message to an agent and return the resulting Task.

        If agent_url is not provided, the registry picks the first healthy
        agent that advertises the requested skill_id.

        Args:
            message:   The message to send.
            skill_id:  Optional — route to an agent with this skill.
            agent_url: Optional — send directly to this agent URL.

        Returns:
            The Task returned by the remote agent.

        Raises:
            ValueError: No suitable agent found.
            AgentUnreachableError: Agent did not respond.
        """
        url = agent_url or self._resolve_agent(skill_id)

        await self.bus.publish(self.EVENT_TASK_SENT, {
            "agent_url": url,
            "skill_id":  skill_id,
        })

        try:
            async with A2AHttpClient(url) as client:
                task = await client.send_message(message, skill_id=skill_id)

            await self.bus.publish(self.EVENT_TASK_COMPLETED, {"task_id": task.id})
            return task

        except Exception as exc:
            await self.bus.publish(self.EVENT_TASK_FAILED, {
                "agent_url": url,
                "error":     str(exc),
            })
            raise

    # ── Workflow shortcuts ────────────────────────────────────────────────────

    async def sequential(
        self,
        agent_urls: list[str],
        message: Message,
        skill_ids: list[str | None] | None = None,
    ) -> OrchestratorResult:
        """
        Run a sequential workflow across the given agents.
        Each agent receives the previous agent's output as its input.
        """
        result = await self._get_orchestrator().sequential(
            agent_urls=agent_urls,
            initial_message=message,
            skill_ids=skill_ids,
        )
        await self.bus.publish(self.EVENT_WORKFLOW_DONE, {
            "mode":    "sequential",
            "success": result.succeeded,
            "steps":   len(result.steps),
        })
        return result

    async def parallel(
        self,
        agent_urls: list[str],
        message: Message,
        skill_ids: list[str | None] | None = None,
    ) -> OrchestratorResult:
        """
        Run all agents concurrently with the same input message.
        """
        result = await self._get_orchestrator().parallel(
            agent_urls=agent_urls,
            message=message,
            skill_ids=skill_ids,
        )
        await self.bus.publish(self.EVENT_WORKFLOW_DONE, {
            "mode":    "parallel",
            "success": result.succeeded,
            "steps":   len(result.steps),
        })
        return result

    async def dag(
        self,
        nodes: list[DAGNode],
        message: Message,
    ) -> OrchestratorResult:
        """
        Run a DAG workflow where each node runs when its dependencies complete.
        """
        result = await self._get_orchestrator().dag(
            nodes=nodes,
            initial_message=message,
        )
        await self.bus.publish(self.EVENT_WORKFLOW_DONE, {
            "mode":    "dag",
            "success": result.succeeded,
            "steps":   len(result.steps),
        })
        return result

    # ── EventBus shortcut ─────────────────────────────────────────────────────

    def on(self, event: str) -> Callable[[EventHandler], EventHandler]:
        """
        Decorator shortcut for subscribing to an event.

        Usage:
            @network.on("task.completed")
            async def handle(event: str, data: dict) -> None:
                print(data)
        """
        def decorator(handler: EventHandler) -> EventHandler:
            self.bus.subscribe(event, handler)
            return handler
        return decorator

    # ── Health ────────────────────────────────────────────────────────────────

    async def health_check(self) -> dict[str, bool]:
        """Ping all registered agents and return a url→healthy map."""
        return await self.registry.check_all_health()

    def summary(self) -> dict[str, Any]:
        """Return a summary of the network state."""
        return self.registry.summary()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _resolve_agent(self, skill_id: str | None) -> str:
        """
        Find the URL of a healthy agent for the given skill.

        Args:
            skill_id: Skill to look for. If None, returns any healthy agent.

        Raises:
            ValueError: No suitable agent found.
        """
        if skill_id:
            matches = self.registry.find_by_skill(skill_id)
            if not matches:
                raise ValueError(
                    f"No healthy agent found with skill '{skill_id}'. "
                    "Register an agent that advertises this skill first."
                )
            return str(matches[0].url).rstrip("/")

        healthy = self.registry.list_healthy()
        if not healthy:
            raise ValueError(
                "No healthy agents registered in this network. "
                "Call await network.add(url) first."
            )
        return str(healthy[0].url).rstrip("/")

    def _get_orchestrator(self) -> Orchestrator:
        """Return (or lazily create) the Orchestrator with network's runner."""
        if self._orchestrator is None:
            self._orchestrator = Orchestrator(runner=self._run_agent)
        return self._orchestrator

    async def _run_agent(self, agent_url: str, message: Message) -> Task:
        """The runner callable passed to the Orchestrator."""
        async with A2AHttpClient(agent_url) as client:
            return await client.send_message(message)
