"""
nexus_a2a/core/orchestrator.py

Orchestrator — drives multi-agent workflows.

Three execution modes:
  sequential  — agents run one after another, each receiving the
                previous agent's output as its next input.
  parallel    — all agents receive the same input and run concurrently,
                results collected when all finish.
  dag         — directed acyclic graph; each node runs when all its
                upstream dependencies have completed. Raises on cycles.

Design principles:
  - The Orchestrator does NOT communicate over the network itself.
    It delegates every agent call to a callable you provide (the "runner").
    This keeps it fully testable without a live server.
  - Cycle detection happens at DAG build time, before any agent is called.
  - Every workflow run returns an OrchestratorResult with per-step details
    so callers can inspect what happened.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from nexus_a2a.models.task import Message, Task

logger = logging.getLogger(__name__)

# Type alias: a function that sends a message to one agent and returns a Task
AgentRunner = Callable[[str, Message], Awaitable[Task]]


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class StepResult:
    """Outcome of a single agent call within a workflow."""
    agent_url:    str
    task:         Task | None = None
    error:        str | None  = None
    duration_sec: float       = 0.0

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.task is not None


@dataclass
class OrchestratorResult:
    """
    Outcome of a complete workflow run.

    Fields:
        mode:         "sequential", "parallel", or "dag".
        steps:        Ordered list of StepResults.
        total_sec:    Wall-clock time for the entire workflow.
        final_output: The last successful Task produced by the workflow.
                      For parallel runs this is the first completed task.
    """
    mode:         str
    steps:        list[StepResult]         = field(default_factory=list)
    total_sec:    float                    = 0.0
    final_output: Task | None             = None

    @property
    def succeeded(self) -> bool:
        return all(s.succeeded for s in self.steps)

    @property
    def failed_steps(self) -> list[StepResult]:
        return [s for s in self.steps if not s.succeeded]


# ── Exceptions ────────────────────────────────────────────────────────────────

class OrchestratorError(Exception):
    """Base class for orchestration errors."""


class WorkflowCycleError(OrchestratorError):
    """Raised when a cycle is detected in a DAG workflow."""

    def __init__(self, cycle: list[str]) -> None:
        super().__init__(
            f"Cycle detected in DAG workflow: {' → '.join(cycle)}"
        )
        self.cycle = cycle


class WorkflowStepError(OrchestratorError):
    """Raised when a step fails and the workflow cannot continue."""

    def __init__(self, agent_url: str, reason: str) -> None:
        super().__init__(f"Step failed for agent '{agent_url}': {reason}")
        self.agent_url = agent_url
        self.reason    = reason


# ── DAG node ──────────────────────────────────────────────────────────────────

@dataclass
class DAGNode:
    """
    One node in a DAG workflow.

    Fields:
        agent_url:    The agent to call at this node.
        depends_on:   List of agent_urls that must complete before this node runs.
        skill_id:     Optional skill to invoke on this agent.
    """
    agent_url:  str
    depends_on: list[str] = field(default_factory=list)
    skill_id:   str | None = None


# ── Orchestrator ──────────────────────────────────────────────────────────────

class Orchestrator:
    """
    Drives multi-agent workflows without being tied to any transport.

    The caller provides a `runner` callable that handles the actual
    network call. This keeps the Orchestrator fully testable.

    Usage:
        async def my_runner(agent_url: str, message: Message) -> Task:
            async with A2AHttpClient(agent_url) as client:
                return await client.send_message(message)

        orch = Orchestrator(runner=my_runner)

        # Run agents one after another
        result = await orch.sequential(
            agent_urls=["http://agent-a:8001", "http://agent-b:8002"],
            initial_message=Message.user_text("Start"),
        )

        # Run agents concurrently
        result = await orch.parallel(
            agent_urls=["http://agent-a:8001", "http://agent-b:8002"],
            message=Message.user_text("Do this"),
        )

        # Run a dependency graph
        result = await orch.dag(
            nodes=[
                DAGNode("http://fetch:8001"),
                DAGNode("http://parse:8002", depends_on=["http://fetch:8001"]),
                DAGNode("http://store:8003", depends_on=["http://parse:8002"]),
            ],
            initial_message=Message.user_text("Process this document"),
        )

    Args:
        runner:         Async callable that sends a message to an agent.
        stop_on_error:  If True (default), abort the workflow when any step fails.
                        If False, continue and collect all results.
    """

    def __init__(
        self,
        runner: AgentRunner,
        stop_on_error: bool = True,
    ) -> None:
        self._runner        = runner
        self._stop_on_error = stop_on_error

    # ── Sequential ────────────────────────────────────────────────────────────

    async def sequential(
        self,
        agent_urls: list[str],
        initial_message: Message,
        skill_ids: list[str | None] | None = None,
    ) -> OrchestratorResult:
        """
        Run agents one after another.

        The output of each agent becomes the input message for the next.
        If an agent fails and stop_on_error=True, the workflow stops there.

        Args:
            agent_urls:      Ordered list of agent URLs to call.
            initial_message: Message sent to the first agent.
            skill_ids:       Optional list of skill IDs (aligned with agent_urls).
                             Use None entries for agents where skill is not needed.

        Returns:
            OrchestratorResult with one StepResult per agent.
        """
        if not agent_urls:
            raise OrchestratorError("agent_urls must not be empty.")

        skills   = skill_ids or [None] * len(agent_urls)
        result   = OrchestratorResult(mode="sequential")
        message  = initial_message
        wall_start = time.monotonic()

        for url, skill in zip(agent_urls, skills, strict=False):
            step = await self._run_step(url, message, skill)
            result.steps.append(step)

            if not step.succeeded:
                logger.warning("Sequential step failed at %s: %s", url, step.error)
                if self._stop_on_error:
                    break
            else:
                # Pass this agent's output as the next agent's input
                result.final_output = step.task
                if step.task and step.task.latest_message():
                    message = step.task.latest_message()  # type: ignore[assignment]
                else:
                    # No reply message — forward the original input
                    message = initial_message

        result.total_sec = time.monotonic() - wall_start
        logger.info(
            "Sequential workflow done: %d steps, %.2fs, success=%s",
            len(result.steps), result.total_sec, result.succeeded,
        )
        return result

    # ── Parallel ──────────────────────────────────────────────────────────────

    async def parallel(
        self,
        agent_urls: list[str],
        message: Message,
        skill_ids: list[str | None] | None = None,
    ) -> OrchestratorResult:
        """
        Run all agents concurrently with the same input message.

        All agents receive the same message simultaneously.
        Results are collected when all agents finish (or fail).

        Args:
            agent_urls: List of agent URLs to call concurrently.
            message:    The message sent to every agent.
            skill_ids:  Optional list of skill IDs aligned with agent_urls.

        Returns:
            OrchestratorResult with one StepResult per agent,
            in the order they were provided (not completion order).
        """
        if not agent_urls:
            raise OrchestratorError("agent_urls must not be empty.")

        skills     = skill_ids or [None] * len(agent_urls)
        wall_start = time.monotonic()

        # Launch all steps concurrently
        step_coros = [
            self._run_step(url, message, skill)
            for url, skill in zip(agent_urls, skills, strict=False)
        ]
        steps: list[StepResult] = list(
            await asyncio.gather(*step_coros, return_exceptions=False)
        )

        result = OrchestratorResult(
            mode="parallel",
            steps=steps,
            total_sec=time.monotonic() - wall_start,
        )

        # final_output = first successful task
        for step in steps:
            if step.succeeded:
                result.final_output = step.task
                break

        logger.info(
            "Parallel workflow done: %d agents, %.2fs, failed=%d",
            len(steps), result.total_sec, len(result.failed_steps),
        )
        return result

    # ── DAG ───────────────────────────────────────────────────────────────────

    async def dag(
        self,
        nodes: list[DAGNode],
        initial_message: Message,
    ) -> OrchestratorResult:
        """
        Run a directed acyclic graph of agents.

        Each node runs as soon as all its dependencies have completed.
        Independent nodes run concurrently.
        Raises WorkflowCycleError if the graph contains a cycle.

        Args:
            nodes:           List of DAGNode definitions.
            initial_message: Message sent to root nodes (nodes with no dependencies).

        Returns:
            OrchestratorResult with StepResults in execution order.

        Raises:
            WorkflowCycleError: If a cycle is detected before execution starts.
        """
        if not nodes:
            raise OrchestratorError("nodes must not be empty.")

        node_map = {n.agent_url: n for n in nodes}
        self._detect_cycle(node_map)

        wall_start   = time.monotonic()
        completed:   dict[str, StepResult] = {}  # url → StepResult
        result       = OrchestratorResult(mode="dag")

        # Topological execution: keep looping until all nodes are done
        pending = list(nodes)

        while pending:
            # Find nodes whose dependencies are all completed successfully
            ready = [
                n for n in pending
                if all(
                    dep in completed and completed[dep].succeeded
                    for dep in n.depends_on
                )
            ]

            if not ready:
                # No node is ready — check if any dependency failed
                failed_deps = [
                    dep
                    for n in pending
                    for dep in n.depends_on
                    if dep in completed and not completed[dep].succeeded
                ]
                if failed_deps and self._stop_on_error:
                    break
                # Safety: avoid infinite loop if graph is somehow stuck
                raise OrchestratorError(
                    "DAG execution stalled — no nodes ready and no failures detected. "
                    "Check that all dependency URLs match node agent_urls exactly."
                )

            # Run ready nodes concurrently
            step_coros = [
                self._run_step(n.agent_url, initial_message, n.skill_id)
                for n in ready
            ]
            new_steps: list[StepResult] = list(
                await asyncio.gather(*step_coros)
            )

            for node, step in zip(ready, new_steps, strict=False):
                completed[node.agent_url] = step
                result.steps.append(step)
                pending.remove(node)

                if not step.succeeded and self._stop_on_error:
                    logger.warning("DAG step failed at %s: %s", node.agent_url, step.error)

        result.total_sec    = time.monotonic() - wall_start
        result.final_output = next(
            (s.task for s in reversed(result.steps) if s.succeeded), None
        )

        logger.info(
            "DAG workflow done: %d nodes, %.2fs, success=%s",
            len(result.steps), result.total_sec, result.succeeded,
        )
        return result

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _run_step(
        self,
        agent_url: str,
        message: Message,
        skill_id: str | None,
    ) -> StepResult:
        """Call one agent and wrap the result in a StepResult."""
        start = time.monotonic()
        try:
            task = await self._runner(agent_url, message)
            return StepResult(
                agent_url=agent_url,
                task=task,
                duration_sec=time.monotonic() - start,
            )
        except Exception as exc:
            logger.error("Step error at %s: %s", agent_url, exc)
            return StepResult(
                agent_url=agent_url,
                error=str(exc),
                duration_sec=time.monotonic() - start,
            )

    @staticmethod
    def _detect_cycle(node_map: dict[str, DAGNode]) -> None:
        """
        Detect cycles using depth-first search (DFS).
        Raises WorkflowCycleError if a cycle is found.
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        colour: dict[str, int] = {url: WHITE for url in node_map}
        parent: dict[str, str | None] = {url: None for url in node_map}

        def dfs(url: str) -> None:
            colour[url] = GRAY
            node = node_map.get(url)
            if node:
                for dep in node.depends_on:
                    if dep not in colour:
                        continue
                    if colour[dep] == GRAY:
                        # Reconstruct the cycle path
                        cycle = [dep, url]
                        cur = url
                        while parent[cur] and parent[cur] != dep:
                            cur = parent[cur]  # type: ignore[assignment]
                            cycle.append(cur)
                        cycle.append(dep)
                        raise WorkflowCycleError(list(reversed(cycle)))
                    if colour[dep] == WHITE:
                        parent[dep] = url
                        dfs(dep)
            colour[url] = BLACK

        for url in node_map:
            if colour[url] == WHITE:
                dfs(url)
