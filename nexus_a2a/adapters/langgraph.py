"""
nexus_a2a/adapters/langgraph.py

LangGraphAdapter — wraps a compiled LangGraph graph so it can receive
A2A Tasks and return AdapterResults.

LangGraph agents are compiled graphs with a `.ainvoke(input_dict)` method.
The adapter:
  1. Extracts the latest user message from the Task.
  2. Calls graph.ainvoke({"messages": [{"role": "user", "content": text}]}).
  3. Reads the last message from the output state.
  4. Returns an AdapterResult.

LangGraph is an optional dependency — the import is deferred so users
who don't use LangGraph don't need to install it.

Install: pip install nexus-a2a langgraph
"""

from __future__ import annotations

import logging
from typing import Any

from nexus_a2a.adapters.base import (
    AdapterConfigError,
    AdapterResult,
    BaseAdapter,
)
from nexus_a2a.models.task import Task

logger = logging.getLogger(__name__)


class LangGraphAdapter(BaseAdapter):
    """
    Wraps a compiled LangGraph graph as an A2A-compatible agent.

    Usage:
        from langgraph.graph import StateGraph
        # ... build and compile your graph ...
        graph = builder.compile()

        adapter = LangGraphAdapter(
            agent=graph,
            input_key="messages",      # key in the state dict for input
            output_key="messages",     # key in the state dict for output
        )

        # Use with TaskManager
        manager = TaskManager()
        task    = await manager.create(Message.user_text("What is 2+2?"))
        await manager.start(task.id)

        result = await adapter.execute(task)
        await manager.complete(task.id, artifact=result.to_artifact())

    Args:
        agent:      A compiled LangGraph graph (has .ainvoke()).
        input_key:  State dict key for the input messages list.
                    Default: "messages".
        output_key: State dict key for reading the agent's reply.
                    Default: "messages" (reads last message in the list).
        config:     Optional LangGraph run config passed to ainvoke()
                    (e.g. {"configurable": {"thread_id": "abc"}}).
    """

    framework_name = "langgraph"

    def __init__(
        self,
        agent: Any,
        input_key:  str = "messages",
        output_key: str = "messages",
        config: dict[str, Any] | None = None,
    ) -> None:
        self._input_key  = input_key
        self._output_key = output_key
        self._run_config = config or {}
        # BaseAdapter.__init__ calls validate()
        super().__init__(agent=agent, config=config)

    def validate(self) -> None:
        """Assert the graph has an ainvoke method."""
        super().validate()
        if not hasattr(self.agent, "ainvoke"):
            raise AdapterConfigError(
                "LangGraphAdapter requires a compiled LangGraph graph "
                "with an 'ainvoke' method. "
                "Make sure you called graph = builder.compile()."
            )

    async def execute(self, task: Task) -> AdapterResult:
        """
        Invoke the LangGraph graph with the task's latest message.

        The graph receives:
            {input_key: [{"role": "user", "content": "<message text>"}]}

        The reply is read from:
            output_state[output_key][-1].content  (last message in list)

        Returns:
            AdapterResult with the graph's text reply.
        """
        input_text = self.extract_input(task)
        if not input_text:
            return self.make_error("Task has no message content to process.")

        graph_input = {
            self._input_key: [{"role": "user", "content": input_text}]
        }

        logger.debug(
            "LangGraphAdapter: invoking graph for task %s input=%r",
            task.id, input_text[:80],
        )

        try:
            output_state: dict[str, Any] = await self.agent.ainvoke(
                graph_input,
                config=self._run_config or None,
            )
        except Exception as exc:
            logger.error("LangGraphAdapter error: %s", exc)
            raise self._wrap_exception(exc) from exc

        # Extract the reply from the output state
        output_text = self._extract_output(output_state)

        logger.debug(
            "LangGraphAdapter: task %s completed, output=%r",
            task.id, output_text[:80],
        )

        return self.make_result(
            output=output_text,
            artifact_name="langgraph_result",
            metadata={
                "framework":  "langgraph",
                "task_id":    task.id,
                "input_key":  self._input_key,
                "output_key": self._output_key,
            },
        )

    def _extract_output(self, state: dict[str, Any]) -> str:
        """
        Pull the agent's reply text out of the LangGraph output state.

        Tries (in order):
          1. state[output_key][-1].content  (LangChain message object)
          2. state[output_key][-1]["content"]  (plain dict message)
          3. str(state[output_key])  (fallback)
          4. str(state)  (last resort)
        """
        messages = state.get(self._output_key)

        if not messages:
            return str(state)

        last = messages[-1]

        # LangChain message object
        if hasattr(last, "content"):
            return str(last.content)

        # Plain dict
        if isinstance(last, dict) and "content" in last:
            return str(last["content"])

        return str(last)
