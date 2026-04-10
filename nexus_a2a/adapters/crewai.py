"""
nexus_a2a/adapters/crewai.py

CrewAIAdapter — wraps a CrewAI Crew so it can receive A2A Tasks
and return AdapterResults.

CrewAI crews are run with crew.kickoff(inputs={"input": text}).
The adapter:
  1. Extracts the latest user message from the Task.
  2. Calls crew.kickoff_async(inputs={input_key: text}).
  3. Reads the CrewOutput.raw string.
  4. Returns an AdapterResult.

CrewAI is an optional dependency.
Install: pip install nexus-a2a crewai
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


class CrewAIAdapter(BaseAdapter):
    """
    Wraps a CrewAI Crew as an A2A-compatible agent.

    Usage:
        from crewai import Crew, Agent, Task as CrewTask, Process

        crew = Crew(
            agents=[researcher, writer],
            tasks=[research_task, write_task],
            process=Process.sequential,
        )

        adapter = CrewAIAdapter(agent=crew, input_key="topic")
        result  = await adapter.execute(a2a_task)

    Args:
        agent:     A CrewAI Crew object (has .kickoff_async()).
        input_key: Key used in the inputs dict passed to kickoff_async().
                   Default: "input".
        config:    Optional extra kwargs passed to kickoff_async().
    """

    framework_name = "crewai"

    def __init__(
        self,
        agent: Any,
        input_key: str = "input",
        config: dict[str, Any] | None = None,
    ) -> None:
        self._input_key = input_key
        super().__init__(agent=agent, config=config)

    def validate(self) -> None:
        """Assert the crew has a kickoff_async method."""
        super().validate()
        if not (
            hasattr(self.agent, "kickoff_async")
            or hasattr(self.agent, "kickoff")
        ):
            raise AdapterConfigError(
                "CrewAIAdapter requires a CrewAI Crew object with "
                "a 'kickoff_async' or 'kickoff' method."
            )

    async def execute(self, task: Task) -> AdapterResult:
        """
        Kick off the CrewAI crew with the task's latest message.

        The crew receives:
            inputs={input_key: "<message text>"}

        The reply is read from CrewOutput.raw.

        Returns:
            AdapterResult with the crew's final output text.
        """
        input_text = self.extract_input(task)
        if not input_text:
            return self.make_error("Task has no message content to process.")

        inputs = {self._input_key: input_text}

        logger.debug(
            "CrewAIAdapter: kicking off crew for task %s input=%r",
            task.id, input_text[:80],
        )

        try:
            # Prefer async kickoff; fall back to sync wrapped in executor
            if hasattr(self.agent, "kickoff_async"):
                crew_output = await self.agent.kickoff_async(inputs=inputs)
            else:
                import asyncio
                loop = asyncio.get_event_loop()
                crew_output = await loop.run_in_executor(
                    None,
                    lambda: self.agent.kickoff(inputs=inputs),
                )
        except Exception as exc:
            logger.error("CrewAIAdapter error: %s", exc)
            raise self._wrap_exception(exc) from exc

        output_text = self._extract_output(crew_output)

        logger.debug(
            "CrewAIAdapter: task %s completed output=%r",
            task.id, output_text[:80],
        )

        return self.make_result(
            output=output_text,
            artifact_name="crew_result",
            metadata={
                "framework": "crewai",
                "task_id":   task.id,
                "input_key": self._input_key,
            },
        )

    @staticmethod
    def _extract_output(crew_output: Any) -> str:
        """
        Extract text from a CrewOutput object.

        Tries (in order):
          1. crew_output.raw           (CrewOutput attribute)
          2. crew_output.final_output  (older CrewAI versions)
          3. str(crew_output)          (fallback)
        """
        if hasattr(crew_output, "raw"):
            return str(crew_output.raw)
        if hasattr(crew_output, "final_output"):
            return str(crew_output.final_output)
        return str(crew_output)
