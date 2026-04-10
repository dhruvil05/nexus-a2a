"""
nexus_a2a/adapters/google_adk.py

GoogleADKAdapter — wraps a Google Agent Development Kit (ADK) agent
so it can receive A2A Tasks and return AdapterResults.

ADK agents are run with:
    runner = Runner(agent=agent, app_name="...", session_service=...)
    async for event in runner.run_async(user_id=..., session_id=..., new_message=...):
        if event.is_final_response():
            output = event.content.parts[0].text

The adapter handles the Runner setup and event loop internally.

Install: pip install nexus-a2a google-adk
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from nexus_a2a.adapters.base import (
    AdapterConfigError,
    AdapterResult,
    BaseAdapter,
)
from nexus_a2a.models.task import Task

logger = logging.getLogger(__name__)


class GoogleADKAdapter(BaseAdapter):
    """
    Wraps a Google ADK Agent as an A2A-compatible agent.

    Usage:
        from google.adk.agents import Agent
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService

        adk_agent = Agent(name="my_agent", model="gemini-2.0-flash", ...)
        session_svc = InMemorySessionService()

        adapter = GoogleADKAdapter(
            agent=adk_agent,
            app_name="my_app",
            session_service=session_svc,
        )
        result = await adapter.execute(a2a_task)

    Args:
        agent:           A Google ADK Agent object.
        app_name:        Application name passed to the ADK Runner.
        session_service: ADK session service instance.
                         Defaults to InMemorySessionService if not provided
                         (requires google-adk installed).
        user_id:         User ID for ADK session. Default: "nexus-a2a-user".
        config:          Optional extra settings.
    """

    framework_name = "google_adk"

    def __init__(
        self,
        agent: Any,
        app_name: str = "nexus-a2a",
        session_service: Any = None,
        user_id: str = "nexus-a2a-user",
        config: dict[str, Any] | None = None,
    ) -> None:
        self._app_name        = app_name
        self._session_service = session_service
        self._user_id         = user_id
        super().__init__(agent=agent, config=config)

    def validate(self) -> None:
        """Assert the agent has a name attribute (basic ADK Agent check)."""
        super().validate()
        if not hasattr(self.agent, "name"):
            raise AdapterConfigError(
                "GoogleADKAdapter requires a Google ADK Agent object "
                "with a 'name' attribute."
            )

    async def execute(self, task: Task) -> AdapterResult:
        """
        Run the ADK agent with the task's latest message.

        Creates a new ADK session per task, runs the agent, and
        collects the final response event.

        Returns:
            AdapterResult with the agent's text reply.
        """
        input_text = self.extract_input(task)
        if not input_text:
            return self.make_error("Task has no message content to process.")

        try:
            runner, session_id = await self._build_runner(task.id)
            output_text        = await self._run_agent(runner, session_id, input_text)
        except Exception as exc:
            logger.error("GoogleADKAdapter error: %s", exc)
            raise self._wrap_exception(exc) from exc

        logger.debug(
            "GoogleADKAdapter: task %s completed output=%r",
            task.id, output_text[:80],
        )

        return self.make_result(
            output=output_text,
            artifact_name="adk_result",
            metadata={
                "framework":  "google_adk",
                "task_id":    task.id,
                "app_name":   self._app_name,
                "session_id": str(task.id),
            },
        )

    async def _build_runner(self, task_id: str) -> tuple[Any, str]:
        """Create the ADK Runner and a fresh session for this task."""
        try:
            from google.adk.runners import Runner
            from google.adk.sessions import InMemorySessionService
        except ImportError as exc:
            raise AdapterConfigError(
                "google-adk is not installed. "
                "Run: pip install nexus-a2a google-adk"
            ) from exc

        session_svc = self._session_service or InMemorySessionService()
        session_id  = str(uuid.uuid4())

        await session_svc.create_session(
            app_name=self._app_name,
            user_id=self._user_id,
            session_id=session_id,
        )

        runner = Runner(
            agent=self.agent,
            app_name=self._app_name,
            session_service=session_svc,
        )
        return runner, session_id

    async def _run_agent(
        self,
        runner: Any,
        session_id: str,
        input_text: str,
    ) -> str:
        """Stream ADK events and return the final response text."""
        try:
            from google.genai import types as genai_types

            new_message = genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=input_text)],
            )
        except ImportError:
            # Fallback: build a plain dict if google-genai not available
            new_message = {"role": "user", "parts": [{"text": input_text}]}

        output_parts: list[str] = []

        async for event in runner.run_async(
            user_id=self._user_id,
            session_id=session_id,
            new_message=new_message,
        ):
            if hasattr(event, "is_final_response") and event.is_final_response():
                if hasattr(event, "content") and event.content:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            output_parts.append(part.text)
                break

        return " ".join(output_parts) if output_parts else ""
