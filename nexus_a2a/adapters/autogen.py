"""
nexus_a2a/adapters/autogen.py

AutoGenAdapter — wraps a Microsoft AutoGen agent (or ConversableAgent)
so it can receive A2A Tasks and return AdapterResults.

AutoGen agents are run via:
    result = await agent.a_initiate_chat(recipient, message=text, max_turns=1)

The adapter uses a UserProxyAgent as the sender and your agent as
the recipient, captures the last message, and returns it.

Install: pip install nexus-a2a pyautogen
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


class AutoGenAdapter(BaseAdapter):
    """
    Wraps a Microsoft AutoGen ConversableAgent as an A2A-compatible agent.

    Usage:
        import autogen

        assistant = autogen.AssistantAgent(
            name="assistant",
            llm_config={"config_list": [{"model": "gpt-4", "api_key": "..."}]},
        )

        adapter = AutoGenAdapter(agent=assistant, max_turns=1)
        result  = await adapter.execute(a2a_task)

    Args:
        agent:      An AutoGen ConversableAgent (the AI assistant to run).
        max_turns:  Maximum conversation turns per execution. Default: 1.
        human_input_mode: Proxy agent input mode. Default: "NEVER"
                          (fully automated, no human prompts).
        config:     Optional extra settings.
    """

    framework_name = "autogen"

    def __init__(
        self,
        agent: Any,
        max_turns: int = 1,
        human_input_mode: str = "NEVER",
        config: dict[str, Any] | None = None,
    ) -> None:
        self._max_turns        = max_turns
        self._human_input_mode = human_input_mode
        super().__init__(agent=agent, config=config)

    def validate(self) -> None:
        """Assert the agent has a_initiate_chat or initiate_chat."""
        super().validate()
        has_async = hasattr(self.agent, "a_initiate_chat")
        has_sync  = hasattr(self.agent, "initiate_chat")
        if not (has_async or has_sync):
            raise AdapterConfigError(
                "AutoGenAdapter requires an AutoGen ConversableAgent "
                "with 'a_initiate_chat' or 'initiate_chat' method."
            )

    async def execute(self, task: Task) -> AdapterResult:
        """
        Run the AutoGen agent for the task's latest message.

        Creates a temporary UserProxyAgent as the sender, initiates
        a chat with max_turns=N, and reads the last reply.

        Returns:
            AdapterResult with the agent's last reply as text.
        """
        input_text = self.extract_input(task)
        if not input_text:
            return self.make_error("Task has no message content to process.")

        try:
            proxy  = self._build_proxy()
            output = await self._run_chat(proxy, input_text)
        except Exception as exc:
            logger.error("AutoGenAdapter error: %s", exc)
            raise self._wrap_exception(exc) from exc

        logger.debug(
            "AutoGenAdapter: task %s completed output=%r",
            task.id, output[:80],
        )

        return self.make_result(
            output=output,
            artifact_name="autogen_result",
            metadata={
                "framework":  "autogen",
                "task_id":    task.id,
                "max_turns":  self._max_turns,
            },
        )

    def _build_proxy(self) -> Any:
        """Build a temporary UserProxyAgent for sending messages."""
        try:
            import autogen
        except ImportError as exc:
            raise AdapterConfigError(
                "pyautogen is not installed. "
                "Run: pip install nexus-a2a pyautogen"
            ) from exc

        return autogen.UserProxyAgent(
            name="nexus_proxy",
            human_input_mode=self._human_input_mode,
            max_consecutive_auto_reply=self._max_turns,
            code_execution_config=False,   # disable code exec for safety
        )

    async def _run_chat(self, proxy: Any, input_text: str) -> str:
        """Initiate chat and extract the last assistant reply."""
        if hasattr(proxy, "a_initiate_chat"):
            chat_result = await proxy.a_initiate_chat(
                self.agent,
                message=input_text,
                max_turns=self._max_turns,
            )
        else:
            import asyncio
            loop = asyncio.get_event_loop()
            chat_result = await loop.run_in_executor(
                None,
                lambda: proxy.initiate_chat(
                    self.agent,
                    message=input_text,
                    max_turns=self._max_turns,
                ),
            )

        return self._extract_last_reply(chat_result)

    @staticmethod
    def _extract_last_reply(chat_result: Any) -> str:
        """
        Pull the last assistant message from a ChatResult.

        Tries (in order):
          1. chat_result.summary           (AutoGen v0.4+)
          2. chat_result.chat_history[-1]["content"]  (message list)
          3. str(chat_result)              (fallback)
        """
        if hasattr(chat_result, "summary") and chat_result.summary:
            return str(chat_result.summary)

        history = getattr(chat_result, "chat_history", None)
        if history:
            last = history[-1]
            if isinstance(last, dict):
                return str(last.get("content", ""))

        return str(chat_result)
