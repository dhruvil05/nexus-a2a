"""
nexus_a2a/adapters/base.py

BaseAdapter — the interface every framework adapter must implement.

An adapter's job is exactly one thing:
  Take an agent object built with framework X (LangGraph, CrewAI, etc.)
  and make it behave like an A2A AgentExecutor — so it can receive
  A2A Tasks and return results without knowing anything about the protocol.

Every concrete adapter:
  1. Inherits from BaseAdapter.
  2. Implements `execute(task) → AdapterResult`.
  3. Optionally overrides `cancel(task_id)`.
  4. Calls `self.validate()` in __init__ to catch misconfiguration early.

Why this abstraction matters:
  - The rest of the package (TaskManager, Orchestrator, AgentNetwork)
    never imports LangGraph, CrewAI, or any other framework directly.
  - New adapters can be added by the community without touching core code.
  - Each adapter is independently testable with a mock agent object.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from nexus_a2a.models.task import Artifact, Part, PartType, Task

# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class AdapterResult:
    """
    The output of one adapter execution.

    Fields:
        output:    The text or structured result produced by the agent.
        artifact:  Optional Artifact built from the output. If None,
                   the adapter layer builds a default text artifact.
        metadata:  Any framework-specific extras (token counts, run IDs, etc.)
                   Stored for observability — not used by the protocol.
        error:     Set if the framework raised an exception.
    """
    output:   str | None          = None
    artifact: Artifact | None     = None
    metadata: dict[str, Any]      = field(default_factory=dict)
    error:    str | None          = None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    def to_artifact(self, name: str = "result") -> Artifact:
        """
        Return self.artifact if already set, otherwise build a default
        text artifact from self.output.
        """
        if self.artifact:
            return self.artifact
        return Artifact(
            name=name,
            parts=[Part(
                type=PartType.TEXT,
                content=self.output or "",
            )],
        )


# ── Exceptions ────────────────────────────────────────────────────────────────

class AdapterError(Exception):
    """Base class for adapter errors."""


class AdapterConfigError(AdapterError):
    """Raised when an adapter is misconfigured at construction time."""


class AdapterExecutionError(AdapterError):
    """Raised when the underlying framework raises during execution."""

    def __init__(self, framework: str, reason: str) -> None:
        super().__init__(f"[{framework}] Execution failed: {reason}")
        self.framework = framework
        self.reason    = reason


# ── BaseAdapter ───────────────────────────────────────────────────────────────

class BaseAdapter(ABC):
    """
    Abstract base class for all framework adapters.

    Subclass this to wrap any AI agent framework.

    Minimal implementation:

        class MyFrameworkAdapter(BaseAdapter):
            framework_name = "myframework"

            def validate(self) -> None:
                if self.agent is None:
                    raise AdapterConfigError("agent must not be None")

            async def execute(self, task: Task) -> AdapterResult:
                input_text = task.latest_message().text()
                result = await self.agent.run(input_text)
                return AdapterResult(output=str(result))

    Args:
        agent:  The framework-specific agent object to wrap.
        config: Optional dict of adapter-level settings.
    """

    # Subclasses set this to identify themselves in logs and errors
    framework_name: str = "unknown"

    def __init__(
        self,
        agent: Any,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.agent  = agent
        self.config = config or {}
        self.validate()

    # ── Interface ─────────────────────────────────────────────────────────────

    def validate(self) -> None:
        """
        Assert the adapter is correctly configured.
        Called automatically in __init__.
        Override to add framework-specific checks.

        Raises:
            AdapterConfigError: If the configuration is invalid.
        """
        if self.agent is None:
            raise AdapterConfigError(
                f"{self.__class__.__name__}: 'agent' must not be None."
            )

    @abstractmethod
    async def execute(self, task: Task) -> AdapterResult:
        """
        Run the wrapped agent for the given Task.

        The adapter extracts the input from task.latest_message(),
        calls the underlying framework, and returns an AdapterResult.

        Args:
            task: The A2A Task to process. Read task.latest_message()
                  to get the user's input.

        Returns:
            AdapterResult with output text and optional artifact.

        Raises:
            AdapterExecutionError: If the framework raises.
        """

    async def cancel(self, task_id: str) -> None:
        """
        Attempt to cancel an in-progress execution.

        Most frameworks do not support cancellation — the default
        implementation is a no-op. Override for frameworks that do.

        Args:
            task_id: The A2A task ID being cancelled.
        """
        return None

    # ── Shared helpers ────────────────────────────────────────────────────────

    def extract_input(self, task: Task) -> str:
        """
        Pull the latest user message text out of a Task.
        Convenience method so every adapter doesn't repeat this pattern.

        Returns:
            The text content of the latest message, or "" if none.
        """
        msg = task.latest_message()
        return msg.text() if msg else ""

    def make_result(
        self,
        output: str,
        artifact_name: str = "result",
        metadata: dict[str, Any] | None = None,
    ) -> AdapterResult:
        """
        Build a successful AdapterResult with a text artifact.
        Convenience factory so adapters don't repeat boilerplate.
        """
        artifact = Artifact(
            name=artifact_name,
            parts=[Part(type=PartType.TEXT, content=output)],
        )
        return AdapterResult(
            output=output,
            artifact=artifact,
            metadata=metadata or {},
        )

    def make_error(self, reason: str) -> AdapterResult:
        """Build a failed AdapterResult."""
        return AdapterResult(error=reason)

    def _wrap_exception(self, exc: Exception) -> AdapterExecutionError:
        """Wrap a framework exception into an AdapterExecutionError."""
        return AdapterExecutionError(
            framework=self.framework_name,
            reason=str(exc),
        )
