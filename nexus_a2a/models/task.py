"""
nexus_a2a/models/task.py

Pydantic models for the A2A task lifecycle — the core unit of
work exchanged between agents.

Lifecycle:
    SUBMITTED → WORKING → COMPLETED
                        → FAILED
                        → CANCELLED
              → INPUT_REQUIRED → (client replies) → WORKING
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

# ── Helpers ───────────────────────────────────────────────────────────────────

def _new_id() -> str:
    """Generate a short, unique ID using uuid4."""
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    """Return the current UTC time (timezone-aware)."""
    return datetime.now(UTC)


# ── Enums ─────────────────────────────────────────────────────────────────────

class TaskState(str, Enum):
    """
    All possible states a Task can be in.
    Mirrors the A2A protocol specification's task lifecycle.
    """
    SUBMITTED       = "submitted"       # Client sent the task, not yet picked up
    WORKING         = "working"         # Agent is actively processing
    INPUT_REQUIRED  = "input_required"  # Agent needs more info from client
    COMPLETED       = "completed"       # Task finished successfully
    FAILED          = "failed"          # Task ended with an error
    CANCELLED       = "cancelled"       # Task was cancelled by client or system

    @property
    def is_terminal(self) -> bool:
        """True if the task cannot transition to any further state."""
        return self in {
            TaskState.COMPLETED,
            TaskState.FAILED,
            TaskState.CANCELLED,
        }


class PartType(str, Enum):
    """The kind of content carried inside a Part."""
    TEXT = "text"
    JSON = "json"
    FILE = "file"


class MessageRole(str, Enum):
    """Who sent this message — the human-side client or the agent."""
    USER  = "user"
    AGENT = "agent"


# ── Content building blocks ───────────────────────────────────────────────────

class Part(BaseModel):
    """
    The smallest unit of content inside a Message or Artifact.
    Every piece of data — text, JSON payload, file reference — is a Part.

    Examples:
        Part(type=PartType.TEXT, content="Hello, agent!")
        Part(type=PartType.JSON, content={"query": "AI papers 2025"})
        Part(type=PartType.FILE, content="https://example.com/doc.pdf", mime_type="application/pdf")
    """
    type:      PartType
    content:   Any = Field(description="The actual payload — string, dict, or URL.")
    mime_type: str | None = Field(
        default=None,
        description="MIME type hint — mainly used when type=FILE.",
    )

    @field_validator("content", mode="before")
    @classmethod
    def _content_not_none(cls, v: Any) -> Any:
        if v is None:
            raise ValueError("Part.content must not be None.")
        return v


class Message(BaseModel):
    """
    One turn of communication between a client and an agent.
    A task is made up of one or more Messages going back and forth.

    Example:
        Message(
            role=MessageRole.USER,
            parts=[Part(type=PartType.TEXT, content="Summarise this article.")],
        )
    """
    id:         str         = Field(default_factory=_new_id)
    role:       MessageRole
    parts:      list[Part]  = Field(min_length=1)
    created_at: datetime    = Field(default_factory=_utcnow)

    model_config = {"use_enum_values": True}

    # ── Convenience constructors ──────────────────────────────────────────────

    @classmethod
    def user_text(cls, text: str) -> Message:
        """Shortcut: create a plain-text user message."""
        return cls(
            role=MessageRole.USER,
            parts=[Part(type=PartType.TEXT, content=text)],
        )

    @classmethod
    def agent_text(cls, text: str) -> Message:
        """Shortcut: create a plain-text agent reply."""
        return cls(
            role=MessageRole.AGENT,
            parts=[Part(type=PartType.TEXT, content=text)],
        )

    def text(self) -> str:
        """
        Extract all TEXT parts and join them into a single string.
        Useful for simple agents that only deal with plain text.
        """
        return " ".join(
            str(p.content)
            for p in self.parts
            if p.type == PartType.TEXT
        )


class Artifact(BaseModel):
    """
    An immutable output produced by an agent when a task completes.
    Unlike Messages (which are conversational), Artifacts are final results
    — documents, structured data, files.

    Example:
        Artifact(
            name="summary",
            description="Summarised article",
            parts=[Part(type=PartType.TEXT, content="The article discusses...")],
        )
    """
    id:          str        = Field(default_factory=_new_id)
    name:        str        = Field(min_length=1, max_length=128)
    description: str | None = None
    parts:       list[Part] = Field(min_length=1)
    created_at:  datetime   = Field(default_factory=_utcnow)


# ── Primary model ─────────────────────────────────────────────────────────────

class Task(BaseModel):
    """
    The core unit of work in A2A — a stateful, trackable job
    assigned to a remote agent.

    Created by the TaskManager when a client sends a message.
    Progresses through TaskState until it reaches a terminal state.

    Example:
        task = Task.create(
            skill_id="web_search",
            initial_message=Message.user_text("Find AI papers from 2025"),
        )
    """

    # Identity
    id:         str      = Field(default_factory=_new_id)
    context_id: str      = Field(
        default_factory=_new_id,
        description="Groups related tasks together in one conversation context.",
    )
    skill_id:   str | None = Field(
        default=None,
        description="Which agent skill this task targets.",
    )

    # State
    state:      TaskState = Field(default=TaskState.SUBMITTED)
    error:      str | None = Field(
        default=None,
        description="Human-readable error message — populated when state=FAILED.",
    )

    # Content
    history:   list[Message]  = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)

    # Timestamps
    created_at:  datetime = Field(default_factory=_utcnow)
    updated_at:  datetime = Field(default_factory=_utcnow)

    model_config = {"use_enum_values": False}  # keep enum objects, not raw strings

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        initial_message: Message,
        skill_id: str | None = None,
        context_id: str | None = None,
    ) -> Task:
        """
        Create a new Task in SUBMITTED state with an initial user message.

        Args:
            initial_message: The first Message from the client.
            skill_id:        Optional skill this task targets.
            context_id:      Optional context ID to group related tasks.
        """
        return cls(
            skill_id=skill_id,
            context_id=context_id or _new_id(),
            state=TaskState.SUBMITTED,
            history=[initial_message],
        )

    # ── State transitions ─────────────────────────────────────────────────────

    # Valid transitions: which states can move to which
    _TRANSITIONS: dict[TaskState, set[TaskState]] = {
        TaskState.SUBMITTED:      {TaskState.WORKING, TaskState.CANCELLED},
        TaskState.WORKING:        {TaskState.COMPLETED, TaskState.FAILED,
                                   TaskState.INPUT_REQUIRED, TaskState.CANCELLED},
        TaskState.INPUT_REQUIRED: {TaskState.WORKING, TaskState.CANCELLED},
        TaskState.COMPLETED:      set(),   # terminal
        TaskState.FAILED:         set(),   # terminal
        TaskState.CANCELLED:      set(),   # terminal
    }

    def transition(self, new_state: TaskState, error: str | None = None) -> None:
        """
        Move the task to a new state, validating the transition is legal.

        Args:
            new_state: The state to move to.
            error:     Required when transitioning to FAILED.

        Raises:
            ValueError: If the transition is not allowed.
        """
        allowed = self._TRANSITIONS.get(self.state, set())
        if new_state not in allowed:
            raise ValueError(
                f"Cannot transition task {self.id!r} "
                f"from {self.state.value!r} to {new_state.value!r}. "
                f"Allowed: {[s.value for s in allowed]}"
            )
        if new_state == TaskState.FAILED and not error:
            raise ValueError("Must provide an error message when transitioning to FAILED.")

        self.state      = new_state
        self.error      = error
        self.updated_at = _utcnow()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def add_message(self, message: Message) -> None:
        """Append a message to the task's conversation history."""
        self.history.append(message)
        self.updated_at = _utcnow()

    def add_artifact(self, artifact: Artifact) -> None:
        """Attach a completed output artifact to this task."""
        self.artifacts.append(artifact)
        self.updated_at = _utcnow()

    def is_done(self) -> bool:
        """Return True if the task has reached a terminal state."""
        return self.state.is_terminal

    def latest_message(self) -> Message | None:
        """Return the most recent message in history, or None if empty."""
        return self.history[-1] if self.history else None
