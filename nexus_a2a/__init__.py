"""
nexus_a2a — Developer-friendly A2A multi-agent communication for Python.

Public API for Phase 1. Import everything you need from here:

    from nexus_a2a import agent, get_card
    from nexus_a2a import AgentCard, AgentSkill, AgentCapabilities
    from nexus_a2a import Task, TaskState, Message, Artifact, Part
"""

# ── Version ───────────────────────────────────────────────────────────────────
__version__ = "0.1.0"

# ── Decorator — the primary developer entry point ─────────────────────────────
from nexus_a2a.decorators import agent, get_card

# ── Agent models ──────────────────────────────────────────────────────────────
from nexus_a2a.models.agent import (
    AgentAuthentication,
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    AuthScheme,
    InputMode,
    OutputMode,
)

# ── Task models ───────────────────────────────────────────────────────────────
from nexus_a2a.models.task import (
    Artifact,
    Message,
    MessageRole,
    Part,
    PartType,
    Task,
    TaskState,
)

# ── What gets exported when someone does: from nexus_a2a import * ─────────────
__all__ = [
    # Decorator
    "agent",
    "get_card",
    # Agent models
    "AgentCard",
    "AgentSkill",
    "AgentCapabilities",
    "AgentAuthentication",
    "AuthScheme",
    "InputMode",
    "OutputMode",
    # Task models
    "Task",
    "TaskState",
    "Message",
    "MessageRole",
    "Part",
    "PartType",
    "Artifact",
]
