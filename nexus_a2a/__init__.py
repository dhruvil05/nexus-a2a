"""
nexus_a2a — Developer-friendly A2A multi-agent communication for Python.

Public API for Phase 1. Import everything you need from here:

    from nexus_a2a import agent, get_card
    from nexus_a2a import AgentCard, AgentSkill, AgentCapabilities
    from nexus_a2a import Task, TaskState, Message, Artifact, Part
"""

# ── Version ───────────────────────────────────────────────────────────────────
__version__ = "0.3.0"

# ── Decorator — the primary developer entry point ─────────────────────────────
# ── Phase 2: Core engine ──────────────────────────────────────────────────────
from nexus_a2a.core.registry import AgentRegistry
from nexus_a2a.core.task_manager import (
    TaskAlreadyDoneError,
    TaskManager,
    TaskNotFoundError,
)
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

# ── Phase 3: Security ─────────────────────────────────────────────────────────
from nexus_a2a.security.auth import (
    AgentCredentialConfig,
    AuthError,
    AuthManager,
    ExpiredCredentialsError,
    InvalidCredentialsError,
    MissingCredentialsError,
)
from nexus_a2a.security.rate_limiter import RateLimitConfig, RateLimiter, RateLimitError
from nexus_a2a.security.trust import (
    AgentNotAllowedError,
    SkillNotAllowedError,
    TrustBoundary,
)
from nexus_a2a.security.validator import PayloadValidator, ValidatorConfig
from nexus_a2a.storage.task_store import InMemoryTaskStore
from nexus_a2a.transport.http_client import (
    A2AHttpClient,
    AgentUnreachableError,
    RemoteAgentError,
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
    # Phase 2 — Core engine
    "AgentRegistry",
    "TaskManager",
    "TaskNotFoundError",
    "TaskAlreadyDoneError",
    "InMemoryTaskStore",
    "A2AHttpClient",
    "AgentUnreachableError",
    "RemoteAgentError",
    # Phase 3 — Security
    "AuthManager",
    "AgentCredentialConfig",
    "AuthError",
    "MissingCredentialsError",
    "InvalidCredentialsError",
    "ExpiredCredentialsError",
    "TrustBoundary",
    "AgentNotAllowedError",
    "SkillNotAllowedError",
    "RateLimiter",
    "RateLimitConfig",
    "RateLimitError",
    "PayloadValidator",
    "ValidatorConfig",
    # Task models
    "Task",
    "TaskState",
    "Message",
    "MessageRole",
    "Part",
    "PartType",
    "Artifact",
]
