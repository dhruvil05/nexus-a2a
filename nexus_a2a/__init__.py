"""
nexus_a2a — Developer-friendly A2A multi-agent communication for Python.

Public API for Phase 1. Import everything you need from here:

    from nexus_a2a import agent, get_card
    from nexus_a2a import AgentCard, AgentSkill, AgentCapabilities
    from nexus_a2a import Task, TaskState, Message, Artifact, Part
"""

# ── Version ───────────────────────────────────────────────────────────────────
__version__ = "1.0.0"

# ── Decorator — the primary developer entry point ─────────────────────────────
from nexus_a2a.adapters.autogen import AutoGenAdapter

# ── Phase 5: Adapters + observability ────────────────────────────────────────
from nexus_a2a.adapters.base import (
    AdapterConfigError,
    AdapterError,
    AdapterExecutionError,
    AdapterResult,
    BaseAdapter,
)
from nexus_a2a.adapters.crewai import CrewAIAdapter
from nexus_a2a.adapters.google_adk import GoogleADKAdapter
from nexus_a2a.adapters.langgraph import LangGraphAdapter

# ── Phase 4: Orchestration + streaming ───────────────────────────────────────
from nexus_a2a.core.orchestrator import (
    DAGNode,
    Orchestrator,
    OrchestratorError,
    OrchestratorResult,
    StepResult,
    WorkflowCycleError,
    WorkflowStepError,
)

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
from nexus_a2a.network import AgentNetwork, EventBus

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
from nexus_a2a.storage.audit_logger import AuditEntry, AuditEvent, AuditLogger
from nexus_a2a.storage.metrics import MetricsCollector, MetricsSnapshot
from nexus_a2a.storage.redis_store import RedisTaskStore
from nexus_a2a.storage.task_store import InMemoryTaskStore
from nexus_a2a.transport.http_client import (
    A2AHttpClient,
    AgentUnreachableError,
    RemoteAgentError,
)
from nexus_a2a.transport.sse import (
    SSEFormatter,
    SSEStreamer,
    StreamEvent,
    StreamEventType,
)
from nexus_a2a.transport.webhook import (
    WebhookConfig,
    WebhookDeliveryError,
    WebhookDispatcher,
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
    # Phase 4 — Orchestration + streaming
    "AgentNetwork",
    "EventBus",
    "Orchestrator",
    "DAGNode",
    "OrchestratorResult",
    "StepResult",
    "OrchestratorError",
    "WorkflowCycleError",
    "WorkflowStepError",
    "SSEStreamer",
    "SSEFormatter",
    "StreamEvent",
    "StreamEventType",
    "WebhookDispatcher",
    "WebhookConfig",
    "WebhookDeliveryError",
    # Phase 5 — Adapters + observability
    "BaseAdapter",
    "AdapterResult",
    "AdapterError",
    "AdapterConfigError",
    "AdapterExecutionError",
    "LangGraphAdapter",
    "CrewAIAdapter",
    "GoogleADKAdapter",
    "AutoGenAdapter",
    "RedisTaskStore",
    "AuditLogger",
    "AuditEvent",
    "AuditEntry",
    "MetricsCollector",
    "MetricsSnapshot",
    # Task models
    "Task",
    "TaskState",
    "Message",
    "MessageRole",
    "Part",
    "PartType",
    "Artifact",
]
