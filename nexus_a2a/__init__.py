"""
nexus_a2a — Developer-friendly A2A multi-agent communication for Python.

Public API for Phase 1. Import everything you need from here:

    from nexus_a2a import agent, get_card
    from nexus_a2a import AgentCard, AgentSkill, AgentCapabilities
    from nexus_a2a import Task, TaskState, Message, Artifact, Part
"""

# ── Version ───────────────────────────────────────────────────────────────────
__version__ = "1.2.0"

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

# ── v1.2: Config ─────────────────────────────────────────────────────────────
from nexus_a2a.config import (
    AgentConfig,
    ConfigError,
    NetworkConfig,
    NexusConfig,
    ObservabilityConfig,
    ReliabilityConfig,
    SecurityConfig,
    SkillConfig,
    StorageConfig,
)

# ── v1.2: AgentServer ────────────────────────────────────────────────────────
from nexus_a2a.core.agent_server import AgentServer
from nexus_a2a.core.dead_letter import DeadLetterQueue, DLQEntry, ReplayResult

# ── v1.2: GracefulShutdown ───────────────────────────────────────────────────
from nexus_a2a.core.graceful_shutdown import GracefulShutdown
from nexus_a2a.core.input_handler import (
    InputHandler,
    InputTimeoutError,
    NoInputWaiterError,
)

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
from nexus_a2a.core.registry import AgentRegistry, RegistryEntry
from nexus_a2a.core.task_manager import (
    TaskAlreadyDoneError,
    TaskManager,
    TaskNotFoundError,
    TaskTimeoutError,
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
from nexus_a2a.security.capability_guard import (
    CapabilityGuard,
    CapabilityMismatchError,
    CapabilityNotSupportedError,
)

# ── v1.2: MutualTLS ──────────────────────────────────────────────────────────
from nexus_a2a.security.mtls import (
    CertInfo,
    MtlsCertificateError,
    MtlsConfigError,
    MtlsError,
    MutualTLSConfig,
    build_client_ssl_context,
    build_server_ssl_context,
    verify_peer_certificate,
)
from nexus_a2a.security.rate_limiter import RateLimitConfig, RateLimiter, RateLimitError
from nexus_a2a.security.trust import (
    AgentNotAllowedError,
    SkillNotAllowedError,
    TrustBoundary,
    TrustError,
)
from nexus_a2a.security.validator import (
    BlankTextPartError,
    InvalidPartError,
    PayloadTooLargeError,
    PayloadValidator,
    TooManyPartsError,
    ValidatorConfig,
)
from nexus_a2a.storage.audit_logger import AuditEntry, AuditEvent, AuditLogger
from nexus_a2a.storage.metrics import MetricsCollector, MetricsSnapshot

# ── v1.2: PostgresTaskStore ──────────────────────────────────────────────────
from nexus_a2a.storage.postgres_store import PostgresTaskStore
from nexus_a2a.storage.redis_store import RedisTaskStore
from nexus_a2a.storage.task_store import AbstractTaskStore, InMemoryTaskStore
from nexus_a2a.transport.http_client import (
    A2AHttpClient,
    AgentCardFetchError,
    AgentUnreachableError,
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    RemoteAgentError,
    RetryConfig,
    TransportError,
)
from nexus_a2a.transport.sse import (
    SSEFormatter,
    SSEStreamer,
    StreamEvent,
    StreamEventType,
)

# ── v1.1.0: reliability + DX upgrades ────────────────────────────────────────
from nexus_a2a.transport.tracing import TRACE_ID_HEADER, Span, Trace, Tracer, TraceStore
from nexus_a2a.transport.webhook import (
    DeliveryRecord,
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
    "RegistryEntry",
    "TaskManager",
    "TaskNotFoundError",
    "TaskAlreadyDoneError",
    "TaskTimeoutError",
    "InMemoryTaskStore",
    "AbstractTaskStore",
    "A2AHttpClient",
    "AgentUnreachableError",
    "AgentCardFetchError",
    "RemoteAgentError",
    "TransportError",
    "CircuitBreaker",
    "CircuitOpenError",
    "CircuitState",
    "RetryConfig",
    "InputHandler",
    "InputTimeoutError",
    "NoInputWaiterError",
    "DeadLetterQueue",
    "DLQEntry",
    "ReplayResult",
    # Phase 3 — Security
    "AuthManager",
    "AgentCredentialConfig",
    "AuthError",
    "MissingCredentialsError",
    "InvalidCredentialsError",
    "ExpiredCredentialsError",
    "TrustBoundary",
    "TrustError",
    "AgentNotAllowedError",
    "SkillNotAllowedError",
    "CapabilityGuard",
    "CapabilityMismatchError",
    "CapabilityNotSupportedError",
    "RateLimiter",
    "RateLimitConfig",
    "RateLimitError",
    "PayloadValidator",
    "ValidatorConfig",
    "PayloadTooLargeError",
    "TooManyPartsError",
    "InvalidPartError",
    "BlankTextPartError",
    "MutualTLSConfig",
    "CertInfo",
    "MtlsError",
    "MtlsConfigError",
    "MtlsCertificateError",
    "build_client_ssl_context",
    "build_server_ssl_context",
    "verify_peer_certificate",
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
    "DeliveryRecord",
    "Tracer",
    "Trace",
    "Span",
    "TraceStore",
    "TRACE_ID_HEADER",
    "GracefulShutdown",
    "AgentServer",
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
    "PostgresTaskStore",
    "AuditLogger",
    "AuditEvent",
    "AuditEntry",
    "MetricsCollector",
    "MetricsSnapshot",
    # Config
    "NexusConfig",
    "AgentConfig",
    "NetworkConfig",
    "SecurityConfig",
    "SkillConfig",
    "StorageConfig",
    "ReliabilityConfig",
    "ObservabilityConfig",
    "ConfigError",
    # Task models
    "Task",
    "TaskState",
    "Message",
    "MessageRole",
    "Part",
    "PartType",
    "Artifact",
]
