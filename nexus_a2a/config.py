"""
nexus_a2a/config.py

nexus.toml parser and zero-config wiring for AgentNetwork.

Usage:
    network = AgentNetwork.from_config("nexus.toml")

Or programmatically:
    cfg = NexusConfig.from_file("nexus.toml")
    network = cfg.build_network()

Environment variable overrides (container-friendly):
    NEXUS_AGENT_NAME        — overrides [agent].name
    NEXUS_AGENT_URL         — overrides [agent].url
    NEXUS_AUTH_SCHEME       — overrides [security].auth_scheme
    NEXUS_AUTH_SECRET       — JWT secret or API key value
    NEXUS_STORAGE_BACKEND   — overrides [storage].backend (memory|redis|postgres)
    NEXUS_STORAGE_URL       — overrides [storage].url
    NEXUS_TASK_TIMEOUT      — overrides [reliability].task_timeout_sec
    NEXUS_LOG_LEVEL         — overrides [observability].log_level
    NEXUS_RATE_LIMIT        — overrides [security].rate_limit
    NEXUS_PUSH_SECRET       — overrides [push].signing_secret
    NEXUS_ADMIN_TOKEN       — overrides [ops].admin_token
    NEXUS_OPS_PORT          — overrides [ops].port
    NEXUS_OPS_URL           — overrides [ops].url

All NEXUS_* env vars are read after the TOML file is parsed,
so they always take precedence — ideal for container deployments.

Design:
    Uses tomllib (stdlib in Python 3.11+). No extra deps needed.
    Validates at parse time — fails fast with a clear ConfigError.
    Returns typed dataclasses, not raw dicts — type checker friendly.
"""

from __future__ import annotations

import difflib
import logging
import os
import tomllib
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nexus_a2a.models.agent import AuthScheme

logger = logging.getLogger(__name__)


# ── Exceptions ────────────────────────────────────────────────────────────────


class ConfigError(Exception):
    """
    Raised when nexus.toml is missing a required field, has an invalid
    value, or fails environment variable override parsing.

    Includes the offending key path so developers can fix it immediately.
    """

    def __init__(self, message: str, key: str | None = None) -> None:
        location = f" (key: '{key}')" if key else ""
        super().__init__(f"nexus.toml config error{location}: {message}")
        self.key = key


def _number(raw: dict[str, Any], key: str, default: float, path: str) -> float:
    """Read a numeric key, turning a bad value into a ConfigError with its path."""
    value = raw.get(key, default)
    if isinstance(value, bool):
        raise ConfigError(f"'{key}' must be a number, got a boolean", path)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"'{key}' must be a number, got {value!r}", path) from exc


class ConfigWarning(UserWarning):
    """A nexus.toml problem that does not stop the config from loading."""


# Every key the parser reads. Anything else is ignored — which, for a security
# setting, silently leaves it unset — so unknown keys are reported.
KNOWN_KEYS: dict[str, frozenset[str]] = {
    "agent": frozenset({"name", "description", "version", "url", "streaming",
                        "skills"}),
    "network": frozenset({"agents"}),
    "reliability": frozenset({"task_timeout_sec", "max_retries", "retry_on",
                              "circuit_breaker_threshold", "circuit_recovery_sec",
                              "base_delay_sec", "max_delay_sec"}),
    "security": frozenset({"auth_scheme", "auth_secret", "trust_mode",
                           "rate_limit", "rate_burst", "max_payload_bytes",
                           "allow_insecure"}),
    "storage": frozenset({"backend", "url", "ttl_sec", "task_retention_sec",
                          "max_tasks"}),
    "observability": frozenset({"tracing", "metrics", "log_level"}),
    "push": frozenset({"signing_secret", "allow_private_urls", "max_retries"}),
    "ops": frozenset({"port", "host", "url", "admin_token"}),
    "dev": frozenset({"agents"}),  # read by `nexus dev`
}
_SKILL_KEYS = frozenset({"id", "name", "description", "tags", "examples"})


def _unknown_key_messages(raw: dict[str, Any]) -> list[str]:
    """Describe every key the parser would silently ignore."""

    def suggest(name: str, choices: frozenset[str]) -> str:
        close = difflib.get_close_matches(name, sorted(choices), n=1, cutoff=0.6)
        return f" (did you mean '{close[0]}'?)" if close else ""

    messages: list[str] = []
    sections = frozenset(KNOWN_KEYS)
    for section, body in raw.items():
        if section not in KNOWN_KEYS:
            messages.append(
                f"unknown section [{section}]{suggest(section, sections)}"
            )
            continue
        if not isinstance(body, dict):
            continue
        for key in body:
            if key not in KNOWN_KEYS[section]:
                messages.append(
                    f"unknown key '{section}.{key}'"
                    f"{suggest(key, KNOWN_KEYS[section])}"
                )
    for index, skill in enumerate(raw.get("agent", {}).get("skills", []) or []):
        if isinstance(skill, dict):
            for key in skill:
                if key not in _SKILL_KEYS:
                    messages.append(
                        f"unknown key 'agent.skills[{index}].{key}'"
                        f"{suggest(key, _SKILL_KEYS)}"
                    )
    return messages


# ── Section dataclasses ───────────────────────────────────────────────────────


@dataclass
class SkillConfig:
    """One entry in [[agent.skills]]."""

    id: str
    name: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)


@dataclass
class AgentConfig:
    """[agent] section."""

    name: str = "agent"
    description: str = ""
    version: str = "1.0.0"
    url: str = "http://localhost:8000"
    streaming: bool = False
    skills: list[SkillConfig] = field(default_factory=list)


@dataclass
class NetworkConfig:
    """[network] section."""

    agents: list[str] = field(default_factory=list)


@dataclass
class ReliabilityConfig:
    """[reliability] section."""

    task_timeout_sec: float = 120.0
    max_retries: int = 3
    retry_on: list[int] = field(default_factory=lambda: [500, 502, 503, 504])
    circuit_breaker_threshold: int = 5
    circuit_recovery_sec: float = 30.0
    base_delay_sec: float = 1.0
    max_delay_sec: float = 30.0


@dataclass
class SecurityConfig:
    """[security] section."""

    auth_scheme: str = "none"  # "none" | "jwt" | "api_key"
    auth_secret: str = ""  # JWT secret or API key value
    trust_mode: str = "off"  # "strict" | "warn" | "off"
    # Requests per second allowed per caller; 0 disables rate limiting.
    rate_limit: float = 0.0
    rate_burst: int = 20
    # Largest accepted request body in bytes; 0 disables payload validation.
    max_payload_bytes: int = 0
    # Silence the warning for serving a public address with no security.
    # In 2.0 that combination becomes an error unless this is set.
    allow_insecure: bool = False


@dataclass
class StorageConfig:
    """[storage] section."""

    backend: str = "memory"  # "memory" | "redis" | "postgres"
    url: str = ""  # redis:// or postgres:// connection URL
    ttl_sec: int = 3600  # TTL for Redis keys (ignored for memory/postgres)
    # In-memory backend only: how long a finished task stays retrievable, and
    # the most tasks held. 0 means "no limit" for either.
    task_retention_sec: float = 3600.0
    max_tasks: int = 10_000


@dataclass
class PushConfig:
    """[push] section — outbound webhook delivery."""

    signing_secret: str = ""  # HMAC-SHA256 key; empty sends unsigned
    allow_private_urls: bool = False  # local development only
    max_retries: int = 3


@dataclass
class OpsConfig:
    """
    [ops] section — the admin/metrics server and where the CLI finds it.

    port:        Start an AgentServer on this port alongside the agent. 0 = off.
    host:        Bind host for it. Empty = same host as the agent.
    url:         Where `nexus trace` / `nexus replay` send admin requests.
    admin_token: Token the ops server requires, and the CLI sends.
    """

    port: int = 0
    host: str = ""
    url: str = ""
    admin_token: str = ""


@dataclass
class ObservabilityConfig:
    """[observability] section."""

    tracing: bool = True
    metrics: bool = True
    log_level: str = "INFO"


@dataclass
class NexusConfig:
    """
    Fully parsed and validated representation of nexus.toml.

    All fields have sensible defaults so a minimal TOML works:

        [agent]
        name = "MyAgent"
        url  = "http://localhost:8001"

    Everything else is optional and has safe defaults.
    """

    agent: AgentConfig = field(default_factory=AgentConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    reliability: ReliabilityConfig = field(default_factory=ReliabilityConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    push: PushConfig = field(default_factory=PushConfig)
    ops: OpsConfig = field(default_factory=OpsConfig)

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def from_file(cls, path: str | Path = "nexus.toml") -> NexusConfig:
        """
        Parse a nexus.toml file and apply environment variable overrides.

        Args:
            path: Path to the TOML file. Defaults to 'nexus.toml' in
                  the current working directory.

        Raises:
            ConfigError: File not found, invalid TOML, or bad values.

        Returns:
            Fully validated NexusConfig instance.
        """
        resolved = Path(path).resolve()

        if not resolved.exists():
            raise ConfigError(
                f"File not found: '{resolved}'. "
                "Create a nexus.toml file or pass an explicit path.",
            )

        try:
            with open(resolved, "rb") as fh:
                raw: dict[str, Any] = tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"Invalid TOML syntax: {exc}") from exc

        config = cls._parse(raw)
        config._apply_env_overrides()
        config._validate()

        logger.debug("NexusConfig loaded from '%s'", resolved)
        return config

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> NexusConfig:
        """
        Parse config from a raw Python dict (useful for testing).

        Args:
            raw: Dict matching the nexus.toml structure.

        Raises:
            ConfigError: Invalid or missing values.

        Returns:
            Fully validated NexusConfig instance.
        """
        config = cls._parse(raw)
        config._apply_env_overrides()
        config._validate()
        return config

    # ── Internal: parsing ─────────────────────────────────────────────────────

    @classmethod
    def _parse(cls, raw: dict[str, Any]) -> NexusConfig:
        """Convert raw TOML dict to a NexusConfig with typed sections."""
        for message in _unknown_key_messages(raw):
            # Ignored keys are how a typo like `auth_schem` leaves an agent
            # open. nexus-a2a 2.0 will reject them outright.
            warnings.warn(
                f"nexus.toml: {message} — it is ignored.",
                ConfigWarning,
                stacklevel=3,
            )
        return cls(
            agent=cls._parse_agent(raw.get("agent", {})),
            network=cls._parse_network(raw.get("network", {})),
            reliability=cls._parse_reliability(raw.get("reliability", {})),
            security=cls._parse_security(raw.get("security", {})),
            storage=cls._parse_storage(raw.get("storage", {})),
            observability=cls._parse_observability(raw.get("observability", {})),
            push=cls._parse_push(raw.get("push", {})),
            ops=cls._parse_ops(raw.get("ops", {})),
        )

    @staticmethod
    def _parse_agent(raw: dict[str, Any]) -> AgentConfig:
        skills: list[SkillConfig] = []
        for i, s in enumerate(raw.get("skills", [])):
            if "id" not in s:
                raise ConfigError(
                    "Missing 'id' in skill entry", f"agent.skills[{i}].id"
                )
            if "name" not in s:
                raise ConfigError(
                    "Missing 'name' in skill entry", f"agent.skills[{i}].name"
                )
            skills.append(
                SkillConfig(
                    id=s["id"],
                    name=s["name"],
                    description=s.get("description", ""),
                    tags=s.get("tags", []),
                    examples=s.get("examples", []),
                )
            )

        return AgentConfig(
            name=raw.get("name", ""),
            description=raw.get("description", ""),
            version=raw.get("version", "1.0.0"),
            url=raw.get("url", "http://localhost:8000"),
            streaming=bool(raw.get("streaming", False)),
            skills=skills,
        )

    @staticmethod
    def _parse_network(raw: dict[str, Any]) -> NetworkConfig:
        agents = raw.get("agents", [])
        if not isinstance(agents, list):
            raise ConfigError("'agents' must be a list of URLs", "network.agents")
        for i, a in enumerate(agents):
            if not isinstance(a, str):
                raise ConfigError(
                    f"Agent URL at index {i} must be a string, got {type(a).__name__}",
                    f"network.agents[{i}]",
                )
        return NetworkConfig(agents=agents)

    @staticmethod
    def _parse_reliability(raw: dict[str, Any]) -> ReliabilityConfig:
        retry_on = raw.get("retry_on", [500, 502, 503, 504])
        if not isinstance(retry_on, list):
            raise ConfigError(
                "'retry_on' must be a list of HTTP status codes", "reliability.retry_on"
            )

        return ReliabilityConfig(
            task_timeout_sec=float(raw.get("task_timeout_sec", 120.0)),
            max_retries=int(raw.get("max_retries", 3)),
            retry_on=[int(code) for code in retry_on],
            circuit_breaker_threshold=int(raw.get("circuit_breaker_threshold", 5)),
            circuit_recovery_sec=float(raw.get("circuit_recovery_sec", 30.0)),
            base_delay_sec=float(raw.get("base_delay_sec", 1.0)),
            max_delay_sec=float(raw.get("max_delay_sec", 30.0)),
        )

    @staticmethod
    def _parse_security(raw: dict[str, Any]) -> SecurityConfig:
        scheme = raw.get("auth_scheme", "none").lower()
        valid_schemes = {"none", "jwt", "api_key"}
        if scheme not in valid_schemes:
            raise ConfigError(
                f"Invalid auth_scheme '{scheme}'. Must be one of: {valid_schemes}",
                "security.auth_scheme",
            )

        trust = raw.get("trust_mode", "off").lower()
        valid_trust = {"strict", "warn", "off"}
        if trust not in valid_trust:
            raise ConfigError(
                f"Invalid trust_mode '{trust}'. Must be one of: {valid_trust}",
                "security.trust_mode",
            )

        return SecurityConfig(
            auth_scheme=scheme,
            auth_secret=raw.get("auth_secret", ""),
            trust_mode=trust,
            rate_limit=_number(raw, "rate_limit", 0.0, "security.rate_limit"),
            rate_burst=int(_number(raw, "rate_burst", 20, "security.rate_burst")),
            max_payload_bytes=int(
                _number(raw, "max_payload_bytes", 0, "security.max_payload_bytes")
            ),
            allow_insecure=bool(raw.get("allow_insecure", False)),
        )

    @staticmethod
    def _parse_storage(raw: dict[str, Any]) -> StorageConfig:
        backend = raw.get("backend", "memory").lower()
        valid_backends = {"memory", "redis", "postgres"}
        if backend not in valid_backends:
            raise ConfigError(
                f"Invalid storage backend '{backend}'. Must be one of: {valid_backends}",
                "storage.backend",
            )
        return StorageConfig(
            backend=backend,
            url=raw.get("url", ""),
            ttl_sec=int(raw.get("ttl_sec", 3600)),
            task_retention_sec=_number(
                raw, "task_retention_sec", 3600.0, "storage.task_retention_sec"
            ),
            max_tasks=int(_number(raw, "max_tasks", 10_000, "storage.max_tasks")),
        )

    @staticmethod
    def _parse_push(raw: dict[str, Any]) -> PushConfig:
        return PushConfig(
            signing_secret=raw.get("signing_secret", ""),
            allow_private_urls=bool(raw.get("allow_private_urls", False)),
            max_retries=int(_number(raw, "max_retries", 3, "push.max_retries")),
        )

    @staticmethod
    def _parse_ops(raw: dict[str, Any]) -> OpsConfig:
        return OpsConfig(
            port=int(_number(raw, "port", 0, "ops.port")),
            host=raw.get("host", ""),
            url=raw.get("url", ""),
            admin_token=raw.get("admin_token", ""),
        )

    @staticmethod
    def _parse_observability(raw: dict[str, Any]) -> ObservabilityConfig:
        level = raw.get("log_level", "INFO").upper()
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if level not in valid_levels:
            raise ConfigError(
                f"Invalid log_level '{level}'. Must be one of: {valid_levels}",
                "observability.log_level",
            )
        return ObservabilityConfig(
            tracing=bool(raw.get("tracing", True)),
            metrics=bool(raw.get("metrics", True)),
            log_level=level,
        )

    # ── Internal: env overrides ───────────────────────────────────────────────

    def _apply_env_overrides(self) -> None:
        """
        Apply NEXUS_* environment variables on top of parsed TOML.
        Env vars always win — this is the container deployment mechanism.
        """
        # [agent]
        if name := os.environ.get("NEXUS_AGENT_NAME"):
            self.agent.name = name
        if url := os.environ.get("NEXUS_AGENT_URL"):
            self.agent.url = url

        # [security]
        if scheme := os.environ.get("NEXUS_AUTH_SCHEME"):
            scheme_lower = scheme.lower()
            valid = {"none", "jwt", "api_key"}
            if scheme_lower not in valid:
                raise ConfigError(
                    f"NEXUS_AUTH_SCHEME='{scheme}' is invalid. Must be: {valid}",
                    "security.auth_scheme",
                )
            self.security.auth_scheme = scheme_lower
        if secret := os.environ.get("NEXUS_AUTH_SECRET"):
            self.security.auth_secret = secret

        # [storage]
        if backend := os.environ.get("NEXUS_STORAGE_BACKEND"):
            backend_lower = backend.lower()
            valid = {"memory", "redis", "postgres"}
            if backend_lower not in valid:
                raise ConfigError(
                    f"NEXUS_STORAGE_BACKEND='{backend}' is invalid. Must be: {valid}",
                    "storage.backend",
                )
            self.storage.backend = backend_lower
        if storage_url := os.environ.get("NEXUS_STORAGE_URL"):
            self.storage.url = storage_url

        # [reliability]
        if timeout := os.environ.get("NEXUS_TASK_TIMEOUT"):
            try:
                self.reliability.task_timeout_sec = float(timeout)
            except ValueError as exc:
                raise ConfigError(
                    f"NEXUS_TASK_TIMEOUT='{timeout}' is not a valid number",
                    "reliability.task_timeout_sec",
                ) from exc

        # [security] — rate limit
        if rate := os.environ.get("NEXUS_RATE_LIMIT"):
            try:
                self.security.rate_limit = float(rate)
            except ValueError as exc:
                raise ConfigError(
                    f"NEXUS_RATE_LIMIT='{rate}' is not a valid number",
                    "security.rate_limit",
                ) from exc

        # [push]
        if push_secret := os.environ.get("NEXUS_PUSH_SECRET"):
            self.push.signing_secret = push_secret

        # [ops]
        if admin_token := os.environ.get("NEXUS_ADMIN_TOKEN"):
            self.ops.admin_token = admin_token
        if ops_url := os.environ.get("NEXUS_OPS_URL"):
            self.ops.url = ops_url
        if ops_port := os.environ.get("NEXUS_OPS_PORT"):
            try:
                self.ops.port = int(ops_port)
            except ValueError as exc:
                raise ConfigError(
                    f"NEXUS_OPS_PORT='{ops_port}' is not a valid port",
                    "ops.port",
                ) from exc

        # [observability]
        if level := os.environ.get("NEXUS_LOG_LEVEL"):
            level_upper = level.upper()
            valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
            if level_upper not in valid:
                raise ConfigError(
                    f"NEXUS_LOG_LEVEL='{level}' is invalid. Must be: {valid}",
                    "observability.log_level",
                )
            self.observability.log_level = level_upper

    # ── Internal: validation ──────────────────────────────────────────────────

    def _validate(self) -> None:
        """
        Cross-field validation after parsing + env overrides.

        Raises ConfigError for configurations that are internally
        inconsistent regardless of where the values came from.
        """
        # Agent name required
        if not self.agent.name.strip():
            raise ConfigError(
                "Agent name cannot be empty. Set [agent].name in nexus.toml "
                "or NEXUS_AGENT_NAME env var.",
                "agent.name",
            )

        # Agent URL must look like HTTP(S)
        url = self.agent.url.strip()
        if url and not (url.startswith("http://") or url.startswith("https://")):
            raise ConfigError(
                f"Agent URL '{url}' must start with 'http://' or 'https://'.",
                "agent.url",
            )

        # Auth: jwt and api_key require a secret
        if self.security.auth_scheme in ("jwt", "api_key"):
            if not self.security.auth_secret.strip():
                raise ConfigError(
                    f"auth_scheme='{self.security.auth_scheme}' requires a secret. "
                    "Set [security].auth_secret in nexus.toml or NEXUS_AUTH_SECRET env var.",
                    "security.auth_secret",
                )

        # Storage: redis/postgres require a URL
        if self.storage.backend in ("redis", "postgres"):
            if not self.storage.url.strip():
                raise ConfigError(
                    f"storage.backend='{self.storage.backend}' requires a connection URL. "
                    "Set [storage].url in nexus.toml or NEXUS_STORAGE_URL env var.",
                    "storage.url",
                )

        # Reliability: sanity checks
        if self.reliability.task_timeout_sec <= 0:
            raise ConfigError(
                "task_timeout_sec must be greater than 0.",
                "reliability.task_timeout_sec",
            )
        if self.reliability.max_retries < 0:
            raise ConfigError(
                "max_retries cannot be negative.",
                "reliability.max_retries",
            )
        # Security limits
        if self.security.rate_limit < 0:
            raise ConfigError(
                "rate_limit cannot be negative (0 disables it).",
                "security.rate_limit",
            )
        if self.security.rate_limit > 0 and self.security.rate_burst < 1:
            raise ConfigError(
                "rate_burst must be at least 1 when rate_limit is set.",
                "security.rate_burst",
            )
        if self.security.max_payload_bytes < 0:
            raise ConfigError(
                "max_payload_bytes cannot be negative (0 disables it).",
                "security.max_payload_bytes",
            )

        # Storage limits
        if self.storage.task_retention_sec < 0:
            raise ConfigError(
                "task_retention_sec cannot be negative (0 means keep forever).",
                "storage.task_retention_sec",
            )
        if self.storage.max_tasks < 0:
            raise ConfigError(
                "max_tasks cannot be negative (0 means no cap).",
                "storage.max_tasks",
            )

        # Ops server
        if not 0 <= self.ops.port <= 65535:
            raise ConfigError(
                f"ops.port={self.ops.port} is not a valid port (0 disables it).",
                "ops.port",
            )

        if self.reliability.circuit_breaker_threshold < 1:
            raise ConfigError(
                "circuit_breaker_threshold must be at least 1.",
                "reliability.circuit_breaker_threshold",
            )

    # ── Build helpers ─────────────────────────────────────────────────────────

    def build_task_store(self) -> Any:
        """
        Instantiate the configured storage backend.

        Returns:
            InMemoryTaskStore, RedisTaskStore, or PostgresTaskStore
            depending on [storage].backend.

        Raises:
            ConfigError: Backend package not installed.
            ImportError: re-raised with a helpful install message.
        """
        backend = self.storage.backend

        if backend == "memory":
            from nexus_a2a.storage.task_store import InMemoryTaskStore

            return InMemoryTaskStore(
                retention_sec=self.storage.task_retention_sec or None,
                max_tasks=self.storage.max_tasks or None,
            )

        if backend == "redis":
            try:
                from nexus_a2a.storage.redis_store import RedisTaskStore
            except ImportError as exc:
                raise ConfigError(
                    "RedisTaskStore requires the 'redis' extra. "
                    "Install with: pip install nexus-a2a[redis]",
                    "storage.backend",
                ) from exc
            return RedisTaskStore(url=self.storage.url, ttl=self.storage.ttl_sec)

        if backend == "postgres":
            try:
                from nexus_a2a.storage.postgres_store import PostgresTaskStore
            except ImportError as exc:
                raise ConfigError(
                    "PostgresTaskStore requires the 'postgres' extra. "
                    "Install with: pip install nexus-a2a[postgres]",
                    "storage.backend",
                ) from exc
            return PostgresTaskStore(dsn=self.storage.url)

        # Should never reach here — _parse_storage already validated backend
        raise ConfigError(f"Unknown storage backend: '{backend}'", "storage.backend")

    def build_retry_config(self) -> Any:
        """Build a RetryConfig from [reliability] settings."""
        from nexus_a2a.transport.http_client import RetryConfig

        return RetryConfig(
            max_retries=self.reliability.max_retries,
            retry_on=set(self.reliability.retry_on),
            base_delay=self.reliability.base_delay_sec,
            max_delay=self.reliability.max_delay_sec,
        )

    def build_circuit_breaker(self) -> Any:
        """Build a CircuitBreaker from [reliability] settings."""
        from nexus_a2a.transport.http_client import CircuitBreaker

        return CircuitBreaker(
            failure_threshold=self.reliability.circuit_breaker_threshold,
            recovery_timeout=self.reliability.circuit_recovery_sec,
        )

    def build_auth_manager(self) -> Any:
        """
        Build an AuthManager from [security] settings.

        Returns an AuthManager with the configured default scheme.
        For per-agent credentials, call manager.register_agent() manually
        after building the network.
        """
        from nexus_a2a.security.auth import AgentCredentialConfig, AuthManager

        scheme_map = {
            "none": AuthScheme.NONE,
            "jwt": AuthScheme.JWT,
            "api_key": AuthScheme.API_KEY,
        }
        scheme = scheme_map[self.security.auth_scheme]
        manager = AuthManager()

        # The configured secret applies to every caller not registered
        # individually. Before 1.9.0 this was registered under the literal URL
        # "*", which the lookup never matched, so the secret was never checked
        # (and before 1.5.0, every caller was let in without one).
        # Per-agent credentials can still be added with register_agent().
        if scheme != AuthScheme.NONE and self.security.auth_secret:
            cred_kwargs: dict[str, Any] = {"scheme": scheme}
            if scheme == AuthScheme.JWT:
                cred_kwargs["jwt_secret"] = self.security.auth_secret
            elif scheme == AuthScheme.API_KEY:
                cred_kwargs["api_key"] = self.security.auth_secret
            manager.set_default(AgentCredentialConfig(**cred_kwargs))

        return manager

    def build_security(self, server_url: str) -> Any:
        """
        Build the SecurityMiddleware described by [security].

        Args:
            server_url: This agent's own URL — the target of trust rules.

        trust_mode:
            off     No trust checks.
            strict  Only [network].agents may call this agent.
            warn    Same rules, but violations are logged and allowed. Use it
                    to find out what strict would break before enforcing it.
        """
        from nexus_a2a.security.middleware import SecurityMiddleware
        from nexus_a2a.security.rate_limiter import RateLimitConfig, RateLimiter
        from nexus_a2a.security.trust import TrustBoundary
        from nexus_a2a.security.validator import PayloadValidator, ValidatorConfig

        auth = None
        if self.security.auth_scheme != "none":
            auth = self.build_auth_manager()

        trust = None
        if self.security.trust_mode != "off":
            trust = TrustBoundary()
            for caller in self.network.agents:
                trust.allow(caller, server_url)
            if not self.network.agents and self.security.trust_mode == "strict":
                logger.warning(
                    "trust_mode='strict' with an empty [network].agents list "
                    "refuses every caller."
                )

        rate_limiter: Any = None
        if self.security.rate_limit > 0:
            rate_cfg = RateLimitConfig(
                rate=self.security.rate_limit, burst=self.security.rate_burst
            )
            if self.storage.backend == "redis":
                from nexus_a2a.security.redis_rate_limiter import RedisRateLimiter

                # Shared across replicas, so N processes enforce ONE limit.
                rate_limiter = RedisRateLimiter(
                    url=self.storage.url, default_config=rate_cfg
                )
            else:
                rate_limiter = RateLimiter(rate_cfg)

        validator = None
        if self.security.max_payload_bytes > 0:
            validator = PayloadValidator(
                ValidatorConfig(max_bytes=self.security.max_payload_bytes)
            )

        return SecurityMiddleware(
            auth=auth,
            trust=trust,
            rate_limiter=rate_limiter,
            validator=validator,
            server_url=server_url if trust is not None else None,
            trust_warn_only=self.security.trust_mode == "warn",
        )

    def build_runtime(
        self,
        agent: Any,
        host: str | None = None,
        port: int | None = None,
    ) -> Any:
        """
        Build everything needed to serve `agent` as nexus.toml describes.

        Args:
            agent: An @agent-decorated class or instance.
            host:  Bind host. Default: the host in [agent].url.
            port:  Bind port. Default: the port in [agent].url.

        Returns:
            An AgentRuntime. Nothing is connected or started until
            `await runtime.start()` (or `async with runtime:`).
        """
        from urllib.parse import urlparse

        from nexus_a2a.core.a2a_server import A2AServer
        from nexus_a2a.core.agent_server import AgentServer
        from nexus_a2a.core.task_manager import TaskManager
        from nexus_a2a.network import AgentNetwork
        from nexus_a2a.runtime import AgentRuntime
        from nexus_a2a.transport.webhook import WebhookConfig

        parsed = urlparse(self.agent.url)
        bind_host = host or parsed.hostname or "127.0.0.1"
        bind_port = port if port is not None else (parsed.port or 8000)
        server_url = self.agent.url.rstrip("/")

        resources: list[Any] = []

        store = self.build_task_store()
        if hasattr(store, "connect"):
            resources.append(store)

        manager = TaskManager(
            store=store, timeout_sec=self.reliability.task_timeout_sec
        )

        security = self.build_security(server_url)
        if hasattr(security.rate_limiter, "connect"):
            resources.append(security.rate_limiter)

        push_store = None
        dlq_store = None
        if self.storage.backend == "redis":
            from nexus_a2a.storage.dlq_store import RedisDLQStore
            from nexus_a2a.storage.push_store import RedisPushStore

            push_store = RedisPushStore(url=self.storage.url)
            resources.append(push_store)
            if self.ops.port:
                dlq_store = RedisDLQStore(url=self.storage.url)
                resources.append(dlq_store)

        server = A2AServer(
            agent,
            host=bind_host,
            port=bind_port,
            task_manager=manager,
            security=security,
            public_url=server_url,
            push_config=WebhookConfig(
                signing_secret=self.push.signing_secret or None,
                allow_private_urls=self.push.allow_private_urls,
                max_retries=self.push.max_retries,
            ),
            push_store=push_store,
            allow_insecure=self.security.allow_insecure,
        )

        ops = None
        if self.ops.port:
            network = AgentNetwork(task_manager=manager, dlq_store=dlq_store)
            ops = AgentServer(
                network=network,
                host=self.ops.host or bind_host,
                port=self.ops.port,
                admin_token=self.ops.admin_token or None,
            )

        return AgentRuntime(
            server=server,
            ops=ops,
            resources=resources,
            watchdog=manager,
        )

    def configure_logging(self) -> None:
        """Apply [observability].log_level to the root nexus_a2a logger."""
        level = getattr(logging, self.observability.log_level, logging.INFO)
        logging.getLogger("nexus_a2a").setLevel(level)
        logger.debug("Logging level set to %s", self.observability.log_level)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialise config back to a dict (useful for logging/debugging).
        Secrets are redacted for safety.
        """
        return {
            "agent": {
                "name": self.agent.name,
                "description": self.agent.description,
                "version": self.agent.version,
                "url": self.agent.url,
                "streaming": self.agent.streaming,
                "skills": [
                    {
                        "id": s.id,
                        "name": s.name,
                        "description": s.description,
                        "tags": s.tags,
                    }
                    for s in self.agent.skills
                ],
            },
            "network": {
                "agents": self.network.agents,
            },
            "reliability": {
                "task_timeout_sec": self.reliability.task_timeout_sec,
                "max_retries": self.reliability.max_retries,
                "retry_on": self.reliability.retry_on,
                "circuit_breaker_threshold": self.reliability.circuit_breaker_threshold,
                "circuit_recovery_sec": self.reliability.circuit_recovery_sec,
            },
            "security": {
                "auth_scheme": self.security.auth_scheme,
                "auth_secret": "***REDACTED***" if self.security.auth_secret else "",
                "trust_mode": self.security.trust_mode,
                "rate_limit": self.security.rate_limit,
                "rate_burst": self.security.rate_burst,
                "max_payload_bytes": self.security.max_payload_bytes,
                "allow_insecure": self.security.allow_insecure,
            },
            "storage": {
                "backend": self.storage.backend,
                "url": self.storage.url,
                "ttl_sec": self.storage.ttl_sec,
                "task_retention_sec": self.storage.task_retention_sec,
                "max_tasks": self.storage.max_tasks,
            },
            "push": {
                "signing_secret": "***REDACTED***" if self.push.signing_secret else "",
                "allow_private_urls": self.push.allow_private_urls,
                "max_retries": self.push.max_retries,
            },
            "ops": {
                "port": self.ops.port,
                "host": self.ops.host,
                "url": self.ops.url,
                "admin_token": "***REDACTED***" if self.ops.admin_token else "",
            },
            "observability": {
                "tracing": self.observability.tracing,
                "metrics": self.observability.metrics,
                "log_level": self.observability.log_level,
            },
        }
