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

All NEXUS_* env vars are read after the TOML file is parsed,
so they always take precedence — ideal for container deployments.

Design:
    Uses tomllib (stdlib in Python 3.11+). No extra deps needed.
    Validates at parse time — fails fast with a clear ConfigError.
    Returns typed dataclasses, not raw dicts — type checker friendly.
"""

from __future__ import annotations

import logging
import os
import tomllib
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

    name: str
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


@dataclass
class StorageConfig:
    """[storage] section."""

    backend: str = "memory"  # "memory" | "redis" | "postgres"
    url: str = ""  # redis:// or postgres:// connection URL
    ttl_sec: int = 3600  # TTL for Redis keys (ignored for memory/postgres)


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
        return cls(
            agent=cls._parse_agent(raw.get("agent", {})),
            network=cls._parse_network(raw.get("network", {})),
            reliability=cls._parse_reliability(raw.get("reliability", {})),
            security=cls._parse_security(raw.get("security", {})),
            storage=cls._parse_storage(raw.get("storage", {})),
            observability=cls._parse_observability(raw.get("observability", {})),
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
            except ValueError:
                raise ConfigError(
                    f"NEXUS_TASK_TIMEOUT='{timeout}' is not a valid number",
                    "reliability.task_timeout_sec",
                )

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

            return InMemoryTaskStore()

        if backend == "redis":
            try:
                from nexus_a2a.storage.redis_store import RedisTaskStore
            except ImportError:
                raise ConfigError(
                    "RedisTaskStore requires the 'redis' extra. "
                    "Install with: pip install nexus-a2a[redis]",
                    "storage.backend",
                )
            return RedisTaskStore(url=self.storage.url, ttl=self.storage.ttl_sec)

        if backend == "postgres":
            try:
                from nexus_a2a.storage.postgres_store import (
                    PostgresTaskStore,  # type: ignore[import]
                )
            except ImportError:
                raise ConfigError(
                    "PostgresTaskStore requires the 'postgres' extra. "
                    "Install with: pip install nexus-a2a[postgres]",
                    "storage.backend",
                )
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

        # For non-NONE schemes, register a wildcard default credential.
        # Developers can override per-agent creds by calling register_agent().
        if scheme != AuthScheme.NONE and self.security.auth_secret:
            cred_kwargs: dict[str, Any] = {"scheme": scheme}
            if scheme == AuthScheme.JWT:
                cred_kwargs["jwt_secret"] = self.security.auth_secret
            elif scheme == AuthScheme.API_KEY:
                cred_kwargs["api_key"] = self.security.auth_secret
            cred = AgentCredentialConfig(**cred_kwargs)
            manager.register_agent("*", cred)

        return manager

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
            },
            "storage": {
                "backend": self.storage.backend,
                "url": self.storage.url,
                "ttl_sec": self.storage.ttl_sec,
            },
            "observability": {
                "tracing": self.observability.tracing,
                "metrics": self.observability.metrics,
                "log_level": self.observability.log_level,
            },
        }
