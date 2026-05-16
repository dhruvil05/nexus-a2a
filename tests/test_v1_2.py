"""
tests/test_v1_2.py

Tests for v1.2.0 — config.py (NexusConfig) and AgentNetwork.from_config().

Coverage:
  - NexusConfig.from_dict(): all sections, defaults, invalid values
  - NexusConfig.from_file(): file loading, missing file error
  - Environment variable overrides (NEXUS_* vars)
  - Cross-field validation (auth secret required, storage URL required, etc.)
  - build_task_store(): memory, redis (import error), postgres (import error)
  - build_retry_config() and build_circuit_breaker()
  - configure_logging()
  - to_dict() secret redaction
  - AgentNetwork.from_config() happy path
"""

from __future__ import annotations

import logging
import os
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def minimal_raw() -> dict:
    """Minimal valid raw config dict (only required fields)."""
    return {"agent": {"name": "TestAgent"}}


def full_raw() -> dict:
    """Full raw config dict with all sections populated."""
    return {
        "agent": {
            "name":        "ResearchAgent",
            "description": "Searches the web.",
            "version":     "2.0.0",
            "url":         "http://localhost:8001",
            "streaming":   False,
            "skills": [
                {
                    "id":          "web_search",
                    "name":        "Web Search",
                    "description": "Searches the web.",
                    "tags":        ["search"],
                    "examples":    ["Search for AI news"],
                }
            ],
        },
        "network": {
            "agents": ["http://agent-a:8001", "http://agent-b:8002"],
        },
        "reliability": {
            "task_timeout_sec":          60.0,
            "max_retries":               5,
            "retry_on":                  [500, 503],
            "circuit_breaker_threshold": 3,
            "circuit_recovery_sec":      15.0,
            "base_delay_sec":            0.5,
            "max_delay_sec":             20.0,
        },
        "security": {
            "auth_scheme": "jwt",
            "auth_secret": "super-secret-key",
            "trust_mode":  "strict",
        },
        "storage": {
            "backend": "memory",
            "url":     "",
            "ttl_sec": 7200,
        },
        "observability": {
            "tracing":   True,
            "metrics":   False,
            "log_level": "DEBUG",
        },
    }


# ── NexusConfig.from_dict() — happy paths ─────────────────────────────────────

class TestNexusConfigFromDict:

    def test_minimal_config_uses_defaults(self):
        cfg = NexusConfig.from_dict(minimal_raw())

        assert cfg.agent.name == "TestAgent"
        assert cfg.agent.url == "http://localhost:8000"
        assert cfg.agent.version == "1.0.0"
        assert cfg.agent.streaming is False
        assert cfg.agent.skills == []

        assert cfg.network.agents == []
        assert cfg.reliability.task_timeout_sec == 120.0
        assert cfg.reliability.max_retries == 3
        assert cfg.security.auth_scheme == "none"
        assert cfg.storage.backend == "memory"
        assert cfg.observability.log_level == "INFO"

    def test_full_config_parsed_correctly(self):
        cfg = NexusConfig.from_dict(full_raw())

        assert cfg.agent.name == "ResearchAgent"
        assert cfg.agent.description == "Searches the web."
        assert cfg.agent.version == "2.0.0"
        assert cfg.agent.url == "http://localhost:8001"
        assert len(cfg.agent.skills) == 1
        assert cfg.agent.skills[0].id == "web_search"
        assert cfg.agent.skills[0].name == "Web Search"
        assert cfg.agent.skills[0].tags == ["search"]

        assert cfg.network.agents == ["http://agent-a:8001", "http://agent-b:8002"]
        assert cfg.reliability.task_timeout_sec == 60.0
        assert cfg.reliability.max_retries == 5
        assert cfg.reliability.retry_on == [500, 503]
        assert cfg.reliability.circuit_breaker_threshold == 3

        assert cfg.security.auth_scheme == "jwt"
        assert cfg.security.auth_secret == "super-secret-key"
        assert cfg.security.trust_mode == "strict"

        assert cfg.storage.backend == "memory"
        assert cfg.storage.ttl_sec == 7200

        assert cfg.observability.tracing is True
        assert cfg.observability.metrics is False
        assert cfg.observability.log_level == "DEBUG"

    def test_auth_scheme_case_insensitive(self):
        raw = minimal_raw()
        raw["security"] = {"auth_scheme": "JWT", "auth_secret": "mysecret"}
        cfg = NexusConfig.from_dict(raw)
        assert cfg.security.auth_scheme == "jwt"

    def test_log_level_case_insensitive(self):
        raw = minimal_raw()
        raw["observability"] = {"log_level": "warning"}
        cfg = NexusConfig.from_dict(raw)
        assert cfg.observability.log_level == "WARNING"

    def test_skill_without_optional_fields(self):
        raw = minimal_raw()
        raw["agent"]["skills"] = [{"id": "my_skill", "name": "My Skill"}]
        cfg = NexusConfig.from_dict(raw)
        skill = cfg.agent.skills[0]
        assert skill.id == "my_skill"
        assert skill.description == ""
        assert skill.tags == []
        assert skill.examples == []

    def test_multiple_skills(self):
        raw = minimal_raw()
        raw["agent"]["skills"] = [
            {"id": "skill_a", "name": "Skill A"},
            {"id": "skill_b", "name": "Skill B"},
        ]
        cfg = NexusConfig.from_dict(raw)
        assert len(cfg.agent.skills) == 2
        assert cfg.agent.skills[1].id == "skill_b"

    def test_redis_storage_with_url(self):
        raw = minimal_raw()
        raw["storage"] = {"backend": "redis", "url": "redis://localhost:6379"}
        cfg = NexusConfig.from_dict(raw)
        assert cfg.storage.backend == "redis"
        assert cfg.storage.url == "redis://localhost:6379"

    def test_postgres_storage_with_url(self):
        raw = minimal_raw()
        raw["storage"] = {
            "backend": "postgres",
            "url": "postgresql://user:pass@localhost/db",
        }
        cfg = NexusConfig.from_dict(raw)
        assert cfg.storage.backend == "postgres"

    def test_api_key_auth_with_secret(self):
        raw = minimal_raw()
        raw["security"] = {"auth_scheme": "api_key", "auth_secret": "my-key"}
        cfg = NexusConfig.from_dict(raw)
        assert cfg.security.auth_scheme == "api_key"


# ── NexusConfig.from_dict() — validation errors ───────────────────────────────

class TestNexusConfigValidationErrors:

    def test_empty_agent_name_raises(self):
        raw = {"agent": {"name": ""}}
        with pytest.raises(ConfigError, match="name"):
            NexusConfig.from_dict(raw)

    def test_whitespace_agent_name_raises(self):
        raw = {"agent": {"name": "   "}}
        with pytest.raises(ConfigError, match="name"):
            NexusConfig.from_dict(raw)

    def test_missing_agent_section_raises(self):
        # No [agent] section at all — agent.name defaults to "" → ConfigError
        with pytest.raises(ConfigError, match="name"):
            NexusConfig.from_dict({})

    def test_invalid_agent_url_raises(self):
        raw = {"agent": {"name": "Agent", "url": "ftp://wrong"}}
        with pytest.raises(ConfigError, match="url"):
            NexusConfig.from_dict(raw)

    def test_jwt_without_secret_raises(self):
        raw = minimal_raw()
        raw["security"] = {"auth_scheme": "jwt", "auth_secret": ""}
        with pytest.raises(ConfigError, match="secret"):
            NexusConfig.from_dict(raw)

    def test_api_key_without_secret_raises(self):
        raw = minimal_raw()
        raw["security"] = {"auth_scheme": "api_key"}
        with pytest.raises(ConfigError, match="secret"):
            NexusConfig.from_dict(raw)

    def test_redis_without_url_raises(self):
        raw = minimal_raw()
        raw["storage"] = {"backend": "redis", "url": ""}
        with pytest.raises(ConfigError, match="url"):
            NexusConfig.from_dict(raw)

    def test_postgres_without_url_raises(self):
        raw = minimal_raw()
        raw["storage"] = {"backend": "postgres"}
        with pytest.raises(ConfigError, match="url"):
            NexusConfig.from_dict(raw)

    def test_invalid_auth_scheme_raises(self):
        raw = minimal_raw()
        raw["security"] = {"auth_scheme": "oauth3"}
        with pytest.raises(ConfigError, match="auth_scheme"):
            NexusConfig.from_dict(raw)

    def test_invalid_trust_mode_raises(self):
        raw = minimal_raw()
        raw["security"] = {"trust_mode": "medium"}
        with pytest.raises(ConfigError, match="trust_mode"):
            NexusConfig.from_dict(raw)

    def test_invalid_storage_backend_raises(self):
        raw = minimal_raw()
        raw["storage"] = {"backend": "sqlite"}
        with pytest.raises(ConfigError, match="backend"):
            NexusConfig.from_dict(raw)

    def test_invalid_log_level_raises(self):
        raw = minimal_raw()
        raw["observability"] = {"log_level": "VERBOSE"}
        with pytest.raises(ConfigError, match="log_level"):
            NexusConfig.from_dict(raw)

    def test_skill_missing_id_raises(self):
        raw = minimal_raw()
        raw["agent"]["skills"] = [{"name": "No ID Skill"}]
        with pytest.raises(ConfigError, match="id"):
            NexusConfig.from_dict(raw)

    def test_skill_missing_name_raises(self):
        raw = minimal_raw()
        raw["agent"]["skills"] = [{"id": "no_name"}]
        with pytest.raises(ConfigError, match="name"):
            NexusConfig.from_dict(raw)

    def test_network_agents_not_list_raises(self):
        raw = minimal_raw()
        raw["network"] = {"agents": "http://agent:8001"}
        with pytest.raises(ConfigError, match="agents"):
            NexusConfig.from_dict(raw)

    def test_network_agent_not_string_raises(self):
        raw = minimal_raw()
        raw["network"] = {"agents": [123]}
        with pytest.raises(ConfigError):
            NexusConfig.from_dict(raw)

    def test_zero_timeout_raises(self):
        raw = minimal_raw()
        raw["reliability"] = {"task_timeout_sec": 0}
        with pytest.raises(ConfigError, match="timeout"):
            NexusConfig.from_dict(raw)

    def test_negative_timeout_raises(self):
        raw = minimal_raw()
        raw["reliability"] = {"task_timeout_sec": -5}
        with pytest.raises(ConfigError, match="timeout"):
            NexusConfig.from_dict(raw)

    def test_negative_max_retries_raises(self):
        raw = minimal_raw()
        raw["reliability"] = {"max_retries": -1}
        with pytest.raises(ConfigError, match="retries"):
            NexusConfig.from_dict(raw)

    def test_zero_circuit_threshold_raises(self):
        raw = minimal_raw()
        raw["reliability"] = {"circuit_breaker_threshold": 0}
        with pytest.raises(ConfigError, match="threshold"):
            NexusConfig.from_dict(raw)


# ── NexusConfig.from_file() ───────────────────────────────────────────────────

class TestNexusConfigFromFile:

    def test_loads_valid_toml_file(self, tmp_path: Path):
        toml = textwrap.dedent("""\
            [agent]
            name = "FileAgent"
            url  = "http://localhost:9000"

            [reliability]
            task_timeout_sec = 30
        """)
        p = tmp_path / "nexus.toml"
        p.write_text(toml)

        cfg = NexusConfig.from_file(str(p))
        assert cfg.agent.name == "FileAgent"
        assert cfg.reliability.task_timeout_sec == 30.0

    def test_missing_file_raises_config_error(self, tmp_path: Path):
        with pytest.raises(ConfigError, match="not found"):
            NexusConfig.from_file(str(tmp_path / "does_not_exist.toml"))

    def test_invalid_toml_syntax_raises_config_error(self, tmp_path: Path):
        p = tmp_path / "bad.toml"
        p.write_text("this is not valid toml ===")
        with pytest.raises(ConfigError, match="TOML"):
            NexusConfig.from_file(str(p))

    def test_full_toml_file(self, tmp_path: Path):
        toml = textwrap.dedent("""\
            [agent]
            name        = "ResearchAgent"
            description = "Does research"
            version     = "1.5.0"
            url         = "http://localhost:8001"
            streaming   = false

            [[agent.skills]]
            id          = "search"
            name        = "Web Search"
            description = "Searches the web"
            tags        = ["search", "web"]

            [network]
            agents = ["http://summary:8002"]

            [reliability]
            task_timeout_sec          = 90
            max_retries               = 2
            circuit_breaker_threshold = 4

            [security]
            auth_scheme = "jwt"
            auth_secret = "topsecret"
            trust_mode  = "warn"

            [storage]
            backend = "memory"

            [observability]
            tracing   = true
            log_level = "WARNING"
        """)
        p = tmp_path / "nexus.toml"
        p.write_text(toml)

        cfg = NexusConfig.from_file(str(p))
        assert cfg.agent.name == "ResearchAgent"
        assert cfg.agent.skills[0].id == "search"
        assert cfg.network.agents == ["http://summary:8002"]
        assert cfg.reliability.task_timeout_sec == 90.0
        assert cfg.security.auth_scheme == "jwt"
        assert cfg.security.auth_secret == "topsecret"
        assert cfg.observability.log_level == "WARNING"


# ── Environment variable overrides ────────────────────────────────────────────

class TestEnvOverrides:

    def test_nexus_agent_name_overrides_toml(self):
        with patch.dict(os.environ, {"NEXUS_AGENT_NAME": "EnvAgent"}):
            cfg = NexusConfig.from_dict(minimal_raw())
        assert cfg.agent.name == "EnvAgent"

    def test_nexus_agent_url_overrides_toml(self):
        with patch.dict(os.environ, {"NEXUS_AGENT_URL": "http://env-host:9999"}):
            cfg = NexusConfig.from_dict(minimal_raw())
        assert cfg.agent.url == "http://env-host:9999"

    def test_nexus_auth_scheme_overrides_toml(self):
        with patch.dict(os.environ, {
            "NEXUS_AUTH_SCHEME": "jwt",
            "NEXUS_AUTH_SECRET": "env-secret",
        }):
            cfg = NexusConfig.from_dict(minimal_raw())
        assert cfg.security.auth_scheme == "jwt"
        assert cfg.security.auth_secret == "env-secret"

    def test_nexus_storage_backend_overrides_toml(self):
        with patch.dict(os.environ, {
            "NEXUS_STORAGE_BACKEND": "redis",
            "NEXUS_STORAGE_URL": "redis://env:6379",
        }):
            cfg = NexusConfig.from_dict(minimal_raw())
        assert cfg.storage.backend == "redis"
        assert cfg.storage.url == "redis://env:6379"

    def test_nexus_task_timeout_overrides_toml(self):
        with patch.dict(os.environ, {"NEXUS_TASK_TIMEOUT": "45.5"}):
            cfg = NexusConfig.from_dict(minimal_raw())
        assert cfg.reliability.task_timeout_sec == 45.5

    def test_nexus_log_level_overrides_toml(self):
        with patch.dict(os.environ, {"NEXUS_LOG_LEVEL": "ERROR"}):
            cfg = NexusConfig.from_dict(minimal_raw())
        assert cfg.observability.log_level == "ERROR"

    def test_invalid_nexus_auth_scheme_raises(self):
        with patch.dict(os.environ, {"NEXUS_AUTH_SCHEME": "oauth2"}):
            with pytest.raises(ConfigError, match="NEXUS_AUTH_SCHEME"):
                NexusConfig.from_dict(minimal_raw())

    def test_invalid_nexus_storage_backend_raises(self):
        with patch.dict(os.environ, {"NEXUS_STORAGE_BACKEND": "sqlite"}):
            with pytest.raises(ConfigError, match="NEXUS_STORAGE_BACKEND"):
                NexusConfig.from_dict(minimal_raw())

    def test_invalid_nexus_task_timeout_raises(self):
        with patch.dict(os.environ, {"NEXUS_TASK_TIMEOUT": "not-a-number"}):
            with pytest.raises(ConfigError, match="NEXUS_TASK_TIMEOUT"):
                NexusConfig.from_dict(minimal_raw())

    def test_invalid_nexus_log_level_raises(self):
        with patch.dict(os.environ, {"NEXUS_LOG_LEVEL": "VERBOSE"}):
            with pytest.raises(ConfigError, match="NEXUS_LOG_LEVEL"):
                NexusConfig.from_dict(minimal_raw())

    def test_env_wins_over_toml(self):
        """Env var always takes precedence over TOML value."""
        raw = minimal_raw()
        raw["agent"]["name"] = "TomlAgent"
        with patch.dict(os.environ, {"NEXUS_AGENT_NAME": "EnvAgent"}):
            cfg = NexusConfig.from_dict(raw)
        assert cfg.agent.name == "EnvAgent"

    def test_no_env_vars_uses_toml_values(self):
        """With no env vars set, TOML values are used."""
        raw = minimal_raw()
        raw["agent"]["name"] = "TomlOnly"
        # Ensure NEXUS_AGENT_NAME is not in env
        with patch.dict(os.environ, {}, clear=False):
            env = {k: v for k, v in os.environ.items() if not k.startswith("NEXUS_")}
            with patch.dict(os.environ, env, clear=True):
                cfg = NexusConfig.from_dict(raw)
        assert cfg.agent.name == "TomlOnly"


# ── build_task_store() ────────────────────────────────────────────────────────

class TestBuildTaskStore:

    def test_memory_backend_returns_in_memory_store(self):
        cfg = NexusConfig.from_dict(minimal_raw())
        from nexus_a2a.storage.task_store import InMemoryTaskStore
        store = cfg.build_task_store()
        assert isinstance(store, InMemoryTaskStore)

    def test_redis_backend_raises_without_package(self):
        raw = minimal_raw()
        raw["storage"] = {"backend": "redis", "url": "redis://localhost:6379"}
        cfg = NexusConfig.from_dict(raw)

        # Simulate redis package not installed
        with patch.dict("sys.modules", {"nexus_a2a.storage.redis_store": None}):
            with pytest.raises((ConfigError, ImportError)):
                cfg.build_task_store()

    def test_redis_backend_returns_redis_store_when_available(self):
        raw = minimal_raw()
        raw["storage"] = {"backend": "redis", "url": "redis://localhost:6379"}
        cfg = NexusConfig.from_dict(raw)

        try:
            from nexus_a2a.storage.redis_store import RedisTaskStore
            store = cfg.build_task_store()
            assert isinstance(store, RedisTaskStore)
        except ImportError:
            pytest.skip("redis package not installed")


# ── build_retry_config() ──────────────────────────────────────────────────────

class TestBuildRetryConfig:

    def test_default_retry_config(self):
        cfg = NexusConfig.from_dict(minimal_raw())
        retry = cfg.build_retry_config()
        assert retry.max_retries == 3
        assert 500 in retry.retry_on
        assert 503 in retry.retry_on

    def test_custom_retry_config(self):
        raw = minimal_raw()
        raw["reliability"] = {
            "max_retries": 5,
            "retry_on": [500, 502],
            "base_delay_sec": 2.0,
            "max_delay_sec": 60.0,
        }
        cfg = NexusConfig.from_dict(raw)
        retry = cfg.build_retry_config()
        assert retry.max_retries == 5
        assert retry.retry_on == {500, 502}
        assert retry.base_delay == 2.0
        assert retry.max_delay == 60.0


# ── build_circuit_breaker() ───────────────────────────────────────────────────

class TestBuildCircuitBreaker:

    def test_default_circuit_breaker(self):
        cfg = NexusConfig.from_dict(minimal_raw())
        cb = cfg.build_circuit_breaker()
        assert cb._fail_thresh == 5
        assert cb._recovery == 30.0

    def test_custom_circuit_breaker(self):
        raw = minimal_raw()
        raw["reliability"] = {
            "circuit_breaker_threshold": 10,
            "circuit_recovery_sec": 60.0,
        }
        cfg = NexusConfig.from_dict(raw)
        cb = cfg.build_circuit_breaker()
        assert cb._fail_thresh == 10
        assert cb._recovery == 60.0


# ── configure_logging() ───────────────────────────────────────────────────────

class TestConfigureLogging:

    def test_sets_debug_level(self):
        raw = minimal_raw()
        raw["observability"] = {"log_level": "DEBUG"}
        cfg = NexusConfig.from_dict(raw)
        cfg.configure_logging()
        assert logging.getLogger("nexus_a2a").level == logging.DEBUG

    def test_sets_warning_level(self):
        raw = minimal_raw()
        raw["observability"] = {"log_level": "WARNING"}
        cfg = NexusConfig.from_dict(raw)
        cfg.configure_logging()
        assert logging.getLogger("nexus_a2a").level == logging.WARNING

    def test_default_info_level(self):
        cfg = NexusConfig.from_dict(minimal_raw())
        cfg.configure_logging()
        assert logging.getLogger("nexus_a2a").level == logging.INFO


# ── to_dict() secret redaction ────────────────────────────────────────────────

class TestToDict:

    def test_secret_is_redacted(self):
        raw = minimal_raw()
        raw["security"] = {"auth_scheme": "jwt", "auth_secret": "my-real-secret"}
        cfg = NexusConfig.from_dict(raw)
        d = cfg.to_dict()
        assert d["security"]["auth_secret"] == "***REDACTED***"
        assert "my-real-secret" not in str(d)

    def test_empty_secret_not_redacted(self):
        cfg = NexusConfig.from_dict(minimal_raw())
        d = cfg.to_dict()
        assert d["security"]["auth_secret"] == ""

    def test_skills_serialised(self):
        raw = minimal_raw()
        raw["agent"]["skills"] = [{"id": "s1", "name": "Skill One"}]
        cfg = NexusConfig.from_dict(raw)
        d = cfg.to_dict()
        assert d["agent"]["skills"][0]["id"] == "s1"

    def test_all_top_level_keys_present(self):
        cfg = NexusConfig.from_dict(minimal_raw())
        d = cfg.to_dict()
        expected_keys = {"agent", "network", "reliability", "security", "storage", "observability"}
        assert set(d.keys()) == expected_keys


# ── AgentNetwork.from_config() ────────────────────────────────────────────────

class TestAgentNetworkFromConfig:

    def test_from_config_returns_network(self, tmp_path: Path):
        toml = textwrap.dedent("""\
            [agent]
            name = "MyAgent"
            url  = "http://localhost:8001"
        """)
        p = tmp_path / "nexus.toml"
        p.write_text(toml)

        from nexus_a2a.network import AgentNetwork
        network = AgentNetwork.from_config(str(p))
        assert isinstance(network, AgentNetwork)

    def test_from_config_wires_task_manager(self, tmp_path: Path):
        toml = textwrap.dedent("""\
            [agent]
            name = "MyAgent"

            [reliability]
            task_timeout_sec = 45
        """)
        p = tmp_path / "nexus.toml"
        p.write_text(toml)

        from nexus_a2a.network import AgentNetwork
        network = AgentNetwork.from_config(str(p))
        assert network.task_manager is not None
        assert network.task_manager._timeout_sec == 45.0

    def test_from_config_stores_config_reference(self, tmp_path: Path):
        toml = textwrap.dedent("""\
            [agent]
            name = "ConfigAgent"
            url  = "http://localhost:8002"
        """)
        p = tmp_path / "nexus.toml"
        p.write_text(toml)

        from nexus_a2a.network import AgentNetwork
        network = AgentNetwork.from_config(str(p))
        assert hasattr(network, "_config")
        assert network._config.agent.name == "ConfigAgent"

    def test_from_config_raises_on_invalid_toml(self, tmp_path: Path):
        p = tmp_path / "bad.toml"
        p.write_text("not = valid = toml")
        from nexus_a2a.network import AgentNetwork
        with pytest.raises(ConfigError):
            AgentNetwork.from_config(str(p))

    def test_from_config_raises_on_missing_file(self, tmp_path: Path):
        from nexus_a2a.network import AgentNetwork
        with pytest.raises(ConfigError, match="not found"):
            AgentNetwork.from_config(str(tmp_path / "ghost.toml"))

    def test_from_config_with_memory_storage(self, tmp_path: Path):
        toml = textwrap.dedent("""\
            [agent]
            name = "MemAgent"

            [storage]
            backend = "memory"
        """)
        p = tmp_path / "nexus.toml"
        p.write_text(toml)

        from nexus_a2a.network import AgentNetwork
        from nexus_a2a.storage.task_store import InMemoryTaskStore
        network = AgentNetwork.from_config(str(p))
        assert isinstance(network.task_manager._store, InMemoryTaskStore)