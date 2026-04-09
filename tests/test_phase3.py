"""
tests/test_phase3.py

Tests for Phase 3: AuthManager, TrustBoundary, RateLimiter, PayloadValidator.
Run with:  uv run pytest tests/test_phase3.py -v
"""

from __future__ import annotations

import time
import pytest

from nexus_a2a.models.agent import AuthScheme
from nexus_a2a.models.task import Message, Part, PartType
from nexus_a2a.security.auth import (
    AgentCredentialConfig,
    AuthManager,
    AuthError,
    ExpiredCredentialsError,
    InvalidCredentialsError,
    MissingCredentialsError,
)
from nexus_a2a.security.trust import (
    AgentNotAllowedError,
    SkillNotAllowedError,
    TrustBoundary,
)
from nexus_a2a.security.rate_limiter import (
    RateLimitConfig,
    RateLimitError,
    RateLimiter,
)
from nexus_a2a.security.validator import (
    BlankTextPartError,
    InvalidPartError,
    PayloadTooLargeError,
    PayloadValidator,
    TooManyPartsError,
    ValidatorConfig,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

AGENT_A = "http://agent-a:8001"
AGENT_B = "http://agent-b:8002"
AGENT_C = "http://agent-c:8003"
SECRET   = "test-secret-key-32-chars-minimum!"


# ══════════════════════════════════════════════════════════════════════════════
# AuthManager
# ══════════════════════════════════════════════════════════════════════════════

class TestAuthManagerNone:
    """No auth scheme — all requests pass."""

    async def test_none_scheme_always_passes(self):
        auth = AuthManager()
        auth.register_agent(AGENT_A, AgentCredentialConfig(scheme=AuthScheme.NONE))
        claims = await auth.verify(AGENT_A, {})
        assert claims["scheme"] == "none"

    async def test_unregistered_agent_defaults_to_none(self):
        auth = AuthManager()
        claims = await auth.verify("http://unknown:9999", {})
        assert claims["scheme"] == "none"


class TestAuthManagerApiKey:
    def _auth(self) -> AuthManager:
        auth = AuthManager()
        auth.register_agent(
            AGENT_A,
            AgentCredentialConfig(scheme=AuthScheme.API_KEY, api_key="valid-key"),
        )
        return auth

    async def test_valid_api_key_passes(self):
        auth = self._auth()
        claims = await auth.verify(AGENT_A, {"X-API-Key": "valid-key"})
        assert claims["scheme"] == "api_key"

    async def test_wrong_api_key_raises(self):
        auth = self._auth()
        with pytest.raises(InvalidCredentialsError):
            await auth.verify(AGENT_A, {"X-API-Key": "wrong-key"})

    async def test_missing_api_key_raises(self):
        auth = self._auth()
        with pytest.raises(MissingCredentialsError):
            await auth.verify(AGENT_A, {})

    async def test_custom_header_name(self):
        auth = AuthManager()
        auth.register_agent(
            AGENT_B,
            AgentCredentialConfig(
                scheme=AuthScheme.API_KEY,
                api_key="secret",
                header_name="X-Custom-Key",
            ),
        )
        claims = await auth.verify(AGENT_B, {"X-Custom-Key": "secret"})
        assert claims["scheme"] == "api_key"

    def test_api_key_config_without_key_raises(self):
        auth = AuthManager()
        with pytest.raises(ValueError, match="api_key"):
            auth.register_agent(
                AGENT_A,
                AgentCredentialConfig(scheme=AuthScheme.API_KEY),  # no api_key
            )


class TestAuthManagerJWT:
    def _auth(self) -> AuthManager:
        auth = AuthManager()
        auth.register_agent(
            AGENT_A,
            AgentCredentialConfig(scheme=AuthScheme.JWT, jwt_secret=SECRET),
        )
        return auth

    def _bearer(self, auth: AuthManager, agent_url: str = AGENT_A) -> dict[str, str]:
        token = auth.issue_jwt(agent_url, subject="caller")
        return {"Authorization": f"Bearer {token}"}

    async def test_valid_jwt_passes(self):
        auth = self._auth()
        claims = await auth.verify(AGENT_A, self._bearer(auth))
        assert claims["sub"] == "caller"

    async def test_missing_bearer_raises(self):
        auth = self._auth()
        with pytest.raises(MissingCredentialsError):
            await auth.verify(AGENT_A, {})

    async def test_invalid_token_raises(self):
        auth = self._auth()
        with pytest.raises(InvalidCredentialsError):
            await auth.verify(AGENT_A, {"Authorization": "Bearer totally.fake.token"})

    async def test_expired_token_raises(self):
        auth = self._auth()
        # Issue a token that expired 10 seconds ago
        from jose import jwt as jose_jwt
        payload = {"sub": "caller", "iat": int(time.time()) - 20, "exp": int(time.time()) - 10}
        token = jose_jwt.encode(payload, SECRET, algorithm="HS256")
        with pytest.raises(ExpiredCredentialsError):
            await auth.verify(AGENT_A, {"Authorization": f"Bearer {token}"})

    def test_jwt_config_without_secret_raises(self):
        auth = AuthManager()
        with pytest.raises(ValueError, match="jwt_secret"):
            auth.register_agent(
                AGENT_A,
                AgentCredentialConfig(scheme=AuthScheme.JWT),  # no secret
            )

    def test_issue_jwt_wrong_scheme_raises(self):
        auth = AuthManager()
        auth.register_agent(
            AGENT_A,
            AgentCredentialConfig(scheme=AuthScheme.API_KEY, api_key="k"),
        )
        with pytest.raises(ValueError, match="not 'jwt'"):
            auth.issue_jwt(AGENT_A, subject="x")

    def test_build_auth_headers_jwt(self):
        auth = self._auth()
        headers = auth.build_auth_headers(AGENT_A)
        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Bearer ")

    def test_build_auth_headers_api_key(self):
        auth = AuthManager()
        auth.register_agent(
            AGENT_B,
            AgentCredentialConfig(scheme=AuthScheme.API_KEY, api_key="my-key"),
        )
        headers = auth.build_auth_headers(AGENT_B)
        assert headers["X-API-Key"] == "my-key"

    def test_build_auth_headers_none(self):
        auth = AuthManager()
        auth.register_agent(AGENT_A, AgentCredentialConfig(scheme=AuthScheme.NONE))
        assert auth.build_auth_headers(AGENT_A) == {}


# ══════════════════════════════════════════════════════════════════════════════
# TrustBoundary
# ══════════════════════════════════════════════════════════════════════════════

class TestTrustBoundary:
    def test_default_deny_blocks_unknown(self):
        trust = TrustBoundary(default_allow=False)
        with pytest.raises(AgentNotAllowedError):
            trust.check(AGENT_A, AGENT_B)

    def test_default_allow_permits_unknown(self):
        trust = TrustBoundary(default_allow=True)
        trust.check(AGENT_A, AGENT_B)  # should not raise

    def test_allow_rule_grants_access(self):
        trust = TrustBoundary()
        trust.allow(AGENT_A, AGENT_B)
        trust.check(AGENT_A, AGENT_B)  # should not raise

    def test_allow_wildcard_target(self):
        trust = TrustBoundary()
        trust.allow(AGENT_A, "*")
        trust.check(AGENT_A, AGENT_B)
        trust.check(AGENT_A, AGENT_C)

    def test_block_overrides_allow(self):
        trust = TrustBoundary()
        trust.allow(AGENT_A, AGENT_B)
        trust.block(AGENT_A)
        with pytest.raises(AgentNotAllowedError):
            trust.check(AGENT_A, AGENT_B)

    def test_blocked_target_is_unreachable(self):
        trust = TrustBoundary(default_allow=True)
        trust.block(AGENT_B)
        with pytest.raises(AgentNotAllowedError):
            trust.check(AGENT_A, AGENT_B)

    def test_unblock_restores_access(self):
        trust = TrustBoundary(default_allow=True)
        trust.block(AGENT_A)
        trust.unblock(AGENT_A)
        trust.check(AGENT_A, AGENT_B)  # should not raise

    def test_revoke_removes_rule(self):
        trust = TrustBoundary()
        trust.allow(AGENT_A, AGENT_B)
        trust.revoke(AGENT_A, AGENT_B)
        with pytest.raises(AgentNotAllowedError):
            trust.check(AGENT_A, AGENT_B)

    def test_skill_acl_allows_listed_skill(self):
        trust = TrustBoundary()
        trust.allow(AGENT_A, AGENT_B, skills=["search"])
        trust.check(AGENT_A, AGENT_B, skill_id="search")  # OK

    def test_skill_acl_blocks_unlisted_skill(self):
        trust = TrustBoundary()
        trust.allow(AGENT_A, AGENT_B, skills=["search"])
        with pytest.raises(SkillNotAllowedError):
            trust.check(AGENT_A, AGENT_B, skill_id="summarise")

    def test_no_skill_acl_allows_all_skills(self):
        trust = TrustBoundary()
        trust.allow(AGENT_A, AGENT_B)  # no skills restriction
        trust.check(AGENT_A, AGENT_B, skill_id="any-skill")

    def test_is_allowed_returns_bool(self):
        trust = TrustBoundary()
        trust.allow(AGENT_A, AGENT_B)
        assert trust.is_allowed(AGENT_A, AGENT_B) is True
        assert trust.is_allowed(AGENT_A, AGENT_C) is False

    def test_allowed_targets(self):
        trust = TrustBoundary()
        trust.allow(AGENT_A, AGENT_B)
        trust.allow(AGENT_A, AGENT_C)
        targets = trust.allowed_targets(AGENT_A)
        assert set(targets) == {AGENT_B, AGENT_C}

    def test_summary(self):
        trust = TrustBoundary()
        trust.allow(AGENT_A, AGENT_B)
        trust.block(AGENT_C)
        s = trust.summary()
        assert s["default_allow"] is False
        assert AGENT_C in s["blocked"]


# ══════════════════════════════════════════════════════════════════════════════
# RateLimiter
# ══════════════════════════════════════════════════════════════════════════════

class TestRateLimiter:
    async def test_within_burst_passes(self):
        limiter = RateLimiter(default_config=RateLimitConfig(rate=2, burst=5))
        # 5 requests should all pass (burst=5, bucket starts full)
        for _ in range(5):
            await limiter.check(AGENT_A)

    async def test_exceeding_burst_raises(self):
        limiter = RateLimiter(default_config=RateLimitConfig(rate=1, burst=3))
        for _ in range(3):
            await limiter.check(AGENT_A)
        with pytest.raises(RateLimitError) as exc_info:
            await limiter.check(AGENT_A)
        assert exc_info.value.retry_after > 0

    async def test_custom_limit_per_agent(self):
        limiter = RateLimiter(default_config=RateLimitConfig(rate=100, burst=100))
        # Give AGENT_B a very tight limit
        limiter.set_limit(AGENT_B, RateLimitConfig(rate=1, burst=1))
        await limiter.check(AGENT_B)  # uses 1 token
        with pytest.raises(RateLimitError):
            await limiter.check(AGENT_B)  # bucket empty

    async def test_different_agents_independent(self):
        limiter = RateLimiter(default_config=RateLimitConfig(rate=1, burst=2))
        await limiter.check(AGENT_A)
        await limiter.check(AGENT_A)
        # AGENT_A is now empty but AGENT_B bucket is full
        await limiter.check(AGENT_B)
        await limiter.check(AGENT_B)

    async def test_is_allowed_returns_bool(self):
        limiter = RateLimiter(default_config=RateLimitConfig(rate=1, burst=1))
        assert await limiter.is_allowed(AGENT_A) is True
        assert await limiter.is_allowed(AGENT_A) is False  # bucket empty

    async def test_available_tokens_decreases(self):
        limiter = RateLimiter(default_config=RateLimitConfig(rate=10, burst=10))
        before = await limiter.available_tokens(AGENT_A)
        await limiter.check(AGENT_A)
        after = await limiter.available_tokens(AGENT_A)
        assert after < before

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError, match="rate"):
            RateLimitConfig(rate=0, burst=10)
        with pytest.raises(ValueError, match="burst"):
            RateLimitConfig(rate=1, burst=0)
        # burst < rate is now valid — they are independent concepts
        config = RateLimitConfig(rate=10, burst=5)
        assert config.rate == 10
        assert config.burst == 5

    async def test_remove_limit_falls_back_to_default(self):
        limiter = RateLimiter(default_config=RateLimitConfig(rate=100, burst=100))
        limiter.set_limit(AGENT_A, RateLimitConfig(rate=1, burst=1))
        await limiter.check(AGENT_A)
        limiter.remove_limit(AGENT_A)
        # New bucket from default — should have plenty of tokens
        for _ in range(10):
            await limiter.check(AGENT_A)


# ══════════════════════════════════════════════════════════════════════════════
# PayloadValidator
# ══════════════════════════════════════════════════════════════════════════════

class TestPayloadValidator:
    def _msg(self, text: str = "Hello agent") -> Message:
        return Message.user_text(text)

    def test_valid_message_passes(self):
        v = PayloadValidator()
        result = v.validate(self._msg())
        assert result.parts[0].content == "Hello agent"

    def test_strips_whitespace_from_text(self):
        v = PayloadValidator()
        msg = Message.user_text("  hello  ")
        result = v.validate(msg)
        assert result.parts[0].content == "hello"

    def test_blank_text_after_strip_raises(self):
        v = PayloadValidator()
        msg = Message.user_text("   ")
        with pytest.raises(BlankTextPartError):
            v.validate(msg)

    def test_too_large_payload_raises(self):
        v = PayloadValidator(config=ValidatorConfig(max_bytes=10))
        with pytest.raises(PayloadTooLargeError):
            v.validate(self._msg("This message is definitely longer than 10 bytes"))

    def test_too_many_parts_raises(self):
        v = PayloadValidator(config=ValidatorConfig(max_parts=2))
        msg = Message(
            role="user",
            parts=[
                Part(type=PartType.TEXT, content=f"part {i}")
                for i in range(3)
            ],
        )
        with pytest.raises(TooManyPartsError):
            v.validate(msg)

    def test_no_strip_when_disabled(self):
        v = PayloadValidator(config=ValidatorConfig(strip_text=False))
        msg = Message.user_text("  hello  ")
        result = v.validate(msg)
        assert result.parts[0].content == "  hello  "

    def test_validate_dict_parses_and_validates(self):
        v = PayloadValidator()
        raw = {
            "role": "user",
            "parts": [{"type": "text", "content": "hello"}],
        }
        result = v.validate_dict(raw)
        assert result.parts[0].content == "hello"

    def test_validate_dict_invalid_schema_raises(self):
        v = PayloadValidator()
        with pytest.raises(InvalidPartError):
            v.validate_dict({"not": "a message"})

    def test_json_part_not_stripped(self):
        v = PayloadValidator()
        msg = Message(
            role="user",
            parts=[Part(type=PartType.JSON, content={"key": "value"})],
        )
        result = v.validate(msg)
        assert result.parts[0].content == {"key": "value"}

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError):
            ValidatorConfig(max_bytes=0)
        with pytest.raises(ValueError):
            ValidatorConfig(max_parts=0)