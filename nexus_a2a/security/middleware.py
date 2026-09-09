"""
nexus_a2a/security/middleware.py

SecurityMiddleware — composes the security primitives into a single inbound
check that A2AServer runs before any agent code executes.

Through v1.5.0 the security classes (AuthManager, TrustBoundary, RateLimiter,
PayloadValidator) were correct, tested, standalone building blocks that nothing
in the library ever called — there was no inbound server to enforce them on.
This is the piece that wires them into the request path.

Stages run cheapest-and-most-protective first:

  1. Size     — reject oversized bodies BEFORE parsing them.
  2. Rate     — shed load before spending work on crypto.
  3. Auth     — who is calling?
  4. Trust    — are they allowed to call us, for this skill?
  5. Validate — is the payload well-formed and within limits?

Every stage is optional: pass only the components you want enforced. A
SecurityMiddleware with nothing configured is a no-op, which is A2AServer's
default so an agent works out of the box and hardens incrementally.

Caller identity:
    Auth and trust both need to know WHO is calling. The caller announces
    itself with the 'X-Nexus-Caller' header, which A2AHttpClient sets when
    constructed with caller_url=... . Requests without it are anonymous;
    set require_caller=True to reject them outright.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from nexus_a2a.models.task import Message
from nexus_a2a.security.auth import AuthError, AuthManager
from nexus_a2a.security.rate_limiter import RateLimiter, RateLimitError
from nexus_a2a.security.trust import TrustBoundary, TrustError
from nexus_a2a.security.validator import (
    PayloadTooLargeError,
    PayloadValidator,
    ValidationError_,
)

logger = logging.getLogger(__name__)

# Header a calling agent uses to announce its own base URL.
CALLER_HEADER = "X-Nexus-Caller"

# Rate-limit bucket shared by all callers that did not identify themselves.
# One shared bucket deliberately caps TOTAL anonymous traffic rather than
# handing every unidentified caller its own fresh allowance.
ANONYMOUS_CALLER = "<anonymous>"


# ── Exceptions ────────────────────────────────────────────────────────────────


class MissingCallerError(AuthError):
    """
    Raised when authentication or trust is enforced but the request carried
    no caller identity, so there is nothing to authenticate or authorise.
    """

    def __init__(self, header: str = CALLER_HEADER) -> None:
        super().__init__(
            f"Request did not identify its caller. Send the '{header}' header "
            "(A2AHttpClient sets it when given caller_url=...), or construct "
            "the server's SecurityMiddleware without auth/trust."
        )
        self.header = header


# ── Caller identity ───────────────────────────────────────────────────────────


@dataclass
class CallerIdentity:
    """
    Who made this request, as far as the security layer could establish.

    Fields:
        url:       The caller's announced base URL, or None if anonymous.
        claims:    Claims returned by AuthManager (empty when auth is off).
        anonymous: True when no caller identity was presented.
    """

    url: str | None = None
    claims: dict[str, Any] = field(default_factory=dict)
    anonymous: bool = True

    @property
    def label(self) -> str:
        """A stable string for logs, metrics and rate-limit buckets."""
        return self.url or ANONYMOUS_CALLER


# ── Middleware ────────────────────────────────────────────────────────────────


class SecurityMiddleware:
    """
    Runs the inbound security chain for one A2A server.

    Usage:
        security = SecurityMiddleware(
            auth=auth_manager,
            trust=trust_boundary,
            rate_limiter=RateLimiter(),
            validator=PayloadValidator(),
            server_url="http://my-agent:8001",
        )
        server = A2AServer(MyAgent, security=security)

    Args:
        auth:           Verifies caller credentials. None = no authentication.
        trust:          Enforces caller to this-agent ACLs. None = no ACL check.
        rate_limiter:   Per-caller token bucket. None = unlimited.
        validator:      Payload size/shape checks. None = no validation.
        server_url:     This agent's own URL — the TRUST TARGET. Required when
                        trust is set, since a trust rule is caller to target.
        caller_header:  Header carrying the caller's base URL.
        require_caller: Reject requests with no caller identity, even when
                        auth is not configured.

    Raises:
        ValueError: If trust is configured without server_url.
    """

    def __init__(
        self,
        auth: AuthManager | None = None,
        trust: TrustBoundary | None = None,
        rate_limiter: RateLimiter | None = None,
        validator: PayloadValidator | None = None,
        server_url: str | None = None,
        caller_header: str = CALLER_HEADER,
        require_caller: bool = False,
    ) -> None:
        if trust is not None and not server_url:
            raise ValueError(
                "SecurityMiddleware(trust=...) also needs server_url — a trust "
                "rule is 'caller -> target', and this agent is the target."
            )

        self.auth = auth
        self.trust = trust
        self.rate_limiter = rate_limiter
        self.validator = validator
        self.server_url = server_url.rstrip("/") if server_url else None
        self.caller_header = caller_header
        self.require_caller = require_caller

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        """True if any stage is configured — i.e. this is not a no-op."""
        return (
            any((self.auth, self.trust, self.rate_limiter, self.validator))
            or self.require_caller
        )

    def summary(self) -> dict[str, bool]:
        """Which stages are active. Used by A2AServer's startup log."""
        return {
            "auth": self.auth is not None,
            "trust": self.trust is not None,
            "rate_limit": self.rate_limiter is not None,
            "validation": self.validator is not None,
            "require_caller": self.require_caller,
        }

    # ── Stage 1: size ─────────────────────────────────────────────────────────

    def check_size(self, raw: bytes) -> None:
        """
        Reject an oversized request body before it is parsed.

        Called on the raw bytes so an enormous body never reaches json.loads()
        or Pydantic. No-op when no validator is configured.

        Raises:
            PayloadTooLargeError: Body exceeds the validator's max_bytes.
        """
        if self.validator is None:
            return
        limit = self.validator.max_bytes
        if len(raw) > limit:
            raise PayloadTooLargeError(len(raw), limit)

    # ── Stages 2-4: caller ────────────────────────────────────────────────────

    async def authorize(
        self,
        headers: dict[str, str],
        skill_id: str | None = None,
    ) -> CallerIdentity:
        """
        Rate-limit, authenticate and authorise the caller.

        Args:
            headers:  Inbound HTTP headers.
            skill_id: Skill being invoked, for skill-level trust ACLs.

        Returns:
            The established CallerIdentity.

        Raises:
            RateLimitError:     Caller exceeded its request rate.
            MissingCallerError: Identity required but not presented.
            AuthError:          Credentials missing, invalid, or expired.
            TrustError:         Caller may not call this agent or this skill.
        """
        caller_url = self._read_caller(headers)
        identity = CallerIdentity(url=caller_url, anonymous=caller_url is None)

        # ── 2. Rate limit ─────────────────────────────────────────────────────
        # Before auth so a flood cannot force expensive signature checks.
        if self.rate_limiter is not None:
            await self.rate_limiter.check(identity.label)

        needs_identity = (
            self.require_caller or self.auth is not None or self.trust is not None
        )
        if needs_identity and caller_url is None:
            raise MissingCallerError(self.caller_header)

        # ── 3. Authenticate ───────────────────────────────────────────────────
        if self.auth is not None:
            # caller_url is non-None here: needs_identity was True.
            identity.claims = await self.auth.verify(str(caller_url), headers)

        # ── 4. Trust ──────────────────────────────────────────────────────────
        if self.trust is not None:
            self.trust.check(
                caller_url=str(caller_url),
                target_url=str(self.server_url),
                skill_id=skill_id,
            )

        return identity

    # ── Stage 5: payload ──────────────────────────────────────────────────────

    def validate_message(self, raw_message: Any) -> Message:
        """
        Validate and sanitise the inbound Message.

        Args:
            raw_message: The 'message' object from the JSON-RPC params.

        Returns:
            A validated, sanitised Message.

        Raises:
            ValidationError_: Payload is malformed or violates a limit.
        """
        if self.validator is None:
            return Message.model_validate(raw_message)
        return self.validator.validate_dict(raw_message)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _read_caller(self, headers: dict[str, str]) -> str | None:
        """Read the caller URL header, case-insensitively."""
        raw = headers.get(self.caller_header) or headers.get(
            self.caller_header.lower()
        )
        if raw is None:
            return None
        caller = raw.strip().rstrip("/")
        return caller or None


# ── Status mapping ────────────────────────────────────────────────────────────


def http_status_for(exc: Exception) -> int:
    """
    Map a security failure to the HTTP status A2AServer should return.

    Transport-level rejections get real HTTP status codes rather than JSON-RPC
    error bodies, so proxies, WAFs and dashboards can see them and the client
    does not retry them as if they were server faults.

        401 — no or bad credentials
        403 — authenticated but not permitted
        413 — payload too large
        429 — rate limited
        400 — malformed payload
    """
    if isinstance(exc, RateLimitError):
        return 429
    if isinstance(exc, PayloadTooLargeError):
        return 413
    if isinstance(exc, TrustError):
        return 403
    if isinstance(exc, AuthError):
        return 401
    if isinstance(exc, ValidationError_):
        return 400
    return 500
