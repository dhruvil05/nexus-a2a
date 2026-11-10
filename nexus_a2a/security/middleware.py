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

    The header is self-asserted. It only becomes a VERIFIED identity when the
    credentials prove it: the URL has its own registered credential that
    matched, or the request carries a valid JWT whose 'sub' equals it. Trust
    rules and per-caller rate limits apply only to verified identities;
    everything else is rate-limited by client address, never by the header.
"""

from __future__ import annotations

import logging
import ssl
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import unquote, urlsplit

from nexus_a2a.models.task import Message
from nexus_a2a.security.auth import AuthError, AuthManager
from nexus_a2a.security.rate_limiter import (
    AbstractRateLimiter,
    RateLimitError,
)
from nexus_a2a.security.trust import (
    TrustBoundary,
    TrustError,
    UnverifiedCallerError,
)
from nexus_a2a.security.validator import (
    PayloadTooLargeError,
    PayloadValidator,
    ValidationError_,
)

logger = logging.getLogger(__name__)

# Header a calling agent uses to announce its own base URL.
CALLER_HEADER = "X-Nexus-Caller"

# Rate-limit bucket for requests whose client address is unknown. One shared
# bucket deliberately caps that traffic in total rather than handing each
# request a fresh allowance.
ANONYMOUS_CALLER = "<anonymous>"

# Prefix for rate-limit buckets keyed by network address. Use it to give one
# address a custom limit: limiter.set_limit("ip:10.0.0.5", RateLimitConfig(...)).
IP_BUCKET_PREFIX = "ip:"


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
        verified:  True when the credentials PROVE url (see module docs).
                   False means url is only what the caller claimed.
        client:    Network bucket key for this request ("ip:<addr>").
    """

    url: str | None = None
    claims: dict[str, Any] = field(default_factory=dict)
    anonymous: bool = True
    verified: bool = False
    client: str = ANONYMOUS_CALLER

    @property
    def label(self) -> str:
        """A stable string for logs and metrics (may be unverified)."""
        return self.url or ANONYMOUS_CALLER

    @property
    def rate_key(self) -> str:
        """
        The rate-limit bucket for this request.

        The verified caller URL when there is one, so set_limit(caller_url)
        still applies to it; otherwise the client address. Never an unverified
        header value — rotating that gave every request a fresh bucket.
        """
        if self.verified and self.url:
            return self.url
        return self.client


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
        rate_limiter:   Per-caller token bucket. None = unlimited. Use
                        RedisRateLimiter when limits must hold across
                        replicas — the in-process one multiplies the
                        effective limit by the number of processes.
        validator:      Payload size/shape checks. None = no validation.
        server_url:     This agent's own URL — the TRUST TARGET. Required when
                        trust is set, since a trust rule is caller to target.
        caller_header:  Header carrying the caller's base URL.
        require_caller: Reject requests with no caller identity, even when
                        auth is not configured.
        trust_warn_only: Log trust violations instead of refusing them. For
                        rolling out a trust policy: see what it would block
                        before it blocks anything.
        mtls:           MutualTLSConfig whose CA bundle verifies client
                        certificates. A verified certificate whose SAN names
                        the caller (URI:<caller URL> or DNS:<caller host>)
                        proves the caller's identity, like a per-caller key.
        client_cert_header: Header in which a TLS-terminating proxy forwards
                        the verified client certificate as (URL-escaped) PEM,
                        e.g. nginx's $ssl_client_escaped_cert. Only set this
                        when every request passes through that proxy and it
                        overwrites the header; otherwise it can be forged.
                        The ASGI 'tls' extension is used when the server
                        provides it.
        client_ip_header: Header holding the real client address when the
                        server sits behind a reverse proxy, e.g.
                        "X-Forwarded-For" (its LAST entry is used: the one your
                        proxy appended). Only set this when every request
                        passes through a proxy that sets it; otherwise clients
                        can forge it. Default: the TCP peer address.

    Raises:
        ValueError: If trust is configured without server_url.
    """

    def __init__(
        self,
        auth: AuthManager | None = None,
        trust: TrustBoundary | None = None,
        rate_limiter: AbstractRateLimiter | None = None,
        validator: PayloadValidator | None = None,
        server_url: str | None = None,
        caller_header: str = CALLER_HEADER,
        require_caller: bool = False,
        trust_warn_only: bool = False,
        client_ip_header: str | None = None,
        mtls: Any = None,
        client_cert_header: str | None = None,
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
        self.trust_warn_only = trust_warn_only
        self.client_ip_header = client_ip_header
        self.mtls = mtls
        self.client_cert_header = client_cert_header

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        """True if any stage is configured — i.e. this is not a no-op."""
        return (
            # mtls alone is not listed: it only PROVES identity, it refuses
            # nothing unless trust/require_caller act on that identity.
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
            "trust_warn_only": self.trust_warn_only,
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
        client_host: str | None = None,
        peer_cert_chain: list[str] | None = None,
    ) -> CallerIdentity:
        """
        Rate-limit, authenticate and authorise the caller.

        Args:
            headers:     Inbound HTTP headers.
            skill_id:    Skill being invoked, for skill-level trust ACLs.
            client_host: The TCP peer address (request.client.host). Used to
                         rate-limit callers whose identity is not verified.
            peer_cert_chain: PEM client certificate chain (leaf first) from
                         the ASGI 'tls' extension, if the server provides it.

        Returns:
            The established CallerIdentity.

        Raises:
            RateLimitError:        Caller exceeded its request rate.
            MissingCallerError:    Identity required but not presented.
            AuthError:             Credentials missing, invalid, or expired.
            UnverifiedCallerError: Trust is enforced but the announced caller
                                   URL is not proven by the credentials.
            TrustError:            Caller may not call this agent or skill.
        """
        caller_url = self._read_caller(headers)
        identity = CallerIdentity(
            url=caller_url,
            anonymous=caller_url is None,
            client=self._client_key(headers, client_host),
        )

        # A default credential (one shared key for every caller) does not need
        # to know who is calling. Per-caller credentials and trust rules do.
        auth_needs_identity = self.auth is not None and not self.auth.has_default
        needs_identity = (
            self.require_caller or auth_needs_identity or self.trust is not None
        )

        if self.auth is None:
            # No credential check: limit by address, BEFORE any work.
            await self._rate_limit(identity.client)
            if needs_identity and caller_url is None:
                raise MissingCallerError(self.caller_header)
            identity.verified = self._cert_proves_caller(
                caller_url, headers, peer_cert_chain
            )
        else:
            if needs_identity and caller_url is None:
                await self._rate_limit(identity.client)
                raise MissingCallerError(self.caller_header)
            # Shed an address that has already used up its allowance (on
            # failed or unverified requests) BEFORE paying for signature
            # checks or a JWKS lookup.
            await self._shed_if_exhausted(identity.client)
            try:
                # An anonymous caller here is checked against the default
                # credential; AuthManager still fails closed on a wrong secret.
                identity.claims = await self.auth.verify(caller_url or "", headers)
            except AuthError:
                # Failed attempts spend the ADDRESS's allowance, so guessing
                # credentials is throttled without letting a forged caller
                # header pick (or drain) somebody else's bucket.
                await self._rate_limit(identity.client)
                raise
            identity.verified = self._proves_caller(
                caller_url, identity.claims
            ) or self._cert_proves_caller(caller_url, headers, peer_cert_chain)
            await self._rate_limit(identity.rate_key)

        # ── Trust ─────────────────────────────────────────────────────────────
        if self.trust is not None:
            try:
                if not identity.verified:
                    raise UnverifiedCallerError(
                        str(caller_url), str(self.server_url)
                    )
                self.trust.check(
                    caller_url=str(caller_url),
                    target_url=str(self.server_url),
                    skill_id=skill_id,
                )
            except TrustError as exc:
                if not self.trust_warn_only:
                    raise
                logger.warning("Trust violation allowed (warn-only mode): %s", exc)

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

    async def _rate_limit(self, key: str) -> None:
        if self.rate_limiter is not None:
            await self.rate_limiter.check(key)

    async def _shed_if_exhausted(self, key: str) -> None:
        """Raise RateLimitError if key's bucket is empty, without consuming."""
        if self.rate_limiter is None:
            return
        available = await self.rate_limiter.available_tokens(key)
        if available < 1.0:
            rate = self.rate_limiter.get_config(key).rate
            raise RateLimitError(key, (1.0 - available) / rate)

    def _client_key(self, headers: dict[str, str], client_host: str | None) -> str:
        """Bucket key for the network peer: 'ip:<addr>' or the anonymous one."""
        host = client_host
        if self.client_ip_header:
            raw = headers.get(self.client_ip_header) or headers.get(
                self.client_ip_header.lower()
            )
            if raw:
                # The proxy APPENDS the address it saw; earlier entries came
                # from the client and can be anything.
                host = raw.split(",")[-1].strip() or host
        return f"{IP_BUCKET_PREFIX}{host}" if host else ANONYMOUS_CALLER

    def _cert_proves_caller(
        self,
        caller_url: str | None,
        headers: dict[str, str],
        peer_cert_chain: list[str] | None,
    ) -> bool:
        """
        Does a verified client certificate name the announced caller?

        Needs mtls= (the CA bundle). The chain comes from the ASGI 'tls'
        extension or, behind a TLS-terminating proxy, client_cert_header.
        A certificate that fails verification proves nothing (logged).
        """
        if self.mtls is None or caller_url is None:
            return False
        chain = list(peer_cert_chain or [])
        if not chain and self.client_cert_header:
            raw = headers.get(self.client_cert_header) or headers.get(
                self.client_cert_header.lower()
            )
            if raw:
                chain = [unquote(raw)]
        if not chain:
            return False

        from nexus_a2a.security.mtls import (
            MtlsCertificateError,
            verify_peer_certificate,
        )

        try:
            ders = [ssl.PEM_cert_to_DER_cert(pem.strip()) for pem in chain]
            info = verify_peer_certificate(self.mtls, ders[0], intermediates=ders[1:])
        except (MtlsCertificateError, ValueError) as exc:
            logger.warning("Client certificate not accepted: %s", exc)
            return False

        host = (urlsplit(caller_url).hostname or "").lower()
        for entry in info.san:
            kind, _, value = entry.partition(":")
            if kind == "URI" and value.rstrip("/") == caller_url:
                return True
            if kind == "DNS" and host and value.lower() == host:
                return True
        logger.warning(
            "Client certificate (%s) does not name caller %s in its SAN",
            info.subject,
            caller_url,
        )
        return False

    def _proves_caller(self, caller_url: str | None, claims: dict[str, Any]) -> bool:
        """
        Did the credentials that just verified prove the announced caller URL?

        Yes when the URL is registered with key material no other caller has
        (verify() used it and it matched), or when the credential is a JWT
        whose 'sub' names the URL. Key material shared between callers — one
        JWKS issuer, one HMAC secret, one API key — proves only membership, so
        then the token must name the caller. Any holder of a shared HS256
        secret can still mint a token for any 'sub'; use per-caller keys or an
        issuer that controls 'sub' where peers must not impersonate each other.
        A shared API key carries no identity at all, so it never proves one.
        """
        if caller_url is None or self.auth is None:
            return False
        if self.auth.has_own_credentials(caller_url):
            return True
        if self.auth.jwt_applies(caller_url):
            sub = claims.get("sub")
            return isinstance(sub, str) and sub.rstrip("/") == caller_url
        return False

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
