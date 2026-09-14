"""
nexus_a2a/security/auth.py

AuthManager — verifies credentials on every inbound agent request.

Supports three schemes (matching A2A's AgentAuthentication model):
  - NONE     : no auth required (dev/testing only)
  - API_KEY  : static secret in a request header
  - JWT      : signed JSON Web Token in Authorization: Bearer header

Design principles:
  - Credentials are never logged.
  - Each registered agent can have its OWN scheme and secret —
    Agent A might use JWT while Agent B uses an API key.
  - All verify_* methods are async so they can later call remote
    token introspection endpoints without breaking the interface.
"""

from __future__ import annotations

import hmac
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import jwt
from jwt import ExpiredSignatureError, InvalidTokenError

from nexus_a2a.models.agent import AuthScheme

logger = logging.getLogger(__name__)

# JWT algorithm used when nothing else is configured.
_JWT_ALGORITHM = "HS256"

# Which algorithms each kind of key material may be used with.
#
# This split is the whole defence against algorithm confusion. An RSA public
# key is, by definition, public — so if a verifier configured for RS256 also
# accepted HS256, an attacker could take that public key, use it as an HMAC
# secret, sign their own token, and be believed. The allowed algorithms are
# therefore derived from the CONFIGURED key, never from the token's own `alg`
# header, and the two families never overlap.
_SYMMETRIC_ALGORITHMS = ("HS256", "HS384", "HS512")
_ASYMMETRIC_ALGORITHMS = (
    "RS256", "RS384", "RS512",
    "PS256", "PS384", "PS512",
    "ES256", "ES384", "ES512",
)

# Default header name for API key auth
_DEFAULT_API_KEY_HEADER = "X-API-Key"


# ── Exceptions ────────────────────────────────────────────────────────────────


class AuthError(Exception):
    """Base class for all authentication errors."""


class MissingCredentialsError(AuthError):
    """Raised when expected credentials are absent from the request."""

    def __init__(self, scheme: AuthScheme) -> None:
        super().__init__(
            f"Request is missing credentials required by scheme '{scheme.value}'."
        )
        self.scheme = scheme


class InvalidCredentialsError(AuthError):
    """Raised when credentials are present but invalid (wrong key, bad token, etc.)."""

    def __init__(self, reason: str) -> None:
        # Never include the actual credential value in the message
        super().__init__(f"Invalid credentials: {reason}")
        self.reason = reason


class ExpiredCredentialsError(AuthError):
    """Raised when a JWT token has expired."""

    def __init__(self) -> None:
        super().__init__("Token has expired. Request a new one.")


class UnknownAgentError(AuthError):
    """
    Raised when credentials are checked for an agent that was never
    registered with the AuthManager.

    Authentication fails closed: an unregistered agent is rejected rather
    than silently treated as 'no auth required'. Pass
    AuthManager(allow_unregistered=True) to restore the pre-1.5.0 behaviour.
    """

    def __init__(self, agent_url: str) -> None:
        super().__init__(
            f"No credential config registered for agent '{agent_url}'. "
            "Register it with AuthManager.register_agent(), or construct "
            "AuthManager(allow_unregistered=True) to allow unregistered agents."
        )
        self.agent_url = agent_url


# ── Per-agent credential config ───────────────────────────────────────────────


@dataclass
class AgentCredentialConfig:
    """
    Stores the expected credentials for ONE registered agent.

    Fields:
        scheme:          Which auth scheme this agent requires.
        api_key:         The expected API key (for API_KEY scheme).
        jwt_secret:      Shared secret for symmetric JWTs (HS*). Whoever holds
                         it can both verify AND forge, so prefer a key pair for
                         anything beyond two mutually-trusting agents.
        jwt_public_key:  PEM public key used to VERIFY asymmetric JWTs.
        jwt_private_key: PEM private key used to SIGN outbound JWTs.
        jwks_url:        URL publishing the signer's public keys. Verification
                         resolves the key by the token's `kid`, so rotation is
                         a publish rather than a coordinated secret swap.
        jwt_algorithm:   Algorithm for asymmetric keys. Default: RS256.
                         Ignored when jwt_secret is used (always HS256).
        jwt_audience:    Optional 'aud' claim to require.
        jwt_issuer:      Optional 'iss' claim to require.
        header_name:     Header to read the API key from. Default 'X-API-Key'.

    Exactly one source of verification material may be set: jwt_secret,
    jwt_public_key, or jwks_url. Configuring a secret alongside a public key
    would mean accepting both families, which is the algorithm-confusion hole.
    """

    scheme: AuthScheme = AuthScheme.NONE
    api_key: str | None = None
    jwt_secret: str | None = None
    jwt_public_key: str | None = None
    jwt_private_key: str | None = None
    jwks_url: str | None = None
    jwt_algorithm: str = "RS256"
    jwt_audience: str | None = None
    jwt_issuer: str | None = None
    header_name: str = field(default=_DEFAULT_API_KEY_HEADER)

    @property
    def is_asymmetric(self) -> bool:
        """True when verification uses a public key rather than a secret."""
        return bool(self.jwt_public_key or self.jwks_url)

    def allowed_algorithms(self) -> list[str]:
        """
        Algorithms acceptable for THIS config, from its key material alone.

        Never widen this with anything read out of a token.
        """
        if self.is_asymmetric:
            return [self.jwt_algorithm]
        return [_JWT_ALGORITHM]


# ── AuthManager ───────────────────────────────────────────────────────────────


class AuthManager:
    """
    Verifies inbound credentials for agent-to-agent requests.

    Each agent in your network can be registered with its own credential
    config. AuthManager picks the right verification strategy automatically.

    Usage:
        auth = AuthManager()

        # Register an agent that expects an API key
        auth.register_agent(
            agent_url="http://research-agent:8001",
            config=AgentCredentialConfig(
                scheme=AuthScheme.API_KEY,
                api_key="super-secret-key-123",
            ),
        )

        # Later, verify a request coming from that agent
        headers = {"X-API-Key": "super-secret-key-123"}
        claims = await auth.verify(agent_url="http://research-agent:8001", headers=headers)

        # Issue a JWT for outbound calls to an agent expecting JWT auth
        auth.register_agent(
            agent_url="http://summary-agent:8002",
            config=AgentCredentialConfig(
                scheme=AuthScheme.JWT,
                jwt_secret="my-jwt-secret",
            ),
        )
        token = auth.issue_jwt(agent_url="http://summary-agent:8002",
                               subject="nexus-a2a", expires_in=3600)
    """

    def __init__(self, allow_unregistered: bool = False) -> None:
        """
        Args:
            allow_unregistered: If False (the default), verifying an agent
                that was never registered raises UnknownAgentError. If True,
                unregistered agents fall back to AuthScheme.NONE — the
                pre-1.5.0 behaviour, which fails OPEN and should only be
                used in development.
        """
        # agent_url → AgentCredentialConfig
        self._configs: dict[str, AgentCredentialConfig] = {}
        # jwks_url -> JWKSClient, so key sets are fetched and cached once per
        # URL rather than per request.
        self._jwks_clients: dict[str, Any] = {}
        self._allow_unregistered = allow_unregistered
        if allow_unregistered:
            logger.warning(
                "AuthManager(allow_unregistered=True): requests from agents "
                "that are not registered will bypass authentication."
            )

    # ── Registration ──────────────────────────────────────────────────────────

    def register_agent(
        self,
        agent_url: str,
        config: AgentCredentialConfig,
    ) -> None:
        """
        Register the expected credentials for a remote agent.

        Args:
            agent_url: The agent's base URL (used as the lookup key).
            config:    Credential config for this agent.

        Raises:
            ValueError: If the config is incomplete for the chosen scheme.
        """
        self._validate_config(config)
        self._configs[agent_url.rstrip("/")] = config
        logger.info(
            "Auth registered for agent %s with scheme '%s'",
            agent_url,
            config.scheme.value,
        )

    def unregister_agent(self, agent_url: str) -> None:
        """Remove the credential config for an agent."""
        self._configs.pop(agent_url.rstrip("/"), None)

    # ── Verification (inbound requests) ──────────────────────────────────────

    async def verify(
        self,
        agent_url: str,
        headers: dict[str, str],
    ) -> dict[str, Any]:
        """
        Verify the credentials in an inbound request from agent_url.

        Args:
            agent_url: The URL of the agent that sent the request.
            headers:   The HTTP headers from the incoming request.

        Returns:
            A dict of claims extracted from the credentials.
            For API_KEY: {"scheme": "api_key", "agent_url": ...}
            For JWT:     the decoded JWT payload dict.
            For NONE:    {"scheme": "none"}

        Raises:
            UnknownAgentError:        Agent is not registered (fail-closed).
            MissingCredentialsError:  Expected header not present.
            InvalidCredentialsError:  Credentials present but wrong.
            ExpiredCredentialsError:  JWT has expired.
        """
        config = self._get_config(agent_url)

        match config.scheme:
            case AuthScheme.NONE:
                return {"scheme": "none"}

            case AuthScheme.API_KEY:
                return await self._verify_api_key(agent_url, headers, config)

            case AuthScheme.JWT:
                return await self._verify_jwt(headers, config)

            case _:
                raise InvalidCredentialsError(
                    f"Unsupported auth scheme: {config.scheme.value}"
                )

    # ── Token issuance (outbound requests) ───────────────────────────────────

    def issue_jwt(
        self,
        agent_url: str,
        subject: str,
        expires_in: int = 3600,
        extra_claims: dict[str, Any] | None = None,
    ) -> str:
        """
        Create a signed JWT to authenticate outbound calls to agent_url.

        Args:
            agent_url:    The target agent's URL (must be registered with JWT scheme).
            subject:      The 'sub' claim — typically your agent's identifier.
            expires_in:   Token lifetime in seconds. Default: 1 hour.
            extra_claims: Any additional claims to embed in the token.

        Returns:
            A signed JWT string.

        Raises:
            ValueError: If the agent is not registered or not using JWT scheme.
        """
        try:
            config = self._get_config(agent_url)
        except UnknownAgentError as exc:
            raise ValueError(str(exc)) from exc

        if config.scheme != AuthScheme.JWT:
            raise ValueError(
                f"Agent at '{agent_url}' uses scheme '{config.scheme.value}', not 'jwt'."
            )
        signing_key = config.jwt_private_key or config.jwt_secret
        if not signing_key:
            raise ValueError(
                f"Agent at '{agent_url}' has no signing key: set jwt_private_key "
                "(asymmetric) or jwt_secret (symmetric)."
            )
        algorithm = (
            config.jwt_algorithm if config.jwt_private_key else _JWT_ALGORITHM
        )

        now = int(time.time())
        payload: dict[str, Any] = {
            "sub": subject,
            "iat": now,
            "exp": now + expires_in,
            **(extra_claims or {}),
        }
        if config.jwt_audience:
            payload["aud"] = config.jwt_audience
        if config.jwt_issuer:
            payload["iss"] = config.jwt_issuer

        return jwt.encode(payload, signing_key, algorithm=algorithm)

    def build_auth_headers(
        self,
        agent_url: str,
        subject: str = "nexus-a2a",
    ) -> dict[str, str]:
        """
        Build the HTTP headers needed to authenticate a request to agent_url.

        Convenience method used by the HTTP client before sending a request.

        Args:
            agent_url: The target agent's URL.
            subject:   JWT subject (only used for JWT scheme).

        Returns:
            Dict of headers to merge into the outbound request.
            Returns {} if scheme is NONE or the agent is not registered —
            an outbound call to an agent we hold no credentials for simply
            carries no auth headers. Fail-closed applies to inbound
            verification, not to header construction.
        """
        try:
            config = self._get_config(agent_url)
        except UnknownAgentError:
            return {}

        match config.scheme:
            case AuthScheme.NONE:
                return {}
            case AuthScheme.API_KEY:
                if not config.api_key:
                    return {}
                return {config.header_name: config.api_key}
            case AuthScheme.JWT:
                token = self.issue_jwt(agent_url, subject=subject)
                return {"Authorization": f"Bearer {token}"}
            case _:
                return {}

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _verify_api_key(
        self,
        agent_url: str,
        headers: dict[str, str],
        config: AgentCredentialConfig,
    ) -> dict[str, Any]:
        """Compare the provided API key against the registered one."""
        provided = headers.get(config.header_name) or headers.get(
            config.header_name.lower()
        )
        if not provided:
            raise MissingCredentialsError(AuthScheme.API_KEY)

        # Constant-time comparison to prevent timing attacks
        if not hmac.compare_digest(
            provided.encode(),
            (config.api_key or "").encode(),
        ):
            raise InvalidCredentialsError("API key does not match.")

        return {"scheme": "api_key", "agent_url": agent_url}

    async def _verify_jwt(
        self,
        headers: dict[str, str],
        config: AgentCredentialConfig,
    ) -> dict[str, Any]:
        """Decode and verify a Bearer JWT from the Authorization header."""
        auth_header = headers.get("Authorization") or headers.get("authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise MissingCredentialsError(AuthScheme.JWT)

        token = auth_header[len("Bearer ") :]

        key = await self._verification_key(token, config)

        try:
            # 'audience' is a top-level parameter — passing it inside
            # 'options' silently skips audience validation entirely.
            #
            # 'algorithms' comes from the CONFIG, never the token. Accepting
            # whatever `alg` a token declares is how an attacker signs with a
            # public key as an HMAC secret and gets believed.
            claims: dict[str, Any] = jwt.decode(
                token,
                key,
                algorithms=config.allowed_algorithms(),
                audience=config.jwt_audience,
                issuer=config.jwt_issuer,
            )
            return claims

        except ExpiredSignatureError as err:
            raise ExpiredCredentialsError() from err
        except InvalidTokenError as exc:
            raise InvalidCredentialsError(str(exc)) from exc

    async def _verification_key(
        self,
        token: str,
        config: AgentCredentialConfig,
    ) -> Any:
        """
        Resolve the key this token must be verified against.

        JWKS resolves by the token's `kid`, which only selects WHICH public key
        to try — it can never widen the set of acceptable algorithms.
        """
        if config.jwks_url:
            from nexus_a2a.security.jwks import JWKSError

            client = self._jwks_client(config.jwks_url)
            try:
                return (await client.key_for_token(token)).key
            except JWKSError as exc:
                raise InvalidCredentialsError(str(exc)) from exc

        if config.jwt_public_key:
            return config.jwt_public_key

        return config.jwt_secret or ""

    def _jwks_client(self, url: str) -> Any:
        """Return the cached JWKS client for a URL, creating it on first use."""
        client = self._jwks_clients.get(url)
        if client is None:
            from nexus_a2a.security.jwks import JWKSClient

            client = JWKSClient(url)
            self._jwks_clients[url] = client
        return client

    def _get_config(self, agent_url: str) -> AgentCredentialConfig:
        """
        Return the credential config for the given agent URL.

        Fails CLOSED: an agent that was never registered is rejected, so a
        typo'd or attacker-supplied URL cannot bypass authentication by
        landing on a permissive default.

        Raises:
            UnknownAgentError: Agent is not registered and
                               allow_unregistered is False.
        """
        config = self._configs.get(agent_url.rstrip("/"))
        if config is not None:
            return config

        if self._allow_unregistered:
            return AgentCredentialConfig(scheme=AuthScheme.NONE)

        raise UnknownAgentError(agent_url)

    @staticmethod
    def _validate_config(config: AgentCredentialConfig) -> None:
        """Raise ValueError if the config is missing required fields."""
        if config.scheme == AuthScheme.API_KEY and not config.api_key:
            raise ValueError(
                "API_KEY scheme requires 'api_key' to be set in AgentCredentialConfig."
            )
        if config.scheme != AuthScheme.JWT:
            return

        sources = [
            name
            for name, value in (
                ("jwt_secret", config.jwt_secret),
                ("jwt_public_key", config.jwt_public_key),
                ("jwks_url", config.jwks_url),
            )
            if value
        ]

        if not sources:
            raise ValueError(
                "JWT scheme requires one of 'jwt_secret' (symmetric), "
                "'jwt_public_key' or 'jwks_url' (asymmetric) in "
                "AgentCredentialConfig."
            )

        if len(sources) > 1:
            # Accepting both families at once is the algorithm-confusion hole:
            # a public key doubles as an HMAC secret an attacker already knows.
            raise ValueError(
                f"AgentCredentialConfig sets {' and '.join(sources)}. Choose "
                "exactly one verification source — accepting both a shared "
                "secret and a public key allows algorithm-confusion forgery."
            )

        if config.is_asymmetric and config.jwt_algorithm not in _ASYMMETRIC_ALGORITHMS:
            raise ValueError(
                f"jwt_algorithm={config.jwt_algorithm!r} is not an asymmetric "
                f"algorithm. Choose one of: {', '.join(_ASYMMETRIC_ALGORITHMS)}."
            )
