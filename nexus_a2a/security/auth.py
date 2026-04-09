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
from typing import Any, cast

from jose import JWTError, jwt

from nexus_a2a.models.agent import AuthScheme

logger = logging.getLogger(__name__)

# JWT algorithm used for signing and verification
_JWT_ALGORITHM = "HS256"

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


# ── Per-agent credential config ───────────────────────────────────────────────

@dataclass
class AgentCredentialConfig:
    """
    Stores the expected credentials for ONE registered agent.

    Fields:
        scheme:          Which auth scheme this agent requires.
        api_key:         The expected API key (for API_KEY scheme).
        jwt_secret:      The secret used to verify JWTs (for JWT scheme).
        jwt_audience:    Optional audience claim to validate in JWTs.
        header_name:     Header to read the API key from.
                         Defaults to 'X-API-Key'.
    """
    scheme:       AuthScheme = AuthScheme.NONE
    api_key:      str | None = None
    jwt_secret:   str | None = None
    jwt_audience: str | None = None
    header_name:  str        = field(default=_DEFAULT_API_KEY_HEADER)


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

    def __init__(self) -> None:
        # agent_url → AgentCredentialConfig
        self._configs: dict[str, AgentCredentialConfig] = {}

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
            agent_url, config.scheme.value,
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
        config = self._get_config(agent_url)

        if config.scheme != AuthScheme.JWT:
            raise ValueError(
                f"Agent at '{agent_url}' uses scheme '{config.scheme.value}', not 'jwt'."
            )
        if not config.jwt_secret:
            raise ValueError(
                f"Agent at '{agent_url}' has no jwt_secret configured."
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

        return cast(str, jwt.encode(payload, config.jwt_secret, algorithm=_JWT_ALGORITHM))

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
            Returns {} if scheme is NONE.
        """
        config = self._get_config(agent_url)

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

        token = auth_header[len("Bearer "):]

        try:
            options: dict[str, Any] = {}
            if config.jwt_audience:
                options["audience"] = config.jwt_audience

            claims: dict[str, Any] = jwt.decode(
                token,
                config.jwt_secret or "",
                algorithms=[_JWT_ALGORITHM],
                options=options,
            )
            return claims

        except jwt.ExpiredSignatureError as err:
            raise ExpiredCredentialsError() from err
        except JWTError as exc:
            raise InvalidCredentialsError(str(exc)) from exc

    def _get_config(self, agent_url: str) -> AgentCredentialConfig:
        """
        Return the credential config for the given agent URL.
        Falls back to NONE scheme if the agent is not explicitly registered.
        """
        return self._configs.get(
            agent_url.rstrip("/"),
            AgentCredentialConfig(scheme=AuthScheme.NONE),
        )

    @staticmethod
    def _validate_config(config: AgentCredentialConfig) -> None:
        """Raise ValueError if the config is missing required fields."""
        if config.scheme == AuthScheme.API_KEY and not config.api_key:
            raise ValueError(
                "API_KEY scheme requires 'api_key' to be set in AgentCredentialConfig."
            )
        if config.scheme == AuthScheme.JWT and not config.jwt_secret:
            raise ValueError(
                "JWT scheme requires 'jwt_secret' to be set in AgentCredentialConfig."
            )
