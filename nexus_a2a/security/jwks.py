"""
nexus_a2a/security/jwks.py

JWKSClient — fetches and caches an agent's public signing keys.

Why asymmetric keys matter here:
    With HS256 the verifier holds the same secret as the signer. Any agent that
    can *check* a peer's token can also *mint* one, so in a network of more
    than two parties every verifier is also a forger. Splitting the key pair
    fixes that: a signer keeps the private key, and everyone else only ever
    sees the public half.

    JWKS is how the public half gets distributed — the signer publishes a key
    set at a URL, verifiers fetch it, and rotation is a publish rather than a
    coordinated secret swap.

Caching:
    Key sets are cached for cache_ttl seconds. A token whose `kid` is not in
    the cache triggers one refresh, which is how rotation is picked up without
    waiting for the TTL — but refreshes are rate-limited, because otherwise a
    stream of tokens bearing unknown kids would turn this agent into a
    traffic amplifier pointed at the JWKS endpoint.

    Stale-cache refreshes are single-flight and rate-limited too. If the
    endpoint is down, previously fetched keys keep being served (with a
    warning) until a refresh succeeds; only a client that has never fetched
    any keys fails verification.

Requires: pip install nexus-a2a[jwks]
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Any

import httpx

from nexus_a2a.security.redaction import redact_url

logger = logging.getLogger(__name__)

# How long a fetched key set is trusted before it is re-fetched.
_DEFAULT_CACHE_TTL = 300.0

# Floor between refreshes triggered by an unknown kid. Without it, tokens
# carrying junk kids would each cause a fetch. Stale-cache refreshes use the
# same floor (capped at cache_ttl) after a failed attempt.
_MIN_REFRESH_INTERVAL = 10.0

_DEFAULT_TIMEOUT = 10.0


class JWKSError(Exception):
    """Base class for JWKS retrieval and lookup failures."""


class JWKSFetchError(JWKSError):
    """Raised when the key set could not be retrieved or parsed."""

    def __init__(self, url: str, reason: str) -> None:
        super().__init__(f"Could not fetch JWKS from '{url}': {reason}")
        self.url = url
        self.reason = reason


class JWKSKeyNotFoundError(JWKSError):
    """Raised when no key in the set matches the token's key id."""

    def __init__(self, kid: str | None) -> None:
        super().__init__(
            f"No signing key matching kid={kid!r} in the key set. "
            "The token may be signed by a retired key, or by a different issuer."
        )
        self.kid = kid


class JWKSClient:
    """
    Fetches a JWKS document and resolves a token's signing key from it.

    Usage:
        client = JWKSClient("https://issuer.example.com/.well-known/jwks.json")
        key = await client.key_for_token(token)

    Args:
        url:         Where the key set is published.
        cache_ttl:   Seconds a fetched key set is reused. Default: 300.
        timeout:     HTTP timeout per fetch.
        client:      An existing httpx.AsyncClient to reuse. One is created
                     per fetch when omitted.
        max_stale:   While the endpoint is failing, keep serving the last
                     fetched keys for at most this many seconds past their
                     TTL. Bounds how long a key the issuer has since revoked
                     (removed from the set) stays usable during an outage.
                     Default: 3600.
    """

    def __init__(
        self,
        url: str,
        cache_ttl: float = _DEFAULT_CACHE_TTL,
        timeout: float = _DEFAULT_TIMEOUT,
        client: httpx.AsyncClient | None = None,
        max_stale: float = 3600.0,
    ) -> None:
        self.url = url
        self.cache_ttl = cache_ttl
        self.max_stale = max_stale
        self.timeout = timeout
        self._client = client

        self._keys: dict[str, Any] = {}
        self._fetched_at: float = 0.0
        # -inf = never attempted, so the first refresh is never rate-limited.
        self._last_refresh_attempt: float = -math.inf
        self._last_error: JWKSFetchError | None = None
        self._lock = asyncio.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    async def key_for_token(self, token: str) -> Any:
        """
        Return the signing key for a token, fetching the key set if needed.

        Only the token's `kid` header is consulted — never its `alg`. Letting a
        token choose its own algorithm is the classic JWT forgery; the caller
        decides which algorithms are acceptable from its own configuration.

        Raises:
            JWKSFetchError:       Key set unavailable or malformed.
            JWKSKeyNotFoundError: No key matches the token's kid.
        """
        import jwt

        try:
            header = jwt.get_unverified_header(token)
        except Exception as exc:
            raise JWKSKeyNotFoundError(None) from exc

        kid = header.get("kid")

        if self._is_stale():
            await self._refresh_stale()

        key = self._lookup(kid)
        if key is not None:
            return key

        # Unknown kid: the signer may have rotated. One rate-limited retry.
        # Look up again regardless of who refreshed: a concurrent request may
        # have fetched the rotated key while this one waited on the lock.
        await self._refresh_if_allowed()
        key = self._lookup(kid)
        if key is not None:
            return key

        raise JWKSKeyNotFoundError(kid)

    async def refresh(self) -> int:
        """
        Fetch the key set and replace the cache.

        An explicit call always fetches. Refreshes triggered by verification
        (stale cache, unknown kid) go through rate-limited, single-flight
        paths instead.

        Returns:
            How many keys were loaded.

        Raises:
            JWKSFetchError: The document could not be fetched or parsed.
                The previously cached keys are left untouched.
        """
        async with self._lock:
            return await self._refresh_locked()

    @property
    def cached_key_ids(self) -> list[str]:
        """Key ids currently cached. Useful for diagnostics."""
        return list(self._keys)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _is_stale(self) -> bool:
        if not self._keys:
            return True
        return (time.monotonic() - self._fetched_at) >= self.cache_ttl

    def _lookup(self, kid: str | None) -> Any:
        if kid is not None:
            return self._keys.get(kid)
        # A key set with exactly one key is unambiguous even without a kid.
        if len(self._keys) == 1:
            return next(iter(self._keys.values()))
        return None

    async def _refresh_locked(self) -> int:
        """Fetch and replace the cache. Caller must hold self._lock."""
        self._last_refresh_attempt = time.monotonic()
        try:
            data = await self._fetch()
            keys = self._parse(data)
        except JWKSFetchError as exc:
            self._last_error = exc
            raise
        self._keys = keys
        self._fetched_at = time.monotonic()
        self._last_error = None
        logger.info("JWKS: loaded %d key(s) from %s", len(keys), redact_url(self.url))
        return len(keys)

    def _may_serve_stale(self) -> bool:
        """True while cached keys are within cache_ttl + max_stale of fetch."""
        age = time.monotonic() - self._fetched_at
        if age <= self.cache_ttl + self.max_stale:
            return True
        logger.error(
            "JWKS: keys from %s are %.0fs old and the endpoint is still "
            "failing; refusing to keep serving them",
            redact_url(self.url),
            age,
        )
        return False

    def _stale_refresh_floor(self) -> float:
        # Never wait longer than the configured TTL: a caller asking for a
        # very short TTL gets it, while the default TTL gets the full floor.
        return max(0.0, min(_MIN_REFRESH_INTERVAL, self.cache_ttl))

    async def _refresh_stale(self) -> None:
        """
        Refresh a stale cache — single-flight, rate-limited, stale-while-error.

        - Concurrent callers queue on the lock; once inside, they re-check
          staleness, so only the first one actually fetches.
        - After a failed attempt, no new fetch is made until the floor has
          passed, so an outage does not turn every request into a fetch.
        - If a refresh fails but keys were fetched before, those keys keep
          being served (with a warning) instead of failing every request.

        Raises:
            JWKSFetchError: Only when no keys have ever been fetched.
        """
        async with self._lock:
            if not self._is_stale():
                return  # another request refreshed while this one waited

            elapsed = time.monotonic() - self._last_refresh_attempt
            floor = self._stale_refresh_floor()
            if elapsed < floor:
                if self._keys and self._may_serve_stale():
                    return  # recently attempted and failed: serve stale keys
                reason = self._last_error.reason if self._last_error else "unknown"
                raise JWKSFetchError(
                    redact_url(self.url),
                    f"{reason} (last attempt {elapsed:.1f}s ago; "
                    f"retrying after {floor:.0f}s)",
                )

            try:
                await self._refresh_locked()
            except JWKSFetchError as exc:
                if not self._keys or not self._may_serve_stale():
                    raise
                logger.warning(
                    "JWKS: refresh from %s failed (%s); serving %d previously "
                    "fetched key(s) until the next successful refresh",
                    redact_url(self.url),
                    exc.reason,
                    len(self._keys),
                )

    async def _refresh_if_allowed(self) -> bool:
        """Refresh unless one was attempted too recently. Never raises."""
        async with self._lock:
            # Checked under the lock so that concurrent unknown-kid requests
            # do not each pass the check and then fetch one after another.
            elapsed = time.monotonic() - self._last_refresh_attempt
            if elapsed < _MIN_REFRESH_INTERVAL:
                logger.debug(
                    "JWKS: refresh suppressed, last attempt %.1fs ago", elapsed
                )
                return False
            try:
                await self._refresh_locked()
                return True
            except JWKSFetchError:
                logger.warning("JWKS: refresh for unknown kid failed", exc_info=True)
                return False

    async def _fetch(self) -> dict[str, Any]:
        try:
            if self._client is not None:
                response = await self._client.get(self.url, timeout=self.timeout)
            else:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.get(self.url)
        except httpx.HTTPError as exc:
            raise JWKSFetchError(self.url, str(exc)) from exc

        if response.status_code != 200:
            raise JWKSFetchError(self.url, f"HTTP {response.status_code}")

        try:
            payload = response.json()
        except ValueError as exc:
            raise JWKSFetchError(self.url, f"response is not JSON: {exc}") from exc

        if not isinstance(payload, dict):
            raise JWKSFetchError(self.url, "JWKS document must be an object")
        return payload

    @staticmethod
    def _parse(data: dict[str, Any]) -> dict[str, Any]:
        """
        Turn a JWKS document into {kid: PyJWK}.

        A single unusable key is skipped rather than failing the whole set, so
        one bad entry cannot lock out every other key the issuer publishes.
        """
        try:
            import jwt
        except ImportError as exc:  # pragma: no cover
            raise JWKSFetchError("", "PyJWT is not installed") from exc

        raw_keys = data.get("keys")
        if not isinstance(raw_keys, list):
            raise JWKSFetchError("", "JWKS document has no 'keys' array")

        parsed: dict[str, Any] = {}
        for index, raw in enumerate(raw_keys):
            try:
                key = jwt.PyJWK(raw)
            except Exception:
                logger.warning("JWKS: skipping unusable key at index %d", index)
                continue
            kid = raw.get("kid") if isinstance(raw, dict) else None
            parsed[kid or f"_unnamed_{index}"] = key

        if not parsed:
            raise JWKSFetchError("", "JWKS document contained no usable keys")
        return parsed


__all__ = [
    "JWKSClient",
    "JWKSError",
    "JWKSFetchError",
    "JWKSKeyNotFoundError",
]
