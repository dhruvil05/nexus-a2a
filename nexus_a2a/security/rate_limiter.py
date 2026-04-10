"""
nexus_a2a/security/rate_limiter.py

RateLimiter — prevents any single agent from flooding another with requests.

Algorithm: Token Bucket
  - Each agent gets a bucket that holds up to `burst` tokens.
  - Tokens refill at a rate of `rate` tokens per second.
  - Each request consumes 1 token.
  - If the bucket is empty, the request is rejected immediately (no queuing).

Why token bucket?
  - Allows short bursts (e.g. 10 requests at once) while enforcing
    a sustained average rate.
  - Zero external dependencies — pure Python, fully in-process.
  - Constant-time O(1) per check.

Limitation:
  - In-process only. For distributed rate limiting across multiple
    processes, use RedisRateLimiter (Phase 5).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── Exceptions ────────────────────────────────────────────────────────────────

class RateLimitError(Exception):
    """Raised when an agent exceeds its allowed request rate."""

    def __init__(self, agent_url: str, retry_after: float) -> None:
        super().__init__(
            f"Rate limit exceeded for agent '{agent_url}'. "
            f"Retry after {retry_after:.2f}s."
        )
        self.agent_url   = agent_url
        self.retry_after = retry_after  # seconds until next token available


# ── Token bucket ──────────────────────────────────────────────────────────────

@dataclass
class _TokenBucket:
    """
    Internal token bucket for one agent.

    Fields:
        rate:       Tokens added per second (sustained throughput).
        burst:      Maximum tokens the bucket can hold (peak burst size).
        tokens:     Current token count.
        last_refill: Timestamp of the last refill (monotonic clock).
    """
    rate:        float
    burst:       float
    tokens:      float = field(init=False)
    last_refill: float = field(init=False)

    def __post_init__(self) -> None:
        # Start full so the first burst of requests goes through immediately
        self.tokens      = self.burst
        self.last_refill = time.monotonic()

    def _refill(self) -> None:
        """Add tokens based on elapsed time since last refill."""
        now     = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens      = min(self.burst, self.tokens + elapsed * self.rate)
        self.last_refill = now

    def consume(self) -> float:
        """
        Attempt to consume 1 token.

        Returns:
            0.0  if the token was consumed successfully.
            >0.0 seconds to wait if the bucket is empty.
        """
        self._refill()
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return 0.0
        # Time until the next token becomes available
        return (1.0 - self.tokens) / self.rate


# ── Rate limit config ─────────────────────────────────────────────────────────

@dataclass
class RateLimitConfig:
    """
    Rate limit settings for one agent.

    Args:
        rate:  Requests per second (sustained average). Default: 10 req/s.
        burst: Maximum requests allowed in a single burst. Default: 20.

    Examples:
        RateLimitConfig(rate=1, burst=5)    # 1 req/s, burst up to 5
        RateLimitConfig(rate=10, burst=50)  # 10 req/s, burst up to 50
    """
    rate:  float = 10.0   # tokens per second
    burst: float = 20.0   # max bucket size

    def __post_init__(self) -> None:
        if self.rate <= 0:
            raise ValueError(f"rate must be > 0, got {self.rate}")
        if self.burst <= 0:
            raise ValueError(f"burst must be > 0, got {self.burst}")



# ── RateLimiter ───────────────────────────────────────────────────────────────

class RateLimiter:
    """
    Per-agent token bucket rate limiter.

    Usage:
        limiter = RateLimiter(
            default_config=RateLimitConfig(rate=10, burst=20)
        )

        # Custom limit for a specific agent
        limiter.set_limit(
            agent_url="http://heavy-agent:8003",
            config=RateLimitConfig(rate=2, burst=5),
        )

        # Check on every inbound request
        await limiter.check("http://heavy-agent:8003")  # raises if exceeded

        # Non-raising version
        allowed = await limiter.is_allowed("http://heavy-agent:8003")

    Args:
        default_config: Applied to any agent without a specific config.
                        Default: 10 req/s, burst 20.
    """

    def __init__(
        self,
        default_config: RateLimitConfig | None = None,
    ) -> None:
        self._default = default_config or RateLimitConfig()
        # agent_url → RateLimitConfig
        self._configs: dict[str, RateLimitConfig] = {}
        # agent_url → _TokenBucket (created lazily on first request)
        self._buckets: dict[str, _TokenBucket] = {}
        self._lock = asyncio.Lock()

    # ── Configuration ─────────────────────────────────────────────────────────

    def set_limit(self, agent_url: str, config: RateLimitConfig) -> None:
        """
        Set a custom rate limit for a specific agent.
        Resets the agent's current token bucket.

        Args:
            agent_url: The agent's base URL.
            config:    The rate limit config to apply.
        """
        url = agent_url.rstrip("/")
        self._configs[url] = config
        # Reset the bucket so new limits take effect immediately
        self._buckets.pop(url, None)
        logger.info(
            "Rate limit set for %s: %.1f req/s, burst=%d",
            url, config.rate, config.burst,
        )

    def remove_limit(self, agent_url: str) -> None:
        """Remove a custom limit — agent falls back to the default config."""
        url = agent_url.rstrip("/")
        self._configs.pop(url, None)
        self._buckets.pop(url, None)

    # ── Enforcement ───────────────────────────────────────────────────────────

    async def check(self, agent_url: str) -> None:
        """
        Consume one token for agent_url.
        Raises immediately if the bucket is empty — no waiting.

        Args:
            agent_url: The agent making the request.

        Raises:
            RateLimitError: If the rate limit is exceeded.
        """
        url    = agent_url.rstrip("/")
        bucket = await self._get_or_create_bucket(url)

        async with self._lock:
            retry_after = bucket.consume()

        if retry_after > 0:
            logger.warning("Rate limit exceeded for %s (retry in %.2fs)", url, retry_after)
            raise RateLimitError(agent_url, retry_after)

    async def is_allowed(self, agent_url: str) -> bool:
        """
        Non-raising version of check().
        Returns True if the request is within the rate limit, False if exceeded.
        """
        try:
            await self.check(agent_url)
            return True
        except RateLimitError:
            return False

    # ── Introspection ─────────────────────────────────────────────────────────

    async def available_tokens(self, agent_url: str) -> float:
        """
        Return the current token count for an agent (after refill).
        Useful for monitoring and debugging.
        """
        url    = agent_url.rstrip("/")
        bucket = await self._get_or_create_bucket(url)
        async with self._lock:
            bucket._refill()
            return bucket.tokens

    def get_config(self, agent_url: str) -> RateLimitConfig:
        """Return the effective rate limit config for an agent."""
        return self._configs.get(agent_url.rstrip("/"), self._default)

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _get_or_create_bucket(self, url: str) -> _TokenBucket:
        """Return existing bucket or create one from the agent's config."""
        if url not in self._buckets:
            config = self._configs.get(url, self._default)
            self._buckets[url] = _TokenBucket(rate=config.rate, burst=config.burst)
        return self._buckets[url]
