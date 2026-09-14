"""
nexus_a2a/security/redis_rate_limiter.py

RedisRateLimiter — token bucket rate limiting that holds across processes.

The problem this solves:
    RateLimiter keeps its buckets in a process-local dict. Run three replicas
    behind a load balancer with rate=10/s and callers actually get 30/s — the
    limit silently scales with your deployment, which is the opposite of what
    a limit is for. Nothing warns you; requests just get through.

    RedisRateLimiter keeps one bucket per agent in Redis, so the limit is the
    limit no matter how many replicas share it.

Atomicity:
    Consuming a token is read-modify-write. Doing that from Python across
    replicas races: two processes both read 1 token left, both decide they may
    proceed, and the bucket goes negative. The refill-and-consume therefore
    runs as a Lua script, which Redis executes atomically.

    The script also reads the clock from Redis (TIME) rather than from the
    caller, so one skewed replica cannot rewind a bucket's timestamp and mint
    itself extra tokens.

Memory:
    Every bucket key carries a TTL, so idle agents expire instead of
    accumulating. There is no LRU cap to tune as there is in the in-process
    limiter.

Requires: pip install nexus-a2a[redis], and Redis 7+ (the script calls TIME,
which needs effect-based replication).
"""

from __future__ import annotations

import logging
from typing import Any

from nexus_a2a.security.rate_limiter import (
    AbstractRateLimiter,
    RateLimitConfig,
    RateLimitError,
)

logger = logging.getLogger(__name__)

_KEY_PREFIX = "nexus_a2a:ratelimit:"

# Floor for how long an idle bucket is kept, in seconds. The real TTL is the
# larger of this and the time it takes a bucket to refill completely — expiring
# sooner than that would hand a caller a full bucket early.
_MIN_TTL_SECONDS = 60

# Refill, then try to take one token. Returns seconds to wait: 0 means the
# token was consumed. Keys: bucket. Args: rate, burst.
_CONSUME_SCRIPT = """
local key   = KEYS[1]
local rate  = tonumber(ARGV[1])
local burst = tonumber(ARGV[2])
local ttl   = tonumber(ARGV[3])

-- Redis's clock, not the caller's, so replica skew cannot rewind a bucket.
local t = redis.call('TIME')
local now = tonumber(t[1]) + (tonumber(t[2]) / 1000000)

local state = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(state[1])
local ts = tonumber(state[2])

if tokens == nil or ts == nil then
  -- A new bucket starts full, so a first burst goes straight through.
  tokens = burst
  ts = now
end

local elapsed = now - ts
if elapsed < 0 then
  elapsed = 0
end
tokens = math.min(burst, tokens + (elapsed * rate))

local retry_after = 0
if tokens >= 1 then
  tokens = tokens - 1
else
  retry_after = (1 - tokens) / rate
end

redis.call('HSET', key, 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', key, ttl)

return tostring(retry_after)
"""

# Refill without consuming — used by available_tokens().
_PEEK_SCRIPT = """
local key   = KEYS[1]
local rate  = tonumber(ARGV[1])
local burst = tonumber(ARGV[2])

local state = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(state[1])
local ts = tonumber(state[2])

if tokens == nil or ts == nil then
  return tostring(burst)
end

local t = redis.call('TIME')
local now = tonumber(t[1]) + (tonumber(t[2]) / 1000000)
local elapsed = now - ts
if elapsed < 0 then
  elapsed = 0
end

return tostring(math.min(burst, tokens + (elapsed * rate)))
"""


class RedisRateLimiter(AbstractRateLimiter):
    """
    Token bucket rate limiter backed by Redis, shared across replicas.

    A drop-in replacement for RateLimiter wherever one is accepted, including
    SecurityMiddleware:

        limiter = RedisRateLimiter(url="redis://localhost:6379")
        await limiter.connect()

        security = SecurityMiddleware(rate_limiter=limiter, ...)
        server = A2AServer(MyAgent, security=security)

        await limiter.disconnect()

    Or as an async context manager:

        async with RedisRateLimiter() as limiter:
            await limiter.check("http://caller:9001")

    Per-agent limits set with set_limit() are held in this process: they are
    operator configuration, identical on every replica, and keeping them local
    avoids a Redis round trip on the hot path. Only the bucket state — the part
    that must be shared — lives in Redis.

    Args:
        url:            Redis connection URL.
        default_config: Limit applied to agents with no specific config.
        db:             Redis database number.
        password:       Redis password, if required.
        key_prefix:     Prefix for all bucket keys.
        client:         An existing redis.asyncio client to use instead of
                        opening one. Lets several components share a pool, and
                        lets tests supply a fake. connect() still has to be
                        called, to register the Lua scripts.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        default_config: RateLimitConfig | None = None,
        db: int = 0,
        password: str | None = None,
        key_prefix: str = _KEY_PREFIX,
        client: Any = None,
    ) -> None:
        self._url = url
        self._injected_client = client
        self._default = default_config or RateLimitConfig()
        self._db = db
        self._password = password
        self._prefix = key_prefix

        self._configs: dict[str, RateLimitConfig] = {}
        self._redis: Any = None
        self._consume: Any = None
        self._peek: Any = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """
        Open the connection pool and register the Lua scripts.

        Raises:
            ImportError: redis is not installed (pip install nexus-a2a[redis]).
        """
        try:
            import redis.asyncio as aioredis
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ImportError(
                "RedisRateLimiter requires the redis package. "
                "Install it with: pip install nexus-a2a[redis]"
            ) from exc

        self._redis = self._injected_client or aioredis.from_url(
            self._url,
            db=self._db,
            password=self._password,
            decode_responses=True,
        )
        await self._redis.ping()
        self._consume = self._redis.register_script(_CONSUME_SCRIPT)
        self._peek = self._redis.register_script(_PEEK_SCRIPT)
        logger.info(
            "RedisRateLimiter connected: %s (db=%d)", self._url, self._db
        )

    async def disconnect(self) -> None:
        """
        Close the connection pool.

        An injected client belongs to whoever passed it in, so it is released
        rather than closed.
        """
        if self._redis is not None and self._injected_client is None:
            await self._redis.aclose()
        self._redis = None
        self._consume = None
        self._peek = None

    async def __aenter__(self) -> RedisRateLimiter:
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.disconnect()

    # ── Interface ─────────────────────────────────────────────────────────────

    @property
    def is_distributed(self) -> bool:
        """True — buckets are shared, so the limit holds across replicas."""
        return True

    def set_limit(self, agent_url: str, config: RateLimitConfig) -> None:
        """
        Set a custom limit for one agent.

        Unlike the in-process limiter this does NOT reset the agent's bucket:
        the bucket is shared, and resetting it from one replica would hand
        every other replica a full allowance.
        """
        url = agent_url.rstrip("/")
        self._configs[url] = config
        logger.info(
            "Rate limit set for %s: %.1f req/s, burst=%.0f",
            url,
            config.rate,
            config.burst,
        )

    def remove_limit(self, agent_url: str) -> None:
        """Remove a custom limit — the agent falls back to the default."""
        self._configs.pop(agent_url.rstrip("/"), None)

    def get_config(self, agent_url: str) -> RateLimitConfig:
        """Return the effective config for an agent."""
        return self._configs.get(agent_url.rstrip("/"), self._default)

    async def check(self, agent_url: str) -> None:
        """
        Consume one token for agent_url, atomically across replicas.

        Raises:
            RateLimitError: The shared bucket is empty.
            RuntimeError:   connect() has not been called.
        """
        url = agent_url.rstrip("/")
        config = self.get_config(url)
        script = self._require_script(self._consume)

        raw = await script(
            keys=[self._key(url)],
            args=[config.rate, config.burst, self._ttl_for(config)],
        )
        retry_after = float(raw)

        if retry_after > 0:
            logger.warning(
                "Rate limit exceeded for %s (retry in %.2fs)", url, retry_after
            )
            raise RateLimitError(agent_url, retry_after)

    async def available_tokens(self, agent_url: str) -> float:
        """Current token count for an agent, after refill. Consumes nothing."""
        url = agent_url.rstrip("/")
        config = self.get_config(url)
        script = self._require_script(self._peek)
        raw = await script(keys=[self._key(url)], args=[config.rate, config.burst])
        return float(raw)

    # ── Maintenance ───────────────────────────────────────────────────────────

    async def reset(self, agent_url: str) -> None:
        """Drop an agent's shared bucket, so its next request starts full."""
        client = self._require_client()
        await client.delete(self._key(agent_url.rstrip("/")))

    async def reset_all(self) -> None:
        """Drop every bucket under this prefix. Mainly for tests."""
        client = self._require_client()
        keys = [key async for key in client.scan_iter(match=f"{self._prefix}*")]
        if keys:
            await client.delete(*keys)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _key(self, url: str) -> str:
        return f"{self._prefix}{url}"

    @staticmethod
    def _ttl_for(config: RateLimitConfig) -> int:
        """
        How long an idle bucket is kept.

        Must be at least the time a bucket needs to refill completely —
        expiring earlier would recreate it full and grant a free burst.
        """
        refill_seconds = config.burst / config.rate
        return max(_MIN_TTL_SECONDS, int(refill_seconds * 2) + 1)

    def _require_client(self) -> Any:
        if self._redis is None:
            raise RuntimeError(
                "RedisRateLimiter is not connected. Call await limiter.connect() "
                "first, or use it as an async context manager."
            )
        return self._redis

    def _require_script(self, script: Any) -> Any:
        if script is None:
            raise RuntimeError(
                "RedisRateLimiter is not connected. Call await limiter.connect() "
                "first, or use it as an async context manager."
            )
        return script


__all__ = ["RedisRateLimiter"]
