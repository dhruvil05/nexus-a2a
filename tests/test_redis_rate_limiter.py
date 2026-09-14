"""
tests/test_redis_rate_limiter.py

Distributed rate limiting.

The in-process RateLimiter keeps buckets in a local dict, so three replicas
configured for 10 req/s actually allow 30 — the limit scales with the
deployment, silently. RedisRateLimiter shares one bucket per agent so the limit
is the limit.

These tests run the real Lua script against fakeredis (which executes Lua via
lupa), so the atomic refill-and-consume is genuinely covered rather than
skipped for want of a server. Two limiter instances sharing one client stand in
for two replicas — that is exactly what they are to Redis.
"""

from __future__ import annotations

import asyncio

import pytest

from nexus_a2a.security.middleware import SecurityMiddleware
from nexus_a2a.security.rate_limiter import (
    AbstractRateLimiter,
    RateLimitConfig,
    RateLimiter,
    RateLimitError,
)
from nexus_a2a.security.redis_rate_limiter import (
    _MIN_TTL_SECONDS,
    RedisRateLimiter,
)

fakeredis = pytest.importorskip("fakeredis", reason="fakeredis[lua] not installed")

CALLER = "http://caller:9001"
OTHER = "http://other:9002"


@pytest.fixture
def fake_client():
    import fakeredis.aioredis as fr

    return fr.FakeRedis(decode_responses=True)


async def make_limiter(client, rate=1.0, burst=3.0) -> RedisRateLimiter:
    limiter = RedisRateLimiter(
        default_config=RateLimitConfig(rate=rate, burst=burst),
        client=client,
    )
    await limiter.connect()
    return limiter


# ── Bucket behaviour ──────────────────────────────────────────────────────────


class TestBucketBehaviour:
    async def test_new_bucket_starts_full(self, fake_client):
        limiter = await make_limiter(fake_client, burst=3)
        assert await limiter.available_tokens(CALLER) == 3.0

    async def test_burst_is_allowed_then_refused(self, fake_client):
        limiter = await make_limiter(fake_client, rate=1.0, burst=3)
        for _ in range(3):
            await limiter.check(CALLER)
        with pytest.raises(RateLimitError):
            await limiter.check(CALLER)

    async def test_each_check_consumes_one_token(self, fake_client):
        limiter = await make_limiter(fake_client, rate=0.0001, burst=5)
        await limiter.check(CALLER)
        assert await limiter.available_tokens(CALLER) == pytest.approx(4.0, abs=0.01)

    async def test_retry_after_is_reported(self, fake_client):
        limiter = await make_limiter(fake_client, rate=2.0, burst=1)
        await limiter.check(CALLER)
        with pytest.raises(RateLimitError) as exc:
            await limiter.check(CALLER)
        assert exc.value.retry_after == pytest.approx(0.5, abs=0.1)

    async def test_agents_have_separate_buckets(self, fake_client):
        limiter = await make_limiter(fake_client, rate=1.0, burst=1)
        await limiter.check(CALLER)
        await limiter.check(OTHER)  # must not be affected
        with pytest.raises(RateLimitError):
            await limiter.check(CALLER)

    async def test_tokens_refill_over_time(self, fake_client):
        limiter = await make_limiter(fake_client, rate=50.0, burst=1)
        await limiter.check(CALLER)
        await asyncio.sleep(0.1)  # ~5 tokens' worth at 50/s
        await limiter.check(CALLER)

    async def test_available_tokens_does_not_consume(self, fake_client):
        limiter = await make_limiter(fake_client, rate=0.0001, burst=3)
        before = await limiter.available_tokens(CALLER)
        await limiter.available_tokens(CALLER)
        after = await limiter.available_tokens(CALLER)
        assert before == pytest.approx(after, abs=0.01)

    async def test_is_allowed_does_not_raise(self, fake_client):
        limiter = await make_limiter(fake_client, rate=1.0, burst=1)
        assert await limiter.is_allowed(CALLER) is True
        assert await limiter.is_allowed(CALLER) is False


# ── The point: one bucket across replicas ─────────────────────────────────────


class TestSharedAcrossReplicas:
    async def test_two_replicas_share_one_bucket(self, fake_client):
        """The headline property. In-process limiters would allow 2x here."""
        replica_a = await make_limiter(fake_client, rate=0.0001, burst=2)
        replica_b = await make_limiter(fake_client, rate=0.0001, burst=2)

        await replica_a.check(CALLER)
        await replica_b.check(CALLER)
        with pytest.raises(RateLimitError):
            await replica_a.check(CALLER)
        with pytest.raises(RateLimitError):
            await replica_b.check(CALLER)

    async def test_in_process_limiters_do_not_share(self):
        """Contrast: this is the bug RedisRateLimiter exists to fix."""
        a = RateLimiter(RateLimitConfig(rate=0.0001, burst=1))
        b = RateLimiter(RateLimitConfig(rate=0.0001, burst=1))
        await a.check(CALLER)
        await b.check(CALLER)  # second "replica" grants a whole extra bucket
        with pytest.raises(RateLimitError):
            await a.check(CALLER)

    async def test_replica_sees_tokens_spent_elsewhere(self, fake_client):
        a = await make_limiter(fake_client, rate=0.0001, burst=5)
        b = await make_limiter(fake_client, rate=0.0001, burst=5)
        for _ in range(3):
            await a.check(CALLER)
        assert await b.available_tokens(CALLER) == pytest.approx(2.0, abs=0.05)

    async def test_concurrent_checks_never_over_grant(self, fake_client):
        """
        Atomicity. Twenty concurrent consumers against a 5-token bucket must
        yield exactly 5 successes — a read-modify-write from Python would
        let several through on the same token.
        """
        limiters = [await make_limiter(fake_client, rate=0.0001, burst=5)
                    for _ in range(4)]
        results = await asyncio.gather(
            *(limiters[i % 4].is_allowed(CALLER) for i in range(20))
        )
        assert sum(results) == 5


# ── Configuration ─────────────────────────────────────────────────────────────


class TestConfiguration:
    async def test_default_config_applies(self, fake_client):
        limiter = await make_limiter(fake_client, rate=7.0, burst=9)
        assert limiter.get_config(CALLER).rate == 7.0
        assert limiter.get_config(CALLER).burst == 9

    async def test_set_limit_overrides_default(self, fake_client):
        limiter = await make_limiter(fake_client)
        limiter.set_limit(CALLER, RateLimitConfig(rate=2.0, burst=4))
        assert limiter.get_config(CALLER).burst == 4
        assert limiter.get_config(OTHER).burst == 3

    async def test_remove_limit_restores_default(self, fake_client):
        limiter = await make_limiter(fake_client, burst=3)
        limiter.set_limit(CALLER, RateLimitConfig(rate=2.0, burst=9))
        limiter.remove_limit(CALLER)
        assert limiter.get_config(CALLER).burst == 3

    async def test_set_limit_does_not_reset_the_shared_bucket(self, fake_client):
        """
        Resetting from one replica would hand every other replica a fresh
        allowance — the in-process limiter can reset safely, this one cannot.
        """
        limiter = await make_limiter(fake_client, rate=0.0001, burst=2)
        await limiter.check(CALLER)
        limiter.set_limit(CALLER, RateLimitConfig(rate=0.0001, burst=2))
        assert await limiter.available_tokens(CALLER) == pytest.approx(1.0, abs=0.05)

    async def test_trailing_slash_is_normalised(self, fake_client):
        limiter = await make_limiter(fake_client, rate=0.0001, burst=1)
        await limiter.check("http://caller:9001/")
        with pytest.raises(RateLimitError):
            await limiter.check("http://caller:9001")


# ── Maintenance ───────────────────────────────────────────────────────────────


class TestMaintenance:
    async def test_reset_refills_one_agent(self, fake_client):
        limiter = await make_limiter(fake_client, rate=0.0001, burst=1)
        await limiter.check(CALLER)
        await limiter.reset(CALLER)
        await limiter.check(CALLER)

    async def test_reset_all_clears_every_bucket(self, fake_client):
        limiter = await make_limiter(fake_client, rate=0.0001, burst=1)
        await limiter.check(CALLER)
        await limiter.check(OTHER)
        await limiter.reset_all()
        await limiter.check(CALLER)
        await limiter.check(OTHER)

    async def test_ttl_is_at_least_the_floor(self):
        assert RedisRateLimiter._ttl_for(RateLimitConfig(rate=100, burst=1)) == (
            _MIN_TTL_SECONDS
        )

    async def test_ttl_covers_a_full_refill(self):
        """Expiring sooner would recreate the bucket full and grant a free burst."""
        config = RateLimitConfig(rate=1.0, burst=600)
        assert RedisRateLimiter._ttl_for(config) > 600


# ── Contract ──────────────────────────────────────────────────────────────────


class TestContract:
    def test_implements_the_shared_interface(self):
        assert issubclass(RedisRateLimiter, AbstractRateLimiter)
        assert issubclass(RateLimiter, AbstractRateLimiter)

    def test_reports_itself_distributed(self):
        assert RedisRateLimiter().is_distributed is True
        assert RateLimiter().is_distributed is False

    async def test_operations_require_connect(self):
        limiter = RedisRateLimiter()
        with pytest.raises(RuntimeError, match="not connected"):
            await limiter.check(CALLER)

    async def test_reset_requires_connect(self):
        with pytest.raises(RuntimeError, match="not connected"):
            await RedisRateLimiter().reset(CALLER)

    def test_keys_are_namespaced(self):
        assert RedisRateLimiter()._key(CALLER) == f"nexus_a2a:ratelimit:{CALLER}"

    def test_key_prefix_is_configurable(self):
        limiter = RedisRateLimiter(key_prefix="app:rl:")
        assert limiter._key(CALLER) == f"app:rl:{CALLER}"

    async def test_context_manager_connects_and_releases(self, fake_client):
        async with RedisRateLimiter(client=fake_client) as limiter:
            await limiter.check(CALLER)
        assert limiter._redis is None

    async def test_injected_client_is_not_closed(self, fake_client):
        limiter = await make_limiter(fake_client)
        await limiter.disconnect()
        # The caller still owns it, so it must still work.
        assert await fake_client.ping() is True


# ── Drop-in for SecurityMiddleware ────────────────────────────────────────────


class TestMiddlewareIntegration:
    async def test_accepted_by_security_middleware(self, fake_client):
        limiter = await make_limiter(fake_client, rate=0.0001, burst=1)
        security = SecurityMiddleware(rate_limiter=limiter)

        await security.authorize({"X-Nexus-Caller": CALLER})
        with pytest.raises(RateLimitError):
            await security.authorize({"X-Nexus-Caller": CALLER})

    async def test_reported_in_the_summary(self, fake_client):
        limiter = await make_limiter(fake_client)
        assert SecurityMiddleware(rate_limiter=limiter).summary()["rate_limit"] is True
