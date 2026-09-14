"""
nexus_a2a/storage/push_store.py

PushStore — persists where each task's updates should be POSTed.

Through 1.7.0 push targets lived in a dict on the A2AServer instance. A caller
that registered a webhook and then walked away — which is the entire point of
push notifications — got nothing if the agent restarted mid-task, and nothing
if the next request landed on a different replica. Both are the normal case in
production.

Backends mirror the TaskStore and DLQStore families:
    InMemoryPushStore — zero config, the default, not durable.
    RedisPushStore    — durable and shared across replicas.

A note on secrets: a push config carries the caller's `token`, which the
receiver uses to tell a genuine callback from a forged one. Persisting it means
it is at rest in the backing store, so point RedisPushStore at a Redis you
would be willing to keep credentials in — authenticated, and TLS if it is not
on loopback.
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from nexus_a2a.models.task import PushNotificationConfig

logger = logging.getLogger(__name__)

# Redis key prefix — keeps push targets isolated from tasks and DLQ entries.
_KEY_PREFIX = "nexus_a2a:push:"

# Default TTL: 24 hours. A target is deleted when its task reaches a terminal
# state, so this only catches tasks that never finish at all.
_DEFAULT_TTL_SECONDS = 86_400


# ── Abstract interface ────────────────────────────────────────────────────────


class AbstractPushStore(ABC):
    """
    Interface every push-target backend must implement.

    A2AServer depends only on this, so durability is a constructor argument
    rather than a rewrite.
    """

    @abstractmethod
    async def save(self, task_id: str, config: PushNotificationConfig) -> None:
        """Persist the target for a task (create or overwrite)."""

    @abstractmethod
    async def get(self, task_id: str) -> PushNotificationConfig | None:
        """Return a task's target, or None if none is registered."""

    @abstractmethod
    async def delete(self, task_id: str) -> None:
        """Remove a task's target. Silently succeeds if absent."""

    @abstractmethod
    async def clear(self) -> None:
        """Remove every target."""

    async def count(self) -> int:
        """How many targets are registered. Override for efficiency."""
        raise NotImplementedError


# ── In-memory implementation (the default) ───────────────────────────────────


class InMemoryPushStore(AbstractPushStore):
    """
    Holds targets in a plain dict — identical to the pre-1.8.0 behaviour.

    Targets are lost when the process exits, so a webhook registered before a
    restart is forgotten. Use RedisPushStore when that matters.
    """

    def __init__(self) -> None:
        self._store: dict[str, PushNotificationConfig] = {}
        self._lock = asyncio.Lock()

    async def save(self, task_id: str, config: PushNotificationConfig) -> None:
        async with self._lock:
            self._store[task_id] = config

    async def get(self, task_id: str) -> PushNotificationConfig | None:
        async with self._lock:
            return self._store.get(task_id)

    async def delete(self, task_id: str) -> None:
        async with self._lock:
            self._store.pop(task_id, None)

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()

    async def count(self) -> int:
        async with self._lock:
            return len(self._store)


# ── Redis implementation ──────────────────────────────────────────────────────


class RedisPushStore(AbstractPushStore):
    """
    Redis-backed push targets, so a registered webhook survives a restart and
    is visible to every replica.

    Usage:
        store = RedisPushStore(url="redis://localhost:6379")
        await store.connect()
        server = A2AServer(MyAgent, push_store=store, push_config=...)
        ...
        await store.disconnect()

    Args:
        url:        Redis connection URL.
        ttl:        Seconds before a target expires. Default: 24 hours. Targets
                    are deleted when their task finishes, so this only reaps
                    tasks that never reach a terminal state. 0 disables expiry.
        db:         Redis database number.
        password:   Redis password, if required.
        key_prefix: Prefix for all keys. Default: "nexus_a2a:push:".
        client:     An existing redis.asyncio client to reuse.

    Requires: pip install nexus-a2a[redis]
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        ttl: int = _DEFAULT_TTL_SECONDS,
        db: int = 0,
        password: str | None = None,
        key_prefix: str = _KEY_PREFIX,
        client: Any = None,
    ) -> None:
        self._url = url
        self._ttl = ttl
        self._db = db
        self._password = password
        self._prefix = key_prefix
        self._injected_client = client
        self._redis: Any = client

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """
        Open the Redis connection pool. Call before any store operation.

        Raises:
            ImportError: redis is not installed (pip install nexus-a2a[redis]).
        """
        if self._injected_client is not None:
            self._redis = self._injected_client
            await self._redis.ping()
            return

        try:
            import redis.asyncio as aioredis
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ImportError(
                "RedisPushStore requires the redis package. "
                "Install it with: pip install nexus-a2a[redis]"
            ) from exc

        self._redis = aioredis.from_url(
            self._url,
            db=self._db,
            password=self._password,
            decode_responses=True,
        )
        await self._redis.ping()
        logger.info("RedisPushStore connected: %s (db=%d)", self._url, self._db)

    async def disconnect(self) -> None:
        """Close the pool. An injected client belongs to its owner."""
        if self._redis is not None and self._injected_client is None:
            await self._redis.aclose()
        self._redis = None

    async def __aenter__(self) -> RedisPushStore:
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.disconnect()

    # ── Operations ────────────────────────────────────────────────────────────

    async def save(self, task_id: str, config: PushNotificationConfig) -> None:
        client = self._require_client()
        payload = json.dumps(config.model_dump(mode="json"))
        key = self._key(task_id)
        if self._ttl > 0:
            await client.set(key, payload, ex=self._ttl)
        else:
            await client.set(key, payload)

    async def get(self, task_id: str) -> PushNotificationConfig | None:
        client = self._require_client()
        raw = await client.get(self._key(task_id))
        if not raw:
            return None
        try:
            return PushNotificationConfig.model_validate(json.loads(raw))
        except Exception:
            # A corrupt record must not break delivery for every other task.
            logger.exception(
                "RedisPushStore: unreadable push target for task %s", task_id
            )
            return None

    async def delete(self, task_id: str) -> None:
        client = self._require_client()
        await client.delete(self._key(task_id))

    async def clear(self) -> None:
        client = self._require_client()
        keys = [key async for key in client.scan_iter(match=f"{self._prefix}*")]
        if keys:
            await client.delete(*keys)

    async def count(self) -> int:
        client = self._require_client()
        total = 0
        async for _ in client.scan_iter(match=f"{self._prefix}*"):
            total += 1
        return total

    # ── Internals ─────────────────────────────────────────────────────────────

    def _key(self, task_id: str) -> str:
        return f"{self._prefix}{task_id}"

    def _require_client(self) -> Any:
        if self._redis is None:
            raise RuntimeError(
                "RedisPushStore is not connected. Call await store.connect() "
                "first, or use it as an async context manager."
            )
        return self._redis


__all__ = [
    "AbstractPushStore",
    "InMemoryPushStore",
    "RedisPushStore",
]
