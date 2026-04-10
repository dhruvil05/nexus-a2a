"""
nexus_a2a/storage/redis_store.py

RedisTaskStore — a production-grade TaskStore backend backed by Redis.

Drop-in replacement for InMemoryTaskStore:
    manager = TaskManager(store=RedisTaskStore(url="redis://localhost:6379"))

Features vs InMemoryTaskStore:
  - Persistent: tasks survive process restarts.
  - Distributed: multiple processes/servers share the same task state.
  - TTL: tasks auto-expire after a configurable period.

Requires: pip install nexus-a2a[redis]
"""

from __future__ import annotations

import logging
from typing import Any

from nexus_a2a.models.task import Task
from nexus_a2a.storage.task_store import AbstractTaskStore

logger = logging.getLogger(__name__)

# Redis key prefix — keeps nexus tasks isolated from other app data
_KEY_PREFIX = "nexus_a2a:task:"

# Default TTL: 24 hours. Tasks older than this are auto-deleted by Redis.
_DEFAULT_TTL_SECONDS = 86_400


class RedisTaskStore(AbstractTaskStore):
    """
    Redis-backed TaskStore for production multi-process deployments.

    Usage:
        store   = RedisTaskStore(url="redis://localhost:6379", ttl=3600)
        manager = TaskManager(store=store)

        # Always call connect() before use, disconnect() on shutdown
        await store.connect()
        try:
            ...
        finally:
            await store.disconnect()

        # Or use as async context manager
        async with RedisTaskStore(url="redis://localhost:6379") as store:
            manager = TaskManager(store=store)

    Args:
        url:         Redis connection URL. Default: redis://localhost:6379
        ttl:         Seconds before a task key expires in Redis.
                     Default: 86400 (24 hours). Set to 0 to disable expiry.
        db:          Redis database number. Default: 0.
        password:    Redis password if authentication is required.
        key_prefix:  Prefix for all Redis keys. Default: "nexus_a2a:task:".
    """

    def __init__(
        self,
        url:        str = "redis://localhost:6379",
        ttl:        int = _DEFAULT_TTL_SECONDS,
        db:         int = 0,
        password:   str | None = None,
        key_prefix: str = _KEY_PREFIX,
    ) -> None:
        self._url        = url
        self._ttl        = ttl
        self._db         = db
        self._password   = password
        self._prefix     = key_prefix
        self._redis: Any = None   # set after connect()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """
        Open the Redis connection pool.
        Must be called before any store operations.

        Raises:
            ImportError:     If redis package is not installed.
            ConnectionError: If Redis server is unreachable.
        """
        try:
            import redis.asyncio as aioredis
        except ImportError as exc:
            raise ImportError(
                "redis package is required for RedisTaskStore. "
                "Install it with: pip install nexus-a2a[redis]"
            ) from exc

        self._redis = aioredis.from_url(
            self._url,
            db=self._db,
            password=self._password,
            decode_responses=True,
        )
        # Ping to verify the connection is alive
        await self._redis.ping()
        logger.info("RedisTaskStore: connected to %s (db=%d)", self._url, self._db)

    async def disconnect(self) -> None:
        """Close the Redis connection pool."""
        if self._redis:
            await self._redis.aclose()
            self._redis = None
            logger.info("RedisTaskStore: disconnected")

    # ── Context manager support ───────────────────────────────────────────────

    async def __aenter__(self) -> RedisTaskStore:
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.disconnect()

    # ── AbstractTaskStore interface ───────────────────────────────────────────

    async def save(self, task: Task) -> None:
        """
        Serialise and store a Task in Redis.
        Sets the TTL on every save so the clock resets on activity.
        """
        self._require_connected()
        key  = self._key(task.id)
        data = task.model_dump_json()

        if self._ttl > 0:
            await self._redis.setex(key, self._ttl, data)
        else:
            await self._redis.set(key, data)

        logger.debug("RedisTaskStore: saved task %s", task.id)

    async def get(self, task_id: str) -> Task | None:
        """
        Retrieve a Task from Redis by its ID.
        Returns None if the key does not exist or has expired.
        """
        self._require_connected()
        data = await self._redis.get(self._key(task_id))
        if data is None:
            return None
        return Task.model_validate_json(data)

    async def delete(self, task_id: str) -> None:
        """Delete a Task key from Redis."""
        self._require_connected()
        await self._redis.delete(self._key(task_id))
        logger.debug("RedisTaskStore: deleted task %s", task_id)

    async def list_all(self) -> list[Task]:
        """
        Retrieve all tasks stored under the key prefix.

        Note: Uses Redis SCAN (non-blocking) rather than KEYS
        to avoid blocking the server on large datasets.
        """
        self._require_connected()
        tasks: list[Task] = []
        pattern = f"{self._prefix}*"

        async for key in self._redis.scan_iter(pattern):
            data = await self._redis.get(key)
            if data:
                try:
                    tasks.append(Task.model_validate_json(data))
                except Exception as exc:
                    logger.warning(
                        "RedisTaskStore: could not deserialise key %s: %s",
                        key, exc,
                    )
        return tasks

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _key(self, task_id: str) -> str:
        """Build the Redis key for a task ID."""
        return f"{self._prefix}{task_id}"

    def _require_connected(self) -> None:
        """Raise if connect() has not been called yet."""
        if self._redis is None:
            raise RuntimeError(
                "RedisTaskStore is not connected. "
                "Call await store.connect() before using the store, "
                "or use it as: async with RedisTaskStore(...) as store:"
            )
