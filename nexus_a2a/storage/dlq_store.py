"""
nexus_a2a/storage/dlq_store.py

DLQStore — persists Dead Letter Queue entries.

Through 1.7.0 the DLQ lived entirely in a process-local dict, so a crash or a
redeploy lost every failed task in it. That is the one piece of state in the
library you least want to lose: a DLQ entry exists precisely because work did
NOT complete, and replaying it is the only way that work ever happens.

Backends mirror the TaskStore family:
    InMemoryDLQStore — zero config, the default, not durable.
    RedisDLQStore    — durable and shared across processes.

The DeadLetterQueue writes through to its store and keeps a local view for the
synchronous accessors (count(), pending_entries(), summary()) that AgentServer
and the CLI already depend on. Call load() at startup to rehydrate that view
from the store after a restart.
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nexus_a2a.core.dead_letter import DLQEntry

logger = logging.getLogger(__name__)

# Redis key prefix — keeps DLQ entries isolated from tasks and other app data.
_KEY_PREFIX = "nexus_a2a:dlq:"

# Default TTL: 7 days. A failed task nobody replayed in a week is unlikely to
# be replayed at all, and this stops the queue growing without bound.
_DEFAULT_TTL_SECONDS = 604_800


# ── Abstract interface ────────────────────────────────────────────────────────


class AbstractDLQStore(ABC):
    """
    Interface every DLQ backend must implement.

    DeadLetterQueue depends only on this, never on a concrete class, so
    swapping durability on is a one-line change.
    """

    @abstractmethod
    async def save(self, entry: DLQEntry) -> None:
        """Persist an entry (create or overwrite), keyed by its task id."""

    @abstractmethod
    async def get(self, task_id: str) -> DLQEntry | None:
        """Return the entry for a task id, or None if absent."""

    @abstractmethod
    async def delete(self, task_id: str) -> None:
        """Remove an entry. Silently succeeds if it is not there."""

    @abstractmethod
    async def list_all(self) -> list[DLQEntry]:
        """Return every entry currently stored."""

    @abstractmethod
    async def clear(self) -> None:
        """Remove all entries."""

    async def count(self) -> int:
        """Number of entries stored. Backends may override for efficiency."""
        return len(await self.list_all())


# ── In-memory implementation (the default) ───────────────────────────────────


class InMemoryDLQStore(AbstractDLQStore):
    """
    Holds entries in a plain dict — no external dependencies.

    Identical in behaviour to the pre-1.8.0 DLQ: entries are lost when the
    process exits. Use RedisDLQStore when losing them matters.
    """

    def __init__(self) -> None:
        self._store: dict[str, DLQEntry] = {}
        self._lock = asyncio.Lock()

    async def save(self, entry: DLQEntry) -> None:
        async with self._lock:
            self._store[entry.task_id] = entry

    async def get(self, task_id: str) -> DLQEntry | None:
        async with self._lock:
            return self._store.get(task_id)

    async def delete(self, task_id: str) -> None:
        async with self._lock:
            self._store.pop(task_id, None)

    async def list_all(self) -> list[DLQEntry]:
        async with self._lock:
            return list(self._store.values())

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()

    async def count(self) -> int:
        async with self._lock:
            return len(self._store)


# ── Redis implementation ──────────────────────────────────────────────────────


class RedisDLQStore(AbstractDLQStore):
    """
    Redis-backed DLQ for deployments where a lost failure is a lost job.

    Usage:
        store = RedisDLQStore(url="redis://localhost:6379")
        await store.connect()
        dlq = DeadLetterQueue(store=store)
        await dlq.load()          # rehydrate after a restart
        ...
        await store.disconnect()

        # Or as an async context manager
        async with RedisDLQStore() as store:
            dlq = DeadLetterQueue(store=store)

    Args:
        url:        Redis connection URL.
        ttl:        Seconds before an entry expires. Default: 7 days.
                    Set to 0 to keep entries until they are deleted.
        db:         Redis database number.
        password:   Redis password, if required.
        key_prefix: Prefix for all keys. Default: "nexus_a2a:dlq:".
        client:     An existing redis.asyncio client to reuse. It belongs to
                    its owner and is not closed by disconnect().

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
                "RedisDLQStore requires the redis package. "
                "Install it with: pip install nexus-a2a[redis]"
            ) from exc

        self._redis = aioredis.from_url(
            self._url,
            db=self._db,
            password=self._password,
            decode_responses=True,
        )
        await self._redis.ping()
        logger.info("RedisDLQStore connected: %s (db=%d)", self._url, self._db)

    async def disconnect(self) -> None:
        """Close the Redis connection pool. An injected client is left open."""
        if self._redis is not None and self._injected_client is None:
            await self._redis.aclose()
        self._redis = None

    async def __aenter__(self) -> RedisDLQStore:
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.disconnect()

    # ── Operations ────────────────────────────────────────────────────────────

    async def save(self, entry: DLQEntry) -> None:
        client = self._require_client()
        payload = json.dumps(entry.to_storage_dict())
        key = self._key(entry.task_id)
        if self._ttl > 0:
            await client.set(key, payload, ex=self._ttl)
        else:
            await client.set(key, payload)

    async def get(self, task_id: str) -> DLQEntry | None:
        client = self._require_client()
        raw = await client.get(self._key(task_id))
        return self._decode(raw) if raw else None

    async def delete(self, task_id: str) -> None:
        client = self._require_client()
        await client.delete(self._key(task_id))

    async def list_all(self) -> list[DLQEntry]:
        client = self._require_client()
        keys = [key async for key in client.scan_iter(match=f"{self._prefix}*")]
        if not keys:
            return []
        entries: list[DLQEntry] = []
        for raw in await client.mget(keys):
            if not raw:
                continue
            decoded = self._decode(raw)
            if decoded is not None:
                entries.append(decoded)
        return entries

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
                "RedisDLQStore is not connected. Call await store.connect() "
                "first, or use it as an async context manager."
            )
        return self._redis

    @staticmethod
    def _decode(raw: str) -> DLQEntry | None:
        """
        Rebuild an entry from its stored JSON.

        A single unreadable record must not take down a listing — it is
        skipped with a warning so the rest of the queue stays replayable.
        """
        from nexus_a2a.core.dead_letter import DLQEntry as _DLQEntry

        try:
            return _DLQEntry.from_storage_dict(json.loads(raw))
        except Exception:
            logger.exception("RedisDLQStore: skipping unreadable DLQ record")
            return None


__all__ = [
    "AbstractDLQStore",
    "InMemoryDLQStore",
    "RedisDLQStore",
]
