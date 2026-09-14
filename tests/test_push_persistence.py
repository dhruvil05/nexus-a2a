"""
tests/test_push_persistence.py

Push-target durability.

Registering a webhook is what a caller does precisely because they will NOT sit
and wait for the answer. Through 1.7.0 the target lived in a dict on the
A2AServer instance, so the caller got nothing if the agent restarted mid-task,
and nothing if the follow-up landed on a different replica. Both are the normal
case in production.

A "restart" here is a fresh A2AServer over a store that outlives it — which is
exactly what a restart looks like to the store.
"""

from __future__ import annotations

import httpx
import pytest

from nexus_a2a import agent
from nexus_a2a.core.a2a_server import A2AServer
from nexus_a2a.models.task import Message, NeedsInput, PushNotificationConfig, Task
from nexus_a2a.storage.push_store import (
    AbstractPushStore,
    InMemoryPushStore,
    RedisPushStore,
)
from nexus_a2a.transport.webhook import WebhookConfig, WebhookDispatcher

SERVER_URL = "http://push-agent:8001"
HOOK = "http://127.0.0.1:9999/hook"


@agent(name="Asker", description="Pauses for input.", push_notifications=True,
       url=SERVER_URL)
class AskerAgent:
    async def run(self, task: Task):
        if sum(1 for m in task.history if m.role == "user") == 1:
            return NeedsInput("Confirm?")
        return "finished"


class RecordingDispatcher(WebhookDispatcher):
    """Captures deliveries instead of making HTTP calls."""

    def __init__(self, config: WebhookConfig | None = None) -> None:
        super().__init__(config)
        self.sent: list[tuple[str, str]] = []  # (url, event)

    async def dispatch_silent(self, url, task, event="task_update"):  # type: ignore[override]
        self.sent.append((url, event))
        return None  # type: ignore[return-value]

    def events(self) -> list[str]:
        return [e for _, e in self.sent]


def build_server(store: AbstractPushStore, agent_cls: type = AskerAgent) -> A2AServer:
    server = A2AServer(
        agent_cls,
        push_config=WebhookConfig(allow_private_urls=True),
        push_store=store,
    )
    server.push = RecordingDispatcher(server._push_config)
    return server


async def send(server: A2AServer, text="go", **extra):
    params: dict = {"message": Message.user_text(text).model_dump(mode="json")}
    if "push_notification" in extra:
        params["pushNotification"] = extra.pop("push_notification")
    params.update(extra)
    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        resp = await c.post(
            "/", json={"jsonrpc": "2.0", "id": "1", "method": "message/send",
                       "params": params},
        )
    await server.drain_notifications()
    return resp


# ── InMemoryPushStore ─────────────────────────────────────────────────────────


class TestInMemoryPushStore:
    async def test_save_and_get(self):
        store = InMemoryPushStore()
        await store.save("t1", PushNotificationConfig(url=HOOK, token="tok"))
        got = await store.get("t1")
        assert got.url == HOOK
        assert got.token == "tok"

    async def test_get_missing_returns_none(self):
        assert await InMemoryPushStore().get("nope") is None

    async def test_delete(self):
        store = InMemoryPushStore()
        await store.save("t1", PushNotificationConfig(url=HOOK))
        await store.delete("t1")
        assert await store.get("t1") is None

    async def test_delete_missing_is_silent(self):
        await InMemoryPushStore().delete("nope")

    async def test_count_and_clear(self):
        store = InMemoryPushStore()
        for i in range(3):
            await store.save(f"t{i}", PushNotificationConfig(url=HOOK))
        assert await store.count() == 3
        await store.clear()
        assert await store.count() == 0

    async def test_save_overwrites(self):
        store = InMemoryPushStore()
        await store.save("t1", PushNotificationConfig(url=HOOK))
        await store.save("t1", PushNotificationConfig(url="https://other/hook"))
        assert (await store.get("t1")).url == "https://other/hook"
        assert await store.count() == 1


# ── Durability ────────────────────────────────────────────────────────────────


class TestDurability:
    async def test_target_survives_a_restart(self):
        store = InMemoryPushStore()
        first = build_server(store)
        resp = await send(first, push_notification={"url": HOOK})
        task_id = resp.json()["result"]["id"]
        assert first.push.events() == ["task_input_required"]

        # Process restarts: brand new server, same store.
        revived = build_server(store)
        assert await revived._push_targets.get(task_id) is not None

    async def test_a_task_finished_after_restart_still_notifies(self):
        """The property that actually matters to the caller who walked away."""
        store = InMemoryPushStore()
        first = build_server(store)
        resp = await send(first, push_notification={"url": HOOK})
        task_id = resp.json()["result"]["id"]

        revived = build_server(store)
        # The task itself is in the first server's task manager, so hand the
        # revived server the same one — only the push target is under test.
        revived.tasks = first.tasks
        await send(revived, "yes", taskId=task_id)
        assert revived.push.events() == ["task_completed"]

    async def test_without_a_store_the_target_is_lost(self):
        """Contrast: the pre-1.8.0 behaviour, still the default."""
        first = build_server(InMemoryPushStore())
        resp = await send(first, push_notification={"url": HOOK})
        task_id = resp.json()["result"]["id"]

        separate = build_server(InMemoryPushStore())  # a different store
        assert await separate._push_targets.get(task_id) is None

    async def test_terminal_task_releases_the_target(self):
        store = InMemoryPushStore()
        server = build_server(store)
        await send(server, push_notification={"url": HOOK})
        await send(server, "second", push_notification={"url": HOOK})
        # Both tasks paused, so both targets are still held.
        assert await store.count() == 2

    async def test_completed_task_target_is_gone_from_the_store(self):
        store = InMemoryPushStore()
        server = build_server(store)
        resp = await send(server, push_notification={"url": HOOK})
        task_id = resp.json()["result"]["id"]
        await send(server, "yes", taskId=task_id)
        assert await store.get(task_id) is None
        assert await store.count() == 0

    async def test_default_server_uses_an_in_memory_store(self):
        server = A2AServer(AskerAgent)
        assert isinstance(server._push_targets, InMemoryPushStore)


# ── Store failures are contained ──────────────────────────────────────────────


class BrokenStore(InMemoryPushStore):
    async def get(self, task_id: str):
        raise RuntimeError("store is down")

    async def delete(self, task_id: str) -> None:
        raise RuntimeError("store is down")


class TestStoreFailuresAreContained:
    async def test_unreadable_store_does_not_fail_the_task(self):
        """A store outage must not break the task it was reporting on."""
        server = build_server(BrokenStore())
        resp = await send(server, push_notification={"url": HOOK})
        assert resp.json()["result"]["state"] == "input_required"

    async def test_nothing_is_delivered_when_the_store_is_down(self):
        server = build_server(BrokenStore())
        await send(server, push_notification={"url": HOOK})
        assert server.push.sent == []


# ── Redis backend (contract, no server required) ──────────────────────────────


class TestRedisPushStoreContract:
    def test_implements_the_interface(self):
        assert issubclass(RedisPushStore, AbstractPushStore)

    def test_operations_require_a_connection(self):
        with pytest.raises(RuntimeError, match="not connected"):
            RedisPushStore()._require_client()

    def test_keys_are_namespaced(self):
        assert RedisPushStore()._key("abc") == "nexus_a2a:push:abc"

    def test_key_prefix_is_configurable(self):
        assert RedisPushStore(key_prefix="app:push:")._key("abc") == "app:push:abc"

    async def test_round_trip_through_a_fake_client(self):
        pytest.importorskip("fakeredis")
        import fakeredis.aioredis as fr

        async with RedisPushStore(client=fr.FakeRedis(decode_responses=True)) as store:
            await store.save("t1", PushNotificationConfig(url=HOOK, token="tok"))
            got = await store.get("t1")
            assert got.url == HOOK
            assert got.token == "tok"
            assert await store.count() == 1
            await store.delete("t1")
            assert await store.get("t1") is None

    async def test_corrupt_record_is_skipped_not_fatal(self):
        pytest.importorskip("fakeredis")
        import fakeredis.aioredis as fr

        client = fr.FakeRedis(decode_responses=True)
        async with RedisPushStore(client=client) as store:
            await client.set("nexus_a2a:push:bad", "{not json")
            assert await store.get("bad") is None

    async def test_clear_removes_everything(self):
        pytest.importorskip("fakeredis")
        import fakeredis.aioredis as fr

        async with RedisPushStore(client=fr.FakeRedis(decode_responses=True)) as store:
            for i in range(3):
                await store.save(f"t{i}", PushNotificationConfig(url=HOOK))
            await store.clear()
            assert await store.count() == 0

    async def test_injected_client_is_not_closed(self):
        pytest.importorskip("fakeredis")
        import fakeredis.aioredis as fr

        client = fr.FakeRedis(decode_responses=True)
        store = RedisPushStore(client=client)
        await store.connect()
        await store.disconnect()
        assert await client.ping() is True

    async def test_server_accepts_a_redis_store(self):
        pytest.importorskip("fakeredis")
        import fakeredis.aioredis as fr

        async with RedisPushStore(client=fr.FakeRedis(decode_responses=True)) as store:
            server = build_server(store)
            resp = await send(server, push_notification={"url": HOOK})
            assert resp.json()["result"]["state"] == "input_required"
            assert server.push.events() == ["task_input_required"]
