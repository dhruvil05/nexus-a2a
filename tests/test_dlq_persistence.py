"""
tests/test_dlq_persistence.py

Dead Letter Queue durability.

Through 1.7.0 the DLQ lived in a process-local dict, so a crash or redeploy
lost every failed task in it — the one piece of state you least want to lose,
because an entry exists precisely because work did NOT complete.

A "restart" here means a fresh DeadLetterQueue built over a store that
outlives it. That is exactly what a real restart looks like to the store.
"""

from __future__ import annotations

import pytest

from nexus_a2a.core.dead_letter import DeadLetterQueue, DLQEntry
from nexus_a2a.models.task import Message, Task, TaskState
from nexus_a2a.network import AgentNetwork
from nexus_a2a.storage.dlq_store import (
    AbstractDLQStore,
    InMemoryDLQStore,
    RedisDLQStore,
)


def failed_task(text: str = "work") -> Task:
    task = Task.create(initial_message=Message.user_text(text))
    task.transition(TaskState.WORKING)
    task.transition(TaskState.FAILED, error="boom")
    return task


async def completing_runner(url: str, msg: Message) -> Task:
    task = Task.create(initial_message=msg)
    task.transition(TaskState.WORKING)
    task.transition(TaskState.COMPLETED)
    return task


async def failing_runner(url: str, msg: Message) -> Task:
    raise RuntimeError("still broken")


# ── Entry serialisation ───────────────────────────────────────────────────────


class TestEntrySerialisation:
    def test_round_trip_preserves_every_field(self):
        entry = DLQEntry(
            task=failed_task("original input"),
            error="boom",
            agent_url="http://a:1",
            skill_id="s1",
            retry_count=2,
            last_retry_at=123.0,
            replayed=True,
        )
        revived = DLQEntry.from_storage_dict(entry.to_storage_dict())

        assert revived.task_id == entry.task_id
        assert revived.error == entry.error
        assert revived.agent_url == entry.agent_url
        assert revived.skill_id == entry.skill_id
        assert revived.retry_count == entry.retry_count
        assert revived.last_retry_at == entry.last_retry_at
        assert revived.replayed is True
        assert revived.failed_at == entry.failed_at

    def test_round_trip_preserves_the_original_message(self):
        """Replay is impossible without it — this is the point of storing."""
        entry = DLQEntry(task=failed_task("replay me"), error="boom")
        revived = DLQEntry.from_storage_dict(entry.to_storage_dict())
        assert revived.original_message is not None
        assert revived.original_message.text() == "replay me"

    def test_round_trip_preserves_task_state(self):
        entry = DLQEntry(task=failed_task(), error="boom")
        revived = DLQEntry.from_storage_dict(entry.to_storage_dict())
        assert revived.task.state == TaskState.FAILED

    def test_to_dict_stays_a_summary(self):
        """The display shape must not start carrying the whole task."""
        entry = DLQEntry(task=failed_task(), error="boom")
        assert "task" not in entry.to_dict()
        assert entry.to_dict()["task_id"] == entry.task_id

    def test_storage_dict_is_json_safe(self):
        import json

        json.dumps(DLQEntry(task=failed_task(), error="boom").to_storage_dict())


# ── InMemoryDLQStore ──────────────────────────────────────────────────────────


class TestInMemoryDLQStore:
    async def test_save_and_get(self):
        store = InMemoryDLQStore()
        entry = DLQEntry(task=failed_task(), error="boom")
        await store.save(entry)
        assert (await store.get(entry.task_id)).task_id == entry.task_id

    async def test_get_missing_returns_none(self):
        assert await InMemoryDLQStore().get("nope") is None

    async def test_delete(self):
        store = InMemoryDLQStore()
        entry = DLQEntry(task=failed_task(), error="boom")
        await store.save(entry)
        await store.delete(entry.task_id)
        assert await store.get(entry.task_id) is None

    async def test_delete_missing_is_silent(self):
        await InMemoryDLQStore().delete("nope")

    async def test_list_all_and_count(self):
        store = InMemoryDLQStore()
        for i in range(3):
            await store.save(DLQEntry(task=failed_task(f"t{i}"), error="boom"))
        assert len(await store.list_all()) == 3
        assert await store.count() == 3

    async def test_clear(self):
        store = InMemoryDLQStore()
        await store.save(DLQEntry(task=failed_task(), error="boom"))
        await store.clear()
        assert await store.count() == 0

    async def test_save_overwrites_same_task(self):
        store = InMemoryDLQStore()
        task = failed_task()
        await store.save(DLQEntry(task=task, error="first"))
        await store.save(DLQEntry(task=task, error="second"))
        assert await store.count() == 1
        assert (await store.get(task.id)).error == "second"


# ── Durability ────────────────────────────────────────────────────────────────


class TestDurability:
    async def test_entries_survive_a_restart(self):
        store = InMemoryDLQStore()
        dlq = DeadLetterQueue(store=store)
        await dlq.capture(failed_task("one"), agent_url="http://a:1")
        await dlq.capture(failed_task("two"), agent_url="http://a:1")

        revived = DeadLetterQueue(store=store)
        assert revived.count() == 0, "local view starts empty"
        assert await revived.load() == 2
        assert revived.count() == 2

    async def test_revived_entry_is_replayable(self):
        """The property that actually matters after a crash."""
        store = InMemoryDLQStore()
        dlq = DeadLetterQueue(store=store)
        entry = await dlq.capture(failed_task("replay me"), agent_url="http://a:1")

        revived = DeadLetterQueue(store=store, runner=completing_runner)
        await revived.load()
        result = await revived.replay(entry.task_id)
        assert result.succeeded

    async def test_replay_state_is_persisted(self):
        store = InMemoryDLQStore()
        dlq = DeadLetterQueue(store=store, runner=completing_runner)
        entry = await dlq.capture(failed_task(), agent_url="http://a:1")
        await dlq.replay(entry.task_id)
        assert (await store.get(entry.task_id)).replayed is True

    async def test_failed_replay_persists_retry_count(self):
        store = InMemoryDLQStore()
        dlq = DeadLetterQueue(store=store, runner=failing_runner, retry_delay=0.01)
        entry = await dlq.capture(failed_task(), agent_url="http://a:1")
        await dlq.replay(entry.task_id)
        stored = await store.get(entry.task_id)
        assert stored.retry_count == 1
        assert stored.replayed is False

    async def test_eviction_removes_from_the_store_too(self):
        """Otherwise load() resurrects entries the queue already dropped."""
        store = InMemoryDLQStore()
        dlq = DeadLetterQueue(store=store, max_queue_size=2)
        for i in range(4):
            await dlq.capture(failed_task(f"t{i}"), agent_url="http://a:1")
        assert dlq.count() == 2
        assert await store.count() == 2

    async def test_is_durable_reports_the_truth(self):
        assert DeadLetterQueue(store=InMemoryDLQStore()).is_durable is True
        assert DeadLetterQueue().is_durable is False

    async def test_store_property_exposes_the_backend(self):
        store = InMemoryDLQStore()
        assert DeadLetterQueue(store=store).store is store

    async def test_refresh_is_load(self):
        store = InMemoryDLQStore()
        dlq = DeadLetterQueue(store=store)
        await dlq.capture(failed_task(), agent_url="http://a:1")

        other = DeadLetterQueue(store=store)
        assert await other.refresh() == 1


# ── Purging ───────────────────────────────────────────────────────────────────


class TestPurging:
    async def test_purge_removes_from_both(self):
        store = InMemoryDLQStore()
        dlq = DeadLetterQueue(store=store, runner=completing_runner)
        entry = await dlq.capture(failed_task(), agent_url="http://a:1")
        await dlq.replay(entry.task_id)

        assert await dlq.purge_replayed() == 1
        assert dlq.count() == 0
        assert await store.count() == 0

    async def test_purge_keeps_unreplayed_entries(self):
        store = InMemoryDLQStore()
        dlq = DeadLetterQueue(store=store)
        await dlq.capture(failed_task(), agent_url="http://a:1")
        assert await dlq.purge_replayed() == 0
        assert await store.count() == 1

    async def test_purge_works_without_a_store(self):
        dlq = DeadLetterQueue(runner=completing_runner)
        entry = await dlq.capture(failed_task(), agent_url="http://a:1")
        await dlq.replay(entry.task_id)
        assert await dlq.purge_replayed() == 1

    async def test_sync_clear_leaves_the_store_alone(self):
        """
        clear_replayed() cannot await, so it only touches the local view —
        entries come back on load(). That is why purge_replayed() exists.
        """
        store = InMemoryDLQStore()
        dlq = DeadLetterQueue(store=store, runner=completing_runner)
        entry = await dlq.capture(failed_task(), agent_url="http://a:1")
        await dlq.replay(entry.task_id)

        assert dlq.clear_replayed() == 1
        assert dlq.count() == 0
        assert await store.count() == 1
        assert await DeadLetterQueue(store=store).load() == 1


# ── Existing behaviour is unchanged without a store ───────────────────────────


class TestNoStoreIsUnchanged:
    async def test_capture_and_count(self):
        dlq = DeadLetterQueue()
        await dlq.capture(failed_task(), agent_url="http://a:1")
        assert dlq.count() == 1
        assert dlq.pending_count() == 1

    async def test_load_is_a_no_op(self):
        assert await DeadLetterQueue().load() == 0

    async def test_eviction_still_bounds_the_queue(self):
        dlq = DeadLetterQueue(max_queue_size=2)
        for i in range(5):
            await dlq.capture(failed_task(f"t{i}"), agent_url="http://a:1")
        assert dlq.count() == 2

    async def test_sync_accessors_still_work(self):
        """AgentServer's /dlq and /metrics call these synchronously."""
        dlq = DeadLetterQueue(store=InMemoryDLQStore())
        await dlq.capture(failed_task(), agent_url="http://a:1")
        assert dlq.count() == 1
        assert len(dlq.all_entries()) == 1
        assert len(dlq.pending_entries()) == 1
        assert dlq.get_entry(dlq.all_entries()[0].task_id) is not None
        assert dlq.summary()["total"] == 1


# ── Store failures are contained ──────────────────────────────────────────────


class BrokenStore(InMemoryDLQStore):
    async def save(self, entry: DLQEntry) -> None:
        raise RuntimeError("store is down")

    async def delete(self, task_id: str) -> None:
        raise RuntimeError("store is down")


class TestStoreFailuresAreContained:
    async def test_capture_survives_a_broken_store(self):
        """A store outage must not lose the failure from the local view too."""
        dlq = DeadLetterQueue(store=BrokenStore())
        entry = await dlq.capture(failed_task(), agent_url="http://a:1")
        assert dlq.count() == 1
        assert dlq.get_entry(entry.task_id) is not None

    async def test_purge_survives_a_broken_store(self):
        dlq = DeadLetterQueue(store=BrokenStore(), runner=completing_runner)
        entry = await dlq.capture(failed_task(), agent_url="http://a:1")
        await dlq.replay(entry.task_id)
        assert await dlq.purge_replayed() == 1


# ── Network wiring ────────────────────────────────────────────────────────────


class TestNetworkWiring:
    def test_network_defaults_to_no_store(self):
        assert AgentNetwork().dead_letter_queue.is_durable is False

    def test_network_accepts_a_store(self):
        store = InMemoryDLQStore()
        network = AgentNetwork(dlq_store=store)
        assert network.dead_letter_queue.is_durable is True
        assert network.dead_letter_queue.store is store


# ── Redis backend (no server required) ────────────────────────────────────────


class TestRedisDLQStoreContract:
    def test_implements_the_interface(self):
        assert issubclass(RedisDLQStore, AbstractDLQStore)

    def test_operations_require_a_connection(self):
        store = RedisDLQStore()
        with pytest.raises(RuntimeError, match="not connected"):
            store._require_client()

    def test_key_is_namespaced(self):
        assert RedisDLQStore()._key("abc") == "nexus_a2a:dlq:abc"

    def test_key_prefix_is_configurable(self):
        store = RedisDLQStore(key_prefix="myapp:dlq:")
        assert store._key("abc") == "myapp:dlq:abc"

    def test_unreadable_record_is_skipped_not_fatal(self):
        """One corrupt row must not make the whole queue unreplayable."""
        assert RedisDLQStore._decode("{not json") is None

    def test_decode_rebuilds_an_entry(self):
        import json

        entry = DLQEntry(task=failed_task("x"), error="boom")
        decoded = RedisDLQStore._decode(json.dumps(entry.to_storage_dict()))
        assert decoded is not None
        assert decoded.task_id == entry.task_id
