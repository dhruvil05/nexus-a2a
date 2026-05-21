"""
tests/integration/test_failure_recovery.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Real failure and recovery tests. No mocking. asyncio_mode=auto.

VERIFIED API (from real dead_letter.py source):
  DLQEntry fields: task (Task), error (str), failed_at (float),
                   agent_url, skill_id, retry_count, last_retry_at, replayed
  DLQEntry.task_id    → property: task.id
  DLQEntry.original_message → property: task.history[0] if history else None

  DeadLetterQueue.capture(task, agent_url=None, skill_id=None) → DLQEntry (async)
  DeadLetterQueue.all_entries()     → list[DLQEntry]
  DeadLetterQueue.pending_entries() → list[DLQEntry] (replayed=False)
  DeadLetterQueue.count()           → int
  DeadLetterQueue.pending_count()   → int
  DeadLetterQueue.summary()         → dict

  Task.create(initial_message, skill_id=None, context_id=None) → Task
  Task.transition(new_state, error=None) → None (mutates in place)
  TaskState.FAILED requires error param in transition()
  Message.user_text(text) → Message shortcut
"""

from __future__ import annotations

import asyncio

import pytest

from nexus_a2a.models.task import Message, Task, TaskState
from tests.integration.conftest import AgentServer, get_free_port, make_agent_app


@pytest.fixture
async def input_required_agent():
    port = get_free_port()
    server = AgentServer(
        make_agent_app("InputAgent", "Requires input.",
                       [{"id": "ask", "name": "Ask", "description": "Asks for input."}],
                       require_input=True),
        port,
    )
    await server.start()
    yield server
    await server.stop()


async def test_failed_task_captured_in_dlq(failing_task_agent: AgentServer) -> None:
    """A failed Task (state=failed) can be captured into DLQ via capture()."""
    from nexus_a2a.core.dead_letter import DeadLetterQueue
    from nexus_a2a.transport.http_client import A2AHttpClient

    dlq = DeadLetterQueue()

    async with A2AHttpClient(failing_task_agent.url) as client:
        task = await client.send_message(Message.user_text("fail me"))

    assert task.state == TaskState.FAILED

    # Task must be in FAILED state before capture
    # capture() reads task.error for the DLQEntry.error field
    # If task.error is None, it uses "Unknown error"
    entry = await dlq.capture(task, agent_url=failing_task_agent.url, skill_id="fail")

    assert dlq.count() == 1
    assert entry.task_id == task.id
    assert entry.agent_url == failing_task_agent.url
    assert entry.skill_id == "fail"
    assert entry.error is not None


async def test_dlq_all_entries_returns_captured_tasks(failing_task_agent: AgentServer) -> None:
    """all_entries() returns list of all DLQEntry objects."""
    from nexus_a2a.core.dead_letter import DeadLetterQueue
    from nexus_a2a.transport.http_client import A2AHttpClient

    dlq = DeadLetterQueue()

    async with A2AHttpClient(failing_task_agent.url) as client:
        t1 = await client.send_message(Message.user_text("fail 1"))
        t2 = await client.send_message(Message.user_text("fail 2"))

    await dlq.capture(t1, agent_url=failing_task_agent.url)
    await dlq.capture(t2, agent_url=failing_task_agent.url)

    entries = dlq.all_entries()
    assert len(entries) == 2
    ids = {e.task_id for e in entries}
    assert t1.id in ids
    assert t2.id in ids


async def test_dlq_pending_count(failing_task_agent: AgentServer) -> None:
    """pending_count() tracks non-replayed entries."""
    from nexus_a2a.core.dead_letter import DeadLetterQueue
    from nexus_a2a.transport.http_client import A2AHttpClient

    dlq = DeadLetterQueue()
    assert dlq.count() == 0
    assert dlq.pending_count() == 0

    async with A2AHttpClient(failing_task_agent.url) as client:
        task = await client.send_message(Message.user_text("count me"))

    await dlq.capture(task)
    assert dlq.count() == 1
    assert dlq.pending_count() == 1


async def test_dlq_filter_by_skill(failing_task_agent: AgentServer) -> None:
    """DLQ entries can be filtered by skill_id using all_entries()."""
    from nexus_a2a.core.dead_letter import DeadLetterQueue
    from nexus_a2a.transport.http_client import A2AHttpClient

    dlq = DeadLetterQueue()

    async with A2AHttpClient(failing_task_agent.url) as client:
        t_web = await client.send_message(Message.user_text("web"))
        t_sum = await client.send_message(Message.user_text("sum"))
        t_web2 = await client.send_message(Message.user_text("web2"))

    await dlq.capture(t_web, skill_id="web_search")
    await dlq.capture(t_sum, skill_id="summarize")
    await dlq.capture(t_web2, skill_id="web_search")

    assert dlq.count() == 3

    web_entries = [e for e in dlq.all_entries() if e.skill_id == "web_search"]
    sum_entries = [e for e in dlq.all_entries() if e.skill_id == "summarize"]

    assert len(web_entries) == 2
    assert len(sum_entries) == 1


async def test_dlq_capture_increments_count() -> None:
    """capture() on manually-created failed Tasks increments count."""
    from nexus_a2a.core.dead_letter import DeadLetterQueue

    dlq = DeadLetterQueue()
    assert dlq.count() == 0

    for i in range(3):
        # SUBMITTED → WORKING → FAILED (direct SUBMITTED→FAILED not allowed)
        msg = Message.user_text(f"task {i}")
        task = Task.create(initial_message=msg)
        task.transition(TaskState.WORKING)
        task.transition(TaskState.FAILED, error=f"simulated failure {i}")

        await dlq.capture(task, agent_url="http://fake-agent:9999")

    assert dlq.count() == 3
    assert dlq.pending_count() == 3


async def test_input_required_task_state(input_required_agent: AgentServer) -> None:
    """Agent returns INPUT_REQUIRED state when it needs human input."""
    from nexus_a2a.transport.http_client import A2AHttpClient

    async with A2AHttpClient(input_required_agent.url) as client:
        task = await client.send_message(Message.user_text("start"))

    assert task.state == TaskState.INPUT_REQUIRED


async def test_graceful_shutdown_completes_tasks(echo_agent: AgentServer) -> None:
    """Tasks complete while server runs; server is unreachable after stop."""
    from nexus_a2a.transport.http_client import A2AHttpClient

    async with A2AHttpClient(echo_agent.url) as client:
        task = await client.send_message(Message.user_text("finish me"))

    assert task.state == TaskState.COMPLETED

    await echo_agent.stop()

    import httpx
    with pytest.raises((
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.RemoteProtocolError,
        httpx.ReadError,
        OSError,
    )):
        async with httpx.AsyncClient(timeout=3.0) as client:
            await client.get(f"{echo_agent.url}/health")


async def test_multiple_concurrent_tasks(echo_agent: AgentServer) -> None:
    """10 concurrent tasks all complete successfully."""
    from nexus_a2a.transport.http_client import A2AHttpClient

    async def send_one(i: int) -> Task:
        async with A2AHttpClient(echo_agent.url) as client:
            return await client.send_message(Message.user_text(f"msg-{i}"))

    results = await asyncio.gather(*[send_one(i) for i in range(10)])

    assert len(results) == 10
    assert all(r.state == TaskState.COMPLETED for r in results)