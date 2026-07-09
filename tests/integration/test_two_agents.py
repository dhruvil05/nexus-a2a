"""
tests/integration/test_two_agents.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Real end-to-end tests. No mocking. asyncio_mode=auto.

VERIFIED API (from real source):
  Message.user_text(text)          → Message (classmethod shortcut)
  Part(type=PartType.TEXT, content=...) ← content= not text=!
  Task.state                       → TaskState enum (str enum with values)
  Task.artifacts                   → list[Artifact]
  Task.history                     → list[Message]
  Artifact.parts                   → list[Part]
  Part.content                     → Any
  A2AHttpClient.send_message(msg)  → Task (Pydantic)
  A2AHttpClient.get_task(id)       → Task
  A2AHttpClient.cancel_task(id)    → Task
  AgentRegistry.register_url(url)  → AgentCard (async)
  AgentRegistry.list_all()         → list[AgentCard]
  AgentRegistry.find_by_skill(id)  → list[AgentCard]
  TaskState values: "submitted","working","completed","failed","cancelled","input_required"
"""

from __future__ import annotations

import httpx

from nexus_a2a.models.task import Message, TaskState
from tests.integration.conftest import AgentServer


async def test_send_message_and_get_completed_task(echo_agent: AgentServer) -> None:
    """send_message returns a completed Task with artifact containing echoed text."""
    from nexus_a2a.transport.http_client import A2AHttpClient

    msg = Message.user_text("hello world")

    async with A2AHttpClient(echo_agent.url) as client:
        task = await client.send_message(msg)

    assert task.state == TaskState.COMPLETED
    assert len(task.artifacts) > 0
    artifact_text = str(task.artifacts[0].parts[0].content)
    assert "hello world" in artifact_text
    assert "EchoAgent" in artifact_text


async def test_agent_card_fetched_correctly(echo_agent: AgentServer) -> None:
    """register_url returns AgentCard; list_all() gives it back."""
    from nexus_a2a.core.registry import AgentRegistry

    registry = AgentRegistry()
    card = await registry.register_url(echo_agent.url)

    assert card.name == "EchoAgent"
    assert card.version == "1.3.0"

    all_cards = registry.list_all()
    assert len(all_cards) == 1
    assert all_cards[0].name == "EchoAgent"


async def test_health_endpoint_returns_ok(echo_agent: AgentServer) -> None:
    """/health returns HTTP 200 with status ok."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{echo_agent.url}/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_tasks_get_returns_same_task(echo_agent: AgentServer) -> None:
    """get_task(id) returns the same Task as send_message."""
    from nexus_a2a.transport.http_client import A2AHttpClient

    async with A2AHttpClient(echo_agent.url) as client:
        task = await client.send_message(Message.user_text("ping"))
        fetched = await client.get_task(task.id)

    assert fetched.id == task.id
    assert fetched.state == TaskState.COMPLETED


async def test_tasks_cancel_returns_cancelled_task(echo_agent: AgentServer) -> None:
    """cancel_task returns a Task in CANCELLED state."""
    from nexus_a2a.transport.http_client import A2AHttpClient

    async with A2AHttpClient(echo_agent.url) as client:
        task = await client.send_message(Message.user_text("cancel me"))
        cancelled = await client.cancel_task(task.id)

    assert cancelled.state == TaskState.CANCELLED


async def test_two_agents_discoverable(
    echo_agent: AgentServer, summarizer_agent: AgentServer
) -> None:
    """Registry registers both agents; list_all() returns both AgentCards."""
    from nexus_a2a.core.registry import AgentRegistry

    registry = AgentRegistry()
    await registry.register_url(echo_agent.url)
    await registry.register_url(summarizer_agent.url)

    cards = registry.list_all()
    assert len(cards) == 2
    names = {c.name for c in cards}
    assert names == {"EchoAgent", "SummarizerAgent"}


async def test_find_by_skill_returns_correct_agent(echo_agent: AgentServer) -> None:
    """find_by_skill returns list[AgentCard] matching by skill id."""
    from nexus_a2a.core.registry import AgentRegistry

    registry = AgentRegistry()
    await registry.register_url(echo_agent.url)

    results = registry.find_by_skill("echo")
    assert len(results) > 0
    assert results[0].name == "EchoAgent"


async def test_fetch_agent_card_via_client(echo_agent: AgentServer) -> None:
    """A2AHttpClient.fetch_agent_card() returns parsed AgentCard."""
    from nexus_a2a.transport.http_client import A2AHttpClient

    async with A2AHttpClient(echo_agent.url) as client:
        card = await client.fetch_agent_card()

    assert card.name == "EchoAgent"
    assert len(card.skills) > 0
    assert card.skills[0].id == "echo"


async def test_well_known_endpoint(echo_agent: AgentServer) -> None:
    """/.well-known/agent-card.json returns correct card JSON."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{echo_agent.url}/.well-known/agent-card.json")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "EchoAgent"
    assert data["version"] == "1.3.0"
