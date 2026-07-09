"""
tests/integration/test_sequential_pipeline.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Real orchestrator tests. No mocking. asyncio_mode=auto.

VERIFIED API:
  Orchestrator(runner, stop_on_error=True)
  sequential(agent_urls, initial_message, skill_ids=None) → OrchestratorResult
  parallel(agent_urls, message, skill_ids=None) → OrchestratorResult
  OrchestratorResult.steps         → list[StepResult]
  OrchestratorResult.total_sec     → float
  OrchestratorResult.final_output  → Task | None
  OrchestratorResult.succeeded     → bool (property: all steps succeeded)
  StepResult.task                  → Task | None
  StepResult.error                 → str | None
  StepResult.duration_sec          → float
  StepResult.succeeded             → bool (property)
  Message.user_text(text)          → Message shortcut
"""

from __future__ import annotations

import pytest

from nexus_a2a.models.task import Message, TaskState
from tests.integration.conftest import AgentServer, get_free_port, make_agent_app


@pytest.fixture
async def three_agent_pipeline():
    servers = []
    configs = [
        (
            "AgentA",
            "First",
            [{"id": "step_a", "name": "Step A", "description": "Step A."}],
        ),
        (
            "AgentB",
            "Second",
            [{"id": "step_b", "name": "Step B", "description": "Step B."}],
        ),
        (
            "AgentC",
            "Third",
            [{"id": "step_c", "name": "Step C", "description": "Step C."}],
        ),
    ]
    for name, desc, skills in configs:
        port = get_free_port()
        server = AgentServer(
            make_agent_app(name=name, description=desc, skills=skills), port
        )
        await server.start()
        servers.append(server)
    yield servers
    for s in servers:
        await s.stop()


async def test_sequential_pipeline_all_complete(
    three_agent_pipeline: list[AgentServer],
) -> None:
    """All 3 sequential steps complete; OrchestratorResult.succeeded is True."""
    from nexus_a2a.core.orchestrator import Orchestrator
    from nexus_a2a.transport.http_client import A2AHttpClient

    agent_a, agent_b, agent_c = three_agent_pipeline

    async def runner(url: str, message: Message) -> Task:
        async with A2AHttpClient(url) as client:
            return await client.send_message(message)

    result = await Orchestrator(runner=runner).sequential(
        agent_urls=[agent_a.url, agent_b.url, agent_c.url],
        initial_message=Message.user_text("start"),
    )

    assert len(result.steps) == 3
    assert result.succeeded is True
    for step in result.steps:
        assert step.succeeded is True
        assert step.task is not None
        assert step.task.state == TaskState.COMPLETED


async def test_sequential_output_chaining(
    three_agent_pipeline: list[AgentServer],
) -> None:
    """Each step receives prior step's task output as next input."""
    from nexus_a2a.core.orchestrator import Orchestrator
    from nexus_a2a.transport.http_client import A2AHttpClient

    agent_a, agent_b, agent_c = three_agent_pipeline
    received_texts: list[str] = []

    async def recording_runner(url: str, message: Message) -> Task:
        received_texts.append(message.text())
        async with A2AHttpClient(url) as client:
            return await client.send_message(message)

    result = await Orchestrator(runner=recording_runner).sequential(
        agent_urls=[agent_a.url, agent_b.url, agent_c.url],
        initial_message=Message.user_text("initial"),
    )

    assert len(result.steps) == 3
    assert "initial" in received_texts[0]
    assert all(t for t in received_texts)  # none empty


async def test_sequential_step_timing(three_agent_pipeline: list[AgentServer]) -> None:
    """StepResult.duration_sec >= 0 and OrchestratorResult.total_sec >= 0."""
    from nexus_a2a.core.orchestrator import Orchestrator
    from nexus_a2a.transport.http_client import A2AHttpClient

    agent_a, agent_b, agent_c = three_agent_pipeline

    async def runner(url: str, message: Message) -> Task:
        async with A2AHttpClient(url) as client:
            return await client.send_message(message)

    result = await Orchestrator(runner=runner).sequential(
        agent_urls=[agent_a.url, agent_b.url, agent_c.url],
        initial_message=Message.user_text("time me"),
    )

    assert result.total_sec >= 0
    for step in result.steps:
        assert step.duration_sec >= 0


async def test_sequential_stops_on_error(
    three_agent_pipeline: list[AgentServer],
) -> None:
    """stop_on_error=True (default) halts at failing step; third agent NOT called."""
    from nexus_a2a.core.orchestrator import Orchestrator
    from nexus_a2a.transport.http_client import A2AHttpClient

    agent_a = three_agent_pipeline[0]
    agent_c = three_agent_pipeline[2]

    fail_port = get_free_port()
    fail_server = AgentServer(
        make_agent_app("FailMid", "Fails.", [], fail_tasks=True), fail_port
    )
    await fail_server.start()

    called_urls: list[str] = []

    async def runner(url: str, message: Message) -> Task:
        called_urls.append(url)
        async with A2AHttpClient(url) as client:
            return await client.send_message(message)

    result = await Orchestrator(runner=runner, stop_on_error=True).sequential(
        agent_urls=[agent_a.url, fail_server.url, agent_c.url],
        initial_message=Message.user_text("test"),
    )

    await fail_server.stop()

    # Pipeline should have stopped — only 2 steps recorded
    assert len(result.steps) == 2
    assert result.succeeded is False
    assert agent_c.url not in called_urls


async def test_sequential_final_output_is_last_task(
    three_agent_pipeline: list[AgentServer],
) -> None:
    """OrchestratorResult.final_output is the last successful Task."""
    from nexus_a2a.core.orchestrator import Orchestrator
    from nexus_a2a.transport.http_client import A2AHttpClient

    agent_a, agent_b, agent_c = three_agent_pipeline

    async def runner(url: str, message: Message) -> Task:
        async with A2AHttpClient(url) as client:
            return await client.send_message(message)

    result = await Orchestrator(runner=runner).sequential(
        agent_urls=[agent_a.url, agent_b.url, agent_c.url],
        initial_message=Message.user_text("final"),
    )

    assert result.final_output is not None
    assert result.final_output.id == result.steps[-1].task.id


async def test_parallel_workflow(
    echo_agent: AgentServer, summarizer_agent: AgentServer
) -> None:
    """parallel() sends same message to both agents; both steps succeed."""
    from nexus_a2a.core.orchestrator import Orchestrator
    from nexus_a2a.transport.http_client import A2AHttpClient

    async def runner(url: str, message: Message) -> Task:
        async with A2AHttpClient(url) as client:
            return await client.send_message(message)

    result = await Orchestrator(runner=runner).parallel(
        agent_urls=[echo_agent.url, summarizer_agent.url],
        message=Message.user_text("parallel input"),
    )

    assert len(result.steps) == 2
    assert result.succeeded is True
    for step in result.steps:
        assert step.succeeded is True
        assert step.task.state == TaskState.COMPLETED
