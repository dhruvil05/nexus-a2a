"""
tests/integration/test_a2a_roundtrip.py

End-to-end tests where BOTH halves of the protocol are nexus-a2a's own code:
a real A2AServer on a real port, driven by a real A2AHttpClient over real HTTP.

Every other integration test in this suite talks to a hand-rolled Starlette
mock in conftest.py, because until A2AServer existed the library had no server
to point them at. These tests are the ones that prove two nexus-a2a agents can
actually talk to each other.
"""

from __future__ import annotations

import pytest

from nexus_a2a import agent
from nexus_a2a.core.a2a_server import A2AServer
from nexus_a2a.core.registry import AgentRegistry
from nexus_a2a.models.agent import AuthScheme
from nexus_a2a.models.task import Message, TaskState
from nexus_a2a.security.auth import AgentCredentialConfig, AuthManager
from nexus_a2a.security.middleware import SecurityMiddleware
from nexus_a2a.security.trust import TrustBoundary
from nexus_a2a.transport.http_client import A2AHttpClient, RemoteAgentError

from .conftest import get_free_port

pytestmark = pytest.mark.integration

CALLER_URL = "http://caller:9999"


# ── Agents under test ─────────────────────────────────────────────────────────


def build_echo_agent(url: str) -> type:
    @agent(
        name="EchoAgent",
        description="Echoes the incoming text back.",
        skills=[{"id": "echo", "name": "Echo", "description": "Echo input.",
                 "tags": ["text"]}],
        url=url,
    )
    class EchoAgent:
        async def run(self, task):
            msg = task.latest_message()
            return f"echo: {msg.text() if msg else ''}"

    return EchoAgent


def build_upper_agent(url: str) -> type:
    @agent(
        name="UpperAgent",
        description="Upper-cases the incoming text.",
        skills=[{"id": "upper", "name": "Upper", "description": "Upper-case."}],
        url=url,
    )
    class UpperAgent:
        async def run(self, task):
            msg = task.latest_message()
            return (msg.text() if msg else "").upper()

    return UpperAgent


async def serve(agent_cls_factory, security: SecurityMiddleware | None = None):
    """Start an A2AServer on a free loopback port and return (server, url)."""
    port = get_free_port()
    url = f"http://127.0.0.1:{port}"
    server = A2AServer(
        agent_cls_factory(url), host="127.0.0.1", port=port, security=security
    )
    await server.start()
    return server, url


# ── Discovery ─────────────────────────────────────────────────────────────────


class TestDiscovery:
    async def test_client_fetches_card_from_real_server(self):
        server, url = await serve(build_echo_agent)
        try:
            async with A2AHttpClient(url) as client:
                card = await client.fetch_agent_card()
            assert card.name == "EchoAgent"
            assert card.skill_ids() == ["echo"]
        finally:
            await server.stop()

    async def test_registry_registers_a_real_server(self):
        """AgentRegistry fetches the well-known endpoint — now it exists."""
        server, url = await serve(build_echo_agent)
        try:
            registry = AgentRegistry()
            card = await registry.register_url(url)
            assert card.name == "EchoAgent"
            assert registry.find_by_skill("echo")
        finally:
            await server.stop()


# ── Task round-trip ───────────────────────────────────────────────────────────


class TestTaskRoundTrip:
    async def test_send_message_completes(self):
        server, url = await serve(build_echo_agent)
        try:
            async with A2AHttpClient(url) as client:
                task = await client.send_message(
                    Message.user_text("hello there"), skill_id="echo"
                )
            assert task.state == TaskState.COMPLETED
            assert task.artifacts[0].parts[0].content == "echo: hello there"
        finally:
            await server.stop()

    async def test_get_task_returns_same_task(self):
        server, url = await serve(build_echo_agent)
        try:
            async with A2AHttpClient(url) as client:
                sent = await client.send_message(Message.user_text("x"))
                fetched = await client.get_task(sent.id)
            assert fetched.id == sent.id
            assert fetched.state == TaskState.COMPLETED
        finally:
            await server.stop()

    async def test_get_unknown_task_raises_remote_error(self):
        server, url = await serve(build_echo_agent)
        try:
            async with A2AHttpClient(url) as client:
                with pytest.raises(RemoteAgentError):
                    await client.get_task("does-not-exist")
        finally:
            await server.stop()

    async def test_context_id_is_preserved(self):
        server, url = await serve(build_echo_agent)
        try:
            async with A2AHttpClient(url) as client:
                task = await client.send_message(
                    Message.user_text("x"), context_id="ctx-abc"
                )
            assert task.context_id == "ctx-abc"
        finally:
            await server.stop()

    async def test_trace_header_does_not_break_the_server(self):
        """A2AHttpClient injects trace headers on every call."""
        server, url = await serve(build_echo_agent)
        try:
            async with A2AHttpClient(url, trace_id="trace-123") as client:
                task = await client.send_message(Message.user_text("x"))
            assert task.state == TaskState.COMPLETED
        finally:
            await server.stop()


# ── Two agents chained ────────────────────────────────────────────────────────


class TestTwoRealAgents:
    async def test_output_of_one_feeds_the_other(self):
        echo_server, echo_url = await serve(build_echo_agent)
        upper_server, upper_url = await serve(build_upper_agent)
        try:
            async with A2AHttpClient(echo_url) as client:
                first = await client.send_message(Message.user_text("chain me"))
            text = first.artifacts[0].parts[0].content

            async with A2AHttpClient(upper_url) as client:
                second = await client.send_message(Message.user_text(text))

            assert second.artifacts[0].parts[0].content == "ECHO: CHAIN ME"
        finally:
            await echo_server.stop()
            await upper_server.stop()

    async def test_both_servers_are_independently_discoverable(self):
        echo_server, echo_url = await serve(build_echo_agent)
        upper_server, upper_url = await serve(build_upper_agent)
        try:
            registry = AgentRegistry()
            await registry.register_url(echo_url)
            await registry.register_url(upper_url)
            assert len(registry.list_all()) == 2
            assert registry.find_by_skill("upper")
        finally:
            await echo_server.stop()
            await upper_server.stop()


# ── Failure propagation ───────────────────────────────────────────────────────


class TestFailurePropagation:
    async def test_raising_agent_yields_failed_task_not_transport_error(self):
        def build(url: str) -> type:
            @agent(name="Boom", description="Always fails.", url=url)
            class Boom:
                async def run(self, task):
                    raise RuntimeError("deliberate failure")

            return Boom

        server, url = await serve(build)
        try:
            async with A2AHttpClient(url) as client:
                task = await client.send_message(Message.user_text("x"))
            assert task.state == TaskState.FAILED
            assert "deliberate failure" in task.error
        finally:
            await server.stop()


# ── Security over real HTTP ───────────────────────────────────────────────────


class TestSecurityOverTheWire:
    def _secured(self, server_url: str) -> SecurityMiddleware:
        auth = AuthManager()
        auth.register_agent(
            CALLER_URL,
            AgentCredentialConfig(scheme=AuthScheme.API_KEY, api_key="secret-key"),
        )
        trust = TrustBoundary()
        trust.allow(CALLER_URL, server_url, skills=["echo"])
        return SecurityMiddleware(auth=auth, trust=trust, server_url=server_url)

    async def _serve_secured(self):
        port = get_free_port()
        url = f"http://127.0.0.1:{port}"
        server = A2AServer(
            build_echo_agent(url),
            host="127.0.0.1",
            port=port,
            security=self._secured(url),
        )
        await server.start()
        return server, url

    async def test_authorised_caller_succeeds(self):
        server, url = await self._serve_secured()
        try:
            async with A2AHttpClient(
                url, caller_url=CALLER_URL, headers={"X-API-Key": "secret-key"}
            ) as client:
                task = await client.send_message(
                    Message.user_text("hi"), skill_id="echo"
                )
            assert task.state == TaskState.COMPLETED
        finally:
            await server.stop()

    async def test_anonymous_caller_is_refused(self):
        """No caller_url — the client stays anonymous and is rejected."""
        from nexus_a2a.transport.http_client import AgentUnreachableError

        server, url = await self._serve_secured()
        try:
            async with A2AHttpClient(url) as client:
                with pytest.raises(AgentUnreachableError):
                    await client.send_message(Message.user_text("hi"))
        finally:
            await server.stop()

    async def test_wrong_credential_is_refused(self):
        from nexus_a2a.transport.http_client import AgentUnreachableError

        server, url = await self._serve_secured()
        try:
            async with A2AHttpClient(
                url, caller_url=CALLER_URL, headers={"X-API-Key": "wrong"}
            ) as client:
                with pytest.raises(AgentUnreachableError):
                    await client.send_message(Message.user_text("hi"))
        finally:
            await server.stop()

    async def test_card_endpoint_stays_public(self):
        """Discovery must work before a caller holds credentials."""
        server, url = await self._serve_secured()
        try:
            async with A2AHttpClient(url) as client:
                card = await client.fetch_agent_card()
            assert card.name == "EchoAgent"
        finally:
            await server.stop()


# ── Streaming over real HTTP ──────────────────────────────────────────────────


def build_writer_agent(url: str) -> type:
    @agent(
        name="WriterAgent",
        description="Streams words one at a time.",
        streaming=True,
        skills=[{"id": "write", "name": "Write", "description": "Write text."}],
        url=url,
    )
    class WriterAgent:
        async def run(self, task):
            for word in ["Hello", ", ", "streaming", " ", "world", "!"]:
                yield word

    return WriterAgent


WRITER_WORDS = ["Hello", ", ", "streaming", " ", "world", "!"]


class TestStreamingOverTheWire:
    """
    SSE behaves differently through a real socket than through TestClient —
    chunked transfer, connection reuse, real flushing.
    """

    async def test_chunks_arrive_individually(self):
        from nexus_a2a.transport.sse import StreamEventType

        server, url = await serve(build_writer_agent)
        try:
            async with A2AHttpClient(url) as client:
                chunks = [
                    ev.data["content"]
                    async for ev in client.stream_message(
                        Message.user_text("go"), skill_id="write"
                    )
                    if ev.type == StreamEventType.ARTIFACT_CHUNK
                ]
            assert chunks == WRITER_WORDS
        finally:
            await server.stop()

    async def test_stream_terminates_with_done(self):
        from nexus_a2a.transport.sse import StreamEventType

        server, url = await serve(build_writer_agent)
        try:
            async with A2AHttpClient(url) as client:
                events = [
                    ev.type
                    async for ev in client.stream_message(Message.user_text("go"))
                ]
            assert events[0] == StreamEventType.TASK_CREATED
            assert events[-1] == StreamEventType.DONE
        finally:
            await server.stop()

    async def test_streamed_task_is_retrievable_afterwards(self):
        from nexus_a2a.transport.sse import StreamEventType

        server, url = await serve(build_writer_agent)
        try:
            async with A2AHttpClient(url) as client:
                task_id = None
                async for ev in client.stream_message(Message.user_text("go")):
                    if ev.type == StreamEventType.TASK_CREATED:
                        task_id = ev.data["id"]
                assert task_id is not None
                task = await client.get_task(task_id)
            assert task.state == TaskState.COMPLETED
            assert task.artifacts[0].parts[0].content == "".join(WRITER_WORDS)
        finally:
            await server.stop()

    async def test_streaming_agent_still_works_over_send(self):
        server, url = await serve(build_writer_agent)
        try:
            async with A2AHttpClient(url) as client:
                task = await client.send_message(Message.user_text("go"))
            assert task.state == TaskState.COMPLETED
            assert task.artifacts[0].parts[0].content == "".join(WRITER_WORDS)
        finally:
            await server.stop()

    async def test_non_streaming_agent_served_over_stream(self):
        from nexus_a2a.transport.sse import StreamEventType

        server, url = await serve(build_echo_agent)
        try:
            async with A2AHttpClient(url) as client:
                chunks = [
                    ev.data["content"]
                    async for ev in client.stream_message(Message.user_text("hi"))
                    if ev.type == StreamEventType.ARTIFACT_CHUNK
                ]
            assert chunks == ["echo: hi"]
        finally:
            await server.stop()

    async def test_sse_streamer_observes_a_finished_task(self):
        """The pre-existing SSEStreamer client must work against the server."""
        from nexus_a2a.transport.sse import SSEStreamer, StreamEventType

        server, url = await serve(build_writer_agent)
        try:
            async with A2AHttpClient(url) as client:
                task = await client.send_message(Message.user_text("observe"))
            events = [
                ev.type async for ev in SSEStreamer(url).stream(task_id=task.id)
            ]
            assert events[0] == StreamEventType.TASK_CREATED
            assert events[-1] == StreamEventType.DONE
        finally:
            await server.stop()

    async def test_stream_refused_when_caller_is_unauthorised(self):
        from nexus_a2a.transport.http_client import AgentUnreachableError

        port = get_free_port()
        url = f"http://127.0.0.1:{port}"
        auth = AuthManager()
        auth.register_agent(
            CALLER_URL,
            AgentCredentialConfig(scheme=AuthScheme.API_KEY, api_key="secret-key"),
        )
        server = A2AServer(
            build_writer_agent(url),
            host="127.0.0.1",
            port=port,
            security=SecurityMiddleware(auth=auth, server_url=url),
        )
        await server.start()
        try:
            async with A2AHttpClient(url) as client:
                with pytest.raises(AgentUnreachableError):
                    async for _ in client.stream_message(Message.user_text("x")):
                        pass
        finally:
            await server.stop()


# ── Multi-turn / INPUT_REQUIRED over real HTTP ────────────────────────────────


def build_planner_agent(url: str) -> type:
    from nexus_a2a.models.task import NeedsInput

    @agent(
        name="PlannerAgent",
        description="Asks for a budget, then answers.",
        skills=[{"id": "plan", "name": "Plan", "description": "Make a plan."}],
        url=url,
    )
    class PlannerAgent:
        async def run(self, task):
            turns = sum(1 for m in task.history if m.role == "user")
            if turns == 1:
                return NeedsInput("What is your budget?")
            return f"Plan for {task.history[-1].text()}"

    return PlannerAgent


class TestMultiTurnOverTheWire:
    async def test_full_conversation(self):
        server, url = await serve(build_planner_agent)
        try:
            async with A2AHttpClient(url) as client:
                first = await client.send_message(
                    Message.user_text("plan a trip"), skill_id="plan"
                )
                assert first.state == TaskState.INPUT_REQUIRED
                assert first.history[-1].text() == "What is your budget?"

                second = await client.send_message(
                    Message.user_text("$500"), task_id=first.id
                )
            assert second.id == first.id
            assert second.state == TaskState.COMPLETED
            assert second.artifacts[0].parts[0].content == "Plan for $500"
        finally:
            await server.stop()

    async def test_paused_task_survives_a_separate_client(self):
        """State lives in the store, not in a parked coroutine."""
        server, url = await serve(build_planner_agent)
        try:
            async with A2AHttpClient(url) as client:
                first = await client.send_message(Message.user_text("plan"))
            # A brand-new client and connection continues the same task.
            async with A2AHttpClient(url) as other:
                resumed = await other.send_message(
                    Message.user_text("$900"), task_id=first.id
                )
            assert resumed.state == TaskState.COMPLETED
            assert "900" in resumed.artifacts[0].parts[0].content
        finally:
            await server.stop()

    async def test_continuing_a_completed_task_is_refused(self):
        server, url = await serve(build_planner_agent)
        try:
            async with A2AHttpClient(url) as client:
                first = await client.send_message(Message.user_text("plan"))
                done = await client.send_message(
                    Message.user_text("$500"), task_id=first.id
                )
                assert done.state == TaskState.COMPLETED
                with pytest.raises(RemoteAgentError):
                    await client.send_message(
                        Message.user_text("more"), task_id=done.id
                    )
        finally:
            await server.stop()

    async def test_paused_task_is_visible_via_tasks_get(self):
        server, url = await serve(build_planner_agent)
        try:
            async with A2AHttpClient(url) as client:
                first = await client.send_message(Message.user_text("plan"))
                fetched = await client.get_task(first.id)
            assert fetched.state == TaskState.INPUT_REQUIRED
        finally:
            await server.stop()

    async def test_paused_task_can_be_cancelled(self):
        server, url = await serve(build_planner_agent)
        try:
            async with A2AHttpClient(url) as client:
                first = await client.send_message(Message.user_text("plan"))
                cancelled = await client.cancel_task(first.id)
            assert cancelled.state == TaskState.CANCELLED
        finally:
            await server.stop()
