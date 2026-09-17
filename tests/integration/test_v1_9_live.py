"""
tests/integration/test_v1_9_live.py

1.9.0 behaviour that only shows up over real sockets:

  - GET /stream follows a running task live, including chunks emitted before
    the follower connected, with keep-alives in between;
  - an AgentRuntime built from config serves the agent AND an ops server, with
    the configured secret enforced end to end;
  - `nexus verify` passes against a real, configured server.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from nexus_a2a import A2AHttpClient, Message, agent
from nexus_a2a.config import NexusConfig
from nexus_a2a.core.a2a_server import A2AServer
from nexus_a2a.transport.http_client import AgentUnreachableError
from nexus_a2a.transport.sse import SSEStreamer, StreamEventType
from nexus_a2a.verify import verify_agent

from .conftest import get_free_port

pytestmark = pytest.mark.integration


class TestLiveFollowOverTheWire:
    async def test_follower_sees_every_chunk_as_it_happens(self):
        port = get_free_port()
        url = f"http://127.0.0.1:{port}"

        @agent(name="Slow", description="Streams slowly.", streaming=True, url=url)
        class Slow:
            async def run(self, task):
                for word in ["one", "two", "three"]:
                    await asyncio.sleep(0.25)
                    yield word

        server = A2AServer(Slow, host="127.0.0.1", port=port, stream_heartbeat=0.1)
        await server.start()
        try:
            created = asyncio.Event()
            state: dict = {}

            async def produce():
                async with A2AHttpClient(url) as client:
                    async for ev in client.stream_message(Message.user_text("go")):
                        if ev.type == StreamEventType.TASK_CREATED:
                            state["id"] = ev.data["id"]
                            created.set()

            async def follow():
                await created.wait()
                await asyncio.sleep(0.35)  # join after "one" was already sent
                seen = []
                async for ev in SSEStreamer(url).stream(task_id=state["id"]):
                    seen.append(ev)
                return seen

            _, seen = await asyncio.gather(produce(), follow())
        finally:
            await server.stop()

        chunks = [e.data["content"] for e in seen
                  if e.type == StreamEventType.ARTIFACT_CHUNK]
        assert chunks == ["one", "two", "three"]
        assert any(e.type == StreamEventType.HEARTBEAT for e in seen)
        assert seen[-1].type == StreamEventType.DONE
        assert server._watchers.count() == 0


class TestRuntimeFromConfig:
    def build(self, agent_port: int, ops_port: int):
        url = f"http://127.0.0.1:{agent_port}"

        @agent(name="Configured", description="From nexus.toml.",
               skills=[{"id": "echo", "name": "Echo", "description": "e"}], url=url)
        class Configured:
            async def run(self, task):
                return "echo: " + task.latest_message().text()

        cfg = NexusConfig.from_dict({
            "agent": {"name": "Configured", "url": url},
            "security": {"auth_scheme": "api_key", "auth_secret": "live-key",
                         "rate_limit": 100, "max_payload_bytes": 4096},
            "ops": {"port": ops_port, "admin_token": "ops-token"},
        })
        return cfg.build_runtime(Configured), url

    async def test_secured_agent_and_ops_server(self):
        agent_port, ops_port = get_free_port(), get_free_port()
        runtime, url = self.build(agent_port, ops_port)
        ops = f"http://127.0.0.1:{ops_port}"

        async with runtime:
            async with A2AHttpClient(url) as client:
                with pytest.raises(AgentUnreachableError):
                    await client.send_message(Message.user_text("hi"))

            async with A2AHttpClient(url, headers={"X-API-Key": "live-key"}) as client:
                task = await client.send_message(Message.user_text("hi"))
                assert task.artifacts[0].parts[0].content == "echo: hi"

            async with httpx.AsyncClient() as http:
                card = (await http.get(f"{url}/.well-known/agent-card.json")).json()
                assert card["authentication"]["scheme"] == "api_key"

                assert (await http.get(f"{ops}/health")).status_code == 200
                assert (await http.get(f"{ops}/info")).status_code == 403
                info = await http.get(f"{ops}/info",
                                      headers={"X-Admin-Token": "ops-token"})
                assert info.status_code == 200

                metrics = (await http.get(f"{ops}/metrics")).text
                assert "nexus_a2a_tasks_active" in metrics

    async def test_configured_server_passes_verify(self):
        agent_port, ops_port = get_free_port(), get_free_port()
        runtime, url = self.build(agent_port, ops_port)
        async with runtime:
            report = await verify_agent(url, api_key="live-key")
        failures = [c for c in report.checks if c.status.value == "fail"]
        assert failures == [], failures
