"""
tests/test_client_errors.py

A2AHttpClient refusal errors: a server that answers 401/403/413/429 raises
AgentRefusedError, a subclass of AgentUnreachableError carrying status_code,
instead of a plain "is unreachable" error.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from nexus_a2a import (
    A2AHttpClient,
    AgentRefusedError,
    AgentUnreachableError,
    TransportError,
    agent,
)
from nexus_a2a.core.a2a_server import A2AServer
from nexus_a2a.models.agent import AuthScheme
from nexus_a2a.models.task import Message, Task
from nexus_a2a.security.auth import AgentCredentialConfig, AuthManager
from nexus_a2a.security.middleware import SecurityMiddleware
from nexus_a2a.security.trust import TrustBoundary

AGENT = "http://agent.test:8001"
CALLER = "http://caller.test:9001"


# ── AgentRefusedError (mocked transport) ──────────────────────────────────────


class TestAgentRefusedError:
    def test_is_backwards_compatible(self) -> None:
        exc = AgentRefusedError(AGENT, 401)
        assert isinstance(exc, AgentUnreachableError)
        assert isinstance(exc, TransportError)
        assert exc.status_code == 401
        assert exc.url == AGENT
        # Existing callers that matched on the status in the text still work.
        assert "HTTP 401" in str(exc)
        assert "refused" in str(exc)
        assert "unreachable" not in str(exc)

    @pytest.mark.parametrize("status", [401, 403, 413, 429])
    async def test_refusal_statuses(self, status: int) -> None:
        with respx.mock() as router:
            route = router.post(f"{AGENT}/").respond(
                status, json={"error": "X", "detail": "nope"}
            )
            async with A2AHttpClient(AGENT) as client:
                with pytest.raises(AgentRefusedError) as info:
                    await client.send_message(Message.user_text("hi"))
        assert info.value.status_code == status
        assert info.value.detail == "nope"
        assert route.call_count == 1  # never retried

    async def test_other_4xx_stays_unreachable(self) -> None:
        with respx.mock() as router:
            router.post(f"{AGENT}/").respond(404)
            async with A2AHttpClient(AGENT) as client:
                with pytest.raises(AgentUnreachableError) as info:
                    await client.send_message(Message.user_text("hi"))
        assert not isinstance(info.value, AgentRefusedError)
        assert "HTTP 404 (client error)" in str(info.value)

    async def test_retry_after_is_parsed(self) -> None:
        with respx.mock() as router:
            router.post(f"{AGENT}/").respond(
                429, json={"detail": "slow down"}, headers={"Retry-After": "7"}
            )
            async with A2AHttpClient(AGENT) as client:
                with pytest.raises(AgentRefusedError) as info:
                    await client.get_task("t-1")
        assert info.value.retry_after == 7.0

    async def test_plain_text_body_becomes_detail(self) -> None:
        with respx.mock() as router:
            router.post(f"{AGENT}/").respond(403, text="gateway says no")
            async with A2AHttpClient(AGENT) as client:
                with pytest.raises(AgentRefusedError, match="gateway says no"):
                    await client.get_task("t-1")

    async def test_long_detail_is_truncated(self) -> None:
        with respx.mock() as router:
            router.post(f"{AGENT}/").respond(403, text="x" * 10_000)
            async with A2AHttpClient(AGENT) as client:
                with pytest.raises(AgentRefusedError) as info:
                    await client.get_task("t-1")
        assert len(info.value.detail) == 500

    async def test_stream_refusal(self) -> None:
        with respx.mock() as router:
            router.post(f"{AGENT}/").respond(401, json={"detail": "bad key"})
            async with A2AHttpClient(AGENT) as client:
                with pytest.raises(AgentRefusedError) as info:
                    async for _ in client.stream_message(Message.user_text("hi")):
                        pass
        assert info.value.status_code == 401
        assert info.value.detail == "bad key"

    async def test_stream_other_error_stays_unreachable(self) -> None:
        with respx.mock() as router:
            router.post(f"{AGENT}/").respond(500, text="boom")
            async with A2AHttpClient(AGENT) as client:
                with pytest.raises(AgentUnreachableError) as info:
                    async for _ in client.stream_message(Message.user_text("hi")):
                        pass
        assert not isinstance(info.value, AgentRefusedError)


# ── AgentRefusedError against a real A2AServer ────────────────────────────────


@agent(
    name="Guarded",
    description="Echoes.",
    skills=[{"id": "echo", "name": "Echo", "description": "Echo."}],
    url=AGENT,
)
class Guarded:
    async def run(self, task: Task) -> str:
        return "ok"


def _server(trust_caller: bool) -> A2AServer:
    auth = AuthManager()
    auth.register_agent(
        CALLER, AgentCredentialConfig(scheme=AuthScheme.API_KEY, api_key="right")
    )
    trust = TrustBoundary()
    if trust_caller:
        trust.allow(CALLER, AGENT)
    security = SecurityMiddleware(auth=auth, trust=trust, server_url=AGENT)
    return A2AServer(Guarded, security=security)


async def _send(server: A2AServer, key: str) -> None:
    client = A2AHttpClient(AGENT, caller_url=CALLER, headers={"X-API-Key": key})
    async with client:
        # Route the client's requests into the ASGI app instead of the network.
        assert client._client is not None
        await client._client.aclose()
        client._client = httpx.AsyncClient(
            base_url=AGENT,
            transport=httpx.ASGITransport(app=server.app),
            headers={"X-Nexus-Caller": CALLER, "X-API-Key": key},
        )
        await client.send_message(Message.user_text("hi"))


class TestRefusedEndToEnd:
    async def test_wrong_key_is_refused_401(self) -> None:
        with pytest.raises(AgentRefusedError) as info:
            await _send(_server(trust_caller=True), key="wrong")
        assert info.value.status_code == 401

    async def test_trust_rule_is_refused_403(self) -> None:
        with pytest.raises(AgentRefusedError) as info:
            await _send(_server(trust_caller=False), key="right")
        assert info.value.status_code == 403
        assert info.value.detail  # the server's reason reaches the caller
