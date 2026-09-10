"""
tests/integration/test_push_webhooks.py

Push notifications over real HTTP: a real A2AServer delivering signed webhooks
to a real receiving endpoint on another port.

The headline test is that the receiver can actually verify the signature.
That is the whole point of signing, and it could never have passed before the
canonical-body fix, because the bytes signed were not the bytes sent.
"""

from __future__ import annotations

import asyncio
import json

import pytest
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from nexus_a2a import agent
from nexus_a2a.core.a2a_server import A2AServer
from nexus_a2a.models.task import Message, NeedsInput, TaskState
from nexus_a2a.transport.http_client import A2AHttpClient
from nexus_a2a.transport.webhook import WebhookConfig, WebhookDispatcher

from .conftest import get_free_port

pytestmark = pytest.mark.integration

SECRET = "integration-secret"


# ── Receiving endpoint ────────────────────────────────────────────────────────


class HookReceiver:
    """A real HTTP endpoint that records signed webhook deliveries."""

    def __init__(self, secret: str = SECRET) -> None:
        self.secret = secret
        self.received: list[dict] = []
        self.port = get_free_port()
        self.url = f"http://127.0.0.1:{self.port}/hook"
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        async def hook(request):
            body = await request.body()
            signature = request.headers.get("X-Nexus-Signature-256", "")
            self.received.append(
                {
                    "payload": json.loads(body),
                    "signature_valid": WebhookDispatcher.verify_signature(
                        body, signature, self.secret
                    ),
                    "content_type": request.headers.get("content-type"),
                }
            )
            return JSONResponse({"ok": True})

        app = Starlette(routes=[Route("/hook", hook, methods=["POST"])])
        self._server = uvicorn.Server(
            uvicorn.Config(
                app, host="127.0.0.1", port=self.port, log_level="critical"
            )
        )
        self._task = asyncio.create_task(self._server.serve())
        for _ in range(200):
            if self._server.started:
                return
            await asyncio.sleep(0.02)
        raise RuntimeError("hook receiver did not start")

    async def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._task is not None:
            await self._task

    def events(self) -> list[str]:
        return [r["payload"]["event"] for r in self.received]


# ── Agent under test ──────────────────────────────────────────────────────────


def build_push_agent(url: str) -> type:
    @agent(
        name="PushAgent",
        description="Pauses for confirmation, then finishes.",
        push_notifications=True,
        skills=[{"id": "work", "name": "Work", "description": "Do work."}],
        url=url,
    )
    class PushAgent:
        async def run(self, task):
            turns = sum(1 for m in task.history if m.role == "user")
            if turns == 1:
                return NeedsInput("Confirm?")
            return "all done"

    return PushAgent


async def serve_pushing(**config_kwargs):
    """Start an A2AServer that delivers webhooks to loopback."""
    port = get_free_port()
    url = f"http://127.0.0.1:{port}"
    kwargs = {
        "signing_secret": SECRET,
        "allow_private_urls": True,
        **config_kwargs,
    }
    server = A2AServer(
        build_push_agent(url),
        host="127.0.0.1",
        port=port,
        push_config=WebhookConfig(**kwargs),
    )
    await server.start()
    return server, url


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestSignedDelivery:
    async def test_receiver_can_verify_the_signature(self):
        hook = HookReceiver()
        await hook.start()
        server, url = await serve_pushing()
        try:
            async with A2AHttpClient(url) as client:
                await client.send_message(
                    Message.user_text("go"),
                    push_notification={"url": hook.url, "token": "abc"},
                )
            await server.drain_notifications()
            assert hook.received, "no webhook was delivered"
            assert all(r["signature_valid"] for r in hook.received)
        finally:
            await server.stop()
            await hook.stop()

    async def test_delivery_is_json(self):
        hook = HookReceiver()
        await hook.start()
        server, url = await serve_pushing()
        try:
            async with A2AHttpClient(url) as client:
                await client.send_message(
                    Message.user_text("go"), push_notification={"url": hook.url}
                )
            await server.drain_notifications()
            assert hook.received[0]["content_type"] == "application/json"
        finally:
            await server.stop()
            await hook.stop()

    async def test_payload_carries_the_full_task(self):
        hook = HookReceiver()
        await hook.start()
        server, url = await serve_pushing()
        try:
            async with A2AHttpClient(url) as client:
                task = await client.send_message(
                    Message.user_text("go"), push_notification={"url": hook.url}
                )
            await server.drain_notifications()
            payload = hook.received[0]["payload"]
            assert payload["event"] == "task_input_required"
            assert payload["task_id"] == task.id
            assert payload["state"] == "input_required"
            assert payload["task"]["id"] == task.id
        finally:
            await server.stop()
            await hook.stop()


class TestLifecycleEvents:
    async def test_pause_then_completion(self):
        hook = HookReceiver()
        await hook.start()
        server, url = await serve_pushing()
        try:
            async with A2AHttpClient(url) as client:
                task = await client.send_message(
                    Message.user_text("go"), push_notification={"url": hook.url}
                )
                await server.drain_notifications()
                await client.send_message(Message.user_text("yes"), task_id=task.id)
            await server.drain_notifications()
            assert hook.events() == ["task_input_required", "task_completed"]
        finally:
            await server.stop()
            await hook.stop()

    async def test_cancellation_is_delivered(self):
        hook = HookReceiver()
        await hook.start()
        server, url = await serve_pushing()
        try:
            async with A2AHttpClient(url) as client:
                task = await client.send_message(
                    Message.user_text("go"), push_notification={"url": hook.url}
                )
                await server.drain_notifications()
                await client.cancel_task(task.id)
            await server.drain_notifications()
            assert hook.events()[-1] == "task_cancelled"
        finally:
            await server.stop()
            await hook.stop()

    async def test_registering_later_via_rpc(self):
        hook = HookReceiver()
        await hook.start()
        server, url = await serve_pushing()
        try:
            async with A2AHttpClient(url) as client:
                task = await client.send_message(Message.user_text("go"))
                await server.drain_notifications()
                assert hook.received == []

                await client.set_push_config(task.id, {"url": hook.url})
                readback = await client.get_push_config(task.id)
                assert readback["pushNotificationConfig"]["url"] == hook.url

                await client.send_message(Message.user_text("yes"), task_id=task.id)
            await server.drain_notifications()
            assert hook.events() == ["task_completed"]
        finally:
            await server.stop()
            await hook.stop()


class TestFailuresAreContained:
    async def test_unreachable_webhook_does_not_break_the_task(self):
        """A dead receiver must not fail the task it was reporting on."""
        dead_port = get_free_port()  # nothing listening there
        server, url = await serve_pushing(max_retries=1, base_delay=0.01)
        try:
            async with A2AHttpClient(url) as client:
                task = await client.send_message(
                    Message.user_text("go"),
                    push_notification={"url": f"http://127.0.0.1:{dead_port}/gone"},
                )
            assert task.state == TaskState.INPUT_REQUIRED
            await server.drain_notifications()
        finally:
            await server.stop()

    async def test_delivery_does_not_delay_the_rpc_response(self):
        """
        Delivery is detached, so a slow receiver must not hold up the caller.
        A dead port with retries would take seconds if awaited inline.
        """
        dead_port = get_free_port()
        server, url = await serve_pushing(max_retries=3, base_delay=1.0)
        try:
            async with A2AHttpClient(url) as client:
                started = asyncio.get_running_loop().time()
                await client.send_message(
                    Message.user_text("go"),
                    push_notification={"url": f"http://127.0.0.1:{dead_port}/gone"},
                )
                elapsed = asyncio.get_running_loop().time() - started
            assert elapsed < 2.0, f"response blocked on webhook retries ({elapsed:.1f}s)"
        finally:
            await server.drain_notifications(timeout=15.0)
            await server.stop()
