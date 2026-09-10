"""
tests/test_push_notifications.py

Push notifications: registering a webhook and delivering task updates to it.

Two things here are security controls rather than features:
  - the HMAC signature must be computed over the bytes actually sent, or a
    receiver can never verify it;
  - the webhook URL comes from whoever called the agent, so registering one is
    a server-side request forgery primitive.
"""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import ValidationError
from starlette.testclient import TestClient

from nexus_a2a import agent
from nexus_a2a.core.a2a_server import (
    ERR_INVALID_PARAMS,
    ERR_PUSH_INVALID_URL,
    ERR_PUSH_NOT_SUPPORTED,
    ERR_TASK_NOT_FOUND,
    A2AServer,
)
from nexus_a2a.models.task import Message, NeedsInput, PushNotificationConfig, Task
from nexus_a2a.transport.webhook import (
    WebhookConfig,
    WebhookDispatcher,
    WebhookUrlError,
    _canonical_body,
    sign_body,
    validate_webhook_url,
)

SERVER_URL = "http://push-agent:8001"
HOOK = "http://127.0.0.1:9999/hook"
SECRET = "shared-secret"


# ── Test agents ───────────────────────────────────────────────────────────────


@agent(name="Pusher", description="Supports webhooks.", push_notifications=True,
       url=SERVER_URL)
class PusherAgent:
    async def run(self, task: Task) -> str:
        return "done"


@agent(name="Asker", description="Pauses for input.", push_notifications=True,
       url=SERVER_URL)
class AskerAgent:
    async def run(self, task: Task):
        turns = sum(1 for m in task.history if m.role == "user")
        if turns == 1:
            return NeedsInput("Confirm?")
        return "finished"


@agent(name="Boom", description="Fails.", push_notifications=True, url=SERVER_URL)
class BoomAgent:
    async def run(self, task: Task) -> str:
        raise RuntimeError("kaboom")


@agent(name="NoPush", description="No webhook support.", url=SERVER_URL)
class NoPushAgent:
    async def run(self, task: Task) -> str:
        return "done"


class RecordingDispatcher(WebhookDispatcher):
    """Captures deliveries instead of making HTTP calls."""

    def __init__(self, config: WebhookConfig | None = None) -> None:
        super().__init__(config)
        self.sent: list[tuple[str, str, str]] = []  # (url, task_id, event)

    async def dispatch_silent(self, url, task, event="task_update"):  # type: ignore[override]
        self.sent.append((url, task.id, event))
        return None  # type: ignore[return-value]

    def events(self) -> list[str]:
        return [e for _, _, e in self.sent]


def build_server(agent_cls: type = PusherAgent, **kwargs) -> A2AServer:
    server = A2AServer(
        agent_cls,
        push_config=WebhookConfig(signing_secret=SECRET, allow_private_urls=True),
        **kwargs,
    )
    server.push = RecordingDispatcher(server._push_config)
    return server


def make_client(agent_cls: type = PusherAgent) -> TestClient:
    return TestClient(build_server(agent_cls).app)


def rpc(client: TestClient, method: str, params: dict, rpc_id: str = "1"):
    return client.post(
        "/", json={"jsonrpc": "2.0", "id": rpc_id, "method": method, "params": params}
    )


def _params(text: str, extra: dict) -> dict:
    """Build send params, mapping the Python kwarg name onto the wire name."""
    params = {"message": Message.user_text(text).model_dump(mode="json")}
    if "push_notification" in extra:
        params["pushNotification"] = extra.pop("push_notification")
    params.update(extra)
    return params


def send(client: TestClient, text="go", **extra):
    return rpc(client, "message/send", _params(text, extra))


async def asgi_send(server: A2AServer, text="go", **extra):
    """Drive the server through ASGI so notifications can be awaited."""
    params = _params(text, extra)
    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        resp = await c.post(
            "/", json={"jsonrpc": "2.0", "id": "1", "method": "message/send",
                       "params": params},
        )
    await server.drain_notifications()
    return resp


# ── Signing ───────────────────────────────────────────────────────────────────


class TestSigning:
    def test_signature_round_trips(self):
        body = _canonical_body({"event": "x", "task_id": "abc"})
        sig = sign_body(body, SECRET)
        assert WebhookDispatcher.verify_signature(body, sig, SECRET)

    def test_wrong_secret_fails(self):
        body = _canonical_body({"event": "x"})
        assert not WebhookDispatcher.verify_signature(
            body, sign_body(body, SECRET), "other"
        )

    def test_tampered_payload_fails(self):
        body = _canonical_body({"amount": 1})
        sig = sign_body(body, SECRET)
        assert not WebhookDispatcher.verify_signature(
            _canonical_body({"amount": 1000}), sig, SECRET
        )

    def test_string_payload_accepted(self):
        body = _canonical_body({"a": 1})
        sig = sign_body(body, SECRET)
        assert WebhookDispatcher.verify_signature(body.decode(), sig, SECRET)

    def test_signature_matches_bytes_actually_sent(self):
        """
        Regression: the signature used to be computed over json.dumps(payload)
        while httpx re-encoded json= with compact separators, so the bytes
        differed and verify_signature() could never succeed.
        """
        payload = {"event": "task_completed", "task_id": "abc", "state": "completed"}
        body = _canonical_body(payload)
        sent = httpx.Request("POST", "http://x/", content=body).content
        assert sent == body
        assert WebhookDispatcher.verify_signature(
            sent, sign_body(body, SECRET), SECRET
        )

    def test_canonical_body_is_stable(self):
        """Key order must not change the signature."""
        a = _canonical_body({"b": 2, "a": 1})
        b = _canonical_body({"a": 1, "b": 2})
        assert a == b


# ── URL validation (SSRF) ─────────────────────────────────────────────────────


class TestUrlValidation:
    @pytest.mark.parametrize(
        "url",
        [
            "http://169.254.169.254/latest/meta-data/",  # cloud metadata
            "http://127.0.0.1:8080/admin",
            "http://localhost/x",
            "http://0.0.0.0/x",
            "http://10.0.0.5/internal",
            "http://192.168.1.1/router",
        ],
    )
    def test_non_public_addresses_blocked(self, url: str):
        with pytest.raises(WebhookUrlError):
            validate_webhook_url(url)

    @pytest.mark.parametrize("url", ["ftp://x/y", "file:///etc/passwd", "gopher://x"])
    def test_non_http_schemes_blocked(self, url: str):
        with pytest.raises(WebhookUrlError, match="http and https"):
            validate_webhook_url(url)

    def test_missing_host_blocked(self):
        with pytest.raises(WebhookUrlError, match="no host"):
            validate_webhook_url("http:///nohost")

    def test_unresolvable_host_blocked(self):
        with pytest.raises(WebhookUrlError):
            validate_webhook_url("https://this-host-should-not-resolve.invalid/x")

    def test_allow_private_opts_out(self):
        validate_webhook_url("http://127.0.0.1:9999/hook", allow_private=True)

    def test_allow_private_still_rejects_bad_scheme(self):
        with pytest.raises(WebhookUrlError):
            validate_webhook_url("ftp://127.0.0.1/x", allow_private=True)


# ── Registration ──────────────────────────────────────────────────────────────


class TestRegistration:
    def test_register_with_the_first_message(self):
        resp = send(make_client(), push_notification={"url": HOOK})
        assert "error" not in resp.json()

    def test_register_via_rpc_method(self):
        client = make_client(AskerAgent)
        task_id = send(client).json()["result"]["id"]
        resp = rpc(client, "tasks/pushNotificationConfig/set",
                   {"taskId": task_id, "pushNotificationConfig": {"url": HOOK}})
        assert resp.json()["result"]["pushNotificationConfig"]["url"] == HOOK

    def test_get_returns_registered_config(self):
        client = make_client(AskerAgent)
        task_id = send(client, push_notification={"url": HOOK}).json()["result"]["id"]
        got = rpc(client, "tasks/pushNotificationConfig/get", {"taskId": task_id})
        assert got.json()["result"]["pushNotificationConfig"]["url"] == HOOK

    def test_get_returns_null_when_unregistered(self):
        client = make_client(AskerAgent)
        task_id = send(client).json()["result"]["id"]
        got = rpc(client, "tasks/pushNotificationConfig/get", {"taskId": task_id})
        assert got.json()["result"]["pushNotificationConfig"] is None

    def test_token_is_never_echoed_back(self):
        client = make_client(AskerAgent)
        task_id = send(
            client, push_notification={"url": HOOK, "token": "super-secret"}
        ).json()["result"]["id"]
        got = rpc(client, "tasks/pushNotificationConfig/get", {"taskId": task_id})
        assert "super-secret" not in got.text
        assert got.json()["result"]["pushNotificationConfig"]["hasToken"] is True

    def test_set_on_unknown_task(self):
        body = rpc(make_client(), "tasks/pushNotificationConfig/set",
                   {"taskId": "nope", "pushNotificationConfig": {"url": HOOK}}).json()
        assert body["error"]["code"] == ERR_TASK_NOT_FOUND

    def test_set_without_config(self):
        client = make_client(AskerAgent)
        task_id = send(client).json()["result"]["id"]
        body = rpc(client, "tasks/pushNotificationConfig/set",
                   {"taskId": task_id}).json()
        assert body["error"]["code"] == ERR_INVALID_PARAMS

    def test_config_without_url_rejected(self):
        body = send(make_client(), push_notification={"token": "x"}).json()
        assert body["error"]["code"] == ERR_INVALID_PARAMS


class TestRegistrationRefusals:
    def test_agent_without_capability_refuses(self):
        body = send(make_client(NoPushAgent), push_notification={"url": HOOK}).json()
        assert body["error"]["code"] == ERR_PUSH_NOT_SUPPORTED
        assert "push_notifications=False" in body["error"]["message"]

    def test_ssrf_url_refused(self):
        """A server without allow_private_urls must refuse a metadata URL."""
        server = A2AServer(PusherAgent, push_config=WebhookConfig())
        client = TestClient(server.app)
        body = send(client, push_notification={
            "url": "http://169.254.169.254/latest/meta-data/"}).json()
        assert body["error"]["code"] == ERR_PUSH_INVALID_URL

    def test_task_does_not_run_when_registration_is_refused(self):
        """
        Refusing must abort the whole call — running a task whose updates the
        caller believes they will receive would silently drop them.
        """
        server = A2AServer(PusherAgent, push_config=WebhookConfig())
        client = TestClient(server.app)
        send(client, push_notification={"url": "http://10.0.0.1/x"})
        listed = client.post(
            "/", json={"jsonrpc": "2.0", "id": "2", "method": "tasks/get",
                       "params": {"taskId": "any"}},
        )
        assert listed.json()["error"]["code"] == ERR_TASK_NOT_FOUND


# ── Delivery ──────────────────────────────────────────────────────────────────


class TestDelivery:
    async def test_completion_is_delivered(self):
        server = build_server()
        await asgi_send(server, push_notification={"url": HOOK})
        assert server.push.events() == ["task_completed"]

    async def test_failure_is_delivered(self):
        server = build_server(BoomAgent)
        await asgi_send(server, push_notification={"url": HOOK})
        assert server.push.events() == ["task_failed"]

    async def test_input_required_is_delivered(self):
        server = build_server(AskerAgent)
        await asgi_send(server, push_notification={"url": HOOK})
        assert server.push.events() == ["task_input_required"]

    async def test_pause_then_completion_delivers_both(self):
        server = build_server(AskerAgent)
        resp = await asgi_send(server, push_notification={"url": HOOK})
        task_id = resp.json()["result"]["id"]
        await asgi_send(server, "yes", taskId=task_id)
        assert server.push.events() == ["task_input_required", "task_completed"]

    async def test_nothing_delivered_without_registration(self):
        server = build_server()
        await asgi_send(server)
        assert server.push.sent == []

    async def test_delivery_targets_the_registered_url(self):
        server = build_server()
        await asgi_send(server, push_notification={"url": HOOK})
        assert server.push.sent[0][0] == HOOK

    async def test_target_is_released_once_terminal(self):
        """Otherwise the map grows one entry per task, forever."""
        server = build_server()
        resp = await asgi_send(server, push_notification={"url": HOOK})
        assert resp.json()["result"]["state"] == "completed"
        assert server._push_targets == {}

    async def test_target_is_kept_while_paused(self):
        server = build_server(AskerAgent)
        resp = await asgi_send(server, push_notification={"url": HOOK})
        assert resp.json()["result"]["id"] in server._push_targets


class TestDeliveryFailuresAreContained:
    async def test_broken_webhook_does_not_fail_the_task(self):
        class Exploding(WebhookDispatcher):
            async def dispatch_silent(self, url, task, event="task_update"):  # type: ignore[override]
                raise RuntimeError("webhook host is on fire")

        server = A2AServer(
            PusherAgent,
            push_config=WebhookConfig(allow_private_urls=True),
        )
        server.push = Exploding(server._push_config)
        resp = await asgi_send(server, push_notification={"url": HOOK})
        assert resp.json()["result"]["state"] == "completed"

    async def test_drain_is_safe_with_nothing_pending(self):
        await build_server().drain_notifications()


# ── Capability honesty ────────────────────────────────────────────────────────


class TestCapabilityFlag:
    def test_card_advertises_push_support(self):
        data = make_client().get("/.well-known/agent-card.json").json()
        assert data["capabilities"]["push_notifications"] is True

    def test_card_advertises_absence(self):
        data = make_client(NoPushAgent).get("/.well-known/agent-card.json").json()
        assert data["capabilities"]["push_notifications"] is False


# ── Payload shape ─────────────────────────────────────────────────────────────


class TestPayload:
    def test_payload_carries_task_and_state(self):
        dispatcher = WebhookDispatcher(WebhookConfig())
        task = Task.create(initial_message=Message.user_text("x"))
        payload = dispatcher._build_payload(task, "task_completed")
        assert payload["event"] == "task_completed"
        assert payload["task_id"] == task.id
        assert payload["state"] == task.state.value
        assert payload["task"]["id"] == task.id

    def test_payload_is_json_serialisable(self):
        dispatcher = WebhookDispatcher(WebhookConfig())
        task = Task.create(initial_message=Message.user_text("x"))
        json.dumps(dispatcher._build_payload(task, "task_completed"))


# ── Config model ──────────────────────────────────────────────────────────────


class TestPushNotificationConfigModel:
    def test_url_is_required(self):
        with pytest.raises(ValidationError):
            PushNotificationConfig()  # type: ignore[call-arg]

    def test_token_is_optional(self):
        assert PushNotificationConfig(url=HOOK).token is None

    def test_is_exported(self):
        from nexus_a2a import PushNotificationConfig as Exported

        assert Exported is PushNotificationConfig
