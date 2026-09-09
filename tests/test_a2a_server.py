"""
tests/test_a2a_server.py

A2AServer — the inbound A2A protocol server.

Covers agent resolution, the agent-card endpoint, JSON-RPC dispatch for all
three methods, how run() return values become task results, and the
SecurityMiddleware chain (auth, trust, rate limit, payload).
"""

from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient

from nexus_a2a import agent
from nexus_a2a.adapters.base import AdapterResult
from nexus_a2a.core.a2a_server import (
    AGENT_CARD_PATH,
    ERR_INVALID_PARAMS,
    ERR_INVALID_REQUEST,
    ERR_METHOD_NOT_FOUND,
    ERR_PARSE,
    ERR_TASK_NOT_FOUND,
    A2AServer,
    InvalidAgentError,
    _interpret_output,
)
from nexus_a2a.models.agent import AuthScheme
from nexus_a2a.models.task import Artifact, Message, Part, PartType, Task
from nexus_a2a.security.auth import AgentCredentialConfig, AuthManager
from nexus_a2a.security.middleware import CALLER_HEADER, SecurityMiddleware
from nexus_a2a.security.rate_limiter import RateLimitConfig, RateLimiter
from nexus_a2a.security.trust import TrustBoundary
from nexus_a2a.security.validator import PayloadValidator, ValidatorConfig

SERVER_URL = "http://test-agent:8001"
CALLER_URL = "http://caller-agent:9001"

SKILLS = [{"id": "echo", "name": "Echo", "description": "Echoes input back."}]


# ── Test agents ───────────────────────────────────────────────────────────────


@agent(name="EchoAgent", description="Echoes text.", skills=SKILLS, url=SERVER_URL)
class EchoAgent:
    async def run(self, task: Task) -> str:
        msg = task.latest_message()
        return f"echo: {msg.text() if msg else ''}"


@agent(name="BoomAgent", description="Always raises.", skills=SKILLS, url=SERVER_URL)
class BoomAgent:
    async def run(self, task: Task) -> str:
        raise RuntimeError("agent exploded")


class NotAnAgent:
    async def run(self, task: Task) -> str:
        return "nope"


def make_client(
    agent_cls: type = EchoAgent,
    security: SecurityMiddleware | None = None,
) -> TestClient:
    """Build a TestClient over A2AServer's ASGI app without binding a port."""
    server = A2AServer(agent_cls, security=security)
    return TestClient(server.app)


def rpc(
    client: TestClient,
    method: str,
    params: dict | None = None,
    headers: dict | None = None,
    rpc_id: str = "1",
):
    """Send a JSON-RPC request and return the raw httpx response."""
    return client.post(
        "/",
        json={
            "jsonrpc": "2.0",
            "id": rpc_id,
            "method": method,
            "params": params if params is not None else {},
        },
        headers=headers or {},
    )


def send_text(client: TestClient, text: str = "hello", **kwargs):
    """Shortcut: message/send with a plain-text user message."""
    return rpc(
        client,
        "message/send",
        {"message": Message.user_text(text).model_dump(mode="json")},
        **kwargs,
    )


# ── Agent resolution ──────────────────────────────────────────────────────────


class TestAgentResolution:
    def test_accepts_decorated_class(self):
        server = A2AServer(EchoAgent)
        assert server.card.name == "EchoAgent"

    def test_accepts_instance(self):
        server = A2AServer(EchoAgent())
        assert server.card.name == "EchoAgent"

    def test_rejects_undecorated_class(self):
        with pytest.raises(InvalidAgentError, match="not decorated"):
            A2AServer(NotAnAgent)

    def test_rejects_missing_async_run(self):
        @agent(name="Stub", description="Stub agent.", url=SERVER_URL)
        class Stub:
            async def run(self, task): ...

        # Strip run() after decoration to exercise the server's own guard.
        del Stub.run
        with pytest.raises(InvalidAgentError, match="async def run"):
            A2AServer(Stub)

    def test_host_port_derived_from_card_url(self):
        server = A2AServer(EchoAgent)
        assert server.host == "test-agent"
        assert server.port == 8001

    def test_explicit_host_port_override_card(self):
        server = A2AServer(EchoAgent, host="127.0.0.1", port=9999)
        assert server.host == "127.0.0.1"
        assert server.port == 9999


# ── Agent card endpoint ───────────────────────────────────────────────────────


class TestAgentCardEndpoint:
    def test_returns_200(self):
        assert make_client().get(AGENT_CARD_PATH).status_code == 200

    def test_has_identity_and_skills(self):
        data = make_client().get(AGENT_CARD_PATH).json()
        assert data["name"] == "EchoAgent"
        assert data["description"] == "Echoes text."
        assert [s["id"] for s in data["skills"]] == ["echo"]

    def test_url_reflects_public_url(self):
        server = A2AServer(EchoAgent, public_url="https://public.example.com")
        data = TestClient(server.app).get(AGENT_CARD_PATH).json()
        assert data["url"] == "https://public.example.com"

    def test_card_is_json_round_trippable(self):
        """The client parses this straight into an AgentCard."""
        from nexus_a2a.models.agent import AgentCard

        data = make_client().get(AGENT_CARD_PATH).json()
        assert AgentCard.model_validate(data).name == "EchoAgent"


# ── Health / readiness ────────────────────────────────────────────────────────


class TestProbes:
    def test_health_ok(self):
        resp = make_client().get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_ready_ok(self):
        resp = make_client().get("/ready")
        assert resp.status_code == 200
        assert resp.json()["ready"] is True


# ── JSON-RPC envelope handling ────────────────────────────────────────────────


class TestRpcEnvelope:
    def test_unknown_method(self):
        body = rpc(make_client(), "does/not/exist").json()
        assert body["error"]["code"] == ERR_METHOD_NOT_FOUND

    def test_malformed_json(self):
        resp = make_client().post(
            "/", content=b"{not json", headers={"Content-Type": "application/json"}
        )
        assert resp.json()["error"]["code"] == ERR_PARSE

    def test_non_object_body(self):
        resp = make_client().post("/", json=[1, 2, 3])
        assert resp.json()["error"]["code"] == ERR_INVALID_REQUEST

    def test_missing_method(self):
        resp = make_client().post("/", json={"jsonrpc": "2.0", "id": "1"})
        assert resp.json()["error"]["code"] == ERR_INVALID_REQUEST

    def test_params_must_be_object(self):
        resp = make_client().post(
            "/",
            json={"jsonrpc": "2.0", "id": "1", "method": "tasks/get", "params": [1]},
        )
        assert resp.json()["error"]["code"] == ERR_INVALID_PARAMS

    def test_rpc_id_is_echoed(self):
        body = rpc(make_client(), "does/not/exist", rpc_id="xyz-42").json()
        assert body["id"] == "xyz-42"
        assert body["jsonrpc"] == "2.0"

    def test_rpc_errors_use_http_200(self):
        """A JSON-RPC error is a valid protocol response, not a transport fault."""
        assert rpc(make_client(), "does/not/exist").status_code == 200


# ── message/send ──────────────────────────────────────────────────────────────


class TestMessageSend:
    def test_completes_task(self):
        result = send_text(make_client(), "hello").json()["result"]
        assert result["state"] == "completed"

    def test_artifact_carries_agent_output(self):
        result = send_text(make_client(), "hello").json()["result"]
        assert result["artifacts"][0]["parts"][0]["content"] == "echo: hello"

    def test_history_contains_user_message(self):
        result = send_text(make_client(), "hi there").json()["result"]
        assert result["history"][0]["parts"][0]["content"] == "hi there"

    def test_missing_message_param(self):
        body = rpc(make_client(), "message/send", {}).json()
        assert body["error"]["code"] == ERR_INVALID_PARAMS

    def test_malformed_message_param(self):
        body = rpc(make_client(), "message/send", {"message": {"bad": 1}}).json()
        assert body["error"]["code"] == ERR_INVALID_PARAMS

    def test_skill_and_context_recorded(self):
        client = make_client()
        result = rpc(
            client,
            "message/send",
            {
                "message": Message.user_text("x").model_dump(mode="json"),
                "skillId": "echo",
                "contextId": "ctx-7",
            },
        ).json()["result"]
        assert result["skill_id"] == "echo"
        assert result["context_id"] == "ctx-7"

    def test_agent_exception_becomes_failed_task_not_rpc_error(self):
        """A raising agent must still return a Task the caller can inspect."""
        body = send_text(make_client(BoomAgent)).json()
        assert "error" not in body
        result = body["result"]
        assert result["state"] == "failed"
        assert "agent exploded" in result["error"]

    def test_task_is_retrievable_after_send(self):
        client = make_client()
        task_id = send_text(client).json()["result"]["id"]
        got = rpc(client, "tasks/get", {"taskId": task_id}).json()["result"]
        assert got["id"] == task_id


# ── tasks/get ─────────────────────────────────────────────────────────────────


class TestTasksGet:
    def test_unknown_task(self):
        body = rpc(make_client(), "tasks/get", {"taskId": "nope"}).json()
        assert body["error"]["code"] == ERR_TASK_NOT_FOUND

    def test_missing_task_id(self):
        body = rpc(make_client(), "tasks/get", {}).json()
        assert body["error"]["code"] == ERR_INVALID_PARAMS

    def test_accepts_id_alias(self):
        client = make_client()
        task_id = send_text(client).json()["result"]["id"]
        assert rpc(client, "tasks/get", {"id": task_id}).json()["result"]["id"] == task_id


# ── tasks/cancel ──────────────────────────────────────────────────────────────


class TestTasksCancel:
    def test_unknown_task(self):
        body = rpc(make_client(), "tasks/cancel", {"taskId": "nope"}).json()
        assert body["error"]["code"] == ERR_TASK_NOT_FOUND

    def test_missing_task_id(self):
        body = rpc(make_client(), "tasks/cancel", {}).json()
        assert body["error"]["code"] == ERR_INVALID_PARAMS

    def test_already_terminal_task_is_reported_not_faked(self):
        """EchoAgent completes inline, so the task is terminal by cancel time."""
        client = make_client()
        task_id = send_text(client).json()["result"]["id"]
        body = rpc(client, "tasks/cancel", {"taskId": task_id}).json()
        assert body["error"]["code"] == ERR_INVALID_REQUEST
        assert "terminal" in body["error"]["message"]


# ── run() return-value handling ───────────────────────────────────────────────


class TestInterpretOutput:
    def test_none_yields_nothing(self):
        assert _interpret_output(None) == (None, None, None)

    def test_str_becomes_text_artifact(self):
        artifact, reply, err = _interpret_output("hello")
        assert err is None and reply is None
        assert artifact.parts[0].type == PartType.TEXT
        assert artifact.parts[0].content == "hello"

    def test_dict_becomes_json_artifact(self):
        artifact, _, _ = _interpret_output({"k": "v"})
        assert artifact.parts[0].type == PartType.JSON
        assert artifact.parts[0].content == {"k": "v"}

    def test_list_becomes_json_artifact(self):
        artifact, _, _ = _interpret_output([1, 2])
        assert artifact.parts[0].type == PartType.JSON

    def test_artifact_passes_through(self):
        original = Artifact(
            name="custom", parts=[Part(type=PartType.TEXT, content="x")]
        )
        artifact, _, _ = _interpret_output(original)
        assert artifact is original

    def test_message_becomes_reply(self):
        msg = Message.agent_text("reply text")
        artifact, reply, err = _interpret_output(msg)
        assert artifact is None and err is None
        assert reply is msg

    def test_adapter_result_success(self):
        artifact, _, err = _interpret_output(AdapterResult(output="from adapter"))
        assert err is None
        assert artifact.parts[0].content == "from adapter"

    def test_adapter_result_error_fails_task(self):
        artifact, reply, err = _interpret_output(AdapterResult(error="framework blew up"))
        assert artifact is None and reply is None
        assert err == "framework blew up"

    def test_other_types_are_stringified(self):
        artifact, _, _ = _interpret_output(42)
        assert artifact.parts[0].content == "42"


class TestReturnValuesEndToEnd:
    def _serve(self, run_impl):
        @agent(name="Custom", description="Custom return.", url=SERVER_URL)
        class Custom:
            async def run(self, task: Task):
                return run_impl(task)

        return make_client(Custom)

    def test_none_completes_without_artifact(self):
        result = send_text(self._serve(lambda t: None)).json()["result"]
        assert result["state"] == "completed"
        assert result["artifacts"] == []

    def test_dict_return_is_json_artifact(self):
        result = send_text(self._serve(lambda t: {"ok": True})).json()["result"]
        assert result["artifacts"][0]["parts"][0]["content"] == {"ok": True}

    def test_adapter_result_error_marks_task_failed(self):
        client = self._serve(lambda t: AdapterResult(error="nope"))
        result = send_text(client).json()["result"]
        assert result["state"] == "failed"
        assert result["error"] == "nope"

    def test_message_return_is_appended_to_history(self):
        client = self._serve(lambda t: Message.agent_text("agent says hi"))
        result = send_text(client).json()["result"]
        assert result["state"] == "completed"
        assert result["history"][-1]["parts"][0]["content"] == "agent says hi"


# ── Security: defaults ────────────────────────────────────────────────────────


class TestSecurityDisabledByDefault:
    def test_no_security_is_open(self):
        assert send_text(make_client()).status_code == 200

    def test_default_middleware_reports_disabled(self):
        assert A2AServer(EchoAgent).security.enabled is False


# ── Security: authentication ──────────────────────────────────────────────────


def auth_client(**kwargs) -> TestClient:
    auth = AuthManager()
    auth.register_agent(
        CALLER_URL,
        AgentCredentialConfig(scheme=AuthScheme.API_KEY, api_key="right-key"),
    )
    return make_client(
        security=SecurityMiddleware(auth=auth, server_url=SERVER_URL, **kwargs)
    )


class TestSecurityAuth:
    def test_anonymous_request_rejected_401(self):
        resp = send_text(auth_client())
        assert resp.status_code == 401
        assert resp.json()["error"] == "MissingCallerError"

    def test_wrong_key_rejected_401(self):
        resp = send_text(
            auth_client(),
            headers={CALLER_HEADER: CALLER_URL, "X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401

    def test_missing_key_rejected_401(self):
        resp = send_text(auth_client(), headers={CALLER_HEADER: CALLER_URL})
        assert resp.status_code == 401

    def test_correct_key_accepted(self):
        resp = send_text(
            auth_client(),
            headers={CALLER_HEADER: CALLER_URL, "X-API-Key": "right-key"},
        )
        assert resp.status_code == 200
        assert resp.json()["result"]["state"] == "completed"

    def test_unregistered_caller_rejected_fail_closed(self):
        """AuthManager fails closed, so an unknown caller cannot slip through."""
        resp = send_text(
            auth_client(),
            headers={CALLER_HEADER: "http://stranger:1234", "X-API-Key": "right-key"},
        )
        assert resp.status_code == 401

    def test_agent_never_runs_when_auth_fails(self):
        """
        The refusal must short-circuit before any agent code executes.
        BoomAgent raises if reached, so a 401 with no 'result' proves it wasn't.
        """
        auth = AuthManager()
        auth.register_agent(
            CALLER_URL,
            AgentCredentialConfig(scheme=AuthScheme.API_KEY, api_key="right-key"),
        )
        client = make_client(
            BoomAgent,
            security=SecurityMiddleware(auth=auth, server_url=SERVER_URL),
        )
        resp = send_text(client)
        assert resp.status_code == 401
        assert "result" not in resp.json()

    def test_credential_is_not_echoed_in_error(self):
        resp = send_text(
            auth_client(),
            headers={CALLER_HEADER: CALLER_URL, "X-API-Key": "super-secret-value"},
        )
        assert "super-secret-value" not in resp.text


# ── Security: trust ───────────────────────────────────────────────────────────


class TestSecurityTrust:
    def _client(self, allow_caller: bool, skills: list[str] | None = None):
        trust = TrustBoundary()
        if allow_caller:
            trust.allow(CALLER_URL, SERVER_URL, skills=skills)
        return make_client(
            security=SecurityMiddleware(trust=trust, server_url=SERVER_URL)
        )

    def test_untrusted_caller_rejected_403(self):
        resp = send_text(self._client(allow_caller=False),
                         headers={CALLER_HEADER: CALLER_URL})
        assert resp.status_code == 403

    def test_trusted_caller_allowed(self):
        resp = send_text(self._client(allow_caller=True),
                         headers={CALLER_HEADER: CALLER_URL})
        assert resp.status_code == 200

    def test_skill_outside_acl_rejected_403(self):
        client = self._client(allow_caller=True, skills=["other-skill"])
        resp = rpc(
            client,
            "message/send",
            {
                "message": Message.user_text("x").model_dump(mode="json"),
                "skillId": "echo",
            },
            headers={CALLER_HEADER: CALLER_URL},
        )
        assert resp.status_code == 403

    def test_skill_inside_acl_allowed(self):
        client = self._client(allow_caller=True, skills=["echo"])
        resp = rpc(
            client,
            "message/send",
            {
                "message": Message.user_text("x").model_dump(mode="json"),
                "skillId": "echo",
            },
            headers={CALLER_HEADER: CALLER_URL},
        )
        assert resp.status_code == 200

    def test_trust_requires_server_url(self):
        with pytest.raises(ValueError, match="server_url"):
            SecurityMiddleware(trust=TrustBoundary())


# ── Security: rate limiting ───────────────────────────────────────────────────


class TestSecurityRateLimit:
    def _client(self):
        limiter = RateLimiter(RateLimitConfig(rate=1.0, burst=2))
        return make_client(security=SecurityMiddleware(rate_limiter=limiter))

    def test_burst_allowed_then_throttled(self):
        client = self._client()
        headers = {CALLER_HEADER: CALLER_URL}
        assert send_text(client, headers=headers).status_code == 200
        assert send_text(client, headers=headers).status_code == 200
        assert send_text(client, headers=headers).status_code == 429

    def test_throttled_response_has_retry_after(self):
        client = self._client()
        headers = {CALLER_HEADER: CALLER_URL}
        for _ in range(3):
            resp = send_text(client, headers=headers)
        assert resp.status_code == 429
        assert int(resp.headers["Retry-After"]) >= 1

    def test_limits_are_per_caller(self):
        client = self._client()
        for _ in range(3):
            send_text(client, headers={CALLER_HEADER: CALLER_URL})
        other = send_text(client, headers={CALLER_HEADER: "http://other:1"})
        assert other.status_code == 200


# ── Security: payload ─────────────────────────────────────────────────────────


class TestSecurityPayload:
    def test_oversized_body_rejected_413_before_parsing(self):
        client = make_client(
            security=SecurityMiddleware(
                validator=PayloadValidator(ValidatorConfig(max_bytes=200))
            )
        )
        resp = client.post("/", content=b"x" * 5000,
                           headers={"Content-Type": "application/json"})
        assert resp.status_code == 413

    def test_blank_text_part_rejected_400(self):
        client = make_client(security=SecurityMiddleware(validator=PayloadValidator()))
        resp = send_text(client, "    ")
        assert resp.status_code == 400

    def test_too_many_parts_rejected_400(self):
        client = make_client(
            security=SecurityMiddleware(
                validator=PayloadValidator(ValidatorConfig(max_parts=2))
            )
        )
        message = Message(
            role="user",
            parts=[Part(type=PartType.TEXT, content=f"p{i}") for i in range(5)],
        )
        resp = rpc(
            client, "message/send", {"message": message.model_dump(mode="json")}
        )
        assert resp.status_code == 400

    def test_valid_payload_passes(self):
        client = make_client(security=SecurityMiddleware(validator=PayloadValidator()))
        assert send_text(client, "fine").status_code == 200

    def test_text_is_stripped_by_validator(self):
        client = make_client(security=SecurityMiddleware(validator=PayloadValidator()))
        result = send_text(client, "  padded  ").json()["result"]
        assert result["artifacts"][0]["parts"][0]["content"] == "echo: padded"


# ── Security: full chain ──────────────────────────────────────────────────────


class TestSecurityFullChain:
    def _client(self):
        auth = AuthManager()
        auth.register_agent(
            CALLER_URL,
            AgentCredentialConfig(scheme=AuthScheme.API_KEY, api_key="k"),
        )
        trust = TrustBoundary()
        trust.allow(CALLER_URL, SERVER_URL, skills=["echo"])
        return make_client(
            security=SecurityMiddleware(
                auth=auth,
                trust=trust,
                rate_limiter=RateLimiter(RateLimitConfig(rate=100.0, burst=100)),
                validator=PayloadValidator(),
                server_url=SERVER_URL,
            )
        )

    def _ok_headers(self):
        return {CALLER_HEADER: CALLER_URL, "X-API-Key": "k"}

    def test_fully_authorised_request_succeeds(self):
        resp = rpc(
            self._client(),
            "message/send",
            {
                "message": Message.user_text("hello").model_dump(mode="json"),
                "skillId": "echo",
            },
            headers=self._ok_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["result"]["state"] == "completed"

    def test_summary_reports_which_stages_are_active(self):
        middleware = SecurityMiddleware(
            auth=AuthManager(), validator=PayloadValidator()
        )
        summary = middleware.summary()
        assert summary["auth"] is True
        assert summary["validation"] is True
        assert summary["trust"] is False
        assert summary["rate_limit"] is False

    def test_rate_limit_precedes_auth(self):
        """A flood must be shed before it can force credential checks."""
        limiter = RateLimiter(RateLimitConfig(rate=0.001, burst=1))
        auth = AuthManager()
        client = make_client(
            security=SecurityMiddleware(
                auth=auth, rate_limiter=limiter, server_url=SERVER_URL
            )
        )
        headers = {CALLER_HEADER: CALLER_URL}
        first = send_text(client, headers=headers)
        second = send_text(client, headers=headers)
        # First is refused by auth (unregistered), second by the rate limiter.
        assert first.status_code == 401
        assert second.status_code == 429


# ── Wire-format compatibility ─────────────────────────────────────────────────


class TestWireFormat:
    """The response must round-trip through the client's own models."""

    def test_task_result_parses_as_task(self):
        result = send_text(make_client()).json()["result"]
        assert Task.model_validate(result).state.value == "completed"

    def test_result_is_json_serialisable(self):
        result = send_text(make_client()).json()["result"]
        json.dumps(result)  # must not raise
