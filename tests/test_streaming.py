"""
tests/test_streaming.py

Server-side streaming: the message/stream JSON-RPC method and the
GET /stream?taskId=... endpoint.

Also covers the cross-compatibility rules that keep the two calling styles
independent of how an agent is written:
  - a streaming agent must still work over message/send
  - a non-streaming agent must still work over message/stream
"""

from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient

from nexus_a2a import agent
from nexus_a2a.core.a2a_server import (
    ERR_INVALID_PARAMS,
    STREAM_PATH,
    A2AServer,
    _chunk_text,
    _join_chunks,
    streaming_callable,
)
from nexus_a2a.models.agent import AuthScheme
from nexus_a2a.models.task import Artifact, Message, Part, PartType, Task
from nexus_a2a.security.auth import AgentCredentialConfig, AuthManager
from nexus_a2a.security.middleware import CALLER_HEADER, SecurityMiddleware
from nexus_a2a.security.rate_limiter import RateLimitConfig, RateLimiter
from nexus_a2a.security.validator import PayloadValidator, ValidatorConfig
from nexus_a2a.transport.sse import StreamEventType

SERVER_URL = "http://stream-agent:8001"
CALLER_URL = "http://caller-agent:9001"
WORDS = ["Hello", ", ", "world", "!"]

SKILLS = [{"id": "write", "name": "Write", "description": "Write text."}]


# ── Test agents ───────────────────────────────────────────────────────────────


@agent(name="Writer", description="Streams words.", streaming=True,
       skills=SKILLS, url=SERVER_URL)
class WriterAgent:
    async def run(self, task: Task):
        for word in WORDS:
            yield word


@agent(name="StreamMethod", description="Streams via stream().",
       streaming=True, skills=SKILLS, url=SERVER_URL)
class StreamMethodAgent:
    async def run(self, task: Task) -> str:
        return "".join(WORDS)

    async def stream(self, task: Task):
        for word in WORDS:
            yield word


@agent(name="Plain", description="Does not stream.", skills=SKILLS, url=SERVER_URL)
class PlainAgent:
    async def run(self, task: Task) -> str:
        return "one shot"


@agent(name="BoomStream", description="Fails mid-stream.", streaming=True,
       skills=SKILLS, url=SERVER_URL)
class BoomStreamAgent:
    async def run(self, task: Task):
        yield "partial"
        raise RuntimeError("stream exploded")


@agent(name="JsonStream", description="Streams dicts.", streaming=True,
       skills=SKILLS, url=SERVER_URL)
class JsonStreamAgent:
    async def run(self, task: Task):
        yield {"step": 1}
        yield {"step": 2}


def make_client(
    agent_cls: type = WriterAgent,
    security: SecurityMiddleware | None = None,
) -> TestClient:
    return TestClient(A2AServer(agent_cls, security=security).app)


def parse_sse(text: str) -> list[dict]:
    """Parse an SSE response body into its JSON event payloads, in order."""
    events = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            events.append(json.loads(line[len("data:"):].strip()))
    return events


def stream_rpc(
    client: TestClient,
    text: str = "go",
    params_extra: dict | None = None,
    headers: dict | None = None,
):
    params = {"message": Message.user_text(text).model_dump(mode="json")}
    params.update(params_extra or {})
    return client.post(
        "/",
        json={"jsonrpc": "2.0", "id": "1", "method": "message/stream",
              "params": params},
        headers=headers or {},
    )


def send_rpc(client: TestClient, text: str = "go", headers: dict | None = None):
    return client.post(
        "/",
        json={
            "jsonrpc": "2.0", "id": "1", "method": "message/send",
            "params": {"message": Message.user_text(text).model_dump(mode="json")},
        },
        headers=headers or {},
    )


def types_of(events: list[dict]) -> list[str]:
    return [e["type"] for e in events]


def chunks_of(events: list[dict]) -> list[str]:
    return [
        e["content"] for e in events
        if e["type"] == StreamEventType.ARTIFACT_CHUNK.value
    ]


# ── Agent-shape detection ─────────────────────────────────────────────────────


class TestStreamingCallable:
    def test_async_generator_run_detected(self):
        assert streaming_callable(WriterAgent()) is not None

    def test_stream_method_preferred_over_run(self):
        found = streaming_callable(StreamMethodAgent())
        assert found is not None
        assert found.__name__ == "stream"

    def test_plain_run_is_not_streaming(self):
        assert streaming_callable(PlainAgent()) is None

    def test_streaming_class_attr_is_not_a_generator(self):
        """STREAMING = True means the adapter handles it, not the server."""

        @agent(name="AdapterStream", description="Adapter streams.", url=SERVER_URL)
        class AdapterStream:
            STREAMING = True

            async def run(self, task: Task) -> str:
                return "x"

        assert streaming_callable(AdapterStream()) is None


class TestDecoratorAcceptsAsyncGenerator:
    def test_async_generator_run_is_a_valid_agent(self):
        """iscoroutinefunction() is False for async generators."""
        assert WriterAgent.get_agent_card().name == "Writer"

    def test_server_accepts_async_generator_agent(self):
        assert A2AServer(WriterAgent).card.name == "Writer"


# ── Chunk folding / rendering ─────────────────────────────────────────────────


class TestJoinChunks:
    def test_empty_is_none(self):
        assert _join_chunks([]) is None

    def test_strings_are_concatenated(self):
        assert _join_chunks(["a", "b", "c"]) == "abc"

    def test_single_non_string_passes_through(self):
        assert _join_chunks([{"k": 1}]) == {"k": 1}

    def test_mixed_chunks_become_a_list(self):
        assert _join_chunks(["a", {"k": 1}]) == ["a", {"k": 1}]


class TestChunkText:
    def test_string_passes_through(self):
        assert _chunk_text("hi") == "hi"

    def test_dict_is_json_encoded(self):
        assert json.loads(_chunk_text({"k": 1})) == {"k": 1}

    def test_artifact_renders_parts(self):
        art = Artifact(name="a", parts=[Part(type=PartType.TEXT, content="x")])
        assert _chunk_text(art) == "x"

    def test_message_renders_text(self):
        assert _chunk_text(Message.agent_text("hello")) == "hello"

    def test_none_is_json_null(self):
        assert _chunk_text(None) == "null"


# ── message/stream ────────────────────────────────────────────────────────────


class TestMessageStream:
    def test_content_type_is_event_stream(self):
        resp = stream_rpc(make_client())
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

    def test_event_sequence(self):
        events = types_of(parse_sse(stream_rpc(make_client()).text))
        assert events[0] == "task_created"
        assert events[-2:] == ["task_status", "done"]

    def test_one_chunk_per_yield(self):
        assert chunks_of(parse_sse(stream_rpc(make_client()).text)) == WORDS

    def test_task_created_carries_task_id(self):
        events = parse_sse(stream_rpc(make_client()).text)
        assert "id" in events[0]
        assert events[0]["state"] == "working"

    def test_terminal_status_is_completed(self):
        events = parse_sse(stream_rpc(make_client()).text)
        status = [e for e in events if e["type"] == "task_status"][-1]
        assert status["state"] == "completed"

    def test_chunks_are_indexed(self):
        events = parse_sse(stream_rpc(make_client()).text)
        idx = [e["index"] for e in events if e["type"] == "artifact_chunk"]
        assert idx == list(range(len(WORDS)))

    def test_stream_method_agent_streams(self):
        assert chunks_of(parse_sse(stream_rpc(make_client(StreamMethodAgent)).text)) == WORDS

    def test_json_chunks_are_machine_readable(self):
        chunks = chunks_of(parse_sse(stream_rpc(make_client(JsonStreamAgent)).text))
        assert [json.loads(c) for c in chunks] == [{"step": 1}, {"step": 2}]

    def test_task_is_persisted_and_retrievable(self):
        client = make_client()
        events = parse_sse(stream_rpc(client).text)
        task_id = events[0]["id"]
        got = client.post(
            "/",
            json={"jsonrpc": "2.0", "id": "2", "method": "tasks/get",
                  "params": {"taskId": task_id}},
        ).json()["result"]
        assert got["state"] == "completed"
        assert got["artifacts"][0]["parts"][0]["content"] == "".join(WORDS)

    def test_skill_and_context_are_recorded(self):
        resp = stream_rpc(
            make_client(),
            params_extra={"skillId": "write", "contextId": "ctx-9"},
        )
        created = parse_sse(resp.text)[0]
        assert created["skill_id"] == "write"
        assert created["context_id"] == "ctx-9"


class TestNonStreamingAgentOverStream:
    """A non-streaming agent must still be usable through message/stream."""

    def test_output_arrives_as_one_chunk(self):
        assert chunks_of(parse_sse(stream_rpc(make_client(PlainAgent)).text)) == ["one shot"]

    def test_still_terminates_cleanly(self):
        events = types_of(parse_sse(stream_rpc(make_client(PlainAgent)).text))
        assert events[-2:] == ["task_status", "done"]


class TestStreamingAgentOverSend:
    """A streaming agent must still be usable through message/send."""

    def test_chunks_are_folded_into_one_artifact(self):
        result = send_rpc(make_client()).json()["result"]
        assert result["state"] == "completed"
        assert result["artifacts"][0]["parts"][0]["content"] == "".join(WORDS)

    def test_stream_method_agent_uses_run_for_send(self):
        result = send_rpc(make_client(StreamMethodAgent)).json()["result"]
        assert result["state"] == "completed"

    def test_failing_stream_agent_fails_the_task(self):
        result = send_rpc(make_client(BoomStreamAgent)).json()["result"]
        assert result["state"] == "failed"
        assert "stream exploded" in result["error"]


# ── Failures ──────────────────────────────────────────────────────────────────


class TestStreamFailures:
    def test_mid_stream_error_emits_error_event(self):
        events = parse_sse(stream_rpc(make_client(BoomStreamAgent)).text)
        assert types_of(events)[-1] == "error"
        assert "stream exploded" in events[-1]["message"]

    def test_chunks_before_the_error_are_still_delivered(self):
        events = parse_sse(stream_rpc(make_client(BoomStreamAgent)).text)
        assert chunks_of(events) == ["partial"]

    def test_task_is_marked_failed_after_stream_error(self):
        client = make_client(BoomStreamAgent)
        events = parse_sse(stream_rpc(client).text)
        task_id = events[0]["id"]
        got = client.post(
            "/",
            json={"jsonrpc": "2.0", "id": "2", "method": "tasks/get",
                  "params": {"taskId": task_id}},
        ).json()["result"]
        assert got["state"] == "failed"

    def test_missing_message_is_an_rpc_error_not_a_stream(self):
        resp = make_client().post(
            "/",
            json={"jsonrpc": "2.0", "id": "1", "method": "message/stream",
                  "params": {}},
        )
        assert resp.headers["content-type"].startswith("application/json")
        assert resp.json()["error"]["code"] == ERR_INVALID_PARAMS

    def test_malformed_message_is_an_rpc_error(self):
        resp = stream_rpc(make_client())
        assert resp.status_code == 200  # sanity: the good path streams
        bad = make_client().post(
            "/",
            json={"jsonrpc": "2.0", "id": "1", "method": "message/stream",
                  "params": {"message": {"bad": 1}}},
        )
        assert bad.json()["error"]["code"] == ERR_INVALID_PARAMS


# ── Security runs before the stream opens ─────────────────────────────────────


class TestStreamSecurity:
    """
    Once SSE starts the status line is already sent, so every refusal must
    happen before the first byte of the stream.
    """

    def _auth_client(self, agent_cls: type = WriterAgent) -> TestClient:
        auth = AuthManager()
        auth.register_agent(
            CALLER_URL,
            AgentCredentialConfig(scheme=AuthScheme.API_KEY, api_key="k"),
        )
        return make_client(
            agent_cls, SecurityMiddleware(auth=auth, server_url=SERVER_URL)
        )

    def test_anonymous_stream_rejected_401(self):
        resp = stream_rpc(self._auth_client())
        assert resp.status_code == 401
        assert not resp.headers["content-type"].startswith("text/event-stream")

    def test_wrong_key_rejected_401(self):
        resp = stream_rpc(
            self._auth_client(),
            headers={CALLER_HEADER: CALLER_URL, "X-API-Key": "wrong"},
        )
        assert resp.status_code == 401

    def test_authorised_stream_succeeds(self):
        resp = stream_rpc(
            self._auth_client(),
            headers={CALLER_HEADER: CALLER_URL, "X-API-Key": "k"},
        )
        assert resp.status_code == 200
        assert chunks_of(parse_sse(resp.text)) == WORDS

    def test_agent_never_runs_when_refused(self):
        """BoomStreamAgent would raise if reached."""
        resp = stream_rpc(self._auth_client(BoomStreamAgent))
        assert resp.status_code == 401
        assert "partial" not in resp.text

    def test_rate_limit_refuses_before_streaming(self):
        client = make_client(
            security=SecurityMiddleware(
                rate_limiter=RateLimiter(RateLimitConfig(rate=1.0, burst=1))
            )
        )
        headers = {CALLER_HEADER: CALLER_URL}
        assert stream_rpc(client, headers=headers).status_code == 200
        throttled = stream_rpc(client, headers=headers)
        assert throttled.status_code == 429

    def test_oversized_body_refused_before_streaming(self):
        client = make_client(
            security=SecurityMiddleware(
                validator=PayloadValidator(ValidatorConfig(max_bytes=200))
            )
        )
        resp = client.post(
            "/", content=b"x" * 5000,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 413


# ── GET /stream ───────────────────────────────────────────────────────────────


class TestStreamGetEndpoint:
    def _completed_task_id(self, client: TestClient) -> str:
        return send_rpc(client).json()["result"]["id"]

    def test_returns_event_stream(self):
        client = make_client()
        task_id = self._completed_task_id(client)
        resp = client.get(STREAM_PATH, params={"taskId": task_id})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

    def test_reports_task_state_and_closes(self):
        client = make_client()
        task_id = self._completed_task_id(client)
        events = parse_sse(client.get(STREAM_PATH, params={"taskId": task_id}).text)
        assert types_of(events)[0] == "task_created"
        assert types_of(events)[-1] == "done"

    def test_replays_artifacts(self):
        client = make_client()
        task_id = self._completed_task_id(client)
        events = parse_sse(client.get(STREAM_PATH, params={"taskId": task_id}).text)
        assert chunks_of(events) == ["".join(WORDS)]

    def test_accepts_id_alias(self):
        client = make_client()
        task_id = self._completed_task_id(client)
        assert client.get(STREAM_PATH, params={"id": task_id}).status_code == 200

    def test_missing_task_id_is_400(self):
        assert make_client().get(STREAM_PATH).status_code == 400

    def test_unknown_task_is_404(self):
        resp = make_client().get(STREAM_PATH, params={"taskId": "nope"})
        assert resp.status_code == 404

    def test_security_applies(self):
        auth = AuthManager()
        client = make_client(
            security=SecurityMiddleware(auth=auth, server_url=SERVER_URL)
        )
        resp = client.get(STREAM_PATH, params={"taskId": "anything"})
        assert resp.status_code == 401


# ── Wire format round-trip ────────────────────────────────────────────────────


class TestWireFormat:
    def test_events_parse_with_the_client_parser(self):
        """Everything the server emits must parse via the shared SSE parser."""
        from nexus_a2a.transport.sse import parse_sse_data

        body = stream_rpc(make_client()).text
        raw_lines = [
            line.strip()[len("data:"):].strip()
            for line in body.splitlines()
            if line.strip().startswith("data:")
        ]
        parsed = [parse_sse_data(r) for r in raw_lines]
        assert all(p is not None for p in parsed)
        assert parsed[-1].type == StreamEventType.DONE
        assert parsed[-1].is_terminal

    def test_task_created_event_parses_as_a_task(self):
        from nexus_a2a.transport.sse import parse_sse_data

        first = [
            line.strip()[len("data:"):].strip()
            for line in stream_rpc(make_client()).text.splitlines()
            if line.strip().startswith("data:")
        ][0]
        event = parse_sse_data(first)
        assert event is not None
        assert event.as_task() is not None


@pytest.mark.parametrize("agent_cls", [WriterAgent, StreamMethodAgent, PlainAgent])
def test_every_agent_shape_works_on_both_methods(agent_cls: type):
    """The two calling styles must not depend on how the agent is written."""
    client = make_client(agent_cls)
    assert send_rpc(client).json()["result"]["state"] == "completed"
    assert types_of(parse_sse(stream_rpc(client).text))[-1] == "done"
