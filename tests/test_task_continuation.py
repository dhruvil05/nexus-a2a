"""
tests/test_task_continuation.py

Multi-turn conversations and INPUT_REQUIRED.

These are one mechanism, not two: COMPLETED is terminal in the task state
machine, so an agent cannot add a turn to a finished task. It adds turns by
returning NeedsInput instead of completing, which parks the task at
INPUT_REQUIRED, and the caller answers against the same task id.
"""

from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient

from nexus_a2a import agent
from nexus_a2a.core.a2a_server import (
    ERR_INVALID_PARAMS,
    ERR_TASK_NOT_FOUND,
    ERR_TASK_NOT_WAITING,
    A2AServer,
)
from nexus_a2a.core.input_handler import InputHandler
from nexus_a2a.models.task import Message, NeedsInput, Task
from nexus_a2a.transport.sse import StreamEventType

SERVER_URL = "http://planner:8001"


# ── Test agents ───────────────────────────────────────────────────────────────


def _user_turns(task: Task) -> int:
    return sum(1 for m in task.history if m.role == "user")


@agent(name="Planner", description="Asks two questions, then answers.",
       url=SERVER_URL)
class PlannerAgent:
    async def run(self, task: Task):
        turns = _user_turns(task)
        if turns == 1:
            return NeedsInput("What is your budget?")
        if turns == 2:
            return NeedsInput("Which city?")
        return f"Plan: {task.history[4].text()} on {task.history[2].text()}"


@agent(name="OneShot", description="Never asks.", url=SERVER_URL)
class OneShotAgent:
    async def run(self, task: Task) -> str:
        return "done immediately"


@agent(name="NoMultiTurn", description="Declines continuations.",
       multi_turn=False, url=SERVER_URL)
class NoMultiTurnAgent:
    async def run(self, task: Task):
        if _user_turns(task) == 1:
            return NeedsInput("Anything else?")
        return "second turn"


@agent(name="StreamAsk", description="Streams then asks.", streaming=True,
       url=SERVER_URL)
class StreamAskAgent:
    async def run(self, task: Task):
        if _user_turns(task) == 1:
            yield "thinking..."
            yield NeedsInput("Confirm?")
        else:
            yield "confirmed!"


@agent(name="PromptMessage", description="Asks with a Message object.",
       url=SERVER_URL)
class PromptMessageAgent:
    async def run(self, task: Task):
        if _user_turns(task) == 1:
            return NeedsInput(Message.agent_text("structured prompt"))
        return "ok"


def make_client(agent_cls: type = PlannerAgent, **kwargs) -> TestClient:
    return TestClient(A2AServer(agent_cls, **kwargs).app)


def send(client: TestClient, text: str, task_id: str | None = None, method="message/send"):
    params: dict = {"message": Message.user_text(text).model_dump(mode="json")}
    if task_id:
        params["taskId"] = task_id
    return client.post(
        "/", json={"jsonrpc": "2.0", "id": "1", "method": method, "params": params}
    )


def result_of(resp) -> dict:
    body = resp.json()
    assert "result" in body, body
    return body["result"]


def parse_sse(text: str) -> list[dict]:
    return [
        json.loads(line.strip()[len("data:"):].strip())
        for line in text.splitlines()
        if line.strip().startswith("data:")
    ]


# ── NeedsInput marker ─────────────────────────────────────────────────────────


class TestNeedsInputMarker:
    def test_string_prompt_becomes_agent_message(self):
        msg = NeedsInput("hello?").as_message()
        assert msg.text() == "hello?"
        assert msg.role == "agent"

    def test_message_prompt_passes_through(self):
        original = Message.agent_text("keep me")
        assert NeedsInput(original).as_message() is original

    def test_is_exported_from_package_root(self):
        from nexus_a2a import NeedsInput as Exported

        assert Exported is NeedsInput


# ── Pausing ───────────────────────────────────────────────────────────────────


class TestPausing:
    def test_task_parks_at_input_required(self):
        assert result_of(send(make_client(), "plan"))["state"] == "input_required"

    def test_prompt_is_appended_to_history(self):
        result = result_of(send(make_client(), "plan"))
        assert result["history"][-1]["parts"][0]["content"] == "What is your budget?"
        assert result["history"][-1]["role"] == "agent"

    def test_no_artifact_while_paused(self):
        assert result_of(send(make_client(), "plan"))["artifacts"] == []

    def test_message_object_prompt_works(self):
        result = result_of(send(make_client(PromptMessageAgent), "go"))
        assert result["history"][-1]["parts"][0]["content"] == "structured prompt"

    def test_paused_task_is_retrievable(self):
        client = make_client()
        task_id = result_of(send(client, "plan"))["id"]
        got = client.post(
            "/", json={"jsonrpc": "2.0", "id": "2", "method": "tasks/get",
                       "params": {"taskId": task_id}},
        ).json()["result"]
        assert got["state"] == "input_required"


# ── Resuming ──────────────────────────────────────────────────────────────────


class TestResuming:
    def test_second_turn_asks_again(self):
        client = make_client()
        first = result_of(send(client, "plan"))
        second = result_of(send(client, "$500", task_id=first["id"]))
        assert second["state"] == "input_required"
        assert second["history"][-1]["parts"][0]["content"] == "Which city?"

    def test_third_turn_completes(self):
        client = make_client()
        t = result_of(send(client, "plan"))
        t = result_of(send(client, "$500", task_id=t["id"]))
        t = result_of(send(client, "Lisbon", task_id=t["id"]))
        assert t["state"] == "completed"
        assert t["artifacts"][0]["parts"][0]["content"] == "Plan: Lisbon on $500"

    def test_task_id_is_stable_across_turns(self):
        client = make_client()
        first = result_of(send(client, "plan"))
        second = result_of(send(client, "$500", task_id=first["id"]))
        assert second["id"] == first["id"]

    def test_full_history_accumulates(self):
        client = make_client()
        t = result_of(send(client, "plan"))
        t = result_of(send(client, "$500", task_id=t["id"]))
        t = result_of(send(client, "Lisbon", task_id=t["id"]))
        roles = [m["role"] for m in t["history"]]
        assert roles == ["user", "agent", "user", "agent", "user"]

    def test_agent_sees_earlier_answers(self):
        """The final answer is built from turns 2 and 3."""
        client = make_client()
        t = result_of(send(client, "plan"))
        t = result_of(send(client, "$900", task_id=t["id"]))
        t = result_of(send(client, "Porto", task_id=t["id"]))
        assert "Porto" in t["artifacts"][0]["parts"][0]["content"]
        assert "$900" in t["artifacts"][0]["parts"][0]["content"]

    def test_context_id_survives_continuation(self):
        client = make_client()
        first = client.post(
            "/", json={"jsonrpc": "2.0", "id": "1", "method": "message/send",
                       "params": {
                           "message": Message.user_text("plan").model_dump(mode="json"),
                           "contextId": "ctx-77"}},
        ).json()["result"]
        second = result_of(send(client, "$500", task_id=first["id"]))
        assert second["context_id"] == "ctx-77"


# ── Rejected continuations ────────────────────────────────────────────────────


class TestRejectedContinuations:
    def test_unknown_task_id(self):
        body = send(make_client(), "x", task_id="nope").json()
        assert body["error"]["code"] == ERR_TASK_NOT_FOUND

    def test_completed_task_cannot_be_continued(self):
        client = make_client(OneShotAgent)
        done = result_of(send(client, "go"))
        assert done["state"] == "completed"
        body = send(client, "more", task_id=done["id"]).json()
        assert body["error"]["code"] == ERR_TASK_NOT_WAITING
        assert "completed" in body["error"]["message"]

    def test_error_message_suggests_starting_a_new_task(self):
        client = make_client(OneShotAgent)
        done = result_of(send(client, "go"))
        body = send(client, "more", task_id=done["id"]).json()
        assert "omit 'taskid'" in body["error"]["message"].lower()

    def test_non_string_task_id(self):
        resp = make_client().post(
            "/", json={"jsonrpc": "2.0", "id": "1", "method": "message/send",
                       "params": {
                           "message": Message.user_text("x").model_dump(mode="json"),
                           "taskId": 123}},
        )
        assert resp.json()["error"]["code"] == ERR_INVALID_PARAMS

    def test_agent_declaring_no_multi_turn_refuses(self):
        client = make_client(NoMultiTurnAgent)
        first = result_of(send(client, "go"))
        body = send(client, "again", task_id=first["id"]).json()
        assert body["error"]["code"] == ERR_TASK_NOT_WAITING
        assert "multi_turn=False" in body["error"]["message"]


# ── multi_turn is honoured ────────────────────────────────────────────────────


class TestMultiTurnCapability:
    def test_card_defaults_to_multi_turn_true(self):
        card = A2AServer(PlannerAgent).card
        assert card.capabilities.multi_turn is True

    def test_default_agent_accepts_continuation(self):
        """The default claim must now be truthful."""
        client = make_client()
        first = result_of(send(client, "plan"))
        assert "error" not in send(client, "$500", task_id=first["id"]).json()

    def test_opting_out_is_advertised_on_the_card(self):
        data = make_client(NoMultiTurnAgent).get(
            "/.well-known/agent-card.json"
        ).json()
        assert data["capabilities"]["multi_turn"] is False


# ── Streaming ─────────────────────────────────────────────────────────────────


class TestStreamingContinuation:
    def test_stream_pauses_at_input_required(self):
        events = parse_sse(send(make_client(StreamAskAgent), "go",
                                method="message/stream").text)
        status = [e for e in events if e["type"] == "task_status"][-1]
        assert status["state"] == "input_required"

    def test_prompt_is_emitted_as_a_message_event(self):
        events = parse_sse(send(make_client(StreamAskAgent), "go",
                                method="message/stream").text)
        msgs = [e for e in events if e["type"] == StreamEventType.MESSAGE.value]
        assert msgs and msgs[0]["content"] == "Confirm?"

    def test_chunks_before_the_pause_are_delivered(self):
        events = parse_sse(send(make_client(StreamAskAgent), "go",
                                method="message/stream").text)
        chunks = [e["content"] for e in events if e["type"] == "artifact_chunk"]
        assert chunks == ["thinking..."]

    def test_needs_input_is_not_emitted_as_a_chunk(self):
        events = parse_sse(send(make_client(StreamAskAgent), "go",
                                method="message/stream").text)
        chunks = [e["content"] for e in events if e["type"] == "artifact_chunk"]
        assert not any("NeedsInput" in c for c in chunks)

    def test_stream_still_terminates_with_done(self):
        events = parse_sse(send(make_client(StreamAskAgent), "go",
                                method="message/stream").text)
        assert events[-1]["type"] == "done"

    def test_stream_continuation_completes(self):
        client = make_client(StreamAskAgent)
        first = parse_sse(send(client, "go", method="message/stream").text)
        task_id = first[0]["id"]
        second = parse_sse(
            send(client, "yes", task_id=task_id, method="message/stream").text
        )
        chunks = [e["content"] for e in second if e["type"] == "artifact_chunk"]
        assert chunks == ["confirmed!"]
        status = [e for e in second if e["type"] == "task_status"][-1]
        assert status["state"] == "completed"

    def test_paused_stream_task_can_be_continued_over_send(self):
        """The two methods interoperate on the same task."""
        client = make_client(StreamAskAgent)
        first = parse_sse(send(client, "go", method="message/stream").text)
        resumed = result_of(send(client, "yes", task_id=first[0]["id"]))
        assert resumed["state"] == "completed"


# ── InputHandler suspend path ─────────────────────────────────────────────────


class TestInputHandlerBridge:
    """
    An agent that suspends inside InputHandler.wait_for_input() is resumed by
    the same wire call — the server fires its event instead of re-invoking run().
    """

    def test_server_exposes_an_input_handler(self):
        assert isinstance(A2AServer(PlannerAgent).input_handler, InputHandler)

    def test_custom_input_handler_is_used(self):
        server = A2AServer(PlannerAgent)
        handler = InputHandler(server.tasks)
        server2 = A2AServer(PlannerAgent, input_handler=handler)
        assert server2.input_handler is handler

    async def test_waiting_task_is_resumed_not_rerun(self):
        """submit_reply() path: the parked coroutine finishes the task."""
        import asyncio

        server = A2AServer(PlannerAgent)
        task = await server.tasks.create(initial_message=Message.user_text("hi"))
        await server.tasks.start(task.id)

        handler = server.input_handler
        waiter = asyncio.create_task(
            handler.wait_for_input(task.id, Message.agent_text("q?"), timeout=5.0)
        )
        await asyncio.sleep(0.05)
        assert handler.is_waiting(task.id)

        resumed, error, was_resumed = await server._begin_task(
            Message.user_text("my answer"), {"taskId": task.id}
        )
        assert error is None
        assert was_resumed is True

        reply = await waiter
        assert reply.text() == "my answer"
        assert not handler.is_waiting(task.id)


# ── Regression: single-turn behaviour is unchanged ────────────────────────────


class TestSingleTurnUnaffected:
    def test_agent_that_never_asks_still_completes(self):
        assert result_of(send(make_client(OneShotAgent), "go"))["state"] == "completed"

    def test_omitting_task_id_starts_a_fresh_task(self):
        client = make_client(OneShotAgent)
        first = result_of(send(client, "one"))
        second = result_of(send(client, "two"))
        assert first["id"] != second["id"]


@pytest.mark.parametrize("method", ["message/send", "message/stream"])
def test_pause_and_resume_works_on_both_methods(method: str):
    client = make_client(StreamAskAgent)
    first = send(client, "go", method=method)
    task_id = (
        parse_sse(first.text)[0]["id"]
        if method == "message/stream"
        else result_of(first)["id"]
    )
    second = send(client, "yes", task_id=task_id, method=method)
    if method == "message/stream":
        assert parse_sse(second.text)[-1]["type"] == "done"
    else:
        assert result_of(second)["state"] == "completed"
