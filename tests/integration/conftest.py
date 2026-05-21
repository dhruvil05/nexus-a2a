"""
tests/integration/conftest.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Real in-process Starlette servers for integration tests.
asyncio_mode=auto — plain @pytest.fixture works for async fixtures.

KEY DESIGN INSIGHT:
  StepResult.succeeded = (error is None and task is not None)
  _run_step only sets error when the runner RAISES an exception.
  So fail_tasks=True agents must return a JSON-RPC ERROR response
  (which causes A2AHttpClient to raise RemoteAgentError), NOT a task
  dict with state="failed" (which would be parsed as a valid Task
  and counted as succeeded by the orchestrator).
"""

from __future__ import annotations

import asyncio
import socket
import uuid
from datetime import UTC, datetime
from typing import AsyncGenerator

import pytest
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _make_task_dict(task_id: str, state: str, response_text: str) -> dict:
    """Build dict matching real Task Pydantic schema exactly."""
    artifact_part = {"type": "text", "content": response_text, "mime_type": None}
    agent_message = {
        "id": str(uuid.uuid4()),
        "role": "agent",
        "parts": [artifact_part],
        "created_at": _now(),
    }
    artifact = {
        "id": str(uuid.uuid4()),
        "name": "result",
        "description": "Agent response",
        "parts": [artifact_part],
        "created_at": _now(),
    }
    return {
        "id": task_id,
        "context_id": str(uuid.uuid4()),
        "skill_id": None,
        "state": state,
        "error": None,
        "history": [agent_message] if state == "completed" else [],
        "artifacts": [artifact] if state == "completed" else [],
        "created_at": _now(),
        "updated_at": _now(),
    }


class _Store:
    def __init__(self) -> None:
        self._tasks: dict[str, dict] = {}

    async def save(self, task: dict) -> None:
        self._tasks[task["id"]] = task

    async def get(self, task_id: str) -> dict | None:
        return self._tasks.get(task_id)

    async def set_state(self, task_id: str, state: str) -> None:
        if task_id in self._tasks:
            self._tasks[task_id]["state"] = state
            self._tasks[task_id]["updated_at"] = _now()

    async def all(self) -> list[dict]:
        return list(self._tasks.values())


def make_agent_app(
    name: str,
    description: str,
    skills: list[dict],
    *,
    fail_tasks: bool = False,       # returns JSON-RPC ERROR → raises RemoteAgentError
    fail_as_task: bool = False,     # returns Task with state=failed (for DLQ tests)
    require_input: bool = False,
    streaming: bool = False,
) -> Starlette:
    """
    fail_tasks=True  → JSON-RPC error response → A2AHttpClient raises →
                        StepResult.error set → step.succeeded=False ✓
    fail_as_task=True → Task(state=failed) returned → use for DLQ capture tests
    """
    store = _Store()

    card_data = {
        "name": name,
        "description": description,
        "version": "1.3.0",
        "url": "http://localhost:0",
        "capabilities": {
            "streaming": streaming,
            "push_notifications": False,
            "multi_turn": False,
        },
        "skills": [
            {
                "id": s.get("id", ""),
                "name": s.get("name", ""),
                "description": s.get("description", "No description."),
                "tags": s.get("tags", []),
                "examples": s.get("examples", []),
            }
            for s in skills
        ],
        "input_modes": ["text/plain", "application/json"],
        "output_modes": ["text/plain", "application/json"],
        "authentication": {"scheme": "none"},
    }

    async def agent_card(request: Request) -> JSONResponse:
        return JSONResponse(card_data)

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    async def metrics(request: Request) -> PlainTextResponse:
        tasks = await store.all()
        q = sum(1 for t in tasks if t["state"] not in ("completed", "failed", "cancelled"))
        return PlainTextResponse(f"nexus_task_queue_depth {q}\nnexus_dlq_pending 0\n")

    async def jsonrpc(request: Request) -> JSONResponse:
        body = await request.json()
        method = body.get("method", "")
        rpc_id = body.get("id", 1)
        params = body.get("params", {})

        if method == "message/send":
            # fail_tasks → JSON-RPC error (causes RemoteAgentError in client)
            if fail_tasks:
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": rpc_id,
                    "error": {
                        "code": -32000,
                        "message": "Agent task failed intentionally.",
                    },
                })

            task_id = str(uuid.uuid4())

            # Extract user text (Part.content field)
            user_text = ""
            msg = params.get("message", {})
            if isinstance(msg, dict):
                for part in msg.get("parts", []):
                    if isinstance(part, dict) and part.get("type") == "text":
                        user_text = str(part.get("content", ""))

            if fail_as_task:
                # Returns a real Task dict with state=failed (for DLQ tests)
                task = _make_task_dict(task_id, "failed", "")
                task["error"] = "Task failed intentionally."
            elif require_input:
                task = _make_task_dict(task_id, "input_required", "Waiting for input.")
                task["state"] = "input_required"
            else:
                response_text = f"[{name}] processed: {user_text}"
                task = _make_task_dict(task_id, "completed", response_text)

            await store.save(task)
            return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "result": task})

        if method == "tasks/get":
            task_id = params.get("taskId") or params.get("id")
            task = await store.get(task_id)
            if task is None:
                return JSONResponse({
                    "jsonrpc": "2.0", "id": rpc_id,
                    "error": {"code": -32001, "message": "Task not found"},
                })
            return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "result": task})

        if method == "tasks/cancel":
            task_id = params.get("taskId") or params.get("id")
            if task_id:
                await store.set_state(task_id, "cancelled")
                task = await store.get(task_id)
                if task:
                    task["state"] = "cancelled"
                    return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "result": task})
            cancelled_task = _make_task_dict(str(uuid.uuid4()), "cancelled", "")
            cancelled_task["state"] = "cancelled"
            return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "result": cancelled_task})

        return JSONResponse({
            "jsonrpc": "2.0", "id": rpc_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        })

    return Starlette(routes=[
        Route("/.well-known/agent-card.json", agent_card),
        Route("/health", health),
        Route("/metrics", metrics),
        Route("/", jsonrpc, methods=["POST"]),
    ])


class AgentServer:
    def __init__(self, app: Starlette, port: int) -> None:
        self.app = app
        self.port = port
        self.url = f"http://127.0.0.1:{port}"
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        config = uvicorn.Config(
            self.app, host="127.0.0.1", port=self.port,
            log_level="error", loop="asyncio",
        )
        self._server = uvicorn.Server(config)
        self._task = asyncio.create_task(self._server.serve())
        await self._wait_ready()

    async def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                pass

    async def _wait_ready(self, timeout: float = 5.0) -> None:
        import httpx
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            try:
                async with httpx.AsyncClient() as c:
                    await c.get(f"{self.url}/health", timeout=0.5)
                return
            except Exception:
                await asyncio.sleep(0.05)
        raise TimeoutError(f"Server on port {self.port} not ready in {timeout}s")


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
async def echo_agent() -> AsyncGenerator[AgentServer, None]:
    port = get_free_port()
    server = AgentServer(
        make_agent_app("EchoAgent", "Echoes user input.",
                       [{"id": "echo", "name": "Echo", "description": "Echo input."}]),
        port,
    )
    await server.start()
    yield server
    await server.stop()


@pytest.fixture
async def failing_agent() -> AsyncGenerator[AgentServer, None]:
    """Returns JSON-RPC error → raises RemoteAgentError in client."""
    port = get_free_port()
    server = AgentServer(
        make_agent_app("FailAgent", "Always fails (RPC error).",
                       [{"id": "fail", "name": "Fail", "description": "Fails."}],
                       fail_tasks=True),
        port,
    )
    await server.start()
    yield server
    await server.stop()


@pytest.fixture
async def failing_task_agent() -> AsyncGenerator[AgentServer, None]:
    """Returns Task with state=failed (for DLQ tests needing a real Task)."""
    port = get_free_port()
    server = AgentServer(
        make_agent_app("FailTaskAgent", "Returns failed task.",
                       [{"id": "fail", "name": "Fail", "description": "Fails."}],
                       fail_as_task=True),
        port,
    )
    await server.start()
    yield server
    await server.stop()


@pytest.fixture
async def summarizer_agent() -> AsyncGenerator[AgentServer, None]:
    port = get_free_port()
    server = AgentServer(
        make_agent_app("SummarizerAgent", "Summarizes input.",
                       [{"id": "summarize", "name": "Summarize",
                         "description": "Summarizes text."}]),
        port,
    )
    await server.start()
    yield server
    await server.stop()