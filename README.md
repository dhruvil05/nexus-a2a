# nexus-a2a

> **Production-grade Agent-to-Agent communication for Python.**
> One decorator. Zero boilerplate. Any AI framework.

[![PyPI](https://img.shields.io/pypi/v/nexus-a2a)](https://pypi.org/project/nexus-a2a/)
[![Python](https://img.shields.io/pypi/pyversions/nexus-a2a)](https://pypi.org/project/nexus-a2a/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)



## ❤️ Support

<a href="https://ko-fi.com/dhruvil05" target="_blank">
  <img src="https://storage.ko-fi.com/cdn/kofi3.png?v=3"
       alt="Buy Me a Coffee at Ko-fi"
       height="42">
</a>

## What is nexus-a2a?

nexus-a2a is a Python library that lets AI agents talk to each other over HTTP using the [A2A protocol](https://google.github.io/A2A/). It handles discovery, routing, authentication, retries, circuit breaking, tracing, and failure recovery — so you focus on what your agent actually does.

```
Agent A  ──HTTP/JSON-RPC──▶  Agent B  ──▶  Agent C
   │                             │
   └── auto-retry           DLQ on fail
   └── circuit breaker      trace ID
   └── mTLS                 rate limit
```

---

## Table of Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Core Concepts](#core-concepts)
- [Building Agents](#building-agents)
- [Sending Tasks](#sending-tasks)
- [Agent Registry & Discovery](#agent-registry--discovery)
- [Orchestration](#orchestration)
- [Security](#security)
- [Storage Backends](#storage-backends)
- [Reliability](#reliability)
- [Streaming & Webhooks](#streaming--webhooks)
- [Framework Adapters](#framework-adapters)
- [Observability & CLI](#observability--cli)
- [Configuration (nexus.toml)](#configuration-nexustoml)
- [Error Handling Reference](#error-handling-reference)
- [Testing](#testing)

---

## Installation

```bash
# Core
pip install nexus-a2a

# With Redis task store
pip install "nexus-a2a[redis]"

# With PostgreSQL task store
pip install "nexus-a2a[postgres]"

# Everything
pip install "nexus-a2a[all]"

# For contributors / testing
pip install "nexus-a2a[dev]"
```

Requires **Python 3.11+**.

---

## Quickstart

### 1. Define your agent

```python
from nexus_a2a import agent
from nexus_a2a import Task

@agent(
    name="SummaryAgent",
    description="Summarises text passed to it.",
    skills=[{
        "id": "summarise",
        "name": "Summarise",
        "description": "Returns a one-paragraph summary.",
        "tags": ["nlp", "text"],
    }],
    url="http://localhost:8001",
)
class SummaryAgent:
    async def run(self, task: Task) -> str:
        text = task.latest_message().text()
        return f"Summary: {text[:100]}..."
```

### 2. Start the server

```bash
nexus run --module mypackage.agent:SummaryAgent
# or
python -m mypackage.agent
```

### 3. Send a task from another agent

```python
import asyncio
from nexus_a2a import Message
from nexus_a2a import A2AHttpClient

async def main():
    async with A2AHttpClient("http://localhost:8001") as client:
        task = await client.send_message(Message.user_text("Summarise this long document..."))
        print(task.state)            # TaskState.COMPLETED
        print(task.artifacts[0].parts[0].content)

asyncio.run(main())
```

---

## Core Concepts

### Task state machine

Every task follows a strict state machine. Invalid transitions raise `ValueError`.

```
SUBMITTED ──▶ WORKING ──▶ COMPLETED
                │ ──────▶ FAILED        (requires error= message)
                │ ──────▶ INPUT_REQUIRED ──▶ WORKING
                └───────▶ CANCELLED
```

```python
from nexus_a2a import Task, TaskState, Message

task = Task.create(initial_message=Message.user_text("hello"))
# task.state == TaskState.SUBMITTED

task.transition(TaskState.WORKING)
task.transition(TaskState.COMPLETED)

# ❌ This raises — can't skip WORKING
task2 = Task.create(initial_message=Message.user_text("x"))
task2.transition(TaskState.FAILED)               # ValueError!
task2.transition(TaskState.WORKING)              # ✓
task2.transition(TaskState.FAILED, error="oops") # ✓
```

### Message and Part

```python
from nexus_a2a import Message, Part, PartType, MessageRole

# Shortcut (most common)
msg = Message.user_text("What is the weather today?")

# Full form
msg = Message(
    role=MessageRole.USER,
    parts=[
        Part(type=PartType.TEXT, content="What is the weather today?"),
    ],
)

# Read text back
print(msg.text())   # "What is the weather today?"
```

---

## Building Agents

### `@agent` decorator

The primary way to define an agent. Auto-generates an `AgentCard`.

```python
from nexus_a2a import agent
from nexus_a2a import Task

@agent(
    name="ResearchAgent",
    description="Searches the web and returns summarised results.",
    version="1.0.0",
    url="http://localhost:8001",
    skills=[
        {
            "id": "web_search",
            "name": "Web Search",
            "description": "Searches the web for a given query.",
            "tags": ["search", "web"],
            "examples": ["Find latest AI papers", "Search Python asyncio docs"],
        }
    ],
    streaming=False,
    push_notifications=False,
)
class ResearchAgent:
    async def run(self, task: Task) -> str:
        query = task.latest_message().text()
        # ... call search API ...
        return f"Results for: {query}"
```

Use `@agent` without arguments — class name and docstring become defaults:

```python
@agent
class QuickAgent:
    """A fast agent that does quick tasks."""
    async def run(self, task: Task) -> str:
        return "done"
```

Access the generated card:

```python
card = ResearchAgent.get_agent_card()
print(card.name)           # "ResearchAgent"
print(card.skills[0].id)   # "web_search"

# Or via helper
from nexus_a2a import get_card
card = get_card(ResearchAgent)
```

---

## Sending Tasks

### A2AHttpClient

```python
from nexus_a2a import (
    A2AHttpClient,
    RetryConfig,
    CircuitBreaker,
)
from nexus_a2a import Message, TaskState

async with A2AHttpClient(
    "http://localhost:8001",
    timeout=30.0,
    retry=RetryConfig(
        max_retries=3,
        base_delay=0.5,
        max_delay=30.0,
        jitter=True,
        retry_on={500, 502, 503, 504},
    ),
) as client:

    # Send a task
    task = await client.send_message(Message.user_text("hello"))
    assert task.state == TaskState.COMPLETED

    # Fetch an existing task
    task = await client.get_task(task.id)

    # Cancel a task
    cancelled = await client.cancel_task(task.id)

    # Fetch the agent's card
    card = await client.fetch_agent_card()
    print(card.name, card.skills)
```

### Circuit Breaker

Prevents hammering a failing agent. Automatically opens after N failures
and lets a test request through after the recovery window.

```python
from nexus_a2a import CircuitBreaker, CircuitOpenError

cb = CircuitBreaker(
    failure_threshold=5,    # open after 5 consecutive failures
    recovery_timeout=30.0,  # try again after 30 s
    success_threshold=2,    # close after 2 successes
)

async with A2AHttpClient("http://localhost:8001", circuit_breaker=cb) as client:
    try:
        task = await client.send_message(Message.user_text("ping"))
    except CircuitOpenError as e:
        print(f"Circuit is OPEN. Retry after {e.retry_after:.0f}s")
```

---

## Agent Registry & Discovery

```python
from nexus_a2a import AgentRegistry

registry = AgentRegistry(
    card_ttl_seconds=300.0,       # re-fetch card after 5 min
    health_check_timeout=5.0,
)

# Register by URL (fetches AgentCard automatically)
card = await registry.register_url("http://localhost:8001")

# Register multiple
for url in ["http://agent-a:8001", "http://agent-b:8002"]:
    await registry.register_url(url)

# List all registered agents
cards = registry.list_all()          # list[AgentCard]
healthy = registry.list_healthy()    # only healthy ones

# Find agents that have a specific skill
matches = registry.find_by_skill("web_search")   # list[AgentCard]

# Lookup by name or URL
card = registry.get_by_name("ResearchAgent")
card = registry.get_by_url("http://localhost:8001")

# Health checks
health_map = await registry.check_all_health()  # {"http://...": True/False}
is_up = await registry.check_health("http://localhost:8001")

# Summary
print(registry.summary())
# {"total": 3, "healthy": 2, "unhealthy": 1, "skills": [...]}
```

---

## Orchestration

### Sequential pipeline

Each agent's output becomes the next agent's input.

```python
from nexus_a2a import Orchestrator
from nexus_a2a import A2AHttpClient
from nexus_a2a import Message, TaskState

async def runner(url: str, message: Message) -> Task:
    async with A2AHttpClient(url) as client:
        return await client.send_message(message)

orchestrator = Orchestrator(runner=runner, stop_on_error=True)

result = await orchestrator.sequential(
    agent_urls=["http://agent-a:8001", "http://agent-b:8002", "http://agent-c:8003"],
    initial_message=Message.user_text("Research quantum computing trends"),
)

print(result.succeeded)        # True / False
print(result.total_sec)        # wall-clock seconds
print(result.final_output)     # last Task produced

for step in result.steps:
    print(step.agent_url, step.duration_sec)
    if step.succeeded:
        print("  ✓", step.task.state)
    else:
        print("  ✗", step.error)
```

### Parallel

All agents receive the same input simultaneously.

```python
result = await orchestrator.parallel(
    agent_urls=["http://agent-a:8001", "http://agent-b:8002"],
    message=Message.user_text("Summarise this document"),
)

for step in result.steps:
    print(step.agent_url, "→", step.task.artifacts[0].parts[0].content)
```

### DAG (Directed Acyclic Graph)

Dependency-aware execution. Agents with no pending dependencies run concurrently.

```python
from nexus_a2a import DAGNode

nodes = [
    DAGNode(agent_url="http://fetcher:8001",    depends_on=[]),
    DAGNode(agent_url="http://parser:8002",     depends_on=["http://fetcher:8001"]),
    DAGNode(agent_url="http://summariser:8003", depends_on=["http://fetcher:8001"]),
    DAGNode(agent_url="http://reporter:8004",   depends_on=["http://parser:8002",
                                                             "http://summariser:8003"]),
]

result = await orchestrator.dag(
    nodes=nodes,
    initial_message=Message.user_text("Process this dataset"),
)
```

---

## Security

### Authentication

Three schemes supported. Each registered agent can use a different scheme.

```python
from nexus_a2a import AuthManager, AgentCredentialConfig
from nexus_a2a import AuthScheme

auth = AuthManager()

# API Key
auth.register("http://agent-a:8001", AgentCredentialConfig(
    scheme=AuthScheme.API_KEY,
    api_key="my-secret-key",
    header_name="X-API-Key",    # default
))

# JWT
auth.register("http://agent-b:8002", AgentCredentialConfig(
    scheme=AuthScheme.JWT,
    jwt_secret="super-secret",
))
token = auth.issue_jwt("http://agent-b:8002", expires_in=3600)

# No auth (dev/testing)
auth.register("http://agent-c:8003", AgentCredentialConfig(
    scheme=AuthScheme.NONE,
))

# Verify an incoming request
from nexus_a2a import AuthError
try:
    await auth.verify("http://agent-a:8001", headers={"X-API-Key": "my-secret-key"})
except AuthError as e:
    print("Auth failed:", e)
```

### Rate Limiting

Token-bucket algorithm. In-process, zero dependencies.

```python
from nexus_a2a import RateLimiter, RateLimitConfig, RateLimitError

limiter = RateLimiter()
limiter.configure("http://agent-a:8001", RateLimitConfig(
    rate=10.0,    # 10 requests per second (sustained)
    burst=20,     # allow bursts up to 20
))

try:
    await limiter.check("http://agent-a:8001")
except RateLimitError as e:
    print(f"Slow down! Retry in {e.retry_after:.2f}s")
```

### Trust Boundaries

Default-deny permission matrix between agents.

```python
from nexus_a2a import TrustBoundary

trust = TrustBoundary()

# Allow specific pairs
trust.allow("http://orchestrator:8000", "http://worker-a:8001")
trust.allow("http://orchestrator:8000", "http://worker-b:8002")

# Wildcard: orchestrator can talk to anything
trust.allow("http://orchestrator:8000", "*")

# Block a specific agent
trust.block("http://untrusted:9999", "*")

# Check before sending
if trust.is_allowed("http://orchestrator:8000", "http://worker-a:8001"):
    await client.send_message(msg)
```

### Payload Validation

```python
from nexus_a2a import PayloadValidator

validator = PayloadValidator(
    max_size_bytes=1_000_000,   # 1 MB
    max_parts=50,
)

from nexus_a2a import PayloadTooLargeError, TooManyPartsError, InvalidPartError, BlankTextPartError
try:
    validator.validate(message)
except (PayloadTooLargeError, TooManyPartsError, InvalidPartError, BlankTextPartError) as e:
    print("Invalid payload:", e)
```

### Mutual TLS (mTLS)

Both agents verify each other's certificates.

```python
from nexus_a2a import MutualTLSConfig, build_client_ssl_context

config = MutualTLSConfig(
    cert_file="/certs/agent.crt",
    key_file="/certs/agent.key",
    ca_file="/certs/ca.crt",
)

# Or from environment variables (NEXUS_MTLS_CERT, NEXUS_MTLS_KEY, NEXUS_MTLS_CA)
config = MutualTLSConfig.from_env()

ssl_ctx = build_client_ssl_context(config)

import httpx
async with httpx.AsyncClient(verify=ssl_ctx) as http:
    # all requests use mTLS
    pass
```

---

## Storage Backends

### In-Memory (default)

```python
from nexus_a2a import InMemoryTaskStore

store = InMemoryTaskStore()
```

### Redis

```python
from nexus_a2a import RedisTaskStore

store = RedisTaskStore(
    url="redis://localhost:6379",
    ttl_seconds=86400,    # tasks expire after 24h
)
await store.connect()
```

### PostgreSQL

```python
from nexus_a2a import PostgresTaskStore

store = PostgresTaskStore(dsn="postgresql://user:pass@localhost/nexus")
await store.connect()   # creates tables if not present
```

### Task Manager

Wraps any store with lifecycle operations and a watchdog that auto-fails stuck tasks.

```python
from nexus_a2a import TaskManager

manager = TaskManager(
    store=store,
    timeout_sec=120.0,    # auto-fail tasks stuck in WORKING after 2 min
)

task = await manager.create(message=Message.user_text("hello"))
await manager.start(task.id)
await manager.complete(task.id, artifact_text="done")

# Or fail with reason
await manager.fail(task.id, error="API call timed out")
```

---

## Reliability

### Dead Letter Queue (DLQ)

Failed tasks land in the DLQ for inspection and replay.

```python
from nexus_a2a import DeadLetterQueue

dlq = DeadLetterQueue(
    max_retries=3,
    retry_delay=2.0,       # seconds between retries (exponential backoff)
    max_queue_size=500,
)

# Capture a failed task
entry = await dlq.capture(
    task,
    agent_url="http://worker:8001",
    skill_id="web_search",
)

# Inspect
print(dlq.count())           # total entries
print(dlq.pending_count())   # not yet replayed
entries = dlq.all_entries()  # list[DLQEntry]
pending = dlq.pending_entries()

# Filter by skill
web_failures = [e for e in dlq.all_entries() if e.skill_id == "web_search"]

# Replay a single task
result = await dlq.replay(task_id="abc-123")
print(result.succeeded, result.error)

# Replay all pending
results = await dlq.replay_all()

# Replay filtered by skill or agent
results = await dlq.replay_where(skill_id="web_search")
results = await dlq.replay_where(agent_url="http://worker:8001")

# Failure hook
@dlq.on_failure
async def on_fail(entry):
    print(f"Task {entry.task_id} failed: {entry.error}")

# Clean up replayed entries
removed = dlq.clear_replayed()
print(f"Cleaned {removed} entries")
```

### Input Required (Human-in-the-Loop)

Pause a task, wait for human input, then resume.

```python
from nexus_a2a import InputHandler

handler = InputHandler()

# Pause and wait for input (async — does not block other tasks)
await handler.request_input(task.id, prompt="Please provide your API key:")
response = await handler.wait_for_input(task.id, timeout=300.0)

# From another process / API endpoint — resume the task
await handler.provide_input(task.id, "user-provided-api-key")
```

### Graceful Shutdown

```python
from nexus_a2a import GracefulShutdown

shutdown = GracefulShutdown(
    manager=task_manager,
    drain_timeout=30.0,    # wait up to 30s for WORKING tasks to finish
)

# Registers SIGTERM and SIGINT handlers automatically
shutdown.register()

# Manual shutdown (e.g. from a test or lifecycle hook)
await shutdown.shutdown()
```

---

## Streaming & Webhooks

### SSE Streaming

```python
from nexus_a2a import SSEStreamer, SSEFormatter

# Server side — send events
formatter = SSEFormatter()

# Emit task status
data = formatter.task_status(task_id="abc", state="working")
data = formatter.artifact_chunk(task_id="abc", chunk="partial text...")
data = formatter.done(task_id="abc")

# Client side — consume events
async with A2AHttpClient("http://localhost:8001") as client:
    async for event in SSEStreamer(client).stream(message):
        print(event.type, event.data)
```

### Webhooks

HMAC-SHA256 signed delivery with exponential backoff retries.

```python
from nexus_a2a import WebhookDispatcher

dispatcher = WebhookDispatcher(
    secret="your-webhook-secret",
    max_retries=3,
)

# Dispatch — auto-retries on 5xx, skips 4xx (client error = no retry)
await dispatcher.dispatch(
    url="https://your-app.com/webhooks/nexus",
    event="task.completed",
    payload={"task_id": "abc-123", "result": "..."},
)

# Non-raising version (logs failures silently)
await dispatcher.dispatch_silent(url=..., event=..., payload=...)

# Verify incoming webhook on your server
is_valid = dispatcher.verify_signature(
    payload=request.body,
    signature=request.headers["X-Nexus-Signature-256"],
)
```

---

## Framework Adapters

Wrap existing agents from popular frameworks with zero changes to your existing code.

### LangGraph

```python
from nexus_a2a import LangGraphAdapter

adapter = LangGraphAdapter(graph=compiled_graph)
result = await adapter.run(task)
print(result.text)
```

### CrewAI

```python
from nexus_a2a import CrewAIAdapter

adapter = CrewAIAdapter(crew=my_crew)
result = await adapter.run(task)
```

### AutoGen

```python
from nexus_a2a import AutoGenAdapter

adapter = AutoGenAdapter(agent=my_autogen_agent)
result = await adapter.run(task)
```

### Google ADK

```python
from nexus_a2a import GoogleADKAdapter

adapter = GoogleADKAdapter(agent=my_adk_agent)
result = await adapter.run(task)
```

---

## Observability & CLI

### `nexus` CLI

Install with the package, then:

```bash
# Ping an agent — name, version, skills, latency, health
nexus ping http://localhost:8001

# Inspect full AgentCard
nexus inspect http://localhost:8001

# Network status table (all agents, queue depth, DLQ count)
nexus status --network

# Trace a task — call tree with per-hop latency
nexus trace abc-123-task-id
nexus trace abc-123-task-id --agent http://localhost:8001

# Replay DLQ entries
nexus replay --failed
nexus replay --failed --skill web_search
nexus replay --failed --last 1h
nexus replay --failed --dry-run        # preview without replaying

# Start agent server
nexus run
nexus run --host 0.0.0.0 --port 8080

# JSON output for all commands
nexus --format json status --network
```

### Audit Logger

```python
from nexus_a2a import AuditLogger
import sys

logger = AuditLogger(
    stream=sys.stdout,         # or open("audit.ndjson", "a")
    buffer_size=100,           # flush every 100 events
)

# 8 event types: task_created, task_started, task_completed,
# task_failed, task_cancelled, message_sent, auth_failed, rate_limited
await logger.log("task_completed", {
    "task_id": task.id,
    "agent_url": "http://worker:8001",
    "duration_sec": 1.23,
})
```

### Metrics

```python
from nexus_a2a import MetricsCollector

metrics = MetricsCollector()

# Record events
await metrics.record_task_completed(agent_url="http://worker:8001", duration_sec=1.2)
await metrics.record_task_failed(agent_url="http://worker:8001")
await metrics.record_auth_failure(agent_url="http://worker:8001")

# Query
print(metrics.task_count())               # total tasks
print(metrics.error_rate())               # 0.0 – 1.0
print(metrics.avg_latency_sec())          # float
print(metrics.p99_latency_sec())          # float

# Prometheus text format (expose via /metrics endpoint)
text = metrics.to_prometheus()
```

### Distributed Tracing

```python
from nexus_a2a import Tracer, TraceStore

tracer = Tracer()
trace_store = TraceStore(max_traces=1000)

# Start a trace
trace_id = tracer.new_trace_id()

# Use in client — automatically injected into X-Nexus-Trace-ID header
async with A2AHttpClient(
    "http://localhost:8001",
    trace_id=trace_id,
    trace_store=trace_store,
) as client:
    task = await client.send_message(Message.user_text("hello"))

# Retrieve trace
trace = trace_store.get(trace_id)
print(tracer.format_tree(trace))
```

---

## Configuration (nexus.toml)

Zero-config wiring from a single TOML file.

```toml
[agent]
name        = "ResearchAgent"
description = "Searches the web and summarises results."
version     = "1.0.0"
url         = "http://localhost:8001"

[[agent.skills]]
id          = "web_search"
name        = "Web Search"
description = "Searches the web for a given query."
tags        = ["search", "web"]

[storage]
backend = "redis"               # memory | redis | postgres
url     = "redis://localhost:6379"

[security]
auth_scheme     = "api_key"     # none | api_key | jwt
auth_secret     = "my-secret"
mtls_cert_file  = "/certs/agent.crt"
mtls_key_file   = "/certs/agent.key"
mtls_ca_file    = "/certs/ca.crt"

[reliability]
task_timeout_sec = 120.0
dlq_max_size     = 500
dlq_max_retries  = 3

[observability]
log_level  = "INFO"
tracing    = true

[network]
agents = [
    "http://agent-a:8001",
    "http://agent-b:8002",
]
```

Load programmatically:

```python
from nexus_a2a import AgentNetwork

network = AgentNetwork.from_config("nexus.toml")
```

**Environment variable overrides** (container-friendly):

| Variable | Overrides |
|---|---|
| `NEXUS_AGENT_NAME` | `[agent].name` |
| `NEXUS_AGENT_URL` | `[agent].url` |
| `NEXUS_AUTH_SCHEME` | `[security].auth_scheme` |
| `NEXUS_AUTH_SECRET` | `[security].auth_secret` |
| `NEXUS_STORAGE_BACKEND` | `[storage].backend` |
| `NEXUS_STORAGE_URL` | `[storage].url` |
| `NEXUS_TASK_TIMEOUT` | `[reliability].task_timeout_sec` |
| `NEXUS_LOG_LEVEL` | `[observability].log_level` |

---

## Error Handling Reference

Every error in nexus-a2a is typed. Catch specific exceptions rather than bare `Exception`.

### Transport errors

```python
from nexus_a2a import (
    AgentUnreachableError,   # agent is down / DNS failure
    AgentCardFetchError,     # /.well-known/agent-card.json failed
    RemoteAgentError,        # agent returned JSON-RPC error
    CircuitOpenError,        # circuit breaker is OPEN
    TransportError,          # base class for all transport errors
)

try:
    task = await client.send_message(msg)
except CircuitOpenError as e:
    print(f"Circuit open. Retry in {e.retry_after}s")
except AgentUnreachableError as e:
    print(f"Cannot reach {e.url}: {e.reason}")
except RemoteAgentError as e:
    print(f"Agent error {e.code}: {e.message}")
except TransportError as e:
    print(f"Transport error: {e}")
```

### Auth errors

```python
from nexus_a2a import (
    AuthError,                   # base
    MissingCredentialsError,     # no credentials in request
    InvalidCredentialsError,     # wrong key / bad token
    ExpiredCredentialsError,     # JWT expired
)

try:
    await auth.verify(url, headers)
except ExpiredCredentialsError:
    new_token = auth.issue_jwt(url, expires_in=3600)
except InvalidCredentialsError as e:
    print(f"Auth failed: {e.reason}")
```

### Rate limit errors

```python
from nexus_a2a import RateLimitError

try:
    await limiter.check(agent_url)
except RateLimitError as e:
    await asyncio.sleep(e.retry_after)
    await limiter.check(agent_url)   # retry
```

### Task state errors

```python
from nexus_a2a import TaskNotFoundError, TaskAlreadyDoneError

try:
    await manager.complete(task_id)
except TaskNotFoundError:
    print("Task does not exist")
except TaskAlreadyDoneError:
    print("Task already in terminal state")
```

### Orchestration errors

```python
from nexus_a2a import (
    OrchestratorError,      # base
    WorkflowCycleError,     # DAG has a cycle
    WorkflowStepError,      # individual step failed
)

try:
    result = await orchestrator.dag(nodes, initial_message)
except WorkflowCycleError as e:
    print(f"Cycle detected: {e.cycle}")
```

### Config errors

```python
from nexus_a2a import ConfigError

try:
    network = AgentNetwork.from_config("nexus.toml")
except ConfigError as e:
    print(f"Bad config at key '{e.key}': {e}")
```

---

## Testing

### Integration tests

Real in-process agents on random ports — no HTTP mocking.

```python
import pytest
from starlette.applications import Starlette
import uvicorn

# See tests/integration/conftest.py for full fixture helpers
# Run with:
pytest tests/integration/
```

### Run all tests

```bash
# Unit + integration
pytest

# Skip integration (faster CI)
pytest -m "not integration"

# With coverage
pytest --cov=nexus_a2a --cov-report=term-missing

# Specific test file
pytest tests/integration/test_sequential_pipeline.py -v
```

### Type checking and linting

```bash
mypy nexus_a2a --strict
ruff check nexus_a2a
ruff format nexus_a2a
```

---

## What's in each version

| Version | What it added |
|---|---|
| **v0.1.0** | `@agent` decorator, `Task` state machine, `Message`/`Part`/`Artifact` models |
| **v0.2.0** | `InMemoryTaskStore`, `TaskManager`, `A2AHttpClient`, `AgentRegistry` |
| **v0.3.0** | `AuthManager`, `TrustBoundary`, `RateLimiter`, `PayloadValidator` |
| **v0.4.0** | `Orchestrator` (sequential/parallel/dag), SSE streaming, `WebhookDispatcher`, `AgentNetwork` |
| **v1.0.0** | LangGraph/CrewAI/AutoGen/GoogleADK adapters, `RedisTaskStore`, `AuditLogger`, `MetricsCollector` |
| **v1.1.0** | Task timeout watchdog, `InputHandler`, `DeadLetterQueue`, `CircuitBreaker`, `Tracer`, `CapabilityGuard` |
| **v1.2.0** | `GracefulShutdown`, `AgentServer` (K8s probes), mTLS, `PostgresTaskStore`, `nexus.toml`, CI/CD workflows |
| **v1.3.0** | `nexus` CLI (ping/inspect/status/trace/replay/run), integration test suite, `CHANGELOG.md` |

---

## License

MIT — see [LICENSE](LICENSE).

Built by [HongZiro](https://github.com/dhruvil05).