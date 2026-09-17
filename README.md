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
- [Serving Agents](#serving-agents)
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
- [Preparing for 2.0](#preparing-for-20)

---

## Installation

```bash
# Core
pip install nexus-a2a

# With Redis task store
pip install "nexus-a2a[redis]"

# With PostgreSQL task store
pip install "nexus-a2a[postgres]"

# Asymmetric JWT (RS256/ES256) and JWKS
pip install "nexus-a2a[jwks]"

# With the Google ADK adapter (heavy — ~40 transitive packages)
pip install "nexus-a2a[adk]"

# Everything
pip install "nexus-a2a[all]"

# For contributors / testing
pip install "nexus-a2a[dev]"
```

Requires **Python 3.11+**.

The core install is deliberately small — 20 packages, no known CVEs. LangGraph,
CrewAI, AutoGen and Google ADK are all imported lazily, so install only the
adapter you actually use. (`google-adk` was a required dependency before
v1.5.0; it is now the `adk` extra.)

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
```

That serves the agent over the A2A protocol:

```
Serving agent 'SummaryAgent' on http://localhost:8001
  Agent card: http://localhost:8001/.well-known/agent-card.json
  JSON-RPC:   POST http://localhost:8001/
  Health:     http://localhost:8001/health
  Skills:     summarise
```

Or start it from Python, which is what `nexus run` does under the hood:

```python
from nexus_a2a import A2AServer

server = A2AServer(SummaryAgent, port=8001)
await server.start()
# ... or: async with A2AServer(SummaryAgent, port=8001):
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

## Serving Agents

`A2AServer` is the inbound half of the protocol — it turns an `@agent` class
into a reachable agent.

| Endpoint | Purpose |
|---|---|
| `GET /.well-known/agent-card.json` | Discovery — the card `@agent` built |
| `POST /` | JSON-RPC 2.0: `message/send`, `message/stream`, `tasks/get`, `tasks/cancel`, `tasks/pushNotificationConfig/set` and `/get` (send/stream take an optional `taskId` to continue a paused task) |
| `GET /stream?taskId=` | Observe a task as Server-Sent Events — follows it live while it runs |
| `GET /metrics` | Prometheus task counts (public, no task content) |
| `GET /health` | Liveness probe |
| `GET /ready` | Readiness probe (checks the task store) |

```python
from nexus_a2a import A2AServer

server = A2AServer(SummaryAgent, port=8001)
await server.start()
...
await server.stop()

# Or as an async context manager
async with A2AServer(SummaryAgent, port=8001):
    await asyncio.Event().wait()
```

Host and port default to whatever the agent card's `url` says, so a single
`url="http://localhost:8001"` on the decorator configures both sides.

### What `run()` can return

| Return value | Result |
|---|---|
| `str` | One text `Artifact` |
| `dict` / `list` | One JSON `Artifact` |
| `Artifact` | Used as-is |
| `Message` | Appended to task history as the agent's reply |
| `AdapterResult` | `.to_artifact()`, or task `FAILED` if `.error` is set |
| `None` | Completed with no artifact |

A `run()` that raises is **not** a protocol error: the task is recorded as
`FAILED` with the exception message and returned normally, so the caller can
inspect `task.error` and the DLQ can capture it.

### Multi-turn conversations

An agent asks a follow-up by returning `NeedsInput` instead of a result. The
task parks at `INPUT_REQUIRED` with the question as its last message, and the
caller answers against the **same task id**:

```python
from nexus_a2a import agent, NeedsInput, Task

@agent(name="Planner", description="Plans a trip.", url="http://localhost:8001")
class Planner:
    async def run(self, task: Task):
        turns = sum(1 for m in task.history if m.role == "user")
        if turns == 1:
            return NeedsInput("What is your budget?")
        return f"Plan for {task.history[-1].text()}"
```

```python
async with A2AHttpClient("http://localhost:8001") as client:
    task = await client.send_message(Message.user_text("plan a trip"))
    print(task.state)               # TaskState.INPUT_REQUIRED
    print(task.history[-1].text())  # "What is your budget?"

    task = await client.send_message(Message.user_text("$500"), task_id=task.id)
    print(task.state)               # TaskState.COMPLETED
    print(task.artifacts[0].parts[0].content)   # "Plan for $500"
```

`run()` is re-invoked with the **full history** each turn, so the agent reads
earlier answers straight off `task.history`. There is no limit on turns.

**This is stateless.** The conversation lives in the task store, not in a parked
coroutine — it survives a restart, and a different client or connection can
continue it. Streaming works the same way: pass `task_id=` to `stream_message()`,
and a pause arrives as a `message` event followed by
`task_status: input_required`.

| Situation | Result |
|---|---|
| `taskId` omitted | New task |
| Task is `INPUT_REQUIRED` | Resumes, history preserved |
| Task already finished | `-32002` naming the actual state |
| Unknown `taskId` | `-32001` |
| Agent declares `multi_turn=False` | `-32002`, continuations refused |

> `InputHandler.wait_for_input()` remains for in-process suspension, and the
> same wire call resumes it. Prefer `NeedsInput` over HTTP — `wait_for_input()`
> holds the original request open for its whole timeout.

### Error model

Transport-level rejections that happen *before* dispatch return real HTTP
status codes — `401` auth, `403` trust, `413` too large, `429` rate limit (with
`Retry-After`), `400` malformed. Proxies and dashboards can see them, and the
client will not retry a rejected credential.

Application-level failures *after* dispatch return JSON-RPC error objects —
`-32601` method not found, `-32602` invalid params, `-32001` task not found —
which reach the caller as `RemoteAgentError` with the code intact.

### Enforcing security

The security classes are building blocks; `SecurityMiddleware` is what makes
`A2AServer` actually enforce them. Stages run cheapest-first: **size → rate
limit → auth → trust → payload validation**. Every stage is optional, so an
agent starts open and hardens incrementally.

```python
from nexus_a2a import (
    A2AServer, SecurityMiddleware, AuthManager, AgentCredentialConfig,
    AuthScheme, TrustBoundary, RateLimiter, PayloadValidator,
)

MY_URL = "http://summary-agent:8001"

auth = AuthManager()
auth.register_agent("http://orchestrator:8000", AgentCredentialConfig(
    scheme=AuthScheme.API_KEY, api_key="shared-secret",
))

trust = TrustBoundary()
trust.allow("http://orchestrator:8000", MY_URL, skills=["summarise"])

server = A2AServer(SummaryAgent, security=SecurityMiddleware(
    auth=auth,
    trust=trust,
    rate_limiter=RateLimiter(),
    validator=PayloadValidator(),
    server_url=MY_URL,      # this agent is the trust TARGET
))
```

Callers identify themselves with `caller_url`, which auth and trust both need
(a trust rule is `caller -> target`):

```python
async with A2AHttpClient(
    "http://summary-agent:8001",
    caller_url="http://orchestrator:8000",     # sends X-Nexus-Caller
    headers={"X-API-Key": "shared-secret"},
) as client:
    task = await client.send_message(Message.user_text("..."), skill_id="summarise")
```

Without `caller_url` the call is anonymous, and a server with auth or trust
enabled rejects it with `401`.

> **Ops endpoints live on a separate port.** `/metrics`, `/info`, `/dlq` and
> trace lookup are served by `AgentServer` — the standard app-port / admin-port
> split. Run both if you want protocol and ops on one host.

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
auth.register_agent("http://agent-a:8001", AgentCredentialConfig(
    scheme=AuthScheme.API_KEY,
    api_key="my-secret-key",
    header_name="X-API-Key",    # default
))

# JWT
auth.register_agent("http://agent-b:8002", AgentCredentialConfig(
    scheme=AuthScheme.JWT,
    jwt_secret="super-secret",
))
token = auth.issue_jwt("http://agent-b:8002", expires_in=3600)

# No auth (dev/testing)
auth.register_agent("http://agent-c:8003", AgentCredentialConfig(
    scheme=AuthScheme.NONE,
))

# Verify an incoming request
from nexus_a2a import AuthError
try:
    await auth.verify("http://agent-a:8001", headers={"X-API-Key": "my-secret-key"})
except AuthError as e:
    print("Auth failed:", e)
```

> **Auth fails closed (since v1.5.0).** Verifying an agent that was never
> registered raises `UnknownAgentError` rather than silently passing as
> "no auth required". Before v1.5.0 an unregistered URL — a typo, a
> trailing-slash variant, or an attacker-supplied string — bypassed
> authentication entirely.
>
> ```python
> auth = AuthManager()                              # unregistered -> raises
> auth = AuthManager(allow_unregistered=True)       # old behaviour, dev only
> ```
>
> This applies to inbound verification only. `build_auth_headers()` still
> returns `{}` for unknown agents, so outbound calls to agents you hold no
> credentials for simply carry no auth headers.

### Asymmetric JWT and JWKS

With HS256 the verifier holds the same secret as the signer — so any agent that
can **check** a peer's token can also **mint** one. In a network of more than
two parties, every verifier is a forger. A key pair splits those roles.

```bash
pip install "nexus-a2a[jwks]"
```

The signer keeps the private key:

```python
auth.register_agent("http://summary-agent:8002", AgentCredentialConfig(
    scheme=AuthScheme.JWT,
    jwt_private_key=PRIVATE_PEM,      # signs
    jwt_public_key=PUBLIC_PEM,        # verifies peers
    jwt_algorithm="RS256",            # or ES256, PS256, ...
    jwt_issuer="summary-agent",
))
```

Verifiers hold only the public half — or fetch it from the issuer's key set, so
rotation is a publish rather than a coordinated secret swap:

```python
auth.register_agent("http://summary-agent:8002", AgentCredentialConfig(
    scheme=AuthScheme.JWT,
    jwks_url="https://summary-agent.example.com/.well-known/jwks.json",
    jwt_algorithm="RS256",
    jwt_issuer="summary-agent",
))
```

> **Algorithm confusion is structurally prevented.** An RSA public key is
> public, so a verifier that accepted HS256 *alongside* RS256 would accept a
> token signed with that public key as an HMAC secret — the classic JWT
> forgery. Acceptable algorithms come from the **configured key**, never from
> the token's `alg` header, and the symmetric and asymmetric families never
> overlap. Setting more than one of `jwt_secret` / `jwt_public_key` /
> `jwks_url` is rejected at registration.

| Configured | Accepts |
|---|---|
| `jwt_secret` | HS256 only |
| `jwt_public_key` / `jwks_url` | the one configured asymmetric algorithm |

HS256 keeps working unchanged, and needs no crypto library — `cryptography`
lives in the `jwks` extra so the core install stays at 20 packages.

### Rate Limiting

Token-bucket algorithm. In-process, zero dependencies.

```python
from nexus_a2a import RateLimiter, RateLimitConfig, RateLimitError

limiter = RateLimiter()
limiter.set_limit("http://agent-a:8001", RateLimitConfig(
    rate=10.0,    # 10 requests per second (sustained)
    burst=20,     # allow bursts up to 20
))

try:
    await limiter.check("http://agent-a:8001")
except RateLimitError as e:
    print(f"Slow down! Retry in {e.retry_after:.2f}s")
```

> **`RateLimiter` is per-process.** Three replicas configured for 10 req/s
> allow 30 — the limit scales with your deployment, silently. Use
> `RedisRateLimiter` when the limit has to actually hold:

```python
from nexus_a2a import RedisRateLimiter, SecurityMiddleware

limiter = RedisRateLimiter(url="redis://localhost:6379")
await limiter.connect()

security = SecurityMiddleware(rate_limiter=limiter)   # drop-in
assert limiter.is_distributed          # RateLimiter().is_distributed is False
```

One bucket per agent lives in Redis, and refill-and-consume runs as a Lua
script so it is atomic — a read-modify-write from Python would let two replicas
both spend the same last token. The script reads the clock from Redis, so a
skewed replica cannot rewind a bucket. Requires Redis 7+.

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

### Admin Endpoints

`AgentServer` serves Kubernetes probes and Prometheus metrics publicly, but the
endpoints that expose task data or re-execute work require a token.

| Endpoint | Auth | Purpose |
|---|---|---|
| `GET /health` | public | Liveness probe |
| `GET /ready` | public | Readiness probe |
| `GET /metrics` | public | Prometheus scrape |
| `GET /info` | **admin** | Network topology summary |
| `GET /traces/{id}` | **admin** | Distributed trace lookup |
| `GET /dlq` | **admin** | Dead Letter Queue listing |
| `POST /dlq/replay` | **admin** | Re-execute failed tasks |

```python
server = AgentServer(network=network, port=8080, admin_token="...")
# or: export NEXUS_ADMIN_TOKEN=...
```

```bash
curl -H "X-Admin-Token: $NEXUS_ADMIN_TOKEN" http://agent:8080/dlq
```

> **Admin endpoints are disabled by default (since v1.5.0).** With no
> `admin_token` and no `NEXUS_ADMIN_TOKEN` they return `403`. Before v1.5.0 they
> were served unauthenticated on a `0.0.0.0` bind, so anyone who could reach the
> port could replay every failed task in the queue and read task payloads,
> error strings, and every registered agent URL.

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

store = InMemoryTaskStore()                       # 1 h retention, 10,000 cap
store = InMemoryTaskStore(retention_sec=600, max_tasks=1_000)
store = InMemoryTaskStore(retention_sec=None, max_tasks=None)   # never forget
```

Finished tasks — completed, failed or cancelled — are evicted `retention_sec`
after they finish, and the oldest go first once the store exceeds
`max_tasks`. **Running and paused tasks are never evicted.** Before 1.9.0 this
store kept every task forever, so a long-running server's memory grew with
every request.

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

### Durable Dead Letter Queue

The DLQ holds failed tasks so they can be replayed. By default it lives in
memory, which means a crash or redeploy loses them — and a DLQ entry exists
precisely because that work did *not* complete.

```python
from nexus_a2a import AgentNetwork, RedisDLQStore

store = RedisDLQStore(url="redis://localhost:6379")
await store.connect()

network = AgentNetwork(dlq_store=store)
await network.dead_letter_queue.load()      # rehydrate after a restart
```

```python
dlq = network.dead_letter_queue
dlq.is_durable                   # True
await dlq.load()                 # entries captured before the restart
await dlq.replay_all()           # and they are replayable again
await dlq.purge_replayed()       # removes from the store too
```

### Durable push targets

A webhook is registered by a caller who will *not* wait around — so losing it
on restart defeats the purpose.

```python
from nexus_a2a import A2AServer, RedisPushStore, WebhookConfig

store = RedisPushStore(url="redis://localhost:6379")
await store.connect()

server = A2AServer(
    MyAgent,
    push_store=store,
    push_config=WebhookConfig(signing_secret="shared-secret"),
)
```

Targets are released once a task reaches a terminal state, and carry a TTL for
tasks that never finish.

> A push config holds the caller's `token` — the secret their receiver checks
> callbacks against. Persisting it puts that secret at rest in the store, so
> point `RedisPushStore` at a Redis you would be willing to keep credentials
> in: authenticated, and TLS if it is not on loopback.

---

> `clear_replayed()` is synchronous so it cannot reach an async store — it only
> clears the local view, and entries return on the next `load()`. Use
> `purge_replayed()` when a store is configured.

---

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

An agent streams by writing `run()` as an async generator — every `yield`
becomes one chunk on the wire:

```python
from nexus_a2a import agent, A2AServer, Task

@agent(
    name="Writer",
    description="Writes text a word at a time.",
    streaming=True,
    skills=[{"id": "write", "name": "Write", "description": "Write text."}],
    url="http://localhost:8001",
)
class Writer:
    async def run(self, task: Task):
        for word in ["Hello", ", ", "world", "!"]:
            yield word

await A2AServer(Writer, port=8001).start()
```

Consume it with `stream_message()`:

```python
from nexus_a2a import A2AHttpClient, Message
from nexus_a2a import StreamEventType

async with A2AHttpClient("http://localhost:8001") as client:
    async for event in client.stream_message(Message.user_text("go")):
        if event.type == StreamEventType.ARTIFACT_CHUNK:
            print(event.data["content"], end="", flush=True)
```

**Event sequence:** `task_created` → `artifact_chunk` (one per yield) →
`task_status` → `done`. Iteration stops at the first terminal event.
`task_created` carries the task id, so you can poll or cancel while it runs.

If you prefer a separate method, add `stream()` instead — it wins over `run()`
when both exist. A class attribute `STREAMING = True` declares that a framework
adapter handles streaming internally.

**The two calling styles are interchangeable.** Neither side needs to know how
the other is written:

| | `message/send` | `message/stream` |
|---|---|---|
| **Streaming agent** | chunks folded into one artifact | one chunk per yield |
| **Non-streaming agent** | normal result | whole output as one chunk |

**Observing a task:** `GET /stream?taskId=...` is what `SSEStreamer` targets.
While the task is **running** — served by someone else's `message/send` or
`message/stream` — it is followed live, with keep-alive comments between
events. A follower that joins mid-run first receives everything already
emitted. A finished or paused task is reported and the stream closes. Following
works within one process; a follower on another replica sees only the final
state.

> Security refusals happen **before** the stream opens — once SSE starts the
> status line is already sent and an HTTP status can no longer be signalled. A
> failure after that point arrives as a terminal `error` event and the task is
> marked `FAILED`.

### Push notifications (webhooks)

For callers that will not sit and wait — a long task, a mobile client, a
serverless function, or a task that pauses at `INPUT_REQUIRED` and needs to tell
somebody. The agent POSTs `task_completed`, `task_failed`, `task_cancelled` and
`task_input_required` to a URL you register.

```python
from nexus_a2a import A2AServer, WebhookConfig

server = A2AServer(
    MyAgent,
    push_config=WebhookConfig(signing_secret="shared-secret"),
)
```

Register the callback with the message, or later:

```python
async with A2AHttpClient("http://localhost:8001") as client:
    task = await client.send_message(
        Message.user_text("go"),
        push_notification={"url": "https://me.example.com/hook", "token": "abc"},
    )

    # ...or against an existing task
    await client.set_push_config(task.id, {"url": "https://me.example.com/hook"})
```

Verify the signature on your side:

```python
from nexus_a2a import WebhookDispatcher

async def hook(request):
    body = await request.body()
    ok = WebhookDispatcher.verify_signature(
        body, request.headers["X-Nexus-Signature-256"], "shared-secret"
    )
```

> **Webhook URLs are an SSRF vector.** The URL comes from whoever called the
> agent, and the server then makes a request to it. URLs resolving to private,
> loopback, link-local or reserved addresses are refused, as is any scheme other
> than http/https. Pass `WebhookConfig(allow_private_urls=True)` **only** for
> local development.

| Situation | Result |
|---|---|
| Agent declares `push_notifications=False` | `-32003` |
| URL is private / bad scheme | `-32004` |
| Config missing `url` | `-32602` |

Delivery is detached, so a slow or dead receiver never stalls the response or
fails the task. The registered target is dropped once the task is terminal.
Targets live in memory and do not survive a restart.

#### Dispatching manually

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

# Trace a task (by task id or trace id) — call tree with per-hop latency
nexus trace abc-123-task-id
nexus trace abc-123-task-id --agent http://localhost:8080 --admin-token $TOKEN

# Replay DLQ entries (admin token from --admin-token, NEXUS_ADMIN_TOKEN or [ops])
nexus replay --failed
nexus replay --failed --skill web_search
nexus replay --failed --last 1h
nexus replay --failed --dry-run        # preview without replaying

# Serve an agent over the A2A protocol
nexus run --module mypackage.agent:MyAgent
nexus run --module mypackage.agent:MyAgent --host 0.0.0.0 --port 8080

# Without --module: ops server only (health, metrics, admin)
nexus run

# Check any A2A agent against the protocol and its own card
nexus verify http://localhost:8001
nexus verify http://localhost:8001 --api-key $KEY --strict   # CI-friendly
nexus verify https://agent.example.com --read-only            # never runs the agent

# Run several agents locally in one process
nexus dev --agent agents.research:Research --agent agents.summary:Summary
nexus dev --verify

# JSON output for all commands
nexus --format json status --network
```

> Runnable as a module too, when the console script is not on PATH (a container,
> a CI step, an uninstalled virtualenv):
>
> ```bash
> python -m nexus_a2a.cli ping http://localhost:8001
> ```


### `nexus verify` — conformance checks

Checks any A2A agent, not only nexus-a2a ones, and exits non-zero on failure:

| Group | Checks |
|---|---|
| card | reachable, valid JSON, schema, URL matches where it was served, unique skills, `/health` |
| auth | **the scheme the card advertises is the one enforced**, and supplied credentials work |
| protocol | JSON-RPC error codes for unknown methods, malformed JSON, missing params, unknown tasks |
| task | `message/send`, `tasks/get` and cancelling a finished task |
| stream | `message/stream` really streams, when the card claims it |
| push | push-config methods exist, and **a cloud-metadata webhook URL is refused** |

The auth group matters most. A card that says `api_key` on an agent that
answers anyone means the agent is open. A card that says `none` on an agent
that returns 401 breaks every client that trusted it.

The task group sends one probe message, so the agent does real work;
`--read-only` skips it. If the agent needs credentials you didn't pass, the
checks that need them are **skipped**, not failed. From Python:

```python
from nexus_a2a.verify import verify_agent

report = await verify_agent("http://localhost:8001", api_key="...")
assert report.passed, report.to_dict()
```

### `nexus dev` — several agents at once

```toml
[[dev.agents]]
module = "agents.research:Research"
port   = 8001

[[dev.agents]]
module = "agents.summary:Summary"      # port assigned automatically
```

Each agent binds `127.0.0.1`, and its card advertises that address. Dev agents
are added to `[network].agents`, so `trust_mode = "strict"` lets them call one
another, and their webhooks may target loopback. The rest of `nexus.toml` —
auth, storage and so on — applies to all of them.

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

One file configures the agent `nexus run` serves: auth, trust, rate limits,
storage, webhooks and an ops server. Every key below is read; anything else
triggers a `ConfigWarning` with a "did you mean" suggestion, so a typo can't
silently leave an agent unsecured.

```toml
[agent]                          # optional with --module: the class supplies these
name        = "ResearchAgent"
description = "Searches the web and summarises results."
version     = "1.0.0"
url         = "http://localhost:8001"

[[agent.skills]]
id          = "web_search"
name        = "Web Search"
description = "Searches the web for a given query."
tags        = ["search", "web"]

[security]
auth_scheme       = "api_key"    # none | api_key | jwt
auth_secret       = "change-me"  # one shared credential for every caller
trust_mode        = "strict"     # off | warn | strict ([network].agents only)
rate_limit        = 10           # requests/sec per caller; 0 = off
rate_burst        = 20
max_payload_bytes = 1048576      # 0 = off
allow_insecure    = false        # see "Preparing for 2.0"

[storage]
backend            = "redis"     # memory | redis | postgres
url                = "redis://localhost:6379"
ttl_sec            = 3600        # Redis key TTL
task_retention_sec = 3600        # memory backend: keep finished tasks this long (0 = forever)
max_tasks          = 10000       # memory backend: cap, finished tasks evicted first (0 = no cap)

[push]
signing_secret     = "webhook-hmac-secret"
allow_private_urls = false       # true only for local development
max_retries        = 3

[ops]
port        = 8080               # start the health/metrics/admin server; 0 = off
host        = ""                 # default: same host as the agent
url         = "http://localhost:8080"   # where `nexus trace` / `nexus replay` go
admin_token = "ops-secret"

[reliability]
task_timeout_sec          = 120.0
max_retries               = 3
retry_on                  = [500, 502, 503, 504]
circuit_breaker_threshold = 5
circuit_recovery_sec      = 30.0

[observability]
log_level = "INFO"
tracing   = true
metrics   = true

[network]
agents = ["http://agent-a:8001", "http://agent-b:8002"]
```

Serve it:

```bash
nexus run --module mypackage.agent:ResearchAgent
```

Or build the same thing in Python:

```python
from nexus_a2a import NexusConfig

runtime = NexusConfig.from_file("nexus.toml").build_runtime(ResearchAgent)
async with runtime:          # connects stores, starts agent + ops server
    await runtime.wait()
```

With `backend = "redis"`, the rate limiter, push targets and DLQ use Redis too,
so every replica enforces one limit and sees one queue.

**Environment variable overrides** — these always win over the file:

| Variable | Overrides |
|---|---|
| `NEXUS_AGENT_NAME` / `NEXUS_AGENT_URL` | `[agent].name` / `.url` |
| `NEXUS_AUTH_SCHEME` / `NEXUS_AUTH_SECRET` | `[security].auth_scheme` / `.auth_secret` |
| `NEXUS_RATE_LIMIT` | `[security].rate_limit` |
| `NEXUS_STORAGE_BACKEND` / `NEXUS_STORAGE_URL` | `[storage].backend` / `.url` |
| `NEXUS_PUSH_SECRET` | `[push].signing_secret` |
| `NEXUS_ADMIN_TOKEN` | `[ops].admin_token` |
| `NEXUS_OPS_PORT` / `NEXUS_OPS_URL` | `[ops].port` / `.url` |
| `NEXUS_TASK_TIMEOUT` | `[reliability].task_timeout_sec` |
| `NEXUS_LOG_LEVEL` | `[observability].log_level` |

> **mTLS is configured from the environment, not the file:**
> `NEXUS_MTLS_CERT_FILE`, `NEXUS_MTLS_KEY_FILE`, `NEXUS_MTLS_CA_FILE`, then
> `MutualTLSConfig.from_env()`. Earlier versions of this README showed
> `mtls_*` keys under `[security]`, and `dlq_*` keys under `[reliability]`,
> but nothing ever read them.

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

## Preparing for 2.0

2.0 turns four 1.9 warnings into errors:

- an unsecured agent on a non-loopback address;
- `AuthManager(allow_unregistered=True)`;
- HS256 secrets shorter than 32 bytes;
- unknown `nexus.toml` keys.

**[MIGRATING.md](MIGRATING.md)** explains each one, including how to make your
test suite fail on them now.

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
| **v1.9.0** | `nexus.toml` drives `nexus run` (`AgentRuntime`), `nexus verify`, `nexus dev`, live `GET /stream`, `/metrics` on agents, bounded task store; fixes the never-checked config secret, cards hiding their auth, and four CLI commands broken against real servers |
| **v1.8.0** | Production scale: durable DLQ and push targets, distributed rate limiting (atomic via Lua), asymmetric JWT + JWKS with algorithm-confusion prevention |
| **v1.7.0** | Streaming, multi-turn (`NeedsInput`) and push notifications. Every capability a card advertises is now served; HMAC webhook signing fixed (it had never validated) and webhook URLs are SSRF-checked |
| **v1.6.0** | `A2AServer` (inbound protocol: agent card + JSON-RPC), `SecurityMiddleware`, `caller_url` on the client, `nexus run` actually serves the agent |
| **v1.5.0** | Security hardening: admin endpoints gated, auth fails closed, JWT audience validated, PyJWT replaces python-jose, `google-adk` moved to an extra |
| **v1.2.0** | `GracefulShutdown`, `AgentServer` (K8s probes), mTLS, `PostgresTaskStore`, `nexus.toml`, CI/CD workflows |
| **v1.3.0** | `nexus` CLI (ping/inspect/status/trace/replay/run), integration test suite, `CHANGELOG.md` |

---

## License

MIT — see [LICENSE](LICENSE).

Built by [HongZiro](https://github.com/dhruvil05).