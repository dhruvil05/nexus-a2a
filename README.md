# nexus-a2a

> Developer-friendly Python package for building AI agent-to-agent (A2A) communication with ease.

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![Version](https://img.shields.io/badge/version-1.2.0-teal)](https://github.com/dhruvil05/nexus-a2a)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-alpha-orange)](https://github.com/dhruvil05/nexus-a2a)
[![CI](https://github.com/dhruvil05/nexus-a2a/actions/workflows/ci.yml/badge.svg)](https://github.com/dhruvil05/nexus-a2a/actions/workflows/ci.yml)

---

## What is nexus-a2a?

`nexus-a2a` removes the boilerplate of the [A2A protocol](https://a2a-protocol.org) so you can connect AI agents together in minutes instead of days.

Instead of manually writing Agent Cards, JSON-RPC servers, task lifecycle managers, and discovery clients — you use a single decorator and a few intuitive classes.

```python
from nexus_a2a import agent, AgentNetwork

@agent(name="Researcher", description="Searches the web.", url="http://localhost:8001")
class ResearchAgent:
    async def run(self, task):
        return f"Results for: {task.latest_message().text()}"
```

That's it. No boilerplate. No protocol plumbing.

---

## Install

```bash
pip install nexus-a2a
```

With optional storage backends:

```bash
pip install nexus-a2a[redis]      # RedisTaskStore
pip install nexus-a2a[postgres]   # PostgresTaskStore
pip install nexus-a2a[all]        # all extras
```

Requires Python 3.11 or higher.

---

## What's in the box

### `@agent` decorator

Turn any class into an A2A-compatible agent. The decorator auto-generates
an `AgentCard` from the class metadata and validates the class has a proper
`async def run()` method.

```python
from nexus_a2a import agent, get_card, AgentSkill

@agent(
    name="SummaryAgent",
    description="Summarises long documents into key points.",
    url="http://localhost:8002",
    skills=[
        AgentSkill(id="summarise", name="Summarise", description="Summarises text.")
    ],
    streaming=True,
)
class SummaryAgent:
    async def run(self, task):
        text = task.latest_message().text()
        return f"Summary of: {text[:100]}..."

card = get_card(SummaryAgent)
print(card.name)           # SummaryAgent
print(card.skill_ids())    # ['summarise']
```

---

### `NexusConfig` — zero-config wiring from `nexus.toml`

Define your entire agent network in a single TOML file. Every component
(storage, retry, auth, logging) is wired automatically.

```toml
# nexus.toml
[agent]
name        = "ResearchAgent"
url         = "http://localhost:8001"
description = "Searches the web."

[[agent.skills]]
id   = "web_search"
name = "Web Search"

[network]
agents = ["http://summary-agent:8002"]

[reliability]
task_timeout_sec = 60
max_retries      = 3

[security]
auth_scheme = "jwt"
auth_secret = "your-secret-key"   # or use NEXUS_AUTH_SECRET env var

[storage]
backend = "postgres"
url     = "postgresql://user:pass@localhost/nexus"

[observability]
log_level = "INFO"
```

```python
from nexus_a2a import AgentNetwork

# One line — fully wired network
network = AgentNetwork.from_config("nexus.toml")
```

Environment variables always override TOML values — ideal for containers:

```bash
NEXUS_AGENT_NAME=ResearchAgent
NEXUS_AUTH_SECRET=my-secret
NEXUS_STORAGE_BACKEND=redis
NEXUS_STORAGE_URL=redis://redis:6379
NEXUS_LOG_LEVEL=WARNING
```

---

### `AgentServer` — built-in health and metrics endpoints

Expose Kubernetes-compatible probes and Prometheus metrics in one line.

```python
from nexus_a2a import AgentServer

server = AgentServer(network=network, port=8080)
await server.start()

# GET /health   → liveness probe  (always 200 if process alive)
# GET /ready    → readiness probe (200 when store + registry healthy)
# GET /metrics  → Prometheus text format
# GET /info     → JSON summary of network state
```

Kubernetes config:

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8080
readinessProbe:
  httpGet:
    path: /ready
    port: 8080
```

---

### `GracefulShutdown` — zero-task-loss shutdown

Handles SIGTERM (Kubernetes pod eviction) and SIGINT (Ctrl-C). Drains
in-flight tasks before shutting down — no lost work on deploy.

```python
import asyncio
from nexus_a2a import AgentNetwork, AgentServer, GracefulShutdown

async def main():
    network = AgentNetwork.from_config("nexus.toml")
    server  = AgentServer(network=network, port=8080)

    async with GracefulShutdown(network=network, server=server) as sd:
        await server.start()
        await sd.wait()   # blocks until SIGTERM or SIGINT

asyncio.run(main())
```

Drain behaviour:
- Waits up to `drain_timeout_sec` (default 30s) for active tasks to finish.
- Force-cancels any still-running tasks after the timeout.
- Stops `AgentServer`, stops `TaskManager` watchdog, fires `network.shutdown` event.

---

### `MutualTLS` — agent-to-agent mTLS

Both agents must present a valid certificate before any data is exchanged.

```python
from nexus_a2a import MutualTLSConfig, build_client_ssl_context, build_server_ssl_context

config = MutualTLSConfig(
    cert_file="/certs/agent.crt",
    key_file="/certs/agent.key",
    ca_file="/certs/ca.crt",
)

# Outbound calls (httpx client)
ssl_ctx = build_client_ssl_context(config)
async with httpx.AsyncClient(verify=ssl_ctx) as client:
    resp = await client.post("https://other-agent:8443/tasks/send", ...)

# Inbound connections (uvicorn server)
ssl_ctx = build_server_ssl_context(config)
uvicorn.run(app, host="0.0.0.0", port=8443, ssl=ssl_ctx)
```

Load certs from environment variables (Kubernetes Secrets):

```python
config = MutualTLSConfig.from_env()
# Reads: NEXUS_MTLS_CERT_PEM, NEXUS_MTLS_KEY_PEM, NEXUS_MTLS_CA_PEM (base64)
# Or:    NEXUS_MTLS_CERT_FILE, NEXUS_MTLS_KEY_FILE, NEXUS_MTLS_CA_FILE (paths)
```

---

### `TaskManager`

Creates and drives tasks through their full lifecycle.
Every state transition is validated — illegal moves raise clear errors.

```
SUBMITTED → WORKING → COMPLETED
                     → FAILED
                     → CANCELLED
           → INPUT_REQUIRED → (client replies) → WORKING
```

```python
from nexus_a2a import TaskManager, Message, Artifact, Part, PartType

manager = TaskManager()

task = await manager.create(
    initial_message=Message.user_text("Search for AI papers from 2025"),
    skill_id="web_search",
)
await manager.start(task.id)
await manager.complete(
    task.id,
    artifact=Artifact(
        name="search_results",
        parts=[Part(type=PartType.TEXT, content="Found 10 papers...")],
    ),
)
task = await manager.get(task.id)
print(task.state)   # TaskState.COMPLETED
```

---

### `AgentNetwork` — the top-level API

Ties everything together. Register agents, send tasks, run workflows.

```python
from nexus_a2a import AgentNetwork, Message

network = AgentNetwork()

await network.add("http://research-agent:8001")
await network.add("http://summary-agent:8002")

# Send a single task (auto-routes by skill)
task = await network.send(
    message=Message.user_text("Find AI papers from 2025"),
    skill_id="web_search",
)

# Sequential — output of each agent feeds the next
result = await network.sequential(
    agent_urls=["http://research-agent:8001", "http://summary-agent:8002"],
    message=Message.user_text("Research and summarise AI papers"),
)

# Parallel — all agents run concurrently
result = await network.parallel(
    agent_urls=["http://agent-a:8001", "http://agent-b:8002"],
    message=Message.user_text("Analyse this"),
)

@network.on("task.completed")
async def on_done(event: str, data: dict) -> None:
    print(f"Task {data['task_id']} finished")
```

---

### Storage backends

Three backends — swap without changing any other code.

```python
from nexus_a2a import InMemoryTaskStore, TaskManager
manager = TaskManager(store=InMemoryTaskStore())   # development
```

```python
from nexus_a2a import RedisTaskStore, TaskManager
async with RedisTaskStore(url="redis://localhost:6379", ttl=3600) as store:
    manager = TaskManager(store=store)   # persistent, distributed
```

```python
from nexus_a2a import PostgresTaskStore, TaskManager
async with PostgresTaskStore(dsn="postgresql://user:pass@localhost/nexus") as store:
    manager = TaskManager(store=store)   # ACID, full SQL queryability
    active  = await store.list_by_state("working")
    history = await store.list_by_context("ctx-abc-123")
    await store.delete_older_than(days=30)
```

Or let `nexus.toml` wire the right backend automatically — see `NexusConfig` above.

---

### Security

```python
# Auth — JWT or API key per agent
from nexus_a2a.security.auth import AuthManager, AgentCredentialConfig
from nexus_a2a.models.agent import AuthScheme

auth = AuthManager()
auth.register_agent(
    "http://research-agent:8001",
    AgentCredentialConfig(scheme=AuthScheme.API_KEY, api_key="secret-key"),
)
claims  = await auth.verify("http://research-agent:8001", headers=request.headers)
headers = auth.build_auth_headers("http://research-agent:8001")
```

```python
# TrustBoundary — declare which agents may call which
from nexus_a2a.security.trust import TrustBoundary

trust = TrustBoundary()
trust.allow("http://orchestrator:8000", "http://research-agent:8001")
trust.allow("http://orchestrator:8000", "http://summary-agent:8002",
            skills=["summarise"])
trust.block("http://untrusted:9999")
trust.check(caller_url="http://orchestrator:8000",
            target_url="http://research-agent:8001",
            skill_id="web_search")
```

```python
# RateLimiter — token bucket, per agent
from nexus_a2a.security.rate_limiter import RateLimiter, RateLimitConfig

limiter = RateLimiter(default_config=RateLimitConfig(rate=10, burst=20))
await limiter.check("http://agent:8001")
```

---

### Framework adapters

```python
from nexus_a2a.adapters.langgraph  import LangGraphAdapter
from nexus_a2a.adapters.crewai     import CrewAIAdapter
from nexus_a2a.adapters.google_adk import GoogleADKAdapter
from nexus_a2a.adapters.autogen    import AutoGenAdapter

adapter = LangGraphAdapter(agent=compiled_graph)
result  = await adapter.execute(task)
await manager.complete(task.id, artifact=result.to_artifact())
```

---

### SSE streaming

```python
from nexus_a2a.transport.sse import SSEStreamer, StreamEventType

async with SSEStreamer("http://agent:8001").stream(task_id="abc-123") as events:
    async for event in events:
        if event.type == StreamEventType.ARTIFACT_CHUNK:
            print(event.data.get("content", ""), end="", flush=True)
        if event.is_terminal:
            break
```

---

### Observability

```python
from nexus_a2a.storage.audit_logger import AuditLogger
audit = AuditLogger()
audit.task_created(task)
audit.auth_failure("http://agent:8001", reason="expired token")
```

```python
from nexus_a2a.storage.metrics import MetricsCollector
metrics = MetricsCollector()
with metrics.record_agent_call("http://agent:8001"):
    result = await client.send_message(message)
snap = metrics.snapshot()
print(snap.p99_latency("http://agent:8001"))
```

---

## Data models

| Model | Purpose |
|---|---|
| `AgentCard` | Agent's identity, capabilities, and skills |
| `AgentSkill` | A single capability an agent advertises |
| `Task` | The core unit of work — stateful, trackable |
| `TaskState` | `submitted` → `working` → `completed` / `failed` / `cancelled` |
| `Message` | One turn of conversation between client and agent |
| `Part` | Smallest content unit: text, JSON, or file |
| `Artifact` | Immutable final output produced by an agent |

---

## Error types

| Error | When it's raised |
|---|---|
| `TaskNotFoundError` | Accessing a task ID that doesn't exist |
| `TaskAlreadyDoneError` | Mutating a task in a terminal state |
| `AgentUnreachableError` | Remote agent didn't respond after all retries |
| `RemoteAgentError` | Remote agent returned a JSON-RPC error |
| `AgentCardFetchError` | Agent card endpoint returned invalid data |
| `ConfigError` | Invalid or missing `nexus.toml` field |
| `MtlsConfigError` | mTLS config missing cert, key, or CA |
| `MtlsCertificateError` | Peer certificate failed verification |

---

## Roadmap

| Version | Phase | Status |
|---|---|---|
| `v0.1.0` | Models + `@agent` decorator | ✅ Done |
| `v0.2.0` | TaskManager, Registry, HTTP transport | ✅ Done |
| `v0.3.0` | Security — Auth, TrustBoundary, RateLimiter, Validator | ✅ Done |
| `v0.4.0` | Orchestration — sequential, parallel, DAG + SSE streaming | ✅ Done |
| `v1.0.0` | Framework adapters (LangGraph, CrewAI, ADK) + observability | ✅ Done |
| `v1.1.0` | Reliability — CircuitBreaker, retry, InputHandler, DeadLetterQueue, Tracer | ✅ Done |
| `v1.2.0` | Infrastructure — `nexus.toml` config, AgentServer, GracefulShutdown, mTLS, PostgreSQL | ✅ Done |

---

## Development setup

```bash
git clone https://github.com/dhruvil05/nexus-a2a.git
cd nexus-a2a

# Install all dependencies including dev tools
uv pip install -e ".[dev,all]"

# Run tests (unit only)
pytest tests/ -m "not integration" -v

# Run integration tests (requires PostgreSQL)
NEXUS_TEST_PG_DSN="postgresql://user:pass@localhost/nexus_test" \
  pytest tests/ -m integration -v

# Lint
ruff check nexus_a2a/

# Type check
mypy nexus_a2a/

# Build distribution
uv build
```

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

## Contributing

Issues and pull requests are welcome.
Please open an issue first to discuss any significant changes.