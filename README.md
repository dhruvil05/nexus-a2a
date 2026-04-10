# nexus-a2a

> Developer-friendly Python package for building AI agent-to-agent (A2A) communication with ease.

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![Version](https://img.shields.io/badge/version-1.0.0-teal)](https://github.com/dhruvil05/nexus-a2a)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-alpha-orange)](https://github.com/dhruvil05/nexus-a2a)

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

Requires Python 3.11 or higher.

---

## What's in treasure

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

# Read the auto-generated card
card = get_card(SummaryAgent)
print(card.name)           # SummaryAgent
print(card.skill_ids())    # ['summarise']
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

# Create a task
task = await manager.create(
    initial_message=Message.user_text("Search for AI papers from 2025"),
    skill_id="web_search",
)

# Drive the lifecycle
await manager.start(task.id)
await manager.complete(
    task.id,
    artifact=Artifact(
        name="search_results",
        parts=[Part(type=PartType.TEXT, content="Found 10 papers...")],
    ),
)

# Retrieve it anytime
task = await manager.get(task.id)
print(task.state)   # TaskState.COMPLETED
```

---

### `AgentRegistry`
Discovers and health-checks remote agents. Register a URL once —
the registry fetches the AgentCard automatically.

```python
from nexus_a2a import AgentRegistry

registry = AgentRegistry()

# Register a remote agent by URL (fetches its AgentCard automatically)
card = await registry.register_url("http://research-agent:8001")

# Find agents by skill
agents = registry.find_by_skill("web_search")

# Health check all registered agents
results = await registry.check_all_health()
# {"http://research-agent:8001": True, ...}

# Summary of the network
print(registry.summary())
# {"total": 3, "healthy": 3, "agents": [...]}
```

---

### `A2AHttpClient`
Low-level async HTTP client for sending tasks to remote agents.
Handles JSON-RPC 2.0 envelopes, retries, and error translation.

```python
from nexus_a2a import A2AHttpClient, Message

async with A2AHttpClient("http://research-agent:8001") as client:
    # Fetch the remote agent's card
    card = await client.fetch_agent_card()

    # Send a task
    task = await client.send_message(
        message=Message.user_text("Find AI papers from 2025"),
        skill_id="web_search",
    )

    # Poll for result
    task = await client.get_task(task.id)
    print(task.state)

    # Cancel if needed
    await client.cancel_task(task.id)
```

---

### `InMemoryTaskStore`
Default task storage — zero config, works out of the box.
Swap for `RedisTaskStore` or `PostgresTaskStore` in production (coming in v0.5.0).

```python
from nexus_a2a import TaskManager, InMemoryTaskStore

# Explicit (same as the default)
manager = TaskManager(store=InMemoryTaskStore())
```

---

## Data models

| Model | Purpose |
|---|---|
| `AgentCard` | Agent's identity, capabilities, and skills — served at `/.well-known/agent-card.json` |
| `AgentSkill` | A single capability an agent advertises |
| `AgentCapabilities` | Flags: streaming, push notifications, multi-turn |
| `Task` | The core unit of work — stateful, trackable |
| `TaskState` | Enum: `submitted`, `working`, `input_required`, `completed`, `failed`, `cancelled` |
| `Message` | One turn of conversation between client and agent |
| `Part` | Smallest content unit inside a message: text, JSON, or file |
| `Artifact` | Immutable final output produced by an agent |

---

## Error types

| Error | When it's raised |
|---|---|
| `TaskNotFoundError` | Accessing a task ID that doesn't exist |
| `TaskAlreadyDoneError` | Mutating a task that is already in a terminal state |
| `AgentUnreachableError` | Remote agent server didn't respond after all retries |
| `RemoteAgentError` | Remote agent returned a JSON-RPC error response |
| `AgentCardFetchError` | Agent card endpoint returned invalid data |

---

---

### `AuthManager`
Verifies credentials on every inbound request. Each agent can use its own scheme.

```python
from nexus_a2a.security.auth import AuthManager, AgentCredentialConfig
from nexus_a2a.models.agent import AuthScheme

auth = AuthManager()

# Register an agent that expects an API key
auth.register_agent(
    "http://research-agent:8001",
    AgentCredentialConfig(scheme=AuthScheme.API_KEY, api_key="secret-key"),
)

# Verify an inbound request
claims = await auth.verify("http://research-agent:8001", headers=request.headers)

# Build headers for outbound calls automatically
headers = auth.build_auth_headers("http://research-agent:8001")
```

---

### `TrustBoundary`
Enforces which agents are allowed to call which other agents. Default policy: **deny all**.

```python
from nexus_a2a.security.trust import TrustBoundary

trust = TrustBoundary()

# Allow orchestrator to call research agent (all skills)
trust.allow("http://orchestrator:8000", "http://research-agent:8001")

# Allow orchestrator to call summary agent — but ONLY the 'summarise' skill
trust.allow("http://orchestrator:8000", "http://summary-agent:8002",
            skills=["summarise"])

# Block a rogue agent entirely
trust.block("http://untrusted:9999")

# Check before routing — raises AgentNotAllowedError or SkillNotAllowedError
trust.check(caller_url="http://orchestrator:8000",
            target_url="http://research-agent:8001",
            skill_id="web_search")
```

---

### `RateLimiter`
Token bucket rate limiter — per agent, configurable burst and sustained rate.

```python
from nexus_a2a.security.rate_limiter import RateLimiter, RateLimitConfig

limiter = RateLimiter(default_config=RateLimitConfig(rate=10, burst=20))

# Tighter limit for a heavy agent
limiter.set_limit("http://heavy-agent:8003", RateLimitConfig(rate=2, burst=5))

# Check on every request — raises RateLimitError if exceeded
await limiter.check("http://heavy-agent:8003")

# Non-raising version
allowed = await limiter.is_allowed("http://heavy-agent:8003")
```

---

### `PayloadValidator`
Validates and sanitises every inbound Message before agent logic runs.

```python
from nexus_a2a.security.validator import PayloadValidator, ValidatorConfig

validator = PayloadValidator(
    config=ValidatorConfig(max_bytes=512_000, max_parts=10)
)

# Validate and sanitise — raises on violations, returns clean Message
clean_message = validator.validate(incoming_message)

# Parse from raw dict (e.g. HTTP request body) and validate
clean_message = validator.validate_dict(request_body_dict)
```

---

---

### `AgentNetwork` — the top-level API
Ties everything together. Register agents, send tasks, run workflows.

```python
from nexus_a2a import AgentNetwork, Message

network = AgentNetwork()

# Register agents
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

# Parallel — all agents get the same input, run concurrently
result = await network.parallel(
    agent_urls=["http://agent-a:8001", "http://agent-b:8002"],
    message=Message.user_text("Analyse this"),
)

# Subscribe to network events
@network.on("task.completed")
async def on_done(event: str, data: dict) -> None:
    print(f"Task {data['task_id']} finished")
```

---

### `Orchestrator` — multi-agent workflows
Three execution modes: sequential, parallel, and DAG (dependency graph).

```python
from nexus_a2a.core.orchestrator import Orchestrator, DAGNode

orch = Orchestrator(runner=my_runner)

# DAG: A runs first, then B and C in parallel, then D
result = await orch.dag(
    nodes=[
        DAGNode("http://fetch:8001"),
        DAGNode("http://parse:8002", depends_on=["http://fetch:8001"]),
        DAGNode("http://enrich:8003", depends_on=["http://fetch:8001"]),
        DAGNode("http://store:8004",
                depends_on=["http://parse:8002", "http://enrich:8003"]),
    ],
    initial_message=Message.user_text("Process document"),
)
print(result.succeeded)      # True
print(result.total_sec)      # wall-clock time
print(result.failed_steps)   # [] if all succeeded
```

---

### `EventBus` — async pub/sub
In-process event broadcasting between agents and application code.

```python
from nexus_a2a.network import EventBus

bus = EventBus()

async def on_task_done(event: str, data: dict) -> None:
    print(f"Task {data['task_id']} completed")

bus.subscribe("task.completed", on_task_done)
await bus.publish("task.completed", {"task_id": "abc-123"})
bus.unsubscribe("task.completed", on_task_done)
```

---

### SSE streaming
Stream live task updates from a remote agent as they happen.

```python
from nexus_a2a.transport.sse import SSEStreamer, StreamEventType

streamer = SSEStreamer("http://agent:8001")
async with streamer.stream(task_id="abc-123") as events:
    async for event in events:
        if event.type == StreamEventType.ARTIFACT_CHUNK:
            print(event.data.get("content", ""), end="", flush=True)
        if event.is_terminal:
            break
```

---

### Webhooks — async push notifications
For long-running tasks where the client cannot hold an open connection.

```python
from nexus_a2a.transport.webhook import WebhookDispatcher, WebhookConfig

dispatcher = WebhookDispatcher(
    config=WebhookConfig(signing_secret="my-secret", max_retries=3)
)
await dispatcher.dispatch(
    url="https://client.example.com/hooks/nexus",
    task=completed_task,
    event="task_completed",
)

# On the receiving end — verify the signature
is_valid = WebhookDispatcher.verify_signature(
    payload=request.body,
    signature=request.headers["X-Nexus-Signature-256"],
    secret="my-secret",
)
```

---

---

### Framework adapters — plug in any agent framework
Wrap any existing agent with one class. No protocol knowledge needed.

```python
# LangGraph
from nexus_a2a.adapters.langgraph import LangGraphAdapter
adapter = LangGraphAdapter(agent=compiled_graph)

# CrewAI
from nexus_a2a.adapters.crewai import CrewAIAdapter
adapter = CrewAIAdapter(agent=my_crew, input_key="topic")

# Google ADK
from nexus_a2a.adapters.google_adk import GoogleADKAdapter
adapter = GoogleADKAdapter(agent=adk_agent, app_name="my_app")

# AutoGen
from nexus_a2a.adapters.autogen import AutoGenAdapter
adapter = AutoGenAdapter(agent=assistant, max_turns=1)

# Custom framework — implement BaseAdapter
from nexus_a2a.adapters.base import BaseAdapter, AdapterResult
class MyAdapter(BaseAdapter):
    framework_name = "myframework"
    async def execute(self, task: Task) -> AdapterResult:
        text   = self.extract_input(task)
        output = await self.agent.run(text)
        return self.make_result(str(output))

# Use any adapter with TaskManager
result = await adapter.execute(task)
await manager.complete(task.id, artifact=result.to_artifact())
```

---

### `RedisTaskStore` — production storage
Drop-in replacement for `InMemoryTaskStore`. Persistent and distributed.

```python
from nexus_a2a.storage.redis_store import RedisTaskStore
from nexus_a2a import TaskManager

async with RedisTaskStore(url="redis://localhost:6379", ttl=3600) as store:
    manager = TaskManager(store=store)
    task    = await manager.create(Message.user_text("Hello"))
```

---

### `AuditLogger` — structured audit trail
Every significant event written as JSON lines (NDJSON format).
Plug into Datadog, Loki, CloudWatch, or any log aggregator.

```python
from nexus_a2a.storage.audit_logger import AuditLogger

audit = AuditLogger()   # writes to stdout by default
# or write to a file:
audit = AuditLogger(stream=open("audit.ndjson", "a"))

audit.task_created(task)
audit.agent_called("http://research-agent:8001", task.id, skill_id="search")
audit.auth_failure("http://agent:8001", reason="expired token")
audit.workflow_completed("sequential", total_sec=3.2, steps=3, succeeded=True)

# Inspect in tests
entries = audit.entries_by_event(AuditEvent.AUTH_FAILURE)
```

---

### `MetricsCollector` — operational metrics
Track latency, error rates, and counters. Export to OpenTelemetry.

```python
from nexus_a2a.storage.metrics import MetricsCollector

metrics = MetricsCollector()

metrics.record_task_created()
with metrics.record_agent_call("http://agent:8001"):
    result = await client.send_message(message)

snap = metrics.snapshot()
print(snap.tasks_completed)
print(snap.avg_latency("http://agent:8001"))
print(snap.p99_latency("http://agent:8001"))

# OpenTelemetry export
from opentelemetry import metrics as otel_metrics
meter   = otel_metrics.get_meter("nexus-a2a")
metrics = MetricsCollector.with_otel(meter)
```

---

## Roadmap

| Version | Phase | Status |
|---|---|---|
| `v0.1.0` | Models + `@agent` decorator | ✅ Done |
| `v0.2.0` | TaskManager, Registry, HTTP transport | ✅ Done |
| `v0.3.0` | Security — Auth, TrustBoundary, RateLimiter, Validator | ✅ Done |
| `v0.4.0` | Orchestration — sequential, parallel, DAG workflows + SSE streaming | ✅ Done |
| `v1.0.0` | Framework adapters (LangGraph, CrewAI, ADK) + observability | ✅ Done |

---

## Development setup

```bash
# Clone the repo
git clone https://github.com/dhruvil05/nexus-a2a.git
cd nexus-a2a

# Install all dependencies including dev tools
uv add pydantic httpx starlette uvicorn "python-jose[cryptography]" a2a-sdk
uv add --dev pytest pytest-asyncio ruff mypy

# Run tests
uv run pytest tests/ -v

# Lint
uv run ruff check nexus_a2a/

# Type check
uv run mypy nexus_a2a/
```

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

## Contributing

Issues and pull requests are welcome.
Please open an issue first to discuss any significant changes.