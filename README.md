# nexus-a2a

> Developer-friendly Python package for building AI agent-to-agent (A2A) communication with ease.

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![Version](https://img.shields.io/badge/version-0.2.0-teal)](https://github.com/dhruvil05/nexus-a2a)
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

## What's inside — v0.2.0

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

## Roadmap

| Version | Phase | Status |
|---|---|---|
| `v0.1.0` | Models + `@agent` decorator | ✅ Done |
| `v0.2.0` | TaskManager, Registry, HTTP transport | ✅ Done |
| `v0.3.0` | Security — Auth, TrustBoundary, RateLimiter, Validator | 🔨 Next |
| `v0.4.0` | Orchestration — sequential, parallel, DAG workflows + SSE streaming | 📋 Planned |
| `v1.0.0` | Framework adapters (LangGraph, CrewAI, ADK) + observability | 📋 Planned |

---

## Development setup

```bash
# Clone the repo
git clone https://github.com/yourusername/nexus-a2a.git
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