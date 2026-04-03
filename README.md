# nexus-a2a

A developer-friendly Python package for building AI agent-to-agent (A2A) communication with ease.

## Install

```bash
pip install nexus-a2a
```

## Quick start

```python
from nexus_a2a import agent, AgentNetwork

@agent(name="researcher", skills=["search"])
class ResearchAgent:
    async def run(self, task):
        return f"Result for: {task}"
```

## Status
🚧 Active development — v0.1.0 alpha
