"""
nexus trace <task_id>
~~~~~~~~~~~~~~~~~~~~~
Query TraceStore and render the full distributed call tree with per-hop latency,
status icons (✓/✗), error messages, and yellow highlighting for slow hops (>500ms).

Output modes:
  table (default) — Rich tree rendered to terminal
  json            — Raw trace dict as JSON
"""

from __future__ import annotations

import asyncio
from typing import Optional

import click
import httpx

from nexus_a2a.cli.main import NexusContext, pass_ctx
from nexus_a2a.cli.output import console, print_error, print_warning, render_trace


async def _fetch_trace_remote(agent_url: str, trace_id: str) -> Optional[dict]:
    """Ask a running agent server for a specific trace via GET /traces/<id>."""
    agent_url = agent_url.rstrip("/")
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{agent_url}/traces/{trace_id}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()


def _try_local_trace_store(trace_id: str) -> Optional[dict]:
    """
    Try to read from an in-process TraceStore if this command is run
    inside the same process (e.g. during testing or embedded use).
    Returns None if the store is not accessible.
    """
    try:
        from nexus_a2a.transport.tracing import TraceStore  # type: ignore
        store = TraceStore.instance()  # returns singleton if exists
        raw = store.get(trace_id)
        if raw is None:
            return None
        # Convert Trace object to dict for rendering
        return _trace_to_dict(raw)
    except Exception:
        return None


def _trace_to_dict(trace: object) -> dict:
    """Convert a Trace dataclass/object to a render-friendly dict."""
    try:
        hops = []
        for span in getattr(trace, "spans", []):
            hops.append({
                "url": getattr(span, "agent_url", "unknown"),
                "duration_ms": getattr(span, "duration_ms", None),
                "status": getattr(span, "status", "unknown"),
                "error": getattr(span, "error", None),
                "children": [],
            })
        return {
            "trace_id": getattr(trace, "trace_id", "unknown"),
            "hops": hops,
        }
    except Exception:
        return {"trace_id": str(trace), "hops": []}


@click.command("trace")
@click.argument("task_id")
@click.option(
    "--agent",
    "agent_url",
    default=None,
    metavar="URL",
    help="Agent URL to query for trace data (e.g. http://localhost:8001).",
)
@pass_ctx
def trace(ctx: NexusContext, task_id: str, agent_url: Optional[str]) -> None:
    """Show the distributed call tree for a task.

    \b
    Examples:
      nexus trace abc-123
      nexus trace abc-123 --agent http://localhost:8001
      nexus trace abc-123 --format json
    """
    trace_data: Optional[dict] = None

    # 1. Try local in-process store first (no HTTP needed)
    trace_data = _try_local_trace_store(task_id)

    # 2. If agent URL provided (or in config), query remotely
    if trace_data is None:
        if agent_url is None:
            cfg = ctx.load_config()
            agent_url = cfg.get("agent", {}).get("url")

        if agent_url:
            try:
                trace_data = asyncio.run(_fetch_trace_remote(agent_url, task_id))
            except Exception as e:
                print_error(f"Could not fetch trace from {agent_url}: {e}")
                raise SystemExit(1)

    if trace_data is None:
        print_warning(
            f"No trace found for task_id '{task_id}'. "
            "Make sure tracing=true in nexus.toml and the task was run recently."
        )
        raise SystemExit(1)

    render_trace(trace_data, fmt=ctx.fmt)