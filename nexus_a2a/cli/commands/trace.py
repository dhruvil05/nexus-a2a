"""
nexus trace <id>
~~~~~~~~~~~~~~~~
Query TraceStore and render the full distributed call tree with per-hop latency,
status icons (✓/✗), error messages, and yellow highlighting for slow hops (>500ms).

<id> may be either a trace id or a task id. Traces are stored by trace id, but
the command has always been documented as `nexus trace <task_id>` — before
1.9.0 a task id never matched anything. Each span records the task it
produced, so a task id now resolves to the trace containing it.

Output modes:
  table (default) — Rich tree rendered to terminal
  json            — Raw trace dict as JSON
"""

from __future__ import annotations

import asyncio
from typing import Any

import click
import httpx

from nexus_a2a.cli.admin import (
    admin_headers,
    forbidden_hint,
    resolve_admin_token,
    resolve_ops_url,
)
from nexus_a2a.cli.context import NexusContext, pass_ctx
from nexus_a2a.cli.output import print_error, print_warning, render_trace


class AdminForbiddenError(Exception):
    """The ops server refused the admin request."""


async def _fetch_trace_remote(
    agent_url: str,
    trace_id: str,
    token: str | None = None,
) -> dict[str, Any] | None:
    """Ask a running ops server for a trace via GET /traces/<id>."""
    agent_url = agent_url.rstrip("/")
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            f"{agent_url}/traces/{trace_id}", headers=admin_headers(token)
        )
        if resp.status_code == 404:
            return None
        if resp.status_code == 403:
            raise AdminForbiddenError(forbidden_hint(agent_url, token))
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        return data


def _try_local_trace_store(trace_id: str) -> dict[str, Any] | None:
    """
    Try to read from an in-process TraceStore if this command is run
    inside the same process (e.g. during testing or embedded use).
    Accepts a trace id or a task id. Returns None if nothing matches.
    """
    try:
        from nexus_a2a.transport.tracing import default_store

        raw = default_store.resolve(trace_id)
        if raw is None:
            return None
        return _trace_to_dict(raw)
    except Exception:
        return None


def _trace_to_dict(trace: object) -> dict[str, Any]:
    """Convert a Trace dataclass/object to a render-friendly dict."""
    try:
        hops: list[dict[str, Any]] = []
        for span in getattr(trace, "spans", []):
            hops.append(
                {
                    "url": getattr(span, "agent_url", "unknown"),
                    "duration_ms": getattr(span, "duration_ms", None),
                    "status": getattr(span, "status", "unknown"),
                    "error": getattr(span, "error", None),
                    "children": [],
                }
            )
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
    help="Ops server to query (default: NEXUS_OPS_URL, [ops].url, [agent].url).",
)
@click.option(
    "--admin-token",
    default=None,
    metavar="TOKEN",
    help="Admin token for /traces (default: NEXUS_ADMIN_TOKEN, [ops].admin_token).",
)
@pass_ctx
def trace(
    ctx: NexusContext,
    task_id: str,
    agent_url: str | None,
    admin_token: str | None,
) -> None:
    """Show the distributed call tree for a task or trace id.

    \b
    Examples:
      nexus trace abc-123
      nexus trace abc-123 --agent http://localhost:8080 --admin-token $TOKEN
      nexus trace abc-123 --format json
    """
    trace_data: dict[str, Any] | None = None

    # 1. Try local in-process store first (no HTTP needed)
    trace_data = _try_local_trace_store(task_id)

    # 2. Otherwise ask the ops server
    if trace_data is None:
        cfg = ctx.load_config()
        url = resolve_ops_url(agent_url, cfg)
        token = resolve_admin_token(admin_token, cfg)

        if url:
            try:
                trace_data = asyncio.run(_fetch_trace_remote(url, task_id, token))
            except AdminForbiddenError as e:
                print_error(str(e))
                raise SystemExit(1) from e
            except Exception as e:
                print_error(f"Could not fetch trace from {url}: {e}")
                raise SystemExit(1) from e

    if trace_data is None:
        print_warning(
            f"No trace found for '{task_id}'. "
            "Make sure tracing=true in nexus.toml and the task was run recently."
        )
        raise SystemExit(1)

    render_trace(trace_data, fmt=ctx.fmt)
