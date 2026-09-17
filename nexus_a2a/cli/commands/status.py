"""
nexus status --network
~~~~~~~~~~~~~~~~~~~~~~
Table of all registered agents: name, URL, health (✓/✗),
task queue depth, DLQ pending count, last seen timestamp.
Summary row: total agents, healthy count, total DLQ pending.

Reads agent URLs from nexus.toml [network] section or --agents flag.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import Any

import click
import httpx

from nexus_a2a.cli.context import NexusContext, pass_ctx
from nexus_a2a.cli.output import print_error, print_warning, render_status

# Checked in order; the first one present wins.
QUEUE_DEPTH_METRICS = ("nexus_a2a_tasks_active",)
DLQ_PENDING_METRICS = ("nexus_a2a_dlq_pending",)


def parse_prometheus(text: str) -> dict[str, float]:
    """
    Parse unlabelled Prometheus exposition lines into {name: value}.

    Comment lines and labelled series are skipped; the probe only needs
    simple gauges.
    """
    values: dict[str, float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2 or "{" in parts[0]:
            continue
        try:
            values[parts[0]] = float(parts[1])
        except ValueError:
            continue
    return values


def _first_int(metrics: dict[str, float], names: tuple[str, ...]) -> int | None:
    for name in names:
        if name in metrics:
            return int(metrics[name])
    return None


async def _probe_agent(client: httpx.AsyncClient, url: str) -> dict[str, Any]:
    """Probe a single agent for health, queue depth, and DLQ state."""
    url = url.rstrip("/")
    # None means "this server does not report it", which is rendered as a
    # dash rather than a misleading 0.
    entry: dict[str, Any] = {
        "url": url,
        "name": url,
        "healthy": False,
        "queue_depth": None,
        "dlq_pending": None,
        "last_seen": "—",
    }

    t0 = time.perf_counter()
    try:
        # AgentCard for name
        card_resp = await client.get(f"{url}/.well-known/agent-card.json", timeout=5.0)
        if card_resp.status_code == 200:
            card = card_resp.json()
            entry["name"] = card.get("name", url)

        # /health for liveness
        health_resp = await client.get(f"{url}/health", timeout=5.0)
        entry["healthy"] = health_resp.status_code == 200

        # /metrics — optional. Before 1.9.0 this looked for
        # "nexus_task_queue_depth" and "nexus_dlq_pending", names only the test
        # mock ever emitted; real servers prefix with "nexus_a2a_", so both
        # columns always read 0.
        try:
            metrics_resp = await client.get(f"{url}/metrics", timeout=5.0)
            if metrics_resp.status_code == 200:
                metrics = parse_prometheus(metrics_resp.text)
                entry["queue_depth"] = _first_int(metrics, QUEUE_DEPTH_METRICS)
                entry["dlq_pending"] = _first_int(metrics, DLQ_PENDING_METRICS)
        except Exception:
            pass  # metrics endpoint is optional

        entry["last_seen"] = datetime.now(UTC).strftime("%H:%M:%S UTC")
        entry["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    except Exception as exc:
        entry["error"] = str(exc)

    return entry


async def _probe_all(urls: list[str]) -> list[dict[str, Any]]:
    async with httpx.AsyncClient() as client:
        tasks = [_probe_agent(client, url) for url in urls]
        return await asyncio.gather(*tasks)


def _build_summary(agents: list[dict[str, Any]]) -> dict[str, Any]:
    healthy = sum(1 for a in agents if a.get("healthy"))
    return {
        "total": len(agents),
        "healthy": healthy,
        "unhealthy": len(agents) - healthy,
        "total_dlq": sum(a.get("dlq_pending") or 0 for a in agents),
    }


@click.command("status")
@click.option(
    "--network", is_flag=True, default=False, help="Show all agents from nexus.toml."
)
@click.option(
    "--agents",
    multiple=True,
    metavar="URL",
    help="Agent URLs to probe (repeatable). Overrides nexus.toml.",
)
@pass_ctx
def status(ctx: NexusContext, network: bool, agents: tuple[str, ...]) -> None:
    """Show the health and queue status of all registered agents.

    \b
    Examples:
      nexus status --network
      nexus status --agents http://localhost:8001 --agents http://localhost:8002
    """
    urls: list[str] = list(agents)

    if not urls:
        cfg = ctx.load_config()
        urls = cfg.get("network", {}).get("agents", [])

    if not urls:
        print_warning(
            "No agent URLs found. Use --agents or add [network] agents = [...] to nexus.toml."
        )
        raise SystemExit(0)

    try:
        agent_data = asyncio.run(_probe_all(urls))
    except Exception as e:
        print_error(str(e))
        raise SystemExit(1) from e

    summary = _build_summary(agent_data)
    render_status(agent_data, summary, fmt=ctx.fmt)

    if summary["unhealthy"] > 0:
        raise SystemExit(1)
