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
from datetime import datetime, timezone

import click
import httpx

from nexus_a2a.cli.main import NexusContext, pass_ctx
from nexus_a2a.cli.output import print_error, print_warning, render_status


async def _probe_agent(client: httpx.AsyncClient, url: str) -> dict:
    """Probe a single agent for health, queue depth, and DLQ state."""
    url = url.rstrip("/")
    entry: dict = {
        "url": url,
        "name": url,
        "healthy": False,
        "queue_depth": 0,
        "dlq_pending": 0,
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

        # /metrics for queue depth (Prometheus text — parse task_queue_depth line)
        try:
            metrics_resp = await client.get(f"{url}/metrics", timeout=5.0)
            if metrics_resp.status_code == 200:
                for line in metrics_resp.text.splitlines():
                    if line.startswith("nexus_task_queue_depth"):
                        entry["queue_depth"] = int(float(line.split()[-1]))
                    if line.startswith("nexus_dlq_pending"):
                        entry["dlq_pending"] = int(float(line.split()[-1]))
        except Exception:
            pass  # metrics endpoint is optional

        entry["last_seen"] = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        entry["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    except Exception as exc:
        entry["error"] = str(exc)

    return entry


async def _probe_all(urls: list[str]) -> list[dict]:
    async with httpx.AsyncClient() as client:
        tasks = [_probe_agent(client, url) for url in urls]
        return await asyncio.gather(*tasks)


def _build_summary(agents: list[dict]) -> dict:
    healthy = sum(1 for a in agents if a.get("healthy"))
    return {
        "total": len(agents),
        "healthy": healthy,
        "unhealthy": len(agents) - healthy,
        "total_dlq": sum(a.get("dlq_pending", 0) for a in agents),
    }


@click.command("status")
@click.option("--network", is_flag=True, default=False, help="Show all agents from nexus.toml.")
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
        print_warning("No agent URLs found. Use --agents or add [network] agents = [...] to nexus.toml.")
        raise SystemExit(0)

    try:
        agent_data = asyncio.run(_probe_all(urls))
    except Exception as e:
        print_error(str(e))
        raise SystemExit(1)

    summary = _build_summary(agent_data)
    render_status(agent_data, summary, fmt=ctx.fmt)

    if summary["unhealthy"] > 0:
        raise SystemExit(1)