"""
nexus ping <url>
~~~~~~~~~~~~~~~~
Fetch AgentCard from /.well-known/agent-card.json, hit /health,
report: agent name, version, skills count, round-trip latency, health status.
"""

from __future__ import annotations

import asyncio
import time

import click
import httpx

from nexus_a2a.cli.main import NexusContext, pass_ctx
from nexus_a2a.cli.output import print_error, render_ping


async def _do_ping(url: str) -> dict:
    url = url.rstrip("/")
    result: dict = {"url": url, "healthy": False, "health_endpoint_present": False}

    async with httpx.AsyncClient(timeout=10.0) as client:
        # 1. Fetch AgentCard
        t0 = time.perf_counter()
        try:
            card_resp = await client.get(f"{url}/.well-known/agent-card.json")
            card_resp.raise_for_status()
            card = card_resp.json()
            result["name"] = card.get("name", "Unknown")
            result["version"] = card.get("version", "?")
            result["skills_count"] = len(card.get("skills", []))
        except httpx.HTTPStatusError as e:
            result["error"] = f"AgentCard HTTP {e.response.status_code}"
            return result
        except Exception as e:
            result["error"] = f"AgentCard fetch failed: {e}"
            return result

        # 2. Hit /health
        try:
            health_resp = await client.get(f"{url}/health")
            result["health_endpoint_present"] = True
            result["healthy"] = health_resp.status_code == 200
        except Exception:
            # /health is optional — warn but don't fail
            result["health_endpoint_present"] = False
            result["healthy"] = True  # card fetched OK, assume alive

        result["latency_ms"] = (time.perf_counter() - t0) * 1000

    return result


@click.command("ping")
@click.argument("url")
@pass_ctx
def ping(ctx: NexusContext, url: str) -> None:
    """Ping an agent: fetch its card and check health.

    \b
    Example:
      nexus ping http://localhost:8001
    """
    try:
        result = asyncio.run(_do_ping(url))
    except Exception as e:
        print_error(str(e))
        raise SystemExit(1)

    render_ping(result, fmt=ctx.fmt)

    if not result.get("healthy"):
        raise SystemExit(1)