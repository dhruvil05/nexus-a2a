"""
nexus inspect <url>
~~~~~~~~~~~~~~~~~~~
Pretty-print the full AgentCard of a remote agent:
name, version, URL, capabilities (streaming, push), auth scheme,
input modes, output modes, all skills with descriptions and tags.
"""

from __future__ import annotations

import asyncio

import click
import httpx

from nexus_a2a.cli.main import NexusContext, pass_ctx
from nexus_a2a.cli.output import print_error, render_inspect


async def _fetch_card(url: str) -> dict:
    url = url.rstrip("/")
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{url}/.well-known/agent-card.json")
        resp.raise_for_status()
        return resp.json()


@click.command("inspect")
@click.argument("url")
@pass_ctx
def inspect(ctx: NexusContext, url: str) -> None:
    """Fetch and pretty-print an agent's full AgentCard.

    \b
    Example:
      nexus inspect http://localhost:8001
      nexus inspect http://localhost:8001 --format json
    """
    try:
        card = asyncio.run(_fetch_card(url))
    except httpx.HTTPStatusError as e:
        print_error(f"HTTP {e.response.status_code} from {url}")
        raise SystemExit(1)
    except Exception as e:
        print_error(str(e))
        raise SystemExit(1)

    render_inspect(card, fmt=ctx.fmt)