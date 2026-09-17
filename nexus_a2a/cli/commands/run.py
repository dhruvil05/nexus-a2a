"""
nexus run
~~~~~~~~~
Serve the agent class named by --module, configured from nexus.toml.
Blocks until SIGTERM/SIGINT.

Every section of nexus.toml applies: [security] (auth, trust, rate limit,
payload size), [storage] (backend and retention), [push] (webhook signing),
[ops] (a second server for health, metrics and admin endpoints) and
[reliability] (the task timeout watchdog). Before 1.9.0 none of it was read
here, so `nexus run` could only ever start an open, in-memory server.

Without --module there is no agent to serve, so only the ops server starts.
"""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
from typing import Any

import click

from nexus_a2a.cli.context import NexusContext, pass_ctx
from nexus_a2a.cli.output import console, print_error, print_warning


def _extract_host_port(url: str) -> tuple[str, int]:
    """Extract host and port from a URL string like http://localhost:8001."""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = parsed.hostname or "0.0.0.0"
    port = parsed.port or 8000
    return host, port


def _load_agent_class(module: str) -> Any:
    """Import `package.module:ClassName`. Exits with a message on failure."""
    try:
        mod_path, cls_name = module.rsplit(":", 1)
        mod = importlib.import_module(mod_path)
        return getattr(mod, cls_name)
    except Exception as e:
        print_error(f"Cannot import '{module}': {e}")
        raise SystemExit(1) from e


def build_config(raw: dict[str, Any], agent_class: Any) -> Any:
    """
    Turn raw nexus.toml data into a validated NexusConfig for `agent_class`.

    The class's own card fills in [agent].name and [agent].url when the file
    leaves them out, so a nexus.toml that only configures security or storage
    is enough. Anything the file does set wins.
    """
    from nexus_a2a.config import NexusConfig
    from nexus_a2a.decorators import get_card

    card = get_card(agent_class if isinstance(agent_class, type) else type(agent_class))
    merged = {key: dict(value) if isinstance(value, dict) else value
              for key, value in raw.items()}
    agent_section = merged.setdefault("agent", {})
    agent_section.setdefault("name", card.name)
    agent_section.setdefault("url", str(card.url).rstrip("/"))
    return NexusConfig.from_dict(merged)


def describe_runtime(cfg: Any, runtime: Any) -> list[str]:
    """Human-readable lines describing what the runtime will serve."""
    server = runtime.server
    base = f"http://{server.host}:{server.port}"
    lines = [
        f"[bold green]Serving agent '{server.card.name}' on {base}[/bold green]",
        f"  Agent card: [cyan]{base}/.well-known/agent-card.json[/cyan]",
        f"  JSON-RPC:   [cyan]POST {base}/[/cyan]",
        f"  Health:     [cyan]{base}/health[/cyan]   Metrics: [cyan]{base}/metrics[/cyan]",
        f"  Skills:     [cyan]{', '.join(server.card.skill_ids()) or 'none declared'}[/cyan]",
        f"  Storage:    [cyan]{cfg.storage.backend}[/cyan]",
    ]

    active = [name for name, on in server.security.summary().items() if on]
    if active:
        lines.append(f"  Security:   [cyan]{', '.join(active)}[/cyan]")
    else:
        lines.append(
            "  Security:   [yellow]none[/yellow] [dim]— set [security] in "
            "nexus.toml to require auth, trust or rate limits[/dim]"
        )

    if runtime.ops is not None:
        ops = runtime.ops
        admin = "admin enabled" if ops._admin_token else "admin disabled (no token)"
        lines.append(
            f"  Ops:        [cyan]http://{ops.host}:{ops.port}[/cyan] [dim]({admin})[/dim]"
        )
    return lines


async def _run_agent(
    agent_class: Any,
    raw_config: dict[str, Any],
    host: str | None,
    port: int | None,
) -> None:
    """Serve an @agent class with everything nexus.toml configures."""
    from nexus_a2a.config import ConfigError
    from nexus_a2a.core.a2a_server import InvalidAgentError

    try:
        cfg = build_config(raw_config, agent_class)
        runtime = cfg.build_runtime(agent_class, host=host, port=port)
    except (ConfigError, InvalidAgentError, TypeError) as e:
        print_error(str(e))
        raise SystemExit(1) from e

    cfg.configure_logging()
    for line in describe_runtime(cfg, runtime):
        console.print(line)
    console.print("\n[dim]Press CTRL+C to stop.[/dim]\n")

    async with runtime:
        await runtime.wait()


async def _run_ops_server(host: str, port: int, config_path: str) -> None:
    """
    Start only the ops server (health, readiness, metrics, admin).

    Used when no agent class was given — there is nothing to serve over the
    A2A protocol, but the operational endpoints are still useful.
    """
    try:
        from nexus_a2a.core.agent_server import AgentServer
        from nexus_a2a.network import AgentNetwork
    except ImportError as e:
        print_error(f"nexus_a2a import failed: {e}")
        raise SystemExit(1) from e

    try:
        network = AgentNetwork.from_config(config_path)
    except Exception as e:
        print_warning(
            f"Could not load config from '{config_path}' ({e}); "
            "starting with an empty AgentNetwork."
        )
        network = AgentNetwork()

    # from_config builds but never connects a Redis/Postgres store, so every
    # operation on it would raise "not connected".
    store: Any = network.task_manager._store
    connectable = hasattr(store, "connect")
    if connectable:
        await store.connect()

    # admin_token defaults to NEXUS_ADMIN_TOKEN inside AgentServer.
    server = AgentServer(network=network, host=host, port=port)
    console.print(
        f"[bold green]Starting ops server on http://{host}:{port}[/bold green]"
    )
    console.print(f"  Health:     [cyan]http://{host}:{port}/health[/cyan]")
    console.print(f"  Metrics:    [cyan]http://{host}:{port}/metrics[/cyan]")
    if server._admin_token:
        console.print(
            f"  Admin:      [cyan]http://{host}:{port}/info[/cyan] "
            "[dim](token required)[/dim]"
        )
    else:
        console.print(
            "  Admin:      [dim]/info, /traces, /dlq disabled — "
            "set NEXUS_ADMIN_TOKEN to enable[/dim]"
        )
    console.print("\n[dim]Press CTRL+C to stop.[/dim]\n")

    await server.start()
    try:
        await asyncio.Event().wait()  # block until cancelled (Ctrl+C / SIGTERM)
    finally:
        await server.stop()
        if connectable:
            await store.disconnect()


@click.command("run")
@click.option(
    "--host",
    default=None,
    metavar="HOST",
    help="Override host (default from nexus.toml url).",
)
@click.option(
    "--port",
    default=None,
    type=int,
    metavar="PORT",
    help="Override port (default from nexus.toml url).",
)
@click.option(
    "--module",
    default=None,
    metavar="MODULE:CLASS",
    help="Python dotted path to agent class, e.g. my_package.agent:MyAgent.",
)
@pass_ctx
def run(
    ctx: NexusContext, host: str | None, port: int | None, module: str | None
) -> None:
    """Serve an agent, configured from nexus.toml.

    \b
    Examples:
      nexus run --module mypackage.agent:MyAgent
      nexus run --module mypackage.agent:MyAgent --host 0.0.0.0 --port 8080
      nexus --config prod.toml run --module mypackage.agent:MyAgent
      nexus run            # ops server only
    """
    raw = ctx.load_config()

    if not Path(ctx.config_path).exists() and module:
        print_warning(
            f"No {ctx.config_path} found — serving with defaults: no auth, "
            "in-memory storage."
        )

    try:
        if module:
            agent_class = _load_agent_class(module)
            asyncio.run(_run_agent(agent_class, raw, host, port))
        else:
            print_warning(
                "No --module specified — starting the ops server only (health, "
                "metrics, admin). Pass --module pkg.mod:AgentClass to serve an "
                "agent over the A2A protocol."
            )
            agent_url = raw.get("agent", {}).get("url", "http://localhost:8000")
            cfg_host, cfg_port = _extract_host_port(agent_url)
            asyncio.run(
                _run_ops_server(
                    host or cfg_host, port or cfg_port, str(ctx.config_path)
                )
            )
    except KeyboardInterrupt:
        console.print("\n[dim]Shutting down...[/dim]")
    except SystemExit:
        raise
    except Exception as e:
        print_error(str(e))
        raise SystemExit(1) from e
