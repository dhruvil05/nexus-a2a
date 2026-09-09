"""
nexus run
~~~~~~~~~
Start the agent defined in nexus.toml (or --module) and serve it on the
configured URL. Blocks until SIGTERM/SIGINT.

Equivalent to: uvicorn nexus_a2a.AgentServer --host ... --port ...
but wires everything from nexus.toml automatically.
"""

from __future__ import annotations

import asyncio
import importlib

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


async def _run_agent(agent_class: type, host: str, port: int) -> None:
    """Serve an @agent-decorated class over the A2A protocol."""
    try:
        from nexus_a2a.core.a2a_server import A2AServer, InvalidAgentError
    except ImportError as e:
        print_error(f"nexus_a2a import failed: {e}")
        raise SystemExit(1) from e

    try:
        server = A2AServer(agent_class, host=host, port=port)
    except InvalidAgentError as e:
        print_error(str(e))
        raise SystemExit(1) from e

    console.print(
        f"[bold green]Serving agent '{server.card.name}' "
        f"on http://{host}:{port}[/bold green]"
    )
    console.print(
        f"  Agent card: [cyan]http://{host}:{port}/.well-known/agent-card.json[/cyan]"
    )
    console.print(f"  JSON-RPC:   [cyan]POST http://{host}:{port}/[/cyan]")
    console.print(f"  Health:     [cyan]http://{host}:{port}/health[/cyan]")
    skills = ", ".join(server.card.skill_ids()) or "none declared"
    console.print(f"  Skills:     [cyan]{skills}[/cyan]")
    if not server.security.enabled:
        console.print(
            "  Security:   [dim]none — pass a SecurityMiddleware to enforce "
            "auth, trust, rate limits[/dim]"
        )
    console.print("\n[dim]Press CTRL+C to stop.[/dim]\n")

    await server.start()
    try:
        await asyncio.Event().wait()  # block until cancelled (Ctrl+C / SIGTERM)
    finally:
        await server.stop()


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
    """Start the agent server defined in nexus.toml.

    \b
    Examples:
      nexus run
      nexus run --host 0.0.0.0 --port 8080
      nexus run --module mypackage.agent:MyAgent
    """
    cfg = ctx.load_config()

    # Resolve host/port
    agent_url = cfg.get("agent", {}).get("url", "http://localhost:8000")
    cfg_host, cfg_port = _extract_host_port(agent_url)
    final_host = host or cfg_host
    final_port = port or cfg_port

    # Load agent class
    agent_class = None
    if module:
        try:
            mod_path, cls_name = module.rsplit(":", 1)
            mod = importlib.import_module(mod_path)
            agent_class = getattr(mod, cls_name)
        except Exception as e:
            print_error(f"Cannot import '{module}': {e}")
            raise SystemExit(1) from e
    else:
        print_warning(
            "No --module specified — starting the ops server only (health, "
            "metrics, admin). Pass --module pkg.mod:AgentClass to serve an "
            "agent over the A2A protocol."
        )

    try:
        if agent_class is not None:
            asyncio.run(_run_agent(agent_class, final_host, final_port))
        else:
            asyncio.run(
                _run_ops_server(final_host, final_port, str(ctx.config_path))
            )
    except KeyboardInterrupt:
        console.print("\n[dim]Shutting down...[/dim]")
    except Exception as e:
        print_error(str(e))
        raise SystemExit(1) from e
