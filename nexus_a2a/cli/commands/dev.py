"""
nexus dev
~~~~~~~~~
Run several agents in one process for local development.

Agents come from --agent flags, from [[dev.agents]] in nexus.toml, or both:

    [[dev.agents]]
    module = "agents.research:ResearchAgent"
    port   = 8001

    [[dev.agents]]
    module = "agents.summary:SummaryAgent"     # port assigned automatically

    nexus dev
    nexus dev --agent agents.research:ResearchAgent --agent agents.x:X --verify

Everything else in nexus.toml ([security], [storage], [push], ...) applies to
every agent, with these development overrides:

  - each agent binds 127.0.0.1 and its card advertises that address, so the
    card is correct however the class's own url= was written;
  - every dev agent is added to [network].agents, so trust_mode="strict"
    lets them call one another;
  - webhooks may target loopback, so agents can push to each other;
  - no ops servers start — they would all want the same port — unless
    --dashboard is given: then each agent gets its own ops server on a free
    port with the dashboard on, and one login token is generated for the
    session and printed.

--verify runs `nexus verify` against each agent once they are up.
"""

from __future__ import annotations

import asyncio
import contextlib
import secrets
import socket
from dataclasses import dataclass
from typing import Any

import click

from nexus_a2a.cli.context import NexusContext, pass_ctx
from nexus_a2a.cli.output import console, print_error, print_warning, render_verify

DEV_HOST = "127.0.0.1"


@dataclass
class DevAgent:
    """One agent to serve in dev mode."""

    module: str
    port: int

    @property
    def url(self) -> str:
        return f"http://{DEV_HOST}:{self.port}"


def plan_agents(
    raw: dict[str, Any],
    cli_modules: tuple[str, ...] | list[str],
    base_port: int,
) -> list[DevAgent]:
    """
    Combine config and command-line agents, assigning free ports.

    Explicit ports are honoured; the rest are numbered upward from base_port,
    skipping ports already taken by an explicit entry.

    Raises:
        click.UsageError: A config entry is malformed or two agents share a port.
    """
    entries: list[tuple[str, int | None]] = []

    for index, entry in enumerate(raw.get("dev", {}).get("agents", [])):
        if not isinstance(entry, dict) or not entry.get("module"):
            raise click.UsageError(f"[[dev.agents]] entry {index} needs a 'module'.")
        port = entry.get("port")
        if port is not None and not isinstance(port, int):
            raise click.UsageError(
                f"[[dev.agents]] entry {index}: port must be an integer."
            )
        entries.append((str(entry["module"]), port))

    entries.extend((module, None) for module in cli_modules)

    taken = {port for _, port in entries if port is not None}
    if len(taken) != sum(1 for _, port in entries if port is not None):
        raise click.UsageError("Two [[dev.agents]] entries use the same port.")

    planned: list[DevAgent] = []
    next_port = base_port
    for module, port in entries:
        if port is None:
            while next_port in taken:
                next_port += 1
            port = next_port
            taken.add(port)
        planned.append(DevAgent(module=module, port=port))
    return planned


def dev_config_for(
    raw: dict[str, Any],
    agent: DevAgent,
    peers: list[str],
    dashboard_port: int | None = None,
    dashboard_token: str | None = None,
) -> dict[str, Any]:
    """
    Raw config for one dev agent: the shared file, minus what must differ.

    [agent] is replaced so each agent keeps its own name and gets its own
    address; [ops] is dropped so the agents do not fight over one port. With
    a dashboard port, [ops] is replaced by a private one on that port with
    the dashboard on (content shown unless the file says otherwise — this is
    your own machine and data).
    """
    cfg = {
        key: dict(value) if isinstance(value, dict) else value
        for key, value in raw.items()
        if key not in ("agent", "ops", "dev", "dashboard")
    }
    cfg["agent"] = {"url": agent.url}
    if dashboard_port is not None and dashboard_token:
        cfg["ops"] = {
            "port": dashboard_port,
            "host": DEV_HOST,
            "admin_token": dashboard_token,
        }
        dashboard = dict(raw.get("dashboard", {}))
        dashboard["enabled"] = True
        dashboard.setdefault("show_content", True)
        cfg["dashboard"] = dashboard

    network = cfg.setdefault("network", {})
    known = list(network.get("agents", []))
    network["agents"] = known + [url for url in peers if url not in known]

    push = cfg.setdefault("push", {})
    push.setdefault("allow_private_urls", True)
    return cfg


def build_runtimes(
    raw: dict[str, Any],
    planned: list[DevAgent],
    dashboard_token: str | None = None,
) -> list[tuple[DevAgent, Any]]:
    """
    Import every agent and build its runtime. Raises on the first problem.

    With dashboard_token, each agent also gets an ops server with the
    dashboard on a free port.
    """
    from nexus_a2a.cli.commands.run import _load_agent_class, build_config

    peers = [agent.url for agent in planned]
    runtimes = []
    for agent in planned:
        cls = _load_agent_class(agent.module)
        port = _free_port() if dashboard_token else None
        cfg = build_config(
            dev_config_for(raw, agent, peers, port, dashboard_token), cls
        )
        runtime = cfg.build_runtime(cls, host=DEV_HOST, port=agent.port)
        runtimes.append((agent, runtime))
    return runtimes


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((DEV_HOST, 0))
        return int(sock.getsockname()[1])


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((DEV_HOST, port))
        except OSError:
            return False
    return True


def _describe(runtimes: list[tuple[DevAgent, Any]]) -> None:
    from rich.table import Table

    table = Table(title="nexus dev", header_style="bold cyan")
    table.add_column("Agent", style="cyan", no_wrap=True)
    table.add_column("URL")
    table.add_column("Skills", style="dim")
    table.add_column("Security", style="dim")
    table.add_column("Module", style="dim")
    with_dashboard = any(rt.ops is not None for _, rt in runtimes)
    if with_dashboard:
        table.add_column("Dashboard", style="green")
    for agent, runtime in runtimes:
        server = runtime.server
        active = [n for n, on in server.security.summary().items() if on]
        row = [
            server.card.name,
            agent.url,
            ", ".join(server.card.skill_ids()) or "—",
            ", ".join(active) or "none",
            agent.module,
        ]
        if with_dashboard:
            ops = runtime.ops
            row.append(f"http://{ops.host}:{ops.port}/ui" if ops else "—")
        table.add_row(*row)
    console.print(table)


async def _verify_all(
    runtimes: list[tuple[DevAgent, Any]], raw: dict[str, Any], fmt: str
) -> bool:
    from nexus_a2a.verify import verify_agent

    secret = raw.get("security", {}).get("auth_secret") or None
    scheme = raw.get("security", {}).get("auth_scheme", "none")
    all_ok = True
    for agent, _ in runtimes:
        report = await verify_agent(
            agent.url,
            api_key=secret if scheme == "api_key" else None,
            caller_url=agent.url,
        )
        render_verify(report.to_dict(), fmt=fmt)
        all_ok = all_ok and report.passed
    return all_ok


async def _serve(
    runtimes: list[tuple[DevAgent, Any]],
    raw: dict[str, Any],
    run_verify: bool,
    fmt: str,
    dashboard_token: str | None = None,
) -> None:
    async with contextlib.AsyncExitStack() as stack:
        for _, runtime in runtimes:
            await stack.enter_async_context(runtime)

        _describe(runtimes)
        if dashboard_token:
            console.print(
                "\n[bold]Dashboard login token[/bold] (this session only): "
                f"[green]{dashboard_token}[/green]"
            )
        if run_verify:
            ok = await _verify_all(runtimes, raw, fmt)
            if not ok:
                print_warning("At least one agent failed verification.")
        console.print("\n[dim]Press CTRL+C to stop all agents.[/dim]\n")
        await asyncio.Event().wait()


@click.command("dev")
@click.option(
    "--agent",
    "agents",
    multiple=True,
    metavar="MODULE:CLASS",
    help="Agent to serve (repeatable). Added to any [[dev.agents]] in nexus.toml.",
)
@click.option(
    "--base-port",
    default=8001,
    show_default=True,
    type=int,
    help="First port for agents without an explicit one.",
)
@click.option(
    "--verify",
    "run_verify",
    is_flag=True,
    default=False,
    help="Run conformance checks against each agent once it is up.",
)
@click.option(
    "--dashboard",
    "with_dashboard",
    is_flag=True,
    default=False,
    help="Give each agent a dashboard (http://127.0.0.1:<port>/ui).",
)
@pass_ctx
def dev(
    ctx: NexusContext,
    agents: tuple[str, ...],
    base_port: int,
    run_verify: bool,
    with_dashboard: bool,
) -> None:
    """Run several agents locally in one process.

    \b
    Examples:
      nexus dev
      nexus dev --agent agents.research:Research --agent agents.summary:Summary
      nexus dev --verify
      nexus dev --dashboard
    """
    from nexus_a2a.config import ConfigError

    raw = ctx.load_config()
    planned = plan_agents(raw, agents, base_port)
    if not planned:
        print_error(
            "No agents to run. Pass --agent MODULE:CLASS or add [[dev.agents]] "
            "to nexus.toml."
        )
        raise SystemExit(1)

    busy = [a.port for a in planned if not _port_free(a.port)]
    if busy:
        print_error(f"Port(s) already in use: {', '.join(map(str, busy))}")
        raise SystemExit(1)

    token = secrets.token_urlsafe(24) if with_dashboard else None
    try:
        runtimes = build_runtimes(raw, planned, dashboard_token=token)
    except (ConfigError, TypeError, ValueError) as e:
        print_error(str(e))
        raise SystemExit(1) from e

    try:
        asyncio.run(_serve(runtimes, raw, run_verify, ctx.fmt, token))
    except KeyboardInterrupt:
        console.print("\n[dim]Stopping all agents...[/dim]")
