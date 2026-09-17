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
  - no ops servers start — they would all want the same port.

--verify runs `nexus verify` against each agent once they are up.
"""

from __future__ import annotations

import asyncio
import contextlib
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


def dev_config_for(raw: dict[str, Any], agent: DevAgent, peers: list[str]) -> dict[str, Any]:
    """
    Raw config for one dev agent: the shared file, minus what must differ.

    [agent] is replaced so each agent keeps its own name and gets its own
    address; [ops] is dropped so the agents do not fight over one port.
    """
    cfg = {key: dict(value) if isinstance(value, dict) else value
           for key, value in raw.items() if key not in ("agent", "ops", "dev")}
    cfg["agent"] = {"url": agent.url}

    network = cfg.setdefault("network", {})
    known = list(network.get("agents", []))
    network["agents"] = known + [url for url in peers if url not in known]

    push = cfg.setdefault("push", {})
    push.setdefault("allow_private_urls", True)
    return cfg


def build_runtimes(
    raw: dict[str, Any], planned: list[DevAgent]
) -> list[tuple[DevAgent, Any]]:
    """Import every agent and build its runtime. Raises on the first problem."""
    from nexus_a2a.cli.commands.run import _load_agent_class, build_config

    peers = [agent.url for agent in planned]
    runtimes = []
    for agent in planned:
        cls = _load_agent_class(agent.module)
        cfg = build_config(dev_config_for(raw, agent, peers), cls)
        runtime = cfg.build_runtime(cls, host=DEV_HOST, port=agent.port)
        runtimes.append((agent, runtime))
    return runtimes


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
    for agent, runtime in runtimes:
        server = runtime.server
        active = [n for n, on in server.security.summary().items() if on]
        table.add_row(
            server.card.name,
            agent.url,
            ", ".join(server.card.skill_ids()) or "—",
            ", ".join(active) or "none",
            agent.module,
        )
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
) -> None:
    async with contextlib.AsyncExitStack() as stack:
        for _, runtime in runtimes:
            await stack.enter_async_context(runtime)

        _describe(runtimes)
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
@pass_ctx
def dev(
    ctx: NexusContext,
    agents: tuple[str, ...],
    base_port: int,
    run_verify: bool,
) -> None:
    """Run several agents locally in one process.

    \b
    Examples:
      nexus dev
      nexus dev --agent agents.research:Research --agent agents.summary:Summary
      nexus dev --verify
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

    try:
        runtimes = build_runtimes(raw, planned)
    except (ConfigError, TypeError, ValueError) as e:
        print_error(str(e))
        raise SystemExit(1) from e

    try:
        asyncio.run(_serve(runtimes, raw, run_verify, ctx.fmt))
    except KeyboardInterrupt:
        console.print("\n[dim]Stopping all agents...[/dim]")
