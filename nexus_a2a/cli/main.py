"""
nexus_a2a.cli.main
~~~~~~~~~~~~~~~~~~
'nexus' CLI entry point registered in pyproject.toml as:

    [project.scripts]
    nexus = "nexus_a2a.cli.main:cli"

Global flags:
  --config PATH     Override nexus.toml location (default: ./nexus.toml)
  --verbose / -v    Enable debug-level logging output
  --format          Output format: table (default) | json
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from nexus_a2a.cli.context import NexusContext, pass_ctx

# ── Logging setup ──────────────────────────────────────────────────────────────


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


# ── Root CLI group ────────────────────────────────────────────────────────────


@click.group()
@click.option(
    "--config",
    default="nexus.toml",
    show_default=True,
    type=click.Path(dir_okay=False),
    help="Path to nexus.toml config file.",
)
@click.option(
    "--verbose", "-v", is_flag=True, default=False, help="Enable verbose/debug output."
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["table", "json"], case_sensitive=False),
    default="table",
    show_default=True,
    help="Output format.",
)
@click.version_option(package_name="nexus-a2a", prog_name="nexus")
@click.pass_context
def cli(ctx: click.Context, config: str, verbose: bool, fmt: str) -> None:
    """nexus — Developer CLI for nexus-a2a agent networks.

    \b
    Quick start:
      nexus ping http://localhost:8001
      nexus inspect http://localhost:8001
      nexus status --network
      nexus trace <task_id>
      nexus replay --failed
    """
    _configure_logging(verbose)
    ctx.obj = NexusContext(
        config_path=Path(config),
        verbose=verbose,
        fmt=fmt,
    )


# ── Import and register sub-commands ──────────────────────────────────────────


def _register_commands() -> None:
    from nexus_a2a.cli.commands.inspect import inspect
    from nexus_a2a.cli.commands.ping import ping
    from nexus_a2a.cli.commands.replay import replay
    from nexus_a2a.cli.commands.run import run
    from nexus_a2a.cli.commands.status import status
    from nexus_a2a.cli.commands.trace import trace

    cli.add_command(ping)
    cli.add_command(inspect)
    cli.add_command(status)
    cli.add_command(trace)
    cli.add_command(replay)
    cli.add_command(run)


_register_commands()

# NexusContext and pass_ctx moved to nexus_a2a.cli.context in v1.6.0 to break
# the main <-> commands import cycle. Re-exported here so existing imports of
# 'from nexus_a2a.cli.main import NexusContext, pass_ctx' keep working.
__all__ = ["cli", "NexusContext", "pass_ctx"]


if __name__ == "__main__":
    cli()
