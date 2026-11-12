"""
nexus dashboard
~~~~~~~~~~~~~~~
Open a running agent's dashboard in the browser.

The dashboard is served by the ops server under /ui when [dashboard] enabled
= true (or `nexus dev --dashboard`). Log in with the ops admin token — it is
never put in the URL.

    nexus dashboard                      # [ops].url / NEXUS_OPS_URL
    nexus dashboard --agent http://127.0.0.1:9100
    nexus dashboard --no-browser         # just print the address
"""

from __future__ import annotations

import webbrowser

import click

from nexus_a2a.cli.admin import resolve_ops_url
from nexus_a2a.cli.context import NexusContext, pass_ctx
from nexus_a2a.cli.output import console, print_error


def dashboard_url(ops_url: str) -> str:
    """The dashboard address on an ops server."""
    return ops_url.rstrip("/") + "/ui/"


@click.command("dashboard")
@click.option(
    "--agent",
    "agent_url",
    default=None,
    metavar="URL",
    help="Ops server URL. Default: [ops].url, then NEXUS_OPS_URL.",
)
@click.option(
    "--no-browser",
    is_flag=True,
    default=False,
    help="Print the address instead of opening a browser.",
)
@pass_ctx
def dashboard(ctx: NexusContext, agent_url: str | None, no_browser: bool) -> None:
    """Open the dashboard of a running agent.

    \b
    Examples:
      nexus dashboard
      nexus dashboard --agent http://127.0.0.1:9100
    """
    ops_url = resolve_ops_url(agent_url, ctx.load_config())
    if not ops_url:
        print_error(
            "No ops server address. Pass --agent URL, set NEXUS_OPS_URL, or set "
            "[ops] url in nexus.toml."
        )
        raise SystemExit(1)

    url = dashboard_url(ops_url)
    console.print(f"Dashboard: [cyan]{url}[/cyan]")
    console.print("[dim]Log in with the ops admin token.[/dim]")
    if not no_browser:
        webbrowser.open(url)
