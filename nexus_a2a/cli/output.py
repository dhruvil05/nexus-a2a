"""
nexus_a2a.cli.output
~~~~~~~~~~~~~~~~~~~~
Rich terminal rendering helpers: tables, trees, progress bars, status icons.
All display logic lives here — command modules call these helpers and stay lean.
"""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

# Shared console — all output goes through here so --format json can suppress it
console = Console()
err_console = Console(stderr=True, style="bold red")

# ── Icons ──────────────────────────────────────────────────────────────────────
ICON_OK = "[bold green]✓[/bold green]"
ICON_FAIL = "[bold red]✗[/bold red]"
ICON_WARN = "[bold yellow]⚠[/bold yellow]"
ICON_SLOW = "[bold yellow]⚡[/bold yellow]"  # latency warning


def status_icon(ok: bool) -> str:
    return ICON_OK if ok else ICON_FAIL


# ── Generic helpers ────────────────────────────────────────────────────────────


def print_json(data: Any) -> None:
    """Dump data as pretty JSON to stdout."""
    console.print_json(json.dumps(data, default=str))


def print_error(msg: str) -> None:
    err_console.print(f"[bold red]Error:[/bold red] {msg}")


def print_success(msg: str) -> None:
    console.print(f"{ICON_OK} {msg}")


def print_warning(msg: str) -> None:
    console.print(f"{ICON_WARN} {msg}")


# ── Ping output ────────────────────────────────────────────────────────────────


def render_ping(result: dict[str, Any], fmt: str = "table") -> None:
    """Render nexus ping result."""
    if fmt == "json":
        print_json(result)
        return

    icon = status_icon(result.get("healthy", False))
    latency = result.get("latency_ms")
    latency_str = f"{latency:.1f}ms" if latency is not None else "n/a"

    table = Table(
        title=f"Agent Ping — {result.get('url', '')}",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Field", style="dim", width=20)
    table.add_column("Value")

    table.add_row(
        "Status", f"{icon} {'healthy' if result.get('healthy') else 'unhealthy'}"
    )
    table.add_row("Name", result.get("name", "—"))
    table.add_row("Version", result.get("version", "—"))
    table.add_row("Skills", str(result.get("skills_count", 0)))
    table.add_row("Latency", latency_str)
    table.add_row(
        "Health endpoint", status_icon(result.get("health_endpoint_present", False))
    )

    console.print(table)


# ── Inspect output ─────────────────────────────────────────────────────────────


def render_inspect(card: dict[str, Any], fmt: str = "table") -> None:
    """Pretty-print a full AgentCard."""
    if fmt == "json":
        print_json(card)
        return

    panel_title = f"[bold cyan]{card.get('name', 'Unknown Agent')}[/bold cyan]  v{card.get('version', '?')}"
    console.print(
        Panel(card.get("description", ""), title=panel_title, border_style="cyan")
    )

    # Capabilities
    caps = card.get("capabilities", {})
    cap_table = Table(show_header=False, box=None, padding=(0, 2))
    cap_table.add_column("Key", style="dim")
    cap_table.add_column("Value")
    cap_table.add_row("Streaming", status_icon(caps.get("streaming", False)))
    cap_table.add_row(
        "Push notifications", status_icon(caps.get("push_notifications", False))
    )
    cap_table.add_row("Multi-turn", status_icon(caps.get("multi_turn", False)))
    console.print(Panel(cap_table, title="Capabilities", border_style="dim"))

    # Auth
    auth = card.get("authentication", {})
    auth_table = Table(show_header=False, box=None, padding=(0, 2))
    auth_table.add_column("Key", style="dim")
    auth_table.add_column("Value")
    auth_table.add_row(
        "Scheme", auth.get("schemes", ["none"])[0] if auth.get("schemes") else "none"
    )
    auth_table.add_row("Token URL", auth.get("token_url") or "—")
    auth_table.add_row("Header", auth.get("header_name") or "—")
    console.print(Panel(auth_table, title="Authentication", border_style="dim"))

    # Modes
    modes_table = Table(show_header=False, box=None, padding=(0, 2))
    modes_table.add_column("Key", style="dim")
    modes_table.add_column("Value")
    modes_table.add_row(
        "Input modes", ", ".join(card.get("input_modes", ["text/plain"]))
    )
    modes_table.add_row(
        "Output modes", ", ".join(card.get("output_modes", ["text/plain"]))
    )
    console.print(Panel(modes_table, title="Modes", border_style="dim"))

    # Skills
    skills = card.get("skills", [])
    if skills:
        sk_table = Table(title="Skills", header_style="bold magenta")
        sk_table.add_column("ID", style="cyan", no_wrap=True)
        sk_table.add_column("Name")
        sk_table.add_column("Tags")
        sk_table.add_column("Examples")
        for sk in skills:
            tags = ", ".join(sk.get("tags", []))
            examples = str(len(sk.get("examples", []))) + " example(s)"
            sk_table.add_row(sk.get("id", ""), sk.get("name", ""), tags, examples)
        console.print(sk_table)


# ── Status / network table ─────────────────────────────────────────────────────


def render_status(
    agents: list[dict[str, Any]], summary: dict[str, Any], fmt: str = "table"
) -> None:
    """Render nexus status --network table."""
    if fmt == "json":
        print_json({"agents": agents, "summary": summary})
        return

    table = Table(title="Network Status", header_style="bold cyan")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("URL", style="dim")
    table.add_column("Health", justify="center")
    table.add_column("Queue depth", justify="right")
    table.add_column("DLQ pending", justify="right")
    table.add_column("Last seen", style="dim")

    for agent in agents:
        table.add_row(
            agent.get("name", "—"),
            agent.get("url", "—"),
            status_icon(agent.get("healthy", False)),
            str(agent.get("queue_depth", 0)),
            str(agent.get("dlq_pending", 0)),
            agent.get("last_seen", "—"),
        )

    console.print(table)

    # Summary row
    summary_table = Table(show_header=False, box=None, padding=(0, 2))
    summary_table.add_column("Key", style="dim")
    summary_table.add_column("Value", style="bold")
    summary_table.add_row("Total agents", str(summary.get("total", 0)))
    summary_table.add_row("Healthy", f"[green]{summary.get('healthy', 0)}[/green]")
    summary_table.add_row("Unhealthy", f"[red]{summary.get('unhealthy', 0)}[/red]")
    summary_table.add_row("Total DLQ pending", str(summary.get("total_dlq", 0)))
    console.print(Panel(summary_table, title="Summary", border_style="dim"))


# ── Trace tree ────────────────────────────────────────────────────────────────

SLOW_THRESHOLD_MS = 500  # highlight hops slower than this


def _hop_label(hop: dict[str, Any]) -> Text:
    url = hop.get("url", "unknown")
    latency_ms = hop.get("duration_ms")
    status = hop.get("status", "unknown")
    error = hop.get("error")

    text = Text()
    text.append(url, style="cyan")
    text.append("  ")

    if latency_ms is not None:
        style = "yellow" if latency_ms > SLOW_THRESHOLD_MS else "green"
        text.append(f"{latency_ms:.0f}ms", style=style)
        if latency_ms > SLOW_THRESHOLD_MS:
            text.append(" ⚡", style="yellow")
        text.append("  ")

    if status == "completed":
        text.append("✓ completed", style="bold green")
    elif status == "failed":
        text.append("✗ failed", style="bold red")
        if error:
            text.append(f": {error}", style="red")
    else:
        text.append(status, style="dim")

    return text


def _build_tree(trace_id: str, hops: list[dict[str, Any]]) -> Tree:
    root = Tree(f"[bold]trace: {trace_id}[/bold]")
    for hop in hops:
        label = _hop_label(hop)
        node = root.add(label)
        # Nested children if present (multi-level traces)
        for child in hop.get("children", []):
            node.add(_hop_label(child))
    return root


def render_trace(trace: dict[str, Any], fmt: str = "table") -> None:
    """Render nexus trace <task_id> output."""
    if fmt == "json":
        print_json(trace)
        return

    trace_id = trace.get("trace_id", "unknown")
    hops = trace.get("hops", [])

    if not hops:
        console.print(f"[dim]No hops recorded for trace {trace_id}[/dim]")
        return

    tree = _build_tree(trace_id, hops)
    console.print(tree)

    # Highlight slowest hop
    timed = [h for h in hops if h.get("duration_ms") is not None]
    if timed:
        slowest = max(timed, key=lambda h: h["duration_ms"])
        if slowest["duration_ms"] > SLOW_THRESHOLD_MS:
            console.print(
                f"\n[yellow]Slowest hop:[/yellow] {slowest['url']} — {slowest['duration_ms']:.0f}ms"
            )


# ── Replay output ──────────────────────────────────────────────────────────────


def render_replay_preview(entries: list[dict[str, Any]], fmt: str = "table") -> None:
    """Show DLQ entries before replay confirmation."""
    if fmt == "json":
        print_json(entries)
        return

    table = Table(title="DLQ Entries to Replay", header_style="bold yellow")
    table.add_column("Task ID", style="cyan", no_wrap=True)
    table.add_column("Skill", style="dim")
    table.add_column("Failed at", style="dim")
    table.add_column("Attempts", justify="right")
    table.add_column("Error")

    for entry in entries:
        table.add_row(
            entry.get("task_id", "—"),
            entry.get("skill_id") or "—",
            entry.get("failed_at", "—"),
            str(entry.get("attempts", 1)),
            entry.get("error", "—")[:60],
        )

    console.print(table)


def render_replay_result(succeeded: int, failed: int) -> None:
    console.print(
        f"\nReplay complete: "
        f"[bold green]{succeeded} succeeded[/bold green], "
        f"[bold red]{failed} failed[/bold red]"
    )


# ── Progress bar factory ───────────────────────────────────────────────────────


def make_progress(description: str = "Working") -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn(f"[cyan]{description}[/cyan]"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    )
