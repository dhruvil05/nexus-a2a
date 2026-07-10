"""
nexus replay --failed [--skill web_search] [--last 1h] [--dry-run]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Query a running agent's Dead Letter Queue over HTTP (GET /dlq), apply
filters, show matching entries in a table, confirm, then replay each
one (POST /dlq/replay). Summary: N succeeded, M failed.

DeadLetterQueue has no global/in-process singleton — it lives on the
running AgentServer's AgentNetwork. This command always talks to a
running agent over HTTP; there is no local/offline mode.
"""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime, timedelta
from typing import Any

import click
import httpx

from nexus_a2a.cli.main import NexusContext, pass_ctx
from nexus_a2a.cli.output import (
    console,
    make_progress,
    print_error,
    print_warning,
    render_replay_preview,
    render_replay_result,
)


def _parse_duration(value: str) -> timedelta:
    """Parse '1h', '30m', '2h30m', '7d' into a timedelta."""
    pattern = re.compile(r"(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?")
    m = pattern.fullmatch(value.strip())
    if not m or not any(m.groups()):
        raise click.BadParameter(
            f"Cannot parse duration '{value}'. Use e.g. 1h, 30m, 2h30m, 7d."
        )
    days = int(m.group(1) or 0)
    hours = int(m.group(2) or 0)
    minutes = int(m.group(3) or 0)
    seconds = int(m.group(4) or 0)
    return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)


async def _fetch_dlq_entries(agent_url: str, skill: str | None) -> list[dict[str, Any]]:
    """GET /dlq from a running agent. Raises on network/HTTP errors."""
    agent_url = agent_url.rstrip("/")
    params = {"skill": skill} if skill else {}
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{agent_url}/dlq", params=params)
        resp.raise_for_status()
        data = resp.json()
        result: list[dict[str, Any]] = data.get("entries", [])
        return result


async def _replay_remote(
    agent_url: str, task_ids: list[str], verbose: bool
) -> tuple[int, int]:
    """POST /dlq/replay once per task_id, with a progress bar. Returns (succeeded, failed)."""
    agent_url = agent_url.rstrip("/")
    succeeded = 0
    failed = 0

    async with httpx.AsyncClient(timeout=30.0) as client:
        with make_progress("Replaying tasks") as progress:
            task = progress.add_task("replay", total=len(task_ids))
            for task_id in task_ids:
                try:
                    resp = await client.post(
                        f"{agent_url}/dlq/replay", json={"task_id": task_id}
                    )
                    resp.raise_for_status()
                    body = resp.json()
                    if body.get("succeeded", 0) >= 1:
                        succeeded += 1
                    else:
                        failed += 1
                        if verbose:
                            err = body.get("results", [{}])[0].get("error")
                            console.print(f"[red]  ✗ {task_id}: {err}[/red]")
                except Exception as exc:
                    failed += 1
                    if verbose:
                        console.print(f"[red]  ✗ {task_id}: {exc}[/red]")
                finally:
                    progress.advance(task)

    return succeeded, failed


def _entry_failed_after(entry: dict[str, Any], since: datetime) -> bool:
    """Return True if entry's failed_at (unix timestamp) is after `since`."""
    failed_at = entry.get("failed_at")
    if failed_at is None:
        return True  # no timestamp → include it
    return datetime.fromtimestamp(float(failed_at), tz=UTC) >= since


def _for_display(entry: dict[str, Any]) -> dict[str, Any]:
    """Map the server's DLQEntry.to_dict() shape onto what render_replay_preview expects."""
    display = dict(entry)
    display["attempts"] = entry.get("retry_count", 0)
    failed_at = entry.get("failed_at")
    if isinstance(failed_at, int | float):
        display["failed_at"] = datetime.fromtimestamp(failed_at, tz=UTC).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
    return display


@click.command("replay")
@click.option(
    "--failed", is_flag=True, default=False, help="Replay all failed (DLQ) tasks."
)
@click.option("--skill", default=None, metavar="SKILL_ID", help="Filter by skill ID.")
@click.option(
    "--last",
    default=None,
    metavar="DURATION",
    help="Only tasks failed within duration (e.g. 1h, 30m, 7d).",
)
@click.option(
    "--agent",
    "agent_url",
    default=None,
    metavar="URL",
    help="Agent URL to replay against (default: agent.url from nexus.toml).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Preview matching tasks without replaying.",
)
@click.option(
    "--yes", "-y", is_flag=True, default=False, help="Skip confirmation prompt."
)
@pass_ctx
def replay(
    ctx: NexusContext,
    failed: bool,
    skill: str | None,
    last: str | None,
    agent_url: str | None,
    dry_run: bool,
    yes: bool,
) -> None:
    """Replay failed tasks from a running agent's Dead Letter Queue.

    \b
    Examples:
      nexus replay --failed
      nexus replay --failed --agent http://localhost:8001
      nexus replay --failed --skill web_search
      nexus replay --failed --last 1h
      nexus replay --failed --dry-run
      nexus replay --failed --yes
    """
    if not failed:
        print_warning("Specify --failed to select tasks for replay.")
        raise SystemExit(0)

    # ── Resolve target agent ──────────────────────────────────────────────────
    if agent_url is None:
        cfg = ctx.load_config()
        agent_url = cfg.get("agent", {}).get("url")
    if not agent_url:
        print_error(
            "No agent URL. Pass --agent http://host:port or set agent.url in "
            "nexus.toml."
        )
        raise SystemExit(1)

    # ── Build filter ──────────────────────────────────────────────────────────
    since: datetime | None = None
    if last:
        try:
            delta = _parse_duration(last)
        except click.BadParameter as e:
            print_error(str(e))
            raise SystemExit(1) from e
        since = datetime.now(UTC) - delta

    # ── Fetch matching entries from the running agent ────────────────────────
    try:
        all_entries = asyncio.run(_fetch_dlq_entries(agent_url, skill))
    except Exception as e:
        print_error(f"Failed to read DLQ from {agent_url}: {e}")
        raise SystemExit(1) from e

    entries = all_entries
    if since:
        entries = [e for e in entries if _entry_failed_after(e, since)]

    if not entries:
        print_warning("No matching DLQ entries found.")
        raise SystemExit(0)

    render_replay_preview([_for_display(e) for e in entries], fmt=ctx.fmt)

    if dry_run:
        console.print(
            f"\n[dim]Dry run — {len(entries)} task(s) would be replayed.[/dim]"
        )
        raise SystemExit(0)

    # ── Confirmation ──────────────────────────────────────────────────────────
    if not yes:
        click.confirm(f"\nReplay {len(entries)} task(s)?", abort=True)

    # ── Execute replay ────────────────────────────────────────────────────────
    task_ids = [e["task_id"] for e in entries]
    try:
        succeeded, failed_count = asyncio.run(
            _replay_remote(agent_url, task_ids, ctx.verbose)
        )
    except Exception as e:
        print_error(f"Replay error: {e}")
        raise SystemExit(1) from e

    render_replay_result(succeeded, failed_count)

    if failed_count > 0:
        raise SystemExit(1)
