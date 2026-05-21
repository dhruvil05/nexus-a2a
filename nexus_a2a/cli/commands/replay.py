"""
nexus replay --failed [--skill web_search] [--last 1h] [--dry-run]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Query DeadLetterQueue with filters. Show matching entries in a table.
Confirm before replay (unless --yes). Progress bar during replay.
Summary: N succeeded, M failed.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

import click

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
        raise click.BadParameter(f"Cannot parse duration '{value}'. Use e.g. 1h, 30m, 2h30m, 7d.")
    days = int(m.group(1) or 0)
    hours = int(m.group(2) or 0)
    minutes = int(m.group(3) or 0)
    seconds = int(m.group(4) or 0)
    return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)


def _load_dlq() -> object:
    """Load DeadLetterQueue singleton from the package."""
    try:
        from nexus_a2a.core.dead_letter import DeadLetterQueue  # type: ignore
        return DeadLetterQueue.instance()
    except Exception as e:
        raise RuntimeError(f"Cannot access DeadLetterQueue: {e}") from e


def _dlq_entries_to_dicts(entries: list) -> list[dict]:
    """Convert DLQ entry objects to plain dicts for rendering."""
    result = []
    for entry in entries:
        result.append({
            "task_id": getattr(entry, "task_id", str(entry)),
            "skill_id": getattr(entry, "skill_id", None),
            "failed_at": str(getattr(entry, "failed_at", "—")),
            "attempts": getattr(entry, "attempts", 1),
            "error": str(getattr(entry, "error", "—")),
        })
    return result


async def _replay_entries(dlq: object, entries: list, verbose: bool) -> tuple[int, int]:
    """Replay each entry with a progress bar. Returns (succeeded, failed)."""
    succeeded = 0
    failed = 0

    with make_progress("Replaying tasks") as progress:
        task = progress.add_task("replay", total=len(entries))
        for entry in entries:
            try:
                await dlq.replay(getattr(entry, "task_id", None))
                succeeded += 1
            except Exception as exc:
                failed += 1
                if verbose:
                    console.print(f"[red]  ✗ {getattr(entry, 'task_id', entry)}: {exc}[/red]")
            finally:
                progress.advance(task)

    return succeeded, failed


@click.command("replay")
@click.option("--failed", is_flag=True, default=False, help="Replay all failed (DLQ) tasks.")
@click.option("--skill", default=None, metavar="SKILL_ID", help="Filter by skill ID.")
@click.option("--last", default=None, metavar="DURATION", help="Only tasks failed within duration (e.g. 1h, 30m, 7d).")
@click.option("--dry-run", is_flag=True, default=False, help="Preview matching tasks without replaying.")
@click.option("--yes", "-y", is_flag=True, default=False, help="Skip confirmation prompt.")
@pass_ctx
def replay(
    ctx: NexusContext,
    failed: bool,
    skill: Optional[str],
    last: Optional[str],
    dry_run: bool,
    yes: bool,
) -> None:
    """Replay failed tasks from the Dead Letter Queue.

    \b
    Examples:
      nexus replay --failed
      nexus replay --failed --skill web_search
      nexus replay --failed --last 1h
      nexus replay --failed --dry-run
      nexus replay --failed --yes
    """
    if not failed:
        print_warning("Specify --failed to select tasks for replay.")
        raise SystemExit(0)

    # ── Build filter ──────────────────────────────────────────────────────────
    since: Optional[datetime] = None
    if last:
        try:
            delta = _parse_duration(last)
        except click.BadParameter as e:
            print_error(str(e))
            raise SystemExit(1)
        since = datetime.now(timezone.utc) - delta

    # ── Load DLQ ──────────────────────────────────────────────────────────────
    try:
        dlq = _load_dlq()
    except RuntimeError as e:
        print_error(str(e))
        raise SystemExit(1)

    # ── Fetch matching entries ────────────────────────────────────────────────
    try:
        all_entries = asyncio.run(_get_entries(dlq))
    except Exception as e:
        print_error(f"Failed to read DLQ: {e}")
        raise SystemExit(1)

    # Apply filters
    entries = all_entries
    if skill:
        entries = [e for e in entries if getattr(e, "skill_id", None) == skill]
    if since:
        entries = [
            e for e in entries
            if _entry_failed_after(e, since)
        ]

    if not entries:
        print_warning("No matching DLQ entries found.")
        raise SystemExit(0)

    entry_dicts = _dlq_entries_to_dicts(entries)
    render_replay_preview(entry_dicts, fmt=ctx.fmt)

    if dry_run:
        console.print(f"\n[dim]Dry run — {len(entries)} task(s) would be replayed.[/dim]")
        raise SystemExit(0)

    # ── Confirmation ──────────────────────────────────────────────────────────
    if not yes:
        click.confirm(f"\nReplay {len(entries)} task(s)?", abort=True)

    # ── Execute replay ────────────────────────────────────────────────────────
    try:
        succeeded, failed_count = asyncio.run(_replay_entries(dlq, entries, ctx.verbose))
    except Exception as e:
        print_error(f"Replay error: {e}")
        raise SystemExit(1)

    render_replay_result(succeeded, failed_count)

    if failed_count > 0:
        raise SystemExit(1)


async def _get_entries(dlq: object) -> list:
    """Fetch all DLQ entries (handles both sync and async list() methods)."""
    list_fn = getattr(dlq, "list_all", None) or getattr(dlq, "entries", None)
    if list_fn is None:
        return []
    result = list_fn()
    if asyncio.iscoroutine(result):
        result = await result
    return list(result)


def _entry_failed_after(entry: object, since: datetime) -> bool:
    """Return True if entry's failed_at is after `since`."""
    failed_at = getattr(entry, "failed_at", None)
    if failed_at is None:
        return True  # no timestamp → include it
    if isinstance(failed_at, str):
        try:
            failed_at = datetime.fromisoformat(failed_at)
        except ValueError:
            return True
    if failed_at.tzinfo is None:
        failed_at = failed_at.replace(tzinfo=timezone.utc)
    return failed_at >= since