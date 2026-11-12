"""
nexus_a2a/dashboard/logbuffer.py

RingLogHandler — keeps the most recent log records in memory for the
dashboard's Logs page and streams new ones to it live.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import TYPE_CHECKING, Any

from nexus_a2a.dashboard.events import iso

if TYPE_CHECKING:
    from nexus_a2a.dashboard.events import EventHub

_MAX_MESSAGE = 4000


class RingLogHandler(logging.Handler):
    """
    A logging.Handler that keeps the last `capacity` records.

    Attach it to the "nexus_a2a" logger (Dashboard does this). Thread-safe:
    records may come from any thread.
    """

    def __init__(
        self,
        capacity: int = 5000,
        hub: EventHub | None = None,
        hide_content: bool = True,
    ) -> None:
        super().__init__(level=logging.DEBUG)
        self._lines: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._lines_lock = threading.Lock()
        self._hub = hub
        # With content hidden, exception messages (agents often quote their
        # input in them) are reduced to their type, and records that carry a
        # `nexus_public_message` (set by library calls that log task errors)
        # are shown in that redacted form.
        self.hide_content = hide_content

    def emit(self, record: logging.LogRecord) -> None:
        public = getattr(record, "nexus_public_message", None)
        if self.hide_content and isinstance(public, str):
            message = public
        else:
            try:
                message = record.getMessage()
            except Exception:  # noqa: BLE001 - a broken format string must not raise
                message = str(record.msg)
        if record.exc_info and record.exc_info[1] is not None:
            exc = record.exc_info[1]
            if self.hide_content:
                message = f"{message} [{type(exc).__name__}]"
            else:
                message = f"{message} [{type(exc).__name__}: {exc}]"
        line = {
            "at": iso(record.created),
            "level": record.levelname,
            "logger": record.name,
            "message": message[:_MAX_MESSAGE],
        }
        with self._lines_lock:
            self._lines.append(line)
        if self._hub is not None:
            self._hub.publish("log", line)

    def lines(
        self, min_level: int = logging.DEBUG, query: str = "", limit: int = 200
    ) -> list[dict[str, Any]]:
        """Matching lines, oldest first, at most `limit` (the newest ones)."""
        needle = query.lower().strip()
        with self._lines_lock:
            snapshot = list(self._lines)
        matched = [
            line
            for line in snapshot
            if logging.getLevelName(line["level"]) >= min_level
            and (
                not needle
                or needle in line["message"].lower()
                or needle in line["logger"].lower()
            )
        ]
        return matched[-limit:] if limit > 0 else []


__all__ = ["RingLogHandler"]
