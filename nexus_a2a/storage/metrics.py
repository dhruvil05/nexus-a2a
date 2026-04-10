"""
nexus_a2a/storage/metrics.py

MetricsCollector — records operational metrics for the agent network.

Two modes:
  1. Standalone (default): pure Python in-memory counters.
     Zero dependencies. Good for basic monitoring and tests.

  2. OpenTelemetry (optional): exports metrics to any OTEL-compatible
     backend (Datadog, Prometheus, Grafana, etc.).
     Activated by calling MetricsCollector.with_otel(meter).

Metrics tracked:
  tasks_created       Counter  — total tasks created
  tasks_completed     Counter  — total tasks successfully completed
  tasks_failed        Counter  — total tasks that failed
  agent_call_duration Histogram — latency per agent call in seconds
  agent_errors        Counter  — total agent call errors (by agent_url)
  rate_limit_hits     Counter  — total rate limit rejections
  auth_failures       Counter  — total authentication failures
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── In-memory snapshot ────────────────────────────────────────────────────────

@dataclass
class MetricsSnapshot:
    """
    A point-in-time snapshot of all collected metrics.
    Returned by MetricsCollector.snapshot().
    """
    tasks_created:    int                  = 0
    tasks_completed:  int                  = 0
    tasks_failed:     int                  = 0
    tasks_cancelled:  int                  = 0
    rate_limit_hits:  int                  = 0
    auth_failures:    int                  = 0
    agent_errors:     dict[str, int]       = field(default_factory=dict)
    # agent_url → list of latency floats (seconds)
    call_durations:   dict[str, list[float]] = field(default_factory=dict)

    def avg_latency(self, agent_url: str) -> float | None:
        """Return the mean call latency for an agent, or None if no data."""
        durations = self.call_durations.get(agent_url, [])
        return sum(durations) / len(durations) if durations else None

    def p99_latency(self, agent_url: str) -> float | None:
        """Return the p99 call latency for an agent, or None if no data."""
        durations = sorted(self.call_durations.get(agent_url, []))
        if not durations:
            return None
        idx = max(0, int(len(durations) * 0.99) - 1)
        return durations[idx]

    def total_calls(self) -> int:
        return sum(len(v) for v in self.call_durations.values())


# ── MetricsCollector ──────────────────────────────────────────────────────────

class MetricsCollector:
    """
    Records operational metrics for the nexus-a2a network.

    Usage (standalone — no extra deps):
        metrics = MetricsCollector()

        metrics.record_task_created()
        metrics.record_task_completed()

        with metrics.record_agent_call("http://agent:8001"):
            result = await client.send_message(...)

        snap = metrics.snapshot()
        print(snap.tasks_completed)
        print(snap.avg_latency("http://agent:8001"))

    Usage (OpenTelemetry):
        from opentelemetry import metrics as otel_metrics
        meter   = otel_metrics.get_meter("nexus-a2a")
        metrics = MetricsCollector.with_otel(meter)

    Args:
        max_durations: Max latency samples stored per agent.
                       Oldest samples are dropped when full. Default: 1000.
    """

    def __init__(self, max_durations: int = 1000) -> None:
        self._max_dur  = max_durations
        # Counters
        self._tasks_created:   int = 0
        self._tasks_completed: int = 0
        self._tasks_failed:    int = 0
        self._tasks_cancelled: int = 0
        self._rate_limit_hits: int = 0
        self._auth_failures:   int = 0
        # Per-agent
        self._agent_errors:    dict[str, int]         = defaultdict(int)
        self._call_durations:  dict[str, list[float]] = defaultdict(list)
        # Optional OTEL instruments (set by with_otel())
        self._otel: dict[str, Any] | None = None

    # ── Factory: OpenTelemetry ────────────────────────────────────────────────

    @classmethod
    def with_otel(cls, meter: Any) -> MetricsCollector:
        """
        Create a MetricsCollector that also exports to an OpenTelemetry Meter.

        Args:
            meter: An opentelemetry.metrics.Meter instance.

        Returns:
            MetricsCollector with OTEL export enabled.
        """
        instance = cls()
        try:
            instance._otel = {
                "tasks_created":   meter.create_counter(
                    "nexus_a2a.tasks_created",
                    description="Total A2A tasks created",
                ),
                "tasks_completed": meter.create_counter(
                    "nexus_a2a.tasks_completed",
                    description="Total A2A tasks completed successfully",
                ),
                "tasks_failed":    meter.create_counter(
                    "nexus_a2a.tasks_failed",
                    description="Total A2A tasks that failed",
                ),
                "agent_latency":   meter.create_histogram(
                    "nexus_a2a.agent_call_duration",
                    unit="s",
                    description="A2A agent call latency in seconds",
                ),
                "agent_errors":    meter.create_counter(
                    "nexus_a2a.agent_errors",
                    description="Total agent call errors",
                ),
                "rate_limit_hits": meter.create_counter(
                    "nexus_a2a.rate_limit_hits",
                    description="Total rate limit rejections",
                ),
                "auth_failures":   meter.create_counter(
                    "nexus_a2a.auth_failures",
                    description="Total authentication failures",
                ),
            }
            logger.info("MetricsCollector: OpenTelemetry export enabled")
        except Exception as exc:
            logger.warning("MetricsCollector: OTEL setup failed: %s", exc)
            instance._otel = None

        return instance

    # ── Recording methods ─────────────────────────────────────────────────────

    def record_task_created(self) -> None:
        """Increment the tasks_created counter."""
        self._tasks_created += 1
        self._otel_add("tasks_created", 1)

    def record_task_completed(self) -> None:
        """Increment the tasks_completed counter."""
        self._tasks_completed += 1
        self._otel_add("tasks_completed", 1)

    def record_task_failed(self) -> None:
        """Increment the tasks_failed counter."""
        self._tasks_failed += 1
        self._otel_add("tasks_failed", 1)

    def record_task_cancelled(self) -> None:
        """Increment the tasks_cancelled counter."""
        self._tasks_cancelled += 1

    def record_agent_error(self, agent_url: str) -> None:
        """Increment the error counter for a specific agent."""
        self._agent_errors[agent_url] += 1
        self._otel_add("agent_errors", 1, {"agent_url": agent_url})

    def record_rate_limit_hit(self) -> None:
        """Increment the rate_limit_hits counter."""
        self._rate_limit_hits += 1
        self._otel_add("rate_limit_hits", 1)

    def record_auth_failure(self) -> None:
        """Increment the auth_failures counter."""
        self._auth_failures += 1
        self._otel_add("auth_failures", 1)

    def record_call_duration(self, agent_url: str, duration_sec: float) -> None:
        """
        Record a single agent call latency sample.

        Args:
            agent_url:    The agent that was called.
            duration_sec: How long the call took in seconds.
        """
        durations = self._call_durations[agent_url]
        if len(durations) >= self._max_dur:
            durations.pop(0)
        durations.append(duration_sec)

        if self._otel and "agent_latency" in self._otel:
            try:
                self._otel["agent_latency"].record(
                    duration_sec,
                    {"agent_url": agent_url},
                )
            except Exception:
                pass

    @contextmanager
    def record_agent_call(
        self, agent_url: str
    ) -> Generator[None, None, None]:
        """
        Context manager that automatically records call duration
        and errors for an agent call.

        Usage:
            with metrics.record_agent_call("http://agent:8001"):
                result = await client.send_message(...)
        """
        start = time.monotonic()
        try:
            yield
        except Exception:
            self.record_agent_error(agent_url)
            raise
        finally:
            self.record_call_duration(
                agent_url,
                time.monotonic() - start,
            )

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def snapshot(self) -> MetricsSnapshot:
        """Return a point-in-time snapshot of all collected metrics."""
        return MetricsSnapshot(
            tasks_created=self._tasks_created,
            tasks_completed=self._tasks_completed,
            tasks_failed=self._tasks_failed,
            tasks_cancelled=self._tasks_cancelled,
            rate_limit_hits=self._rate_limit_hits,
            auth_failures=self._auth_failures,
            agent_errors=dict(self._agent_errors),
            call_durations={
                url: list(durs)
                for url, durs in self._call_durations.items()
            },
        )

    def reset(self) -> None:
        """Reset all counters and samples. Mainly useful in tests."""
        self._tasks_created   = 0
        self._tasks_completed = 0
        self._tasks_failed    = 0
        self._tasks_cancelled = 0
        self._rate_limit_hits = 0
        self._auth_failures   = 0
        self._agent_errors.clear()
        self._call_durations.clear()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _otel_add(
        self,
        instrument: str,
        amount: int,
        attrs: dict[str, str] | None = None,
    ) -> None:
        """Safely increment an OTEL counter instrument."""
        if not self._otel or instrument not in self._otel:
            return
        try:
            self._otel[instrument].add(amount, attrs or {})
        except Exception:
            pass
