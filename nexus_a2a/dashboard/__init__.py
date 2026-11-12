"""
nexus_a2a.dashboard — browser dashboard for a running agent (nexus-a2a 2.1).

    from nexus_a2a.dashboard import Dashboard, DashboardConfig

Served by the ops server (AgentServer) under /ui. See API.md in this package
for the HTTP API and nexus.toml's [dashboard] section for settings.
"""

from nexus_a2a.dashboard.app import Dashboard, DashboardConfig
from nexus_a2a.dashboard.events import EventHub
from nexus_a2a.dashboard.logbuffer import RingLogHandler

__all__ = ["Dashboard", "DashboardConfig", "EventHub", "RingLogHandler"]
