"""
nexus_a2a.cli.admin
~~~~~~~~~~~~~~~~~~~
Shared plumbing for CLI commands that talk to an AgentServer's admin
endpoints (/traces, /dlq, /dlq/replay).

Those endpoints have required a token since 1.5.0, but `nexus trace --agent`
and `nexus replay` never sent one, so both returned 403 against every real
server. This module resolves the token and the ops-server URL the same way for
every command.

Resolution order (first match wins):
    token:  --admin-token, NEXUS_ADMIN_TOKEN, [ops].admin_token
    url:    --agent,       NEXUS_OPS_URL,     [ops].url, [agent].url
"""

from __future__ import annotations

import os
from typing import Any

ADMIN_HEADER = "X-Admin-Token"


def resolve_admin_token(explicit: str | None, cfg: dict[str, Any]) -> str | None:
    """Return the admin token to send, or None if none is configured."""
    return (
        explicit
        or os.environ.get("NEXUS_ADMIN_TOKEN")
        or cfg.get("ops", {}).get("admin_token")
        or None
    )


def resolve_ops_url(explicit: str | None, cfg: dict[str, Any]) -> str | None:
    """
    Return the URL of the server that hosts the admin endpoints.

    Falls back to [agent].url last: an A2AServer does not serve /traces or
    /dlq itself, so pointing at it only works when the ops server shares its
    address.
    """
    return (
        explicit
        or os.environ.get("NEXUS_OPS_URL")
        or cfg.get("ops", {}).get("url")
        or cfg.get("agent", {}).get("url")
        or None
    )


def admin_headers(token: str | None) -> dict[str, str]:
    """Headers that authenticate an admin request; empty if no token."""
    return {ADMIN_HEADER: token} if token else {}


def forbidden_hint(url: str, token: str | None) -> str:
    """Explain a 403 from an admin endpoint in terms the user can act on."""
    if token is None:
        return (
            f"{url} refused the request (HTTP 403). Admin endpoints need a token: "
            "pass --admin-token, set NEXUS_ADMIN_TOKEN, or add [ops] admin_token "
            "to nexus.toml."
        )
    return (
        f"{url} rejected the admin token (HTTP 403). Check it matches the "
        "server's admin_token / NEXUS_ADMIN_TOKEN."
    )


__all__ = [
    "ADMIN_HEADER",
    "admin_headers",
    "forbidden_hint",
    "resolve_admin_token",
    "resolve_ops_url",
]
