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

The token from --admin-token / NEXUS_ADMIN_TOKEN is the user's own secret,
while nexus.toml may come from a repository someone else wrote. So that token
is only sent to a URL from nexus.toml when the URL is https or loopback; see
check_token_destination().
"""

from __future__ import annotations

import ipaddress
import os
from typing import Any
from urllib.parse import urlsplit

from nexus_a2a.security.redaction import redact_url

ADMIN_HEADER = "X-Admin-Token"


class UnsafeTokenDestinationError(Exception):
    """The admin token would be sent to an unconfirmed, insecure URL."""


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


def _is_loopback_url(url: str) -> bool:
    """True for URLs (http or https) whose host is this machine."""
    try:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
    except ValueError:
        return False
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def check_token_destination(
    explicit_url: str | None,
    explicit_token: str | None,
    cfg: dict[str, Any],
) -> None:
    """
    Refuse to send the user's admin token to a non-loopback URL taken from
    nexus.toml. https does not help: a hostile nexus.toml in a cloned repo can
    simply name an https host it controls.

    Takes the same arguments as resolve_ops_url() / resolve_admin_token().
    Allowed when any of these hold:
      - there is no URL or no token,
      - the URL came from --agent or NEXUS_OPS_URL (the user chose it),
      - the token came from nexus.toml itself (the file already holds it,
        so sending it to the file's URL reveals nothing new),
      - the URL is on localhost / a loopback address.

    Raises:
        UnsafeTokenDestinationError: with a message telling the user how to
                                     confirm the URL.
    """
    url = resolve_ops_url(explicit_url, cfg)
    token = resolve_admin_token(explicit_token, cfg)
    if not url or not token:
        return
    if explicit_url or os.environ.get("NEXUS_OPS_URL"):
        return
    if not (explicit_token or os.environ.get("NEXUS_ADMIN_TOKEN")):
        return  # token from nexus.toml, same source as the URL
    if _is_loopback_url(url):
        return
    raise UnsafeTokenDestinationError(
        f"Not sending the admin token to {redact_url(url)}: that URL comes from "
        "nexus.toml ([ops].url / [agent].url), not from you, and is not localhost. "
        f"If you trust it, pass it explicitly: --agent {redact_url(url)} "
        "(or set NEXUS_OPS_URL)."
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
    "UnsafeTokenDestinationError",
    "admin_headers",
    "check_token_destination",
    "forbidden_hint",
    "resolve_admin_token",
    "resolve_ops_url",
]
