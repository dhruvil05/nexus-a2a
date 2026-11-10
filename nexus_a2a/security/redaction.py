"""
nexus_a2a/security/redaction.py

Helpers for keeping secrets out of logs and diagnostic output.

Connection URLs routinely carry credentials (redis://:password@host,
postgresql://user:password@host/db) and webhook URLs often carry tokens in
their query string. Anything that logs or prints a URL should pass it through
redact_url() first.
"""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

_MASK = "***"


def redact_url(
    url: str | None, keep_query: bool = False, keep_path: bool = True
) -> str:
    """
    Return url with any userinfo masked and, by default, the query and
    fragment dropped.

        redact_url("redis://:pw@cache:6379/0")      -> "redis://***@cache:6379/0"
        redact_url("https://h/hook?token=abc")       -> "https://h/hook"
        redact_url("https://h/T0/B1/XyZ", keep_path=False) -> "https://h/***"

    Pass keep_path=False for webhook-style URLs, which often carry their
    secret in the path (Slack, Discord, many SaaS hooks).

    Never raises: an unparseable value is replaced by a placeholder rather
    than echoed back, since it may itself be a secret.
    """
    if not url:
        return ""
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        if ":" in host:  # IPv6 literal
            host = f"[{host}]"
        port = f":{parts.port}" if parts.port else ""
    except ValueError:
        return "<unparseable url>"

    netloc = f"{host}{port}"
    if parts.username or parts.password:
        netloc = f"{_MASK}@{netloc}"

    query = parts.query if keep_query else ""
    path = parts.path if keep_path or parts.path in ("", "/") else "/" + _MASK
    return urlunsplit((parts.scheme, netloc, path, query, ""))


__all__ = ["redact_url"]
