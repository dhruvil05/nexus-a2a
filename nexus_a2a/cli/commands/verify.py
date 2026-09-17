"""
nexus verify <url>
~~~~~~~~~~~~~~~~~~
Check any A2A agent against the protocol and against its own agent card.

Exit status is 0 when nothing FAILed (1 otherwise); --strict also fails on
WARN, which suits CI.

The task checks send one probe message (two if the agent streams), so the
agent does real work. Use --read-only to check only discovery, error handling
and auth.
"""

from __future__ import annotations

import asyncio

import click

from nexus_a2a.cli.context import NexusContext, pass_ctx
from nexus_a2a.cli.output import print_error, render_verify


@click.command("verify")
@click.argument("url")
@click.option("--api-key", default=None, metavar="KEY", help="API key to send.")
@click.option(
    "--api-key-header",
    default="X-API-Key",
    show_default=True,
    metavar="NAME",
    help="Header the API key goes in.",
)
@click.option("--bearer", default=None, metavar="TOKEN", help="Bearer token to send.")
@click.option(
    "--caller-url",
    default=None,
    metavar="URL",
    help="Identity to announce (X-Nexus-Caller), for agents enforcing trust.",
)
@click.option("--skill", default=None, metavar="SKILL_ID", help="Skill to probe.")
@click.option(
    "--message",
    default="nexus verify probe",
    show_default=True,
    help="Text of the probe message.",
)
@click.option(
    "--read-only",
    is_flag=True,
    default=False,
    help="Skip checks that make the agent do work.",
)
@click.option(
    "--strict", is_flag=True, default=False, help="Fail on warnings as well."
)
@click.option(
    "--timeout",
    default=15.0,
    show_default=True,
    type=float,
    help="Per-request timeout in seconds.",
)
@pass_ctx
def verify(
    ctx: NexusContext,
    url: str,
    api_key: str | None,
    api_key_header: str,
    bearer: str | None,
    caller_url: str | None,
    skill: str | None,
    message: str,
    read_only: bool,
    strict: bool,
    timeout: float,
) -> None:
    """Check an A2A agent against the protocol and its own card.

    \b
    Examples:
      nexus verify http://localhost:8001
      nexus verify http://localhost:8001 --api-key $KEY --strict
      nexus verify https://agent.example.com --read-only --format json
    """
    from nexus_a2a.verify import verify_agent

    try:
        report = asyncio.run(
            verify_agent(
                url,
                api_key=api_key,
                api_key_header=api_key_header,
                bearer=bearer,
                caller_url=caller_url,
                message=message,
                skill_id=skill,
                read_only=read_only,
                timeout=timeout,
            )
        )
    except Exception as e:
        print_error(f"Could not verify {url}: {e}")
        raise SystemExit(1) from e

    render_verify(report.to_dict(), fmt=ctx.fmt)

    ok = report.passed_strict() if strict else report.passed
    if not ok:
        raise SystemExit(1)
