"""
nexus_a2a/verify.py

Conformance checks for any A2A agent — nexus-a2a or not.

    report = await verify_agent("http://localhost:8001")
    for check in report.checks:
        print(check.status, check.name, check.detail)

Checks are grouped:

    card      discovery document is reachable, valid, and self-consistent
    protocol  JSON-RPC error handling behaves as the spec requires
    task      message/send, tasks/get and tasks/cancel round-trip
    stream    message/stream works, if the card claims streaming
    push      push-config methods answer, if the card claims push
    auth      the auth the card advertises is the auth actually enforced

Each check is PASS, WARN, FAIL or SKIP. A FAIL means the agent breaks the
protocol or contradicts its own card; a WARN is something a well-behaved agent
should do but the spec does not require.

The auth group matters most. An agent whose card says "api_key" but which
answers unauthenticated requests is open to anyone; one whose card says "none"
but refuses them breaks every client that trusted the card.

Side effects: the task group sends ONE probe message (two with streaming), so
the agent does real work. Pass read_only=True to skip anything that invokes it.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

CARD_PATH = "/.well-known/agent-card.json"
_TERMINAL_OR_PAUSED = {"completed", "failed", "cancelled", "input_required"}
_KNOWN_STATES = _TERMINAL_OR_PAUSED | {"submitted", "working"}


class Status(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class CheckResult:
    """The outcome of one conformance check."""

    group: str
    name: str
    status: Status
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass
class VerifyReport:
    """Every check run against one agent."""

    url: str
    checks: list[CheckResult] = field(default_factory=list)
    card: dict[str, Any] | None = None

    def count(self, status: Status) -> int:
        return sum(1 for c in self.checks if c.status == status)

    @property
    def passed(self) -> bool:
        """True when nothing failed."""
        return self.count(Status.FAIL) == 0

    def passed_strict(self) -> bool:
        """True when nothing failed or warned."""
        return self.passed and self.count(Status.WARN) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "passed": self.passed,
            "summary": {s.value: self.count(s) for s in Status},
            "checks": [c.to_dict() for c in self.checks],
        }


class _Checker:
    """Runs the checks and records results. One instance per verification."""

    def __init__(
        self,
        url: str,
        client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        message: str,
        skill_id: str | None,
        read_only: bool,
    ) -> None:
        self.url = url.rstrip("/")
        self.client = client
        self.auth_headers = auth_headers
        self.message = message
        self.skill_id = skill_id
        self.read_only = read_only
        self.report = VerifyReport(url=self.url)
        # Set by check_auth when the agent refuses us: every later check would
        # only be measuring the 401, not the agent.
        self.locked_out: str | None = None

    # ── recording ─────────────────────────────────────────────────────────────

    def record(self, group: str, name: str, status: Status, detail: str = "") -> None:
        self.report.checks.append(CheckResult(group, name, status, detail))

    # ── transport ─────────────────────────────────────────────────────────────

    async def rpc(
        self,
        method: str,
        params: dict[str, Any],
        authenticated: bool = True,
    ) -> httpx.Response:
        body = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": method,
            "params": params,
        }
        headers = dict(self.auth_headers) if authenticated else {}
        return await self.client.post(f"{self.url}/", json=body, headers=headers)

    def message_params(self, text: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "content": text or self.message}],
            }
        }
        if self.skill_id:
            params["skillId"] = self.skill_id
        return params

    # ── card ──────────────────────────────────────────────────────────────────

    async def check_card(self) -> dict[str, Any] | None:
        group = "card"
        try:
            resp = await self.client.get(f"{self.url}{CARD_PATH}")
        except httpx.HTTPError as exc:
            self.record(group, "reachable", Status.FAIL, f"{CARD_PATH}: {exc}")
            return None

        if resp.status_code != 200:
            self.record(group, "reachable", Status.FAIL,
                        f"{CARD_PATH} returned HTTP {resp.status_code}")
            return None
        self.record(group, "reachable", Status.PASS)

        try:
            card = resp.json()
        except ValueError:
            self.record(group, "json", Status.FAIL, "card is not valid JSON")
            return None
        if not isinstance(card, dict):
            self.record(group, "json", Status.FAIL, "card is not a JSON object")
            return None
        self.record(group, "json", Status.PASS)

        from nexus_a2a.models.agent import AgentCard

        try:
            AgentCard.model_validate(card)
        except Exception as exc:
            first = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            self.record(group, "schema", Status.FAIL, first)
        else:
            self.record(group, "schema", Status.PASS)

        card_url = str(card.get("url", ""))
        parsed = urlparse(card_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            self.record(group, "url", Status.FAIL,
                        f"card url {card_url!r} is not an absolute http(s) URL")
        elif card_url.rstrip("/") != self.url:
            self.record(group, "url", Status.WARN,
                        f"card advertises {card_url} but was served from {self.url}; "
                        "clients will call the advertised address")
        else:
            self.record(group, "url", Status.PASS)

        skills = card.get("skills")
        if not skills:
            self.record(group, "skills", Status.WARN,
                        "no skills declared, so discovery by skill cannot find it")
        else:
            ids = [s.get("id") for s in skills if isinstance(s, dict)]
            if len(ids) != len(set(ids)):
                self.record(group, "skills", Status.FAIL, "duplicate skill ids")
            else:
                self.record(group, "skills", Status.PASS, f"{len(ids)} declared")

        try:
            health = await self.client.get(f"{self.url}/health")
            if health.status_code == 200:
                self.record(group, "health", Status.PASS)
            else:
                self.record(group, "health", Status.WARN,
                            f"/health returned HTTP {health.status_code}")
        except httpx.HTTPError:
            self.record(group, "health", Status.WARN, "/health unreachable")

        self.report.card = card
        return card

    # ── protocol ──────────────────────────────────────────────────────────────

    async def check_protocol(self) -> None:
        group = "protocol"

        resp = await self.rpc("nexus/verify/no-such-method", {})
        code = _rpc_error_code(resp)
        if code == -32601:
            self.record(group, "method_not_found", Status.PASS)
        elif code is not None:
            self.record(group, "method_not_found", Status.WARN,
                        f"unknown method returned error {code}, expected -32601")
        else:
            self.record(group, "method_not_found", Status.FAIL,
                        f"unknown method was not rejected (HTTP {resp.status_code})")

        resp = await self.client.post(
            f"{self.url}/",
            content=b"{not json",
            headers={"Content-Type": "application/json", **self.auth_headers},
        )
        code = _rpc_error_code(resp)
        if code == -32700:
            self.record(group, "parse_error", Status.PASS)
        elif code is not None or resp.status_code == 400:
            self.record(group, "parse_error", Status.WARN,
                        f"malformed JSON rejected with {code or resp.status_code}, "
                        "expected -32700")
        else:
            self.record(group, "parse_error", Status.FAIL,
                        f"malformed JSON was not rejected (HTTP {resp.status_code})")

        resp = await self.rpc("message/send", {})
        code = _rpc_error_code(resp)
        if code == -32602:
            self.record(group, "invalid_params", Status.PASS)
        elif code is not None or 400 <= resp.status_code < 500:
            self.record(group, "invalid_params", Status.WARN,
                        f"missing message rejected with {code or resp.status_code}, "
                        "expected -32602")
        else:
            self.record(group, "invalid_params", Status.FAIL,
                        "message/send with no message was accepted")

        resp = await self.rpc("tasks/get", {"taskId": f"verify-{uuid.uuid4()}"})
        if _rpc_error_code(resp) is not None:
            self.record(group, "unknown_task", Status.PASS)
        else:
            self.record(group, "unknown_task", Status.FAIL,
                        "tasks/get for a task that does not exist did not error")

    # ── task ──────────────────────────────────────────────────────────────────

    async def check_task(self) -> dict[str, Any] | None:
        group = "task"
        if self.read_only:
            self.record(group, "send", Status.SKIP, "read-only: agent not invoked")
            return None

        resp = await self.rpc("message/send", self.message_params())
        result = _rpc_result(resp)
        if result is None:
            code = _rpc_error_code(resp)
            self.record(group, "send", Status.FAIL,
                        f"message/send failed: "
                        f"{'error ' + str(code) if code else 'HTTP ' + str(resp.status_code)}")
            return None

        state = str(result.get("state", ""))
        if state not in _KNOWN_STATES:
            self.record(group, "send", Status.FAIL, f"unknown task state {state!r}")
            return None
        if state == "failed":
            self.record(group, "send", Status.WARN,
                        f"the agent failed the probe: {result.get('error')}")
        else:
            self.record(group, "send", Status.PASS, f"state={state}")

        from nexus_a2a.models.task import Task

        try:
            Task.model_validate(result)
        except Exception as exc:
            self.record(group, "task_shape", Status.FAIL, str(exc).splitlines()[0])
        else:
            self.record(group, "task_shape", Status.PASS)

        history = result.get("history") or []
        texts = [
            p.get("content")
            for m in history if isinstance(m, dict)
            for p in m.get("parts", []) if isinstance(p, dict)
        ]
        if self.message in texts:
            self.record(group, "history", Status.PASS)
        else:
            self.record(group, "history", Status.WARN,
                        "task history does not contain the message that was sent")

        task_id = result.get("id")
        got = _rpc_result(await self.rpc("tasks/get", {"taskId": task_id}))
        if got is not None and got.get("id") == task_id:
            self.record(group, "get", Status.PASS)
        else:
            self.record(group, "get", Status.FAIL,
                        "tasks/get did not return the task just created")

        if state in {"completed", "failed", "cancelled"}:
            resp = await self.rpc("tasks/cancel", {"taskId": task_id})
            cancelled = _rpc_result(resp)
            if cancelled is not None and cancelled.get("state") == "cancelled":
                self.record(group, "cancel_finished", Status.FAIL,
                            "cancelling a finished task reported success")
            else:
                self.record(group, "cancel_finished", Status.PASS)
        else:
            self.record(group, "cancel_finished", Status.SKIP,
                        f"probe task is {state}, not finished")

        return result

    # ── stream ────────────────────────────────────────────────────────────────

    async def check_stream(self, card: dict[str, Any]) -> None:
        group = "stream"
        claims = bool(card.get("capabilities", {}).get("streaming"))
        if not claims:
            self.record(group, "message_stream", Status.SKIP,
                        "card does not claim streaming")
            return
        if self.read_only:
            self.record(group, "message_stream", Status.SKIP,
                        "read-only: agent not invoked")
            return

        from nexus_a2a.transport.sse import iter_sse_events

        body = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "message/stream",
            "params": self.message_params(),
        }
        types: list[str] = []
        try:
            async with self.client.stream(
                "POST", f"{self.url}/", json=body,
                headers={"Accept": "text/event-stream", **self.auth_headers},
            ) as resp:
                content_type = resp.headers.get("content-type", "")
                if resp.status_code != 200 or "text/event-stream" not in content_type:
                    await resp.aread()
                    self.record(group, "message_stream", Status.FAIL,
                                f"card claims streaming but message/stream returned "
                                f"HTTP {resp.status_code} ({content_type or 'no type'})")
                    return
                async for event in iter_sse_events(resp):
                    types.append(event.type.value)
                    if event.is_terminal:
                        break
        except httpx.HTTPError as exc:
            self.record(group, "message_stream", Status.FAIL, f"stream broke: {exc}")
            return

        if not types:
            self.record(group, "message_stream", Status.FAIL, "stream carried no events")
        elif types[-1] == "error":
            self.record(group, "message_stream", Status.WARN,
                        "stream ended with an error event")
        elif types[-1] != "done":
            self.record(group, "message_stream", Status.FAIL,
                        "stream closed without a terminal done event")
        else:
            self.record(group, "message_stream", Status.PASS, f"{len(types)} events")

    # ── push ──────────────────────────────────────────────────────────────────

    async def check_push(self, card: dict[str, Any], task: dict[str, Any] | None) -> None:
        group = "push"
        claims = bool(card.get("capabilities", {}).get("push_notifications"))
        if not claims:
            self.record(group, "config_get", Status.SKIP,
                        "card does not claim push notifications")
            return
        if task is None:
            self.record(group, "config_get", Status.SKIP, "no probe task to query")
            return

        resp = await self.rpc(
            "tasks/pushNotificationConfig/get", {"taskId": task.get("id")}
        )
        code = _rpc_error_code(resp)
        if code == -32601:
            self.record(group, "config_get", Status.FAIL,
                        "card claims push notifications but the config methods "
                        "are not implemented")
        elif _rpc_result(resp) is not None:
            self.record(group, "config_get", Status.PASS)
        else:
            self.record(group, "config_get", Status.WARN,
                        f"tasks/pushNotificationConfig/get returned error {code}")

        # A push target is fetched by the AGENT, so it must refuse to be
        # pointed at cloud metadata or its own network.
        resp = await self.rpc(
            "tasks/pushNotificationConfig/set",
            {
                "taskId": task.get("id"),
                "pushNotificationConfig": {
                    "url": "http://169.254.169.254/latest/meta-data/"
                },
            },
        )
        if _rpc_result(resp) is not None:
            self.record(group, "ssrf_guard", Status.FAIL,
                        "agent accepted a cloud-metadata address as a webhook "
                        "target — it can be used as an SSRF proxy")
        else:
            self.record(group, "ssrf_guard", Status.PASS)

    # ── auth ──────────────────────────────────────────────────────────────────

    async def check_auth(self, card: dict[str, Any]) -> None:
        group = "auth"
        scheme = str((card.get("authentication") or {}).get("scheme", "none"))

        # tasks/get for a task that cannot exist: never invokes the agent.
        probe = {"taskId": f"verify-{uuid.uuid4()}"}
        resp = await self.rpc("tasks/get", probe, authenticated=False)
        refused = resp.status_code in (401, 403)

        if scheme == "none":
            if refused:
                self.record(group, "matches_card", Status.FAIL,
                            f"card says no auth is needed, but an unauthenticated "
                            f"request got HTTP {resp.status_code} — clients that "
                            "trust the card will fail")
            else:
                self.record(group, "matches_card", Status.PASS,
                            "no auth advertised, none enforced")
            return

        if not refused:
            self.record(group, "matches_card", Status.FAIL,
                        f"card advertises {scheme!r} but an unauthenticated request "
                        "was served — the agent is open to anyone")
            return
        self.record(group, "matches_card", Status.PASS,
                    f"{scheme} advertised and enforced")

        if not self.auth_headers:
            self.record(group, "credentials", Status.SKIP,
                        "no credentials supplied (--api-key / --bearer)")
            self.locked_out = f"agent requires {scheme} and no credentials were supplied"
            return
        resp = await self.rpc("tasks/get", probe, authenticated=True)
        if resp.status_code in (401, 403):
            self.record(group, "credentials", Status.FAIL,
                        f"the supplied credentials were refused (HTTP {resp.status_code})")
            self.locked_out = "the supplied credentials were refused"
        else:
            self.record(group, "credentials", Status.PASS, "supplied credentials accepted")

    # ── driver ────────────────────────────────────────────────────────────────

    async def run(self) -> VerifyReport:
        card = await self.check_card()
        if card is None:
            return self.report

        await self.check_auth(card)
        if self.locked_out is not None:
            reason = f"not checked: {self.locked_out}"
            for group, name in (
                ("protocol", "error_handling"),
                ("task", "send"),
                ("stream", "message_stream"),
                ("push", "config_get"),
            ):
                self.record(group, name, Status.SKIP, reason)
            return self.report

        await self.check_protocol()
        task = await self.check_task()
        await self.check_stream(card)
        await self.check_push(card, task)
        return self.report


# ── helpers ───────────────────────────────────────────────────────────────────


def _body(resp: httpx.Response) -> dict[str, Any] | None:
    try:
        data = resp.json()
    except (ValueError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _rpc_error_code(resp: httpx.Response) -> int | None:
    body = _body(resp)
    if body is None:
        return None
    error = body.get("error")
    if isinstance(error, dict) and isinstance(error.get("code"), int):
        return int(error["code"])
    return None


def _rpc_result(resp: httpx.Response) -> dict[str, Any] | None:
    if resp.status_code != 200:
        return None
    body = _body(resp)
    if body is None:
        return None
    result = body.get("result")
    return result if isinstance(result, dict) else None


# ── public entry point ────────────────────────────────────────────────────────


async def verify_agent(
    url: str,
    *,
    api_key: str | None = None,
    api_key_header: str = "X-API-Key",
    bearer: str | None = None,
    caller_url: str | None = None,
    headers: dict[str, str] | None = None,
    message: str = "nexus verify probe",
    skill_id: str | None = None,
    read_only: bool = False,
    timeout: float = 15.0,
    client: httpx.AsyncClient | None = None,
) -> VerifyReport:
    """
    Run every conformance check against the agent at `url`.

    Args:
        url:            Base URL of the agent.
        api_key:        Sent in `api_key_header` on authenticated requests.
        bearer:         Sent as `Authorization: Bearer ...`.
        caller_url:     Sent as X-Nexus-Caller, for agents enforcing trust.
        headers:        Any other headers to send on authenticated requests.
        message:        Text of the probe message.
        skill_id:       Skill to target with the probe.
        read_only:      Skip every check that would make the agent do work.
        timeout:        Per-request timeout in seconds.
        client:         An httpx.AsyncClient to use, e.g. with an ASGI transport.
    """
    auth_headers: dict[str, str] = dict(headers or {})
    if api_key:
        auth_headers[api_key_header] = api_key
    if bearer:
        auth_headers["Authorization"] = f"Bearer {bearer}"
    if caller_url:
        auth_headers["X-Nexus-Caller"] = caller_url

    own_client = client is None
    http = client or httpx.AsyncClient(timeout=timeout)
    try:
        checker = _Checker(url, http, auth_headers, message, skill_id, read_only)
        return await checker.run()
    finally:
        if own_client:
            await http.aclose()


__all__ = ["CheckResult", "Status", "VerifyReport", "verify_agent"]
