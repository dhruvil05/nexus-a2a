"""
nexus_a2a/transport/webhook.py

WebhookDispatcher — delivers async task update notifications to a
client-provided URL via HTTP POST.

When to use webhooks vs SSE:
  SSE     — client stays connected, gets real-time updates. Good for
             short-to-medium tasks where the client is online.
  Webhook — agent POSTs updates to a URL. Good for long-running tasks
             (hours/days), mobile clients, serverless functions that
             cannot hold open connections.

Features:
  - Configurable retry with exponential backoff.
  - HMAC-SHA256 signature on every delivery so receivers can verify
    the payload came from a trusted source.
  - Delivery log so you can inspect what was sent and whether it succeeded.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import ipaddress
import json
import logging
import socket
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import httpx

from nexus_a2a.models.task import Task
from nexus_a2a.security.redaction import redact_url

logger = logging.getLogger(__name__)

# Header that carries the HMAC signature of the body alone (kept for
# receivers written against earlier releases).
_SIGNATURE_HEADER = "X-Nexus-Signature-256"
# Signature over "<timestamp>.<body>" — lets a receiver reject replays.
_TIMESTAMPED_SIGNATURE_HEADER = "X-Nexus-Signature-256-Timestamped"
_TIMESTAMP_HEADER = "X-Nexus-Timestamp"
# Unique per delivery attempt group; receivers can de-duplicate on it.
_DELIVERY_ID_HEADER = "X-Nexus-Delivery-Id"
# A2A push-notification token the caller registered for this task.
_NOTIFICATION_TOKEN_HEADER = "X-A2A-Notification-Token"

# Default replay window for verify_timestamped_signature().
_DEFAULT_TOLERANCE_SEC = 300


# ── Exceptions ────────────────────────────────────────────────────────────────


class WebhookDeliveryError(Exception):
    """Raised after all retry attempts have been exhausted."""

    def __init__(self, url: str, attempts: int, last_error: str) -> None:
        super().__init__(
            f"Webhook delivery to '{redact_url(url, keep_path=False)}' failed "
            f"after {attempts} "
            f"attempt(s): {last_error}"
        )
        self.url = url
        self.attempts = attempts
        self.last_error = last_error


class WebhookUrlError(ValueError):
    """Raised when a webhook URL is not acceptable as a delivery target."""

    def __init__(self, url: str, reason: str) -> None:
        super().__init__(
            f"Rejected webhook URL '{redact_url(url, keep_path=False)}': {reason}"
        )
        self.url = url
        self.reason = reason


# ── Signing ───────────────────────────────────────────────────────────────────


def _canonical_body(payload: dict[str, Any]) -> bytes:
    """
    Serialise a payload to the exact bytes that will be sent AND signed.

    Compact separators and sorted keys make the encoding reproducible, so a
    receiver that re-serialises the parsed JSON gets the same bytes back.
    """
    return json.dumps(
        payload, separators=(",", ":"), sort_keys=True, ensure_ascii=False
    ).encode("utf-8")


def sign_body(body: bytes, secret: str) -> str:
    """Return the 'sha256=<hex>' signature for exactly these bytes."""
    return (
        "sha256=" + hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    )


def sign_timestamped(body: bytes, timestamp: str, secret: str) -> str:
    """Signature over b"<timestamp>." + body, binding the delivery time."""
    return sign_body(timestamp.encode("ascii") + b"." + body, secret)


# ── URL validation ────────────────────────────────────────────────────────────


def _parse_webhook_url(url: str) -> str:
    """Check scheme and host; return the hostname."""
    if any(ord(ch) < 0x21 or ord(ch) == 0x7F for ch in url):
        # urlparse silently drops CR/LF/TAB, so a URL carrying them would
        # otherwise validate and then reach the logs verbatim.
        raise WebhookUrlError(url, "control characters or spaces in URL")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise WebhookUrlError(url, "only http and https URLs are accepted")
    if not parsed.hostname:
        raise WebhookUrlError(url, "no host in URL")
    return parsed.hostname


_NAT64_WELL_KNOWN = ipaddress.ip_network("64:ff9b::/96")
_IPV4_COMPATIBLE = ipaddress.ip_network("::/96")


def _proxy_applies(url: str) -> bool:
    """
    True when httpx (trust_env=True) would send this URL through a proxy.

    Pinning rewrites the URL host to an IP; through an HTTP proxy's CONNECT
    tunnel httpcore then verifies TLS against that IP and NO_PROXY / proxy
    allow-lists see an IP. So pinning is skipped when a proxy applies — the
    egress proxy, which does the connecting, must enforce the address policy.
    """
    import urllib.request

    proxies = urllib.request.getproxies_environment()
    scheme = urlparse(url).scheme
    if not (proxies.get(scheme) or proxies.get("all")):
        return False
    host = urlparse(url).hostname or ""
    # Present at runtime on every platform but missing from typeshed.
    bypass = getattr(urllib.request, "proxy_bypass_environment")  # noqa: B009
    return not bypass(host, proxies)


def _non_public_reason(raw: str) -> str | None:
    """
    Why this resolved address must not be a webhook target, or None if it may.

    Uses is_global rather than a deny-list of ranges: a deny-list built from
    is_private/is_reserved/is_link_local misses 100.64.0.0/10 (carrier-grade
    NAT, and Alibaba Cloud's metadata service at 100.100.100.200).
    """
    try:
        addr = ipaddress.ip_address(raw)
    except ValueError:
        return f"unparseable address {raw!r}"
    if isinstance(addr, ipaddress.IPv6Address):
        if addr.ipv4_mapped is not None:
            addr = addr.ipv4_mapped
        elif addr in _NAT64_WELL_KNOWN:
            # 64:ff9b::a.b.c.d reaches a.b.c.d through a NAT64 gateway, and
            # Python calls the whole prefix global — judge the IPv4 inside.
            addr = ipaddress.IPv4Address(int(addr) & 0xFFFFFFFF)
        elif addr in _IPV4_COMPATIBLE and addr.is_global:
            # Deprecated ::a.b.c.d form; never a legitimate webhook target.
            return f"resolves to non-public address {addr} (IPv4-compatible form)"
    if not addr.is_global or addr.is_multicast:
        return f"resolves to non-public address {addr}"
    return None


def _check_addresses(url: str, addresses: list[str]) -> str:
    """Raise unless EVERY address is public; return the first one."""
    if not addresses:
        raise WebhookUrlError(url, "host resolved to no addresses")
    for raw in addresses:
        reason = _non_public_reason(raw)
        if reason is not None:
            raise WebhookUrlError(
                url,
                f"{reason}. Pass allow_private_urls=True only for local development.",
            )
    return addresses[0]


async def _resolve_host(host: str) -> list[str]:
    """Resolve without blocking the event loop. Patched in tests."""
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    return [str(info[4][0]) for info in infos]


def validate_webhook_url(url: str, allow_private: bool = False) -> None:
    """
    Reject a webhook URL that is unsafe to POST to (synchronous check).

    A webhook URL arrives from whoever called the agent, and the server then
    makes a request to it — that is a server-side request forgery primitive.
    Without this check a caller could point an agent at cloud metadata
    (169.254.169.254), at localhost admin ports, or at hosts inside the
    server's private network, and use the agent as a proxy.

    This is a point-in-time check: DNS can change between now and delivery
    (DNS rebinding), so WebhookDispatcher re-resolves and re-checks on every
    attempt and connects to the address it checked. Inside async code prefer
    resolve_webhook_url(), which does not block the event loop.

    Args:
        url:           The candidate webhook URL.
        allow_private: Permit loopback and private ranges. Needed for local
                       development and tests; leave False in production.

    Raises:
        WebhookUrlError: Scheme is not http/https, the host is missing, or the
                         host resolves to a blocked address.
    """
    host = _parse_webhook_url(url)
    if allow_private:
        return
    try:
        addresses = [
            str(info[4][0])
            for info in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        ]
    except (OSError, UnicodeError) as exc:
        raise WebhookUrlError(url, f"host does not resolve: {exc}") from exc
    _check_addresses(url, addresses)


async def resolve_webhook_url(
    url: str,
    allow_private: bool = False,
    timeout: float = 5.0,
) -> str | None:
    """
    Validate a webhook URL and return the public IP to connect to.

    Resolves asynchronously (a slow resolver cannot stall the server) and
    requires EVERY address the name resolves to be public, so a round-robin
    record cannot smuggle one internal address in.

    Returns:
        The vetted IP address, or None when allow_private is True (no pinning
        in development mode).

    Raises:
        WebhookUrlError: Bad scheme or host, resolution failed or timed out,
                         or the host resolves to a non-public address.
    """
    host = _parse_webhook_url(url)
    if allow_private:
        return None
    try:
        addresses = await asyncio.wait_for(_resolve_host(host), timeout=timeout)
    except TimeoutError as exc:
        raise WebhookUrlError(url, "host resolution timed out") from exc
    except (OSError, UnicodeError) as exc:
        raise WebhookUrlError(url, f"host does not resolve: {exc}") from exc
    return _check_addresses(url, addresses)


def _pin_to_address(
    url: str, ip: str
) -> tuple[httpx.URL, dict[str, str], dict[str, Any]]:
    """
    Rewrite url to connect to ip while still presenting the original host.

    The Host header keeps virtual hosting working, and for https the original
    name is used for SNI AND certificate verification (httpcore passes
    sni_hostname as server_hostname), so TLS still authenticates the host the
    caller named — only the TCP destination is fixed to the vetted address.
    """
    original = httpx.URL(url)
    pinned = original.copy_with(host=ip)
    headers = {"Host": original.netloc.decode("ascii")}
    extensions: dict[str, Any] = {}
    if original.scheme == "https":
        extensions["sni_hostname"] = original.raw_host.decode("ascii")
    return pinned, headers, extensions


# ── Delivery record ───────────────────────────────────────────────────────────


@dataclass
class DeliveryRecord:
    """
    Audit trail for one webhook delivery attempt.

    Fields:
        url:         Target webhook URL.
        task_id:     ID of the task whose update was sent.
        event:       The event type string (e.g. "task_status").
        attempts:    How many HTTP attempts were made.
        succeeded:   True if at least one attempt got a 2xx response.
        status_code: HTTP status code of the final attempt.
        error:       Error message if all attempts failed.
        sent_at:     Unix timestamp when the first attempt was made.
    """

    url: str
    task_id: str
    event: str
    attempts: int = 0
    succeeded: bool = False
    status_code: int = 0
    error: str | None = None
    sent_at: float = field(default_factory=time.time)


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class WebhookConfig:
    """
    Delivery settings for the WebhookDispatcher.

    Args:
        max_retries:    Maximum delivery attempts before giving up. Default: 3.
        base_delay:     Initial retry delay in seconds. Doubles each attempt.
        timeout:        Per-attempt HTTP timeout in seconds.
        signing_secret: If set, every payload is signed with HMAC-SHA256 and
                        the signature is placed in X-Nexus-Signature-256.
        allow_private_urls: Permit delivery to loopback and private addresses.
                        Callers supply the webhook URL, so leaving this False
                        stops an agent being used as an SSRF proxy into its own
                        network. Set True for local development and tests.
        max_concurrent: Deliveries in flight at once, per dispatcher. Extra
                        deliveries wait their turn instead of opening sockets.
        max_log_entries: Delivery records kept in memory (oldest dropped).
    """

    max_retries: int = 3
    base_delay: float = 1.0  # seconds; doubles each retry
    timeout: float = 10.0
    signing_secret: str | None = None
    allow_private_urls: bool = False
    max_concurrent: int = 100
    max_log_entries: int = 1000


# ── WebhookDispatcher ─────────────────────────────────────────────────────────


class WebhookDispatcher:
    """
    Delivers task lifecycle updates to client-registered webhook URLs.

    Usage:
        dispatcher = WebhookDispatcher(
            config=WebhookConfig(signing_secret="my-secret")
        )

        # Deliver a task update
        record = await dispatcher.dispatch(
            url="https://client.example.com/webhooks/nexus",
            task=completed_task,
            event="task_completed",
        )

        # Verify a received payload on the client side (replay-safe)
        is_valid = WebhookDispatcher.verify_timestamped_signature(
            payload=request.body,
            signature=request.headers["X-Nexus-Signature-256-Timestamped"],
            timestamp=request.headers["X-Nexus-Timestamp"],
            secret="my-secret",
        )

    Every delivery carries:
        X-Nexus-Delivery-Id                 unique id, for de-duplication
        X-Nexus-Timestamp                   unix seconds when it was signed
        X-A2A-Notification-Token            the token the caller registered
                                            for the task, if any
        X-Nexus-Signature-256               HMAC of the body (signing_secret)
        X-Nexus-Signature-256-Timestamped   HMAC of "<timestamp>.<body>"

    The per-task token is the receiver-specific proof: signing_secret is one
    secret for the whole server, so any receiver that holds it could forge a
    delivery to another receiver; a token only its own receiver knows cannot.

    Args:
        config: Delivery and retry settings.
    """

    def __init__(self, config: WebhookConfig | None = None) -> None:
        self._config = config or WebhookConfig()
        # Recent delivery history — bounded, so a busy agent (or a caller
        # registering many webhooks) cannot grow it without limit.
        self._log: deque[DeliveryRecord] = deque(
            maxlen=max(1, self._config.max_log_entries)
        )
        self._slots = asyncio.Semaphore(max(1, self._config.max_concurrent))

    # ── Public API ────────────────────────────────────────────────────────────

    async def dispatch(
        self,
        url: str,
        task: Task,
        event: str = "task_update",
        token: str | None = None,
    ) -> DeliveryRecord:
        """
        Deliver a task update notification to a webhook URL.

        Retries with exponential backoff on connection errors and 5xx responses.
        Does NOT retry on 4xx responses (client errors — retrying won't help).

        Args:
            url:   The client's webhook endpoint.
            task:  The Task whose current state to deliver.
            event: A string describing what happened (e.g. "task_completed").
            token: The caller's per-task notification token, sent back as
                   X-A2A-Notification-Token so the receiver can verify it.

        Returns:
            DeliveryRecord describing the outcome.

        Raises:
            WebhookDeliveryError: If all retry attempts are exhausted.
        """
        async with self._slots:
            return await self._dispatch(url, task, event, token)

    async def _dispatch(
        self,
        url: str,
        task: Task,
        event: str,
        token: str | None,
    ) -> DeliveryRecord:
        payload = self._build_payload(task, event)
        record = DeliveryRecord(url=url, task_id=task.id, event=event)
        safe_url = redact_url(url, keep_path=False)
        delivery_id = str(uuid.uuid4())
        proxied = _proxy_applies(url)
        if proxied:
            logger.debug(
                "Webhook to %s goes through an HTTP proxy; not pinning the "
                "address (the proxy must enforce egress policy).",
                safe_url,
            )
        # Serialise ONCE and post those exact bytes. Signing a
        # separately-serialised copy and letting httpx re-encode via json=
        # produced different bytes, so every signature failed verification.
        body = _canonical_body(payload)

        last_error = "unknown error"

        async with httpx.AsyncClient(timeout=self._config.timeout) as client:
            for attempt in range(1, self._config.max_retries + 1):
                record.attempts = attempt
                delay = self._config.base_delay * (2 ** (attempt - 1))

                # Re-resolve and re-check on EVERY attempt, then connect to
                # the address that was checked. Validating only at
                # registration let a DNS-rebinding host pass as public and
                # then resolve to 169.254.169.254 or 127.0.0.1 at delivery.
                try:
                    target_ip = await resolve_webhook_url(
                        url,
                        allow_private=self._config.allow_private_urls,
                        timeout=self._config.timeout,
                    )
                except WebhookUrlError as exc:
                    last_error = exc.reason
                    logger.warning(
                        "Webhook refused at delivery for task %s: %s",
                        task.id,
                        exc.reason,
                    )
                    break

                try:
                    headers = self._build_headers(body, delivery_id, token)
                    target: str | httpx.URL = url
                    extensions: dict[str, Any] = {}
                    if target_ip is not None and not proxied:
                        target, host_header, extensions = _pin_to_address(
                            url, target_ip
                        )
                        headers.update(host_header)
                    # One deadline for the whole attempt: httpx's timeout is
                    # per phase, so a receiver trickling bytes could otherwise
                    # hold a delivery (and its slot) open indefinitely.
                    status = await asyncio.wait_for(
                        self._post(client, target, body, headers, extensions),
                        timeout=self._config.timeout,
                    )
                    record.status_code = status

                    if 200 <= status < 300:
                        record.succeeded = True
                        logger.info(
                            "Webhook delivered: url=%s task=%s event=%s attempt=%d",
                            safe_url,
                            task.id,
                            event,
                            attempt,
                        )
                        break

                    # 4xx — client error, don't retry
                    if 400 <= status < 500:
                        last_error = f"HTTP {status} (client error, not retrying)"
                        logger.warning("Webhook 4xx: %s → %s", safe_url, status)
                        break

                    # 5xx — server error, retry
                    last_error = f"HTTP {status}"
                    logger.warning(
                        "Webhook attempt %d/%d failed (%s), retrying in %.1fs",
                        attempt,
                        self._config.max_retries,
                        last_error,
                        delay,
                    )

                except (
                    httpx.ConnectError,
                    httpx.TimeoutException,
                    TimeoutError,
                ) as exc:
                    last_error = str(exc) or type(exc).__name__
                    logger.warning(
                        "Webhook attempt %d/%d connection error: %s, retrying in %.1fs",
                        attempt,
                        self._config.max_retries,
                        last_error,
                        delay,
                    )

                # Wait before next attempt (skip wait on last attempt)
                if attempt < self._config.max_retries and not record.succeeded:
                    await asyncio.sleep(delay)

        if not record.succeeded:
            record.error = last_error
            self._log.append(record)
            raise WebhookDeliveryError(url, record.attempts, last_error)

        self._log.append(record)
        return record

    @staticmethod
    async def _post(
        client: httpx.AsyncClient,
        target: str | httpx.URL,
        body: bytes,
        headers: dict[str, str],
        extensions: dict[str, Any],
    ) -> int:
        """
        POST and return the status code WITHOUT reading the response body.

        Only the status matters; reading the body let a hostile receiver send
        back gigabytes into this process's memory.
        """
        request = client.build_request(
            "POST", target, content=body, headers=headers, extensions=extensions
        )
        response = await client.send(request, stream=True)
        try:
            return int(response.status_code)
        finally:
            await response.aclose()

    async def dispatch_silent(
        self,
        url: str,
        task: Task,
        event: str = "task_update",
        token: str | None = None,
    ) -> DeliveryRecord:
        """
        Same as dispatch() but never raises — logs the error instead.
        Use when webhook delivery failure should not abort the calling workflow.
        """
        try:
            return await self.dispatch(url, task, event, token=token)
        except WebhookDeliveryError as exc:
            logger.error("Silent webhook failure: %s", exc)
            # Return the failed record from the log
            return self._log[-1]

    # ── Signature verification (static — for use on the receiver side) ────────

    @staticmethod
    def verify_signature(
        payload: bytes | str,
        signature: str,
        secret: str,
    ) -> bool:
        """
        Verify an HMAC-SHA256 signature on a received webhook payload.

        Call this on the CLIENT side when you receive a webhook POST. This
        checks X-Nexus-Signature-256, which does not cover the delivery time,
        so a captured delivery can be replayed; prefer
        verify_timestamped_signature().

        Args:
            payload:   The raw request body bytes (or UTF-8 string).
            signature: The value of the X-Nexus-Signature-256 header.
            secret:    The shared signing secret.

        Returns:
            True if the signature is valid, False otherwise.
        """
        if isinstance(payload, str):
            payload = payload.encode("utf-8")

        return hmac.compare_digest(sign_body(payload, secret), signature)

    @staticmethod
    def verify_timestamped_signature(
        payload: bytes | str,
        signature: str,
        timestamp: str,
        secret: str,
        tolerance: float = _DEFAULT_TOLERANCE_SEC,
        now: float | None = None,
    ) -> bool:
        """
        Verify X-Nexus-Signature-256-Timestamped and reject stale deliveries.

        Args:
            payload:   The raw request body bytes (or UTF-8 string).
            signature: The X-Nexus-Signature-256-Timestamped header value.
            timestamp: The X-Nexus-Timestamp header value.
            secret:    The shared signing secret.
            tolerance: Maximum age (and clock skew) accepted, in seconds.
            now:       Override the current time (tests).

        Returns:
            True only if the signature matches AND the timestamp is within
            tolerance of now. De-duplicate on X-Nexus-Delivery-Id to reject
            replays inside the window too.
        """
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        try:
            sent_at = int(timestamp)
        except (TypeError, ValueError):
            return False
        current = time.time() if now is None else now
        if abs(current - sent_at) > tolerance:
            return False
        return hmac.compare_digest(
            sign_timestamped(payload, str(sent_at), secret), signature
        )

    # ── Delivery log ──────────────────────────────────────────────────────────

    def delivery_log(self) -> list[DeliveryRecord]:
        """Return recent delivery records (successful and failed)."""
        return list(self._log)

    def failed_deliveries(self) -> list[DeliveryRecord]:
        """Return only the records where delivery ultimately failed."""
        return [r for r in self._log if not r.succeeded]

    def clear_log(self) -> None:
        """Clear the delivery log. Useful in tests."""
        self._log.clear()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_payload(self, task: Task, event: str) -> dict[str, Any]:
        """Build the JSON payload to deliver."""
        return {
            "event": event,
            "task_id": task.id,
            "state": task.state.value,
            "error": task.error,
            # 'owner' is server-side access control, not the receiver's data.
            "task": task.model_dump(mode="json", exclude={"owner"}),
        }

    def _build_headers(
        self,
        body: bytes,
        delivery_id: str | None = None,
        token: str | None = None,
    ) -> dict[str, str]:
        """
        Build request headers, including the HMAC signatures if configured.

        Takes the exact bytes that will be sent — signing anything else means
        the receiver's verify call cannot succeed.
        """
        timestamp = str(int(time.time()))
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "User-Agent": "nexus-a2a-webhook/2.0",
            _DELIVERY_ID_HEADER: delivery_id or str(uuid.uuid4()),
            _TIMESTAMP_HEADER: timestamp,
        }
        if token:
            headers[_NOTIFICATION_TOKEN_HEADER] = token
        if self._config.signing_secret:
            headers[_SIGNATURE_HEADER] = sign_body(body, self._config.signing_secret)
            headers[_TIMESTAMPED_SIGNATURE_HEADER] = sign_timestamped(
                body, timestamp, self._config.signing_secret
            )
        return headers
