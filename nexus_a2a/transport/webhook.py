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

import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from nexus_a2a.models.task import Task

logger = logging.getLogger(__name__)

# Header that carries the HMAC signature
_SIGNATURE_HEADER = "X-Nexus-Signature-256"


# ── Exceptions ────────────────────────────────────────────────────────────────

class WebhookDeliveryError(Exception):
    """Raised after all retry attempts have been exhausted."""

    def __init__(self, url: str, attempts: int, last_error: str) -> None:
        super().__init__(
            f"Webhook delivery to '{url}' failed after {attempts} attempt(s): {last_error}"
        )
        self.url        = url
        self.attempts   = attempts
        self.last_error = last_error


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
    url:         str
    task_id:     str
    event:       str
    attempts:    int   = 0
    succeeded:   bool  = False
    status_code: int   = 0
    error:       str | None = None
    sent_at:     float = field(default_factory=time.time)


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
    """
    max_retries:    int        = 3
    base_delay:     float      = 1.0    # seconds; doubles each retry
    timeout:        float      = 10.0
    signing_secret: str | None = None


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

        # Verify a received payload on the client side
        is_valid = WebhookDispatcher.verify_signature(
            payload=request.body,
            signature=request.headers["X-Nexus-Signature-256"],
            secret="my-secret",
        )

    Args:
        config: Delivery and retry settings.
    """

    def __init__(self, config: WebhookConfig | None = None) -> None:
        self._config = config or WebhookConfig()
        # Full delivery history — useful for debugging and auditing
        self._log: list[DeliveryRecord] = []

    # ── Public API ────────────────────────────────────────────────────────────

    async def dispatch(
        self,
        url: str,
        task: Task,
        event: str = "task_update",
    ) -> DeliveryRecord:
        """
        Deliver a task update notification to a webhook URL.

        Retries with exponential backoff on connection errors and 5xx responses.
        Does NOT retry on 4xx responses (client errors — retrying won't help).

        Args:
            url:   The client's webhook endpoint.
            task:  The Task whose current state to deliver.
            event: A string describing what happened (e.g. "task_completed").

        Returns:
            DeliveryRecord describing the outcome.

        Raises:
            WebhookDeliveryError: If all retry attempts are exhausted.
        """
        payload = self._build_payload(task, event)
        record  = DeliveryRecord(url=url, task_id=task.id, event=event)

        last_error = "unknown error"

        async with httpx.AsyncClient(timeout=self._config.timeout) as client:
            for attempt in range(1, self._config.max_retries + 1):
                record.attempts = attempt
                delay = self._config.base_delay * (2 ** (attempt - 1))

                try:
                    headers = self._build_headers(payload)
                    response = await client.post(url, json=payload, headers=headers)
                    record.status_code = response.status_code

                    if response.is_success:
                        record.succeeded = True
                        logger.info(
                            "Webhook delivered: url=%s task=%s event=%s attempt=%d",
                            url, task.id, event, attempt,
                        )
                        break

                    # 4xx — client error, don't retry
                    if 400 <= response.status_code < 500:
                        last_error = f"HTTP {response.status_code} (client error, not retrying)"
                        logger.warning("Webhook 4xx: %s → %s", url, response.status_code)
                        break

                    # 5xx — server error, retry
                    last_error = f"HTTP {response.status_code}"
                    logger.warning(
                        "Webhook attempt %d/%d failed (%s), retrying in %.1fs",
                        attempt, self._config.max_retries, last_error, delay,
                    )

                except (httpx.ConnectError, httpx.TimeoutException) as exc:
                    last_error = str(exc)
                    logger.warning(
                        "Webhook attempt %d/%d connection error: %s, retrying in %.1fs",
                        attempt, self._config.max_retries, exc, delay,
                    )

                # Wait before next attempt (skip wait on last attempt)
                if attempt < self._config.max_retries and not record.succeeded:
                    import asyncio
                    await asyncio.sleep(delay)

        if not record.succeeded:
            record.error = last_error
            self._log.append(record)
            raise WebhookDeliveryError(url, record.attempts, last_error)

        self._log.append(record)
        return record

    async def dispatch_silent(
        self,
        url: str,
        task: Task,
        event: str = "task_update",
    ) -> DeliveryRecord:
        """
        Same as dispatch() but never raises — logs the error instead.
        Use when webhook delivery failure should not abort the calling workflow.
        """
        try:
            return await self.dispatch(url, task, event)
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

        Call this on the CLIENT side when you receive a webhook POST.

        Args:
            payload:   The raw request body bytes (or UTF-8 string).
            signature: The value of the X-Nexus-Signature-256 header.
            secret:    The shared signing secret.

        Returns:
            True if the signature is valid, False otherwise.
        """
        if isinstance(payload, str):
            payload = payload.encode("utf-8")

        expected = "sha256=" + hmac.new(
            secret.encode("utf-8"),
            payload,
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(expected, signature)

    # ── Delivery log ──────────────────────────────────────────────────────────

    def delivery_log(self) -> list[DeliveryRecord]:
        """Return all delivery records (successful and failed)."""
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
            "event":   event,
            "task_id": task.id,
            "state":   task.state.value,
            "error":   task.error,
            "task":    task.model_dump(mode="json"),
        }

    def _build_headers(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Build request headers, including HMAC signature if configured."""
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "User-Agent":   "nexus-a2a-webhook/0.4.0",
        }
        if self._config.signing_secret:
            body      = json.dumps(payload).encode("utf-8")
            signature = "sha256=" + hmac.new(
                self._config.signing_secret.encode("utf-8"),
                body,
                hashlib.sha256,
            ).hexdigest()
            headers[_SIGNATURE_HEADER] = signature
        return headers
