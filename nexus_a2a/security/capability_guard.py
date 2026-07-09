"""
nexus_a2a/security/capability_guard.py

CapabilityGuard — enforces that declared AgentCapabilities match
what the agent's implementation actually supports.

The problem this solves:
  Before v1.1, a developer could declare streaming=True on their
  @agent decorator, but nothing validated that their run() method
  actually yielded streaming chunks, or that the client switched
  to SSE mode when it detected the flag. Silent mismatches caused
  confusing bugs.

What CapabilityGuard does:
  1. At registration time: inspects the agent class to verify that
     declared capabilities have corresponding implementation support.
  2. At connection time: validates that a client's expectations match
     what the target agent actually advertises.
  3. Raises clear, descriptive errors instead of silent mismatches.

Usage:
    guard = CapabilityGuard()

    # Check an agent class before registering
    guard.validate_agent_class(MyAgent, card)

    # Check client-server capability compatibility
    guard.validate_compatibility(
        caller_capabilities={"wants_streaming": True},
        agent_card=remote_card,
    )

    # Quick check
    guard.assert_supports_streaming(agent_card)
"""

from __future__ import annotations

import inspect
import logging

from nexus_a2a.models.agent import AgentCard

logger = logging.getLogger(__name__)


# ── Exceptions ────────────────────────────────────────────────────────────────


class CapabilityMismatchError(Exception):
    """
    Raised when a declared capability does not match the implementation.
    """

    def __init__(self, agent_name: str, capability: str, reason: str) -> None:
        super().__init__(
            f"Agent '{agent_name}' declares {capability}=True "
            f"but the implementation does not support it: {reason}"
        )
        self.agent_name = agent_name
        self.capability = capability
        self.reason = reason


class CapabilityNotSupportedError(Exception):
    """
    Raised when a client requests a capability the agent does not offer.
    """

    def __init__(self, agent_name: str, capability: str) -> None:
        super().__init__(
            f"Agent '{agent_name}' does not support '{capability}'. "
            f"Check the agent's AgentCapabilities before connecting."
        )
        self.agent_name = agent_name
        self.capability = capability


# ── CapabilityGuard ───────────────────────────────────────────────────────────


class CapabilityGuard:
    """
    Validates that declared capabilities match actual implementation.

    Enforcement modes:
      "strict" (default): raises CapabilityMismatchError on violation.
      "warn":             logs a warning but does not raise.
      "off":              no checks performed (useful in tests).

    Usage:
        guard = CapabilityGuard(mode="strict")

        # Called automatically by @agent decorator (via registry)
        guard.validate_agent_class(MyAgent, card)

        # Called by AgentNetwork when routing a streaming request
        guard.assert_supports_streaming(remote_card)

        # Called when connecting two agents
        guard.validate_compatibility(
            caller_wants={"streaming": True},
            agent_card=remote_card,
        )
    """

    MODES = {"strict", "warn", "off"}

    def __init__(self, mode: str = "strict") -> None:
        if mode not in self.MODES:
            raise ValueError(f"Invalid mode '{mode}'. Choose from: {self.MODES}")
        self._mode = mode

    # ── Agent class validation ────────────────────────────────────────────────

    def validate_agent_class(
        self,
        agent_cls: type,
        card: AgentCard,
    ) -> list[str]:
        """
        Inspect an agent class and verify its declared capabilities.

        Checks performed:
          streaming=True    → run() must be an async generator (yield chunks)
                              OR the class must have a stream() method.
          push_notifications=True → class must have a webhook_url property
                              or config that provides a delivery URL.

        Args:
            agent_cls: The decorated agent class.
            card:      The AgentCard with declared capabilities.

        Returns:
            List of warning strings (empty if all checks pass).

        Raises:
            CapabilityMismatchError: In strict mode when a check fails.
        """
        warnings: list[str] = []
        caps = card.capabilities

        if self._mode == "off":
            return warnings

        # ── Streaming check ───────────────────────────────────────────────────
        if caps.streaming:
            issue = self._check_streaming(agent_cls)
            if issue:
                warnings.append(issue)
                self._report(
                    agent_name=card.name,
                    capability="streaming",
                    reason=issue,
                )

        # ── Push notifications check ──────────────────────────────────────────
        if caps.push_notifications:
            issue = self._check_push_notifications(agent_cls)
            if issue:
                warnings.append(issue)
                self._report(
                    agent_name=card.name,
                    capability="push_notifications",
                    reason=issue,
                )

        if not warnings:
            logger.debug(
                "CapabilityGuard: '%s' passed all capability checks", card.name
            )

        return warnings

    def validate_compatibility(
        self,
        caller_wants: dict[str, bool],
        agent_card: AgentCard,
    ) -> None:
        """
        Verify that what the caller wants matches what the agent offers.

        Args:
            caller_wants: Dict of {capability_name: True/False} the
                          caller requires.
            agent_card:   The remote agent's card.

        Raises:
            CapabilityNotSupportedError: Agent lacks a required capability.

        Example:
            guard.validate_compatibility(
                caller_wants={"streaming": True},
                agent_card=remote_card,
            )
        """
        caps = agent_card.capabilities

        if caller_wants.get("streaming") and not caps.streaming:
            raise CapabilityNotSupportedError(agent_card.name, "streaming")

        if caller_wants.get("push_notifications") and not caps.push_notifications:
            raise CapabilityNotSupportedError(agent_card.name, "push_notifications")

    # ── Quick assertion helpers ───────────────────────────────────────────────

    def assert_supports_streaming(self, card: AgentCard) -> None:
        """
        Assert the agent supports streaming.
        Raises CapabilityNotSupportedError if not.
        """
        if not card.capabilities.streaming:
            raise CapabilityNotSupportedError(card.name, "streaming")

    def assert_supports_push(self, card: AgentCard) -> None:
        """
        Assert the agent supports push notifications.
        Raises CapabilityNotSupportedError if not.
        """
        if not card.capabilities.push_notifications:
            raise CapabilityNotSupportedError(card.name, "push_notifications")

    def supports_streaming(self, card: AgentCard) -> bool:
        """Return True if the agent card declares streaming support."""
        return card.capabilities.streaming

    def supports_push(self, card: AgentCard) -> bool:
        """Return True if the agent card declares push notification support."""
        return card.capabilities.push_notifications

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _check_streaming(self, agent_cls: type) -> str | None:
        """
        Return a warning string if streaming is declared but not implemented.
        Returns None if the check passes.

        What we look for:
          1. run() is an async generator (uses `yield`)
          2. OR the class has a stream() method
          3. OR the class has a STREAMING = True class attribute
             (explicit override for adapters that handle it internally)
        """
        # Explicit opt-out from check
        if getattr(agent_cls, "STREAMING", False) is True:
            return None

        run_method = getattr(agent_cls, "run", None)
        if run_method is None:
            return "class has no run() method"

        # async generator check
        if inspect.isasyncgenfunction(run_method):
            return None

        # stream() method check
        if hasattr(agent_cls, "stream") and callable(agent_cls.stream):
            return None

        return (
            "run() is not an async generator (no `yield` statements) "
            "and no stream() method found. "
            "Either add `yield` to run(), add a stream() method, "
            "or set STREAMING = True on the class if streaming is "
            "handled internally by the framework adapter."
        )

    def _check_push_notifications(self, agent_cls: type) -> str | None:
        """
        Return a warning string if push_notifications is declared but
        the class has no mechanism to deliver them.

        What we look for:
          1. Class has a webhook_url property or attribute
          2. OR class has a PUSH_NOTIFICATIONS = True class attribute
        """
        if getattr(agent_cls, "PUSH_NOTIFICATIONS", False) is True:
            return None

        if hasattr(agent_cls, "webhook_url"):
            return None

        return (
            "push_notifications=True declared but no webhook_url attribute found. "
            "Add a webhook_url property to the class or set "
            "PUSH_NOTIFICATIONS = True if delivery is handled externally."
        )

    def _report(
        self,
        agent_name: str,
        capability: str,
        reason: str,
    ) -> None:
        """Log or raise depending on the current mode."""
        if self._mode == "off":
            return
        if self._mode == "warn":
            logger.warning(
                "CapabilityGuard [%s]: %s=True but: %s",
                agent_name,
                capability,
                reason,
            )
            return
        # strict
        raise CapabilityMismatchError(agent_name, capability, reason)
