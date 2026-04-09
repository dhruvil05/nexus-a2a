"""
nexus_a2a/security/trust.py

TrustBoundary — controls which agents are allowed to call which other agents.

Solves a critical problem in multi-agent networks:
  Just because Agent A can reach Agent B's URL does not mean it should.
  TrustBoundary enforces an explicit permission matrix.

Two levels of control:
  1. Global allow/block list  — blanket allow or deny by URL pattern.
  2. Per-agent skill ACL      — which skills a caller is allowed to invoke.

Design:
  - Default policy is DENY — unknown agents are blocked unless explicitly allowed.
  - Wildcard "*" grants access to ALL registered agents (use with care).
  - Skill-level ACL is optional — if not set, all skills are accessible
    to any allowed agent.
"""

from __future__ import annotations

import fnmatch
import logging

logger = logging.getLogger(__name__)


# ── Exceptions ────────────────────────────────────────────────────────────────

class TrustError(Exception):
    """Base class for all trust boundary errors."""


class AgentNotAllowedError(TrustError):
    """Raised when a caller agent is not permitted to reach a target agent."""

    def __init__(self, caller_url: str, target_url: str) -> None:
        super().__init__(
            f"Agent '{caller_url}' is not allowed to call '{target_url}'."
        )
        self.caller_url = caller_url
        self.target_url = target_url


class SkillNotAllowedError(TrustError):
    """Raised when a caller is not permitted to invoke a specific skill."""

    def __init__(self, caller_url: str, skill_id: str) -> None:
        super().__init__(
            f"Agent '{caller_url}' is not allowed to invoke skill '{skill_id}'."
        )
        self.caller_url = caller_url
        self.skill_id = skill_id


# ── TrustBoundary ─────────────────────────────────────────────────────────────

class TrustBoundary:
    """
    Enforces an explicit permission matrix between agents.

    Default policy: DENY everything unless explicitly allowed.

    Usage:
        trust = TrustBoundary()

        # Allow one agent to call another
        trust.allow("http://orchestrator:8000", "http://research-agent:8001")

        # Allow with skill restriction — caller can only use 'summarise' skill
        trust.allow(
            "http://orchestrator:8000",
            "http://summary-agent:8002",
            skills=["summarise"],
        )

        # Allow an agent to call ANY registered agent (wildcard)
        trust.allow("http://orchestrator:8000", "*")

        # Block an agent completely (overrides any allow rules)
        trust.block("http://untrusted-agent:9999")

        # Check before routing a request
        trust.check(
            caller_url="http://orchestrator:8000",
            target_url="http://research-agent:8001",
            skill_id="web_search",
        )

    Args:
        default_allow: If True, unknown agents are allowed by default.
                       Keep False (default) in production.
    """

    def __init__(self, default_allow: bool = False) -> None:
        self._default_allow = default_allow

        # caller_url → set of target_url patterns it may call
        # "*" in the set means "any target"
        self._allow_rules: dict[str, set[str]] = {}

        # caller_url → target_url → set of allowed skill IDs (None = all skills)
        self._skill_acl: dict[str, dict[str, set[str] | None]] = {}

        # URLs that are unconditionally blocked (as callers OR targets)
        self._blocked: set[str] = set()

    # ── Rule management ───────────────────────────────────────────────────────

    def allow(
        self,
        caller_url: str,
        target_url: str,
        skills: list[str] | None = None,
    ) -> None:
        """
        Permit caller_url to send requests to target_url.

        Args:
            caller_url: The agent initiating the call.
                        Supports fnmatch wildcards e.g. "http://internal-*".
            target_url: The agent receiving the call. Use "*" for any target.
            skills:     Optional list of skill IDs the caller may invoke on
                        the target. If None, all skills are permitted.

        Example:
            # Full access
            trust.allow("http://orchestrator:8000", "http://agent-a:8001")

            # Skill-restricted access
            trust.allow("http://orchestrator:8000", "http://agent-b:8002",
                        skills=["summarise", "translate"])

            # Wildcard caller pattern
            trust.allow("http://internal-*", "*")
        """
        caller = caller_url.rstrip("/")
        target = target_url.rstrip("/") if target_url != "*" else "*"

        # Register the allow rule
        self._allow_rules.setdefault(caller, set()).add(target)

        # Register the skill ACL
        caller_acl = self._skill_acl.setdefault(caller, {})
        if skills is None:
            caller_acl[target] = None        # None = unrestricted
        else:
            existing = caller_acl.get(target)
            if existing is None:
                caller_acl[target] = set(skills)   # first-time restricted
            else:
                existing.update(skills)             # merge with existing

        logger.debug(
            "Trust rule: %s → %s  skills=%s",
            caller, target, skills or "all",
        )

    def block(self, url: str) -> None:
        """
        Unconditionally block an agent — it cannot call or be called.
        Block rules override all allow rules.

        Args:
            url: The agent's base URL to block.
        """
        self._blocked.add(url.rstrip("/"))
        logger.warning("Agent blocked in TrustBoundary: %s", url)

    def unblock(self, url: str) -> None:
        """Remove a URL from the blocked set."""
        self._blocked.discard(url.rstrip("/"))

    def revoke(self, caller_url: str, target_url: str) -> None:
        """
        Remove a previously granted allow rule.

        Args:
            caller_url: The caller whose permission is being revoked.
            target_url: The target they can no longer call.
        """
        caller = caller_url.rstrip("/")
        target = target_url.rstrip("/") if target_url != "*" else "*"

        self._allow_rules.get(caller, set()).discard(target)
        self._skill_acl.get(caller, {}).pop(target, None)
        logger.info("Trust rule revoked: %s → %s", caller, target)

    # ── Enforcement ───────────────────────────────────────────────────────────

    def check(
        self,
        caller_url: str,
        target_url: str,
        skill_id: str | None = None,
    ) -> None:
        """
        Assert that caller_url is permitted to call target_url.
        Optionally checks skill-level permission too.

        Args:
            caller_url: The agent sending the request.
            target_url: The agent receiving the request.
            skill_id:   The skill being invoked (optional).

        Raises:
            AgentNotAllowedError: Caller is not permitted to reach target.
            SkillNotAllowedError: Caller is not permitted to invoke the skill.
        """
        caller = caller_url.rstrip("/")
        target = target_url.rstrip("/")

        # ── 1. Block check (highest priority) ─────────────────────────────────
        if caller in self._blocked:
            raise AgentNotAllowedError(caller_url, target_url)
        if target in self._blocked:
            raise AgentNotAllowedError(caller_url, target_url)

        # ── 2. Allow check ─────────────────────────────────────────────────────
        if not self._is_allowed(caller, target):
            if not self._default_allow:
                raise AgentNotAllowedError(caller_url, target_url)

        # ── 3. Skill-level check (only if skill_id provided) ──────────────────
        if skill_id:
            self._check_skill(caller, target, skill_id, caller_url)

    def is_allowed(
        self,
        caller_url: str,
        target_url: str,
        skill_id: str | None = None,
    ) -> bool:
        """
        Non-raising version of check(). Returns True/False instead of raising.

        Useful for conditional logic where you want to branch rather than
        catch an exception.
        """
        try:
            self.check(caller_url, target_url, skill_id)
            return True
        except TrustError:
            return False

    # ── Introspection ─────────────────────────────────────────────────────────

    def allowed_targets(self, caller_url: str) -> list[str]:
        """Return all target URLs that caller_url is explicitly allowed to call."""
        caller = caller_url.rstrip("/")
        return list(self._allow_rules.get(caller, set()))

    def blocked_agents(self) -> list[str]:
        """Return all currently blocked agent URLs."""
        return list(self._blocked)

    def summary(self) -> dict[str, object]:
        """Return a human-readable summary of the trust configuration."""
        return {
            "default_allow": self._default_allow,
            "blocked":       list(self._blocked),
            "rules":         [
                {
                    "caller":  caller,
                    "targets": list(targets),
                }
                for caller, targets in self._allow_rules.items()
            ],
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _is_allowed(self, caller: str, target: str) -> bool:
        """
        Check if caller has an allow rule that matches target.
        Supports fnmatch wildcard patterns in caller keys and "*" as target.
        """
        for rule_caller, targets in self._allow_rules.items():
            # Match caller against wildcard patterns
            if fnmatch.fnmatch(caller, rule_caller) or caller == rule_caller:
                if "*" in targets or target in targets:
                    return True
                # Check fnmatch patterns in target set
                for t in targets:
                    if fnmatch.fnmatch(target, t):
                        return True
        return False

    def _check_skill(
        self,
        caller: str,
        target: str,
        skill_id: str,
        caller_url: str,
    ) -> None:
        """
        Check skill-level ACL.
        If no ACL exists for this caller→target pair, all skills are allowed.
        """
        for rule_caller, target_acl in self._skill_acl.items():
            if fnmatch.fnmatch(caller, rule_caller) or caller == rule_caller:
                for rule_target, allowed_skills in target_acl.items():
                    target_matches = (
                        rule_target == "*"
                        or rule_target == target
                        or fnmatch.fnmatch(target, rule_target)
                    )
                    if target_matches:
                        # None means all skills allowed
                        if allowed_skills is not None and skill_id not in allowed_skills:
                            raise SkillNotAllowedError(caller_url, skill_id)
                        return
