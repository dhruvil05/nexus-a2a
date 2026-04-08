"""
nexus_a2a/core/registry.py

AgentRegistry — discovers, stores, and health-checks remote agents.

Responsibilities:
  - Register agents manually (by URL) or from a known AgentCard.
  - Auto-fetch AgentCards from remote servers via their well-known endpoint.
  - Cache cards with a configurable TTL so we don't hit remote servers constantly.
  - Health-check registered agents and mark them available/unavailable.
  - Look up agents by name, URL, or skill ID.

This is the component that makes agent discovery automatic — developers
register a URL once, and the registry handles the rest.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from nexus_a2a.models.agent import AgentCard
from nexus_a2a.transport.http_client import (
    A2AHttpClient,
    AgentCardFetchError,
    AgentUnreachableError,
)

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class RegistryEntry:
    """
    A single record in the registry — one remote agent.

    Fields:
        card:         The agent's AgentCard (capabilities, skills, URL).
        healthy:      True if the last health check succeeded.
        last_seen_at: Unix timestamp of the last successful contact.
        fetch_at:     Unix timestamp when the card was fetched (for TTL tracking).
    """
    card:         AgentCard
    healthy:      bool  = True
    last_seen_at: float = field(default_factory=time.monotonic)
    fetch_at:     float = field(default_factory=time.monotonic)

    def is_stale(self, ttl_seconds: float) -> bool:
        """Return True if the cached card is older than the TTL."""
        return (time.monotonic() - self.fetch_at) > ttl_seconds


# ── Registry ──────────────────────────────────────────────────────────────────

class AgentRegistry:
    """
    Central directory of all known remote agents in the network.

    Usage:
        registry = AgentRegistry()

        # Register by URL — fetches the AgentCard automatically
        card = await registry.register_url("http://research-agent:8001")

        # Or register with a card you already have
        await registry.register_card(card)

        # Look up agents
        agent = registry.get_by_name("ResearchAgent")
        agents = registry.find_by_skill("web_search")
        all_agents = registry.list_healthy()

        # Periodically refresh stale cards
        await registry.refresh_stale()

    Args:
        card_ttl_seconds:      How long before a cached AgentCard is considered
                               stale and needs re-fetching. Default: 5 minutes.
        health_check_timeout:  Seconds to wait when pinging an agent for health.
    """

    def __init__(
        self,
        card_ttl_seconds: float = 300.0,
        health_check_timeout: float = 5.0,
    ) -> None:
        self._ttl     = card_ttl_seconds
        self._hc_timeout = health_check_timeout
        # url → RegistryEntry
        self._entries: dict[str, RegistryEntry] = {}
        self._lock    = asyncio.Lock()

    # ── Registration ──────────────────────────────────────────────────────────

    async def register_url(self, url: str) -> AgentCard:
        """
        Fetch an agent's AgentCard from its well-known endpoint and register it.

        Args:
            url: Base URL of the remote A2A server (e.g. "http://agent:8001").

        Returns:
            The fetched and registered AgentCard.

        Raises:
            AgentCardFetchError:   Card endpoint returned an invalid response.
            AgentUnreachableError: Server did not respond.
        """
        normalised = url.rstrip("/")

        async with A2AHttpClient(normalised) as client:
            card = await client.fetch_agent_card()

        await self._store_entry(normalised, card)
        logger.info("Registered agent '%s' from %s", card.name, normalised)
        return card

    async def register_card(self, card: AgentCard) -> None:
        """
        Register an agent from an AgentCard you already have.
        Useful when the card was obtained out-of-band (config file, service mesh, etc.).

        Args:
            card: A fully populated AgentCard.
        """
        url = str(card.url).rstrip("/")
        await self._store_entry(url, card)
        logger.info("Registered agent '%s' from provided card", card.name)

    async def unregister(self, url: str) -> None:
        """
        Remove an agent from the registry.

        Args:
            url: The agent's base URL (as used during registration).
        """
        normalised = url.rstrip("/")
        async with self._lock:
            removed = self._entries.pop(normalised, None)
        if removed:
            logger.info("Unregistered agent at %s", normalised)

    # ── Lookup ────────────────────────────────────────────────────────────────

    def get_by_url(self, url: str) -> AgentCard | None:
        """Return the AgentCard for a given URL, or None if not registered."""
        entry = self._entries.get(url.rstrip("/"))
        return entry.card if entry else None

    def get_by_name(self, name: str) -> AgentCard | None:
        """
        Return the first registered AgentCard whose name matches exactly.
        Returns None if no match is found.
        """
        for entry in self._entries.values():
            if entry.card.name == name:
                return entry.card
        return None

    def find_by_skill(self, skill_id: str) -> list[AgentCard]:
        """
        Return all healthy agents that advertise the given skill ID.

        Args:
            skill_id: The skill to search for (matches AgentSkill.id).

        Returns:
            List of matching AgentCards (may be empty).
        """
        return [
            entry.card
            for entry in self._entries.values()
            if entry.healthy and entry.card.has_skill(skill_id)
        ]

    def list_all(self) -> list[AgentCard]:
        """Return AgentCards for all registered agents (healthy or not)."""
        return [e.card for e in self._entries.values()]

    def list_healthy(self) -> list[AgentCard]:
        """Return AgentCards for all agents that are currently marked healthy."""
        return [e.card for e in self._entries.values() if e.healthy]

    def is_healthy(self, url: str) -> bool:
        """Return True if the agent at the given URL is marked healthy."""
        entry = self._entries.get(url.rstrip("/"))
        return entry.healthy if entry else False

    # ── Health checks ─────────────────────────────────────────────────────────

    async def check_health(self, url: str) -> bool:
        """
        Ping a registered agent's well-known endpoint and update its health flag.

        Args:
            url: The agent's base URL.

        Returns:
            True if the agent responded successfully, False otherwise.
        """
        normalised = url.rstrip("/")
        try:
            async with A2AHttpClient(normalised, timeout=self._hc_timeout) as client:
                await client.fetch_agent_card()
            healthy = True
        except (AgentUnreachableError, AgentCardFetchError):
            healthy = False

        async with self._lock:
            if normalised in self._entries:
                self._entries[normalised].healthy = healthy
                if healthy:
                    self._entries[normalised].last_seen_at = time.monotonic()

        logger.debug("Health check %s → %s", normalised, "OK" if healthy else "FAIL")
        return healthy

    async def check_all_health(self) -> dict[str, bool]:
        """
        Ping every registered agent concurrently and return a url→healthy map.

        Returns:
            Dict mapping each agent URL to its health status.
        """
        urls = list(self._entries.keys())
        results = await asyncio.gather(
            *(self.check_health(url) for url in urls),
            return_exceptions=True,
        )
        return {
            url: (res if isinstance(res, bool) else False)
            for url, res in zip(urls, results, strict=False)
        }

    # ── Cache refresh ─────────────────────────────────────────────────────────

    async def refresh_stale(self) -> list[str]:
        """
        Re-fetch AgentCards for any registered agents whose cached card
        has exceeded the TTL.

        Returns:
            List of URLs that were successfully refreshed.
        """
        stale_urls = [
            url for url, entry in self._entries.items()
            if entry.is_stale(self._ttl)
        ]

        refreshed: list[str] = []
        for url in stale_urls:
            try:
                await self.register_url(url)
                refreshed.append(url)
                logger.debug("Refreshed stale card for %s", url)
            except (AgentUnreachableError, AgentCardFetchError) as exc:
                logger.warning("Failed to refresh card for %s: %s", url, exc)
                async with self._lock:
                    if url in self._entries:
                        self._entries[url].healthy = False

        return refreshed

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def summary(self) -> dict[str, object]:
        """
        Return a human-readable summary of the registry state.
        Useful for logging and debugging.
        """
        return {
            "total":   len(self._entries),
            "healthy": sum(1 for e in self._entries.values() if e.healthy),
            "agents":  [
                {
                    "name":    e.card.name,
                    "url":     str(e.card.url),
                    "skills":  e.card.skill_ids(),
                    "healthy": e.healthy,
                }
                for e in self._entries.values()
            ],
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _store_entry(self, url: str, card: AgentCard) -> None:
        """Create or overwrite a registry entry under the given URL."""
        async with self._lock:
            self._entries[url] = RegistryEntry(card=card)
