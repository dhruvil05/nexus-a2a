"""
nexus_a2a/decorators.py

The @agent decorator — the primary entry point for developers.
Wraps a class and automatically generates an AgentCard from its metadata.

Usage:
    @agent(
        name="ResearchAgent",
        description="Searches the web and summarises results.",
        skills=[{"id": "search", "name": "Web search", "description": "..."}],
        url="http://localhost:8001",
    )
    class ResearchAgent:
        async def run(self, task: Task) -> str:
            return "result"
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any, Callable, TypeVar, overload, cast

from nexus_a2a.models.agent import (
    AgentAuthentication,
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    AuthScheme,
)

# The decorated class can be any type
C = TypeVar("C", bound=type)
F = TypeVar("F", bound=Callable[..., Any])

# Attribute name we attach to the class so other layers can read the card
_AGENT_CARD_ATTR = "__nexus_agent_card__"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_skill(raw: dict[str, Any] | AgentSkill) -> AgentSkill:
    """Accept either an AgentSkill instance or a plain dict and return AgentSkill."""
    if isinstance(raw, AgentSkill):
        return raw
    return AgentSkill(**raw)


def _has_async_run(cls: type) -> bool:
    """Return True if the class has an async method named 'run'."""
    method = getattr(cls, "run", None)
    return method is not None and inspect.iscoroutinefunction(method)


# ── Public decorator ──────────────────────────────────────────────────────────

@overload
def agent(cls: C) -> C: ...  # called as @agent (no parentheses)

@overload
def agent(
    *,
    name: str | None = None,
    description: str | None = None,
    version: str = "0.1.0",
    url: str = "http://localhost:8000",
    skills: list[dict[str, Any] | AgentSkill] | None = None,
    streaming: bool = False,
    push_notifications: bool = False,
    auth_scheme: AuthScheme = AuthScheme.NONE,
) -> Callable[[C], C]: ...   # called as @agent(...) with arguments


def agent(
    cls: C | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    version: str = "0.1.0",
    url: str = "http://localhost:8000",
    skills: list[dict[str, Any] | AgentSkill] | None = None,
    streaming: bool = False,
    push_notifications: bool = False,
    auth_scheme: AuthScheme = AuthScheme.NONE,
) -> C | Callable[[C], C]:
    """
    Class decorator that registers a class as an A2A-compatible agent
    and automatically generates its AgentCard.

    Can be used in two ways:

    1. With arguments (recommended):
        @agent(
            name="Summariser",
            description="Summarises long documents.",
            skills=[{"id": "summarise", "name": "Summarise", "description": "..."}],
            url="http://localhost:8001",
        )
        class SummariserAgent:
            async def run(self, task): ...

    2. Without arguments (uses class name and docstring as defaults):
        @agent
        class SummariserAgent:
            \"\"\"Summarises long documents.\"\"\"
            async def run(self, task): ...

    After decoration, the class gains:
        - SummariserAgent.__nexus_agent_card__  →  AgentCard instance
        - SummariserAgent.get_agent_card()      →  returns the AgentCard

    Raises:
        TypeError:  If the class does not have an async `run` method.
        ValueError: If name or description cannot be resolved.
    """

    def decorator(klass: C) -> C:
        # ── Resolve metadata ──────────────────────────────────────────────────
        resolved_name = name or klass.__name__
        resolved_desc = (
            description
            or (inspect.getdoc(klass))           # use docstring if available
            or f"A2A agent: {resolved_name}"     # fallback
        )

        if not resolved_desc.strip():
            raise ValueError(
                f"@agent on '{resolved_name}': provide a description= "
                "or add a docstring to the class."
            )

        # ── Validate the class has async run() ───────────────────────────────
        if not _has_async_run(klass):
            raise TypeError(
                f"@agent on '{resolved_name}': the class must define "
                "'async def run(self, task)' method."
            )

        # ── Build skills list ─────────────────────────────────────────────────
        built_skills: list[AgentSkill] = [
            _build_skill(s) for s in (skills or [])
        ]

        # ── Build AgentCard ───────────────────────────────────────────────────
        card = AgentCard(
            name=resolved_name,
            description=resolved_desc,
            version=version,
            url=url,  # type: ignore[arg-type]  # Pydantic coerces str → HttpUrl
            skills=built_skills,
            capabilities=AgentCapabilities(
                streaming=streaming,
                push_notifications=push_notifications,
            ),
            authentication=AgentAuthentication(scheme=auth_scheme),
        )

        # ── Attach card to the class ──────────────────────────────────────────
        setattr(klass, _AGENT_CARD_ATTR, card)

        # ── Add helper class method ───────────────────────────────────────────
        @classmethod  # type: ignore[misc]
        def get_agent_card(klass_inner: type) -> AgentCard:
            """Return the AgentCard generated by @agent."""
            return cast(AgentCard, getattr(klass_inner, _AGENT_CARD_ATTR))

        klass.get_agent_card = get_agent_card  # type: ignore[attr-defined]

        # ── Preserve original class metadata ──────────────────────────────────
        functools.update_wrapper(klass, klass, updated=[])

        return klass

    # Handle both @agent and @agent(...) call styles
    if cls is not None:
        # Called as @agent with no parentheses
        return decorator(cls)

    # Called as @agent(...) with arguments
    return decorator


# ── Public helper ─────────────────────────────────────────────────────────────

def get_card(agent_cls: type) -> AgentCard:
    """
    Retrieve the AgentCard from a decorated agent class.

    Args:
        agent_cls: A class decorated with @agent.

    Returns:
        The AgentCard attached to the class.

    Raises:
        TypeError: If the class was not decorated with @agent.

    Example:
        card = get_card(ResearchAgent)
        print(card.name)
    """
    card = getattr(agent_cls, _AGENT_CARD_ATTR, None)
    if card is None:
        raise TypeError(
            f"'{agent_cls.__name__}' is not decorated with @agent. "
            "Apply @agent before calling get_card()."
        )
    return cast(AgentCard, card)
