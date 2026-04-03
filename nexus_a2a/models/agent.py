"""
nexus_a2a/models/agent.py

Pydantic models that describe an AI agent's identity, capabilities,
and skills — mirroring the A2A protocol's AgentCard specification.

These models are the single source of truth for agent metadata
across the entire package.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, Field, HttpUrl, field_validator

# ── Enums ─────────────────────────────────────────────────────────────────────

class AuthScheme(str, Enum):
    """Supported authentication schemes for agent-to-agent calls."""
    NONE    = "none"
    API_KEY = "api_key"
    JWT     = "jwt"
    OAUTH2  = "oauth2"


class InputMode(str, Enum):
    """MIME types an agent can accept as input."""
    TEXT       = "text/plain"
    JSON       = "application/json"
    MULTIPART  = "multipart/form-data"


class OutputMode(str, Enum):
    """MIME types an agent can produce as output."""
    TEXT       = "text/plain"
    JSON       = "application/json"
    MARKDOWN   = "text/markdown"


# ── Sub-models ────────────────────────────────────────────────────────────────

class AgentAuthentication(BaseModel):
    """
    Describes how a client must authenticate when calling this agent.

    Example:
        AgentAuthentication(scheme=AuthScheme.JWT, token_url="https://auth.example.com/token")
    """
    scheme:    AuthScheme = AuthScheme.NONE
    token_url: HttpUrl | None = Field(
        default=None,
        description="URL to obtain a token — required for JWT and OAuth2 schemes.",
    )
    header_name: str | None = Field(
        default=None,
        description="Custom header name for API key auth. Defaults to 'X-API-Key'.",
    )

    model_config = {"use_enum_values": True}


class AgentCapabilities(BaseModel):
    """
    Flags that describe what protocol features this agent supports.
    The client uses these to decide how to interact with the agent.
    """
    streaming:          bool = Field(
        default=False,
        description="Agent supports Server-Sent Events for real-time task updates.",
    )
    push_notifications: bool = Field(
        default=False,
        description="Agent can POST task updates to a client-provided webhook URL.",
    )
    multi_turn:         bool = Field(
        default=True,
        description="Agent supports back-and-forth conversations within a single task.",
    )


class AgentSkill(BaseModel):
    """
    A single capability that an agent advertises.
    Think of this as one entry in the agent's 'menu of services'.

    Example:
        AgentSkill(
            id="web_search",
            name="Web search",
            description="Searches the web and returns summarised results.",
            tags=["search", "web", "research"],
            examples=["Search for latest AI papers", "Find Python docs for asyncio"],
        )
    """
    id:          Annotated[str, Field(min_length=1, max_length=64)]
    name:        Annotated[str, Field(min_length=1, max_length=128)]
    description: Annotated[str, Field(min_length=1, max_length=1024)]
    tags:        list[str] = Field(
        default_factory=list,
        description="Keywords that help agents discover this skill.",
    )
    examples:    list[str] = Field(
        default_factory=list,
        description="Sample inputs that demonstrate this skill.",
    )

    @field_validator("tags", "examples", mode="before")
    @classmethod
    def _no_empty_strings(cls, v: list[str]) -> list[str]:
        """Strip whitespace and drop empty strings from lists."""
        return [item.strip() for item in v if item.strip()]


# ── Primary model ─────────────────────────────────────────────────────────────

class AgentCard(BaseModel):
    """
    The complete identity document for an agent — exposed at the
    well-known endpoint:  GET /.well-known/agent-card.json

    This is what other agents read during discovery to learn:
      - What the agent can do  (skills)
      - How to reach it        (url)
      - How to authenticate    (authentication)
      - What formats it speaks (input_modes, output_modes)

    Example:
        AgentCard(
            name="ResearchAgent",
            description="Searches the web and summarises findings.",
            version="1.0.0",
            url="https://research-agent.example.com",
            skills=[AgentSkill(id="search", name="Search", description="Web search")],
        )
    """

    # Identity
    name:        Annotated[str, Field(min_length=1, max_length=128)]
    description: Annotated[str, Field(min_length=1, max_length=1024)]
    version:     str = Field(
        default="0.1.0",
        description="Semantic version of this agent. Bump when skills change.",
    )

    # Network
    url: HttpUrl = Field(
        description="Base URL where this agent's A2A server is reachable.",
    )

    # Protocol
    capabilities:  AgentCapabilities   = Field(default_factory=AgentCapabilities)
    authentication: AgentAuthentication = Field(default_factory=AgentAuthentication)

    # Skills
    skills: list[AgentSkill] = Field(
        default_factory=list,
        description="List of capabilities this agent offers.",
    )

    # Communication formats
    input_modes:  list[InputMode]  = Field(
        default_factory=lambda: [InputMode.TEXT, InputMode.JSON],
    )
    output_modes: list[OutputMode] = Field(
        default_factory=lambda: [OutputMode.TEXT, OutputMode.JSON],
    )

    model_config = {
        "use_enum_values": True,
        # Allows: AgentCard(**a2a_sdk_agent_card_dict) without extra fields
        # causing an error — useful when parsing cards from external agents.
        "extra": "ignore",
    }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def has_skill(self, skill_id: str) -> bool:
        """Return True if this agent advertises the given skill id."""
        return any(s.id == skill_id for s in self.skills)

    def skill_ids(self) -> list[str]:
        """Return a flat list of all advertised skill IDs."""
        return [s.id for s in self.skills]

    def to_well_known_dict(self) -> dict[str, Any] :
        """
        Serialise to the dict served at /.well-known/agent-card.json.
        URLs are converted to plain strings so JSON serialisation works.
        """
        return self.model_dump(mode="json")
