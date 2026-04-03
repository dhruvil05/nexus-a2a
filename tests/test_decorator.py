"""
tests/test_decorator.py

Tests for the @agent decorator and get_card() helper.
Run with:  uv run pytest tests/test_decorator.py -v
"""

import pytest

from nexus_a2a import agent, get_card, AgentCard, AgentSkill, AuthScheme
from nexus_a2a.models.task import Task


# ══════════════════════════════════════════════════════════════════════════════
# Basic decoration
# ══════════════════════════════════════════════════════════════════════════════

class TestAgentDecoratorBasic:
    def test_decorator_with_arguments(self):
        @agent(
            name="SearchAgent",
            description="Searches the web.",
            url="http://localhost:8001",
        )
        class SearchAgent:
            async def run(self, task: Task) -> str:
                return "result"

        card = get_card(SearchAgent)
        assert isinstance(card, AgentCard)
        assert card.name == "SearchAgent"
        assert card.description == "Searches the web."

    def test_decorator_without_parentheses_uses_class_name(self):
        @agent
        class MySimpleAgent:
            """Does simple things."""
            async def run(self, task: Task) -> str:
                return "ok"

        card = get_card(MySimpleAgent)
        assert card.name == "MySimpleAgent"
        assert "simple" in card.description.lower()  # from docstring

    def test_get_agent_card_classmethod(self):
        @agent(name="CardAgent", description="Test.", url="http://localhost:8002")
        class CardAgent:
            async def run(self, task: Task) -> str:
                return ""

        # Both access paths should return the same card
        assert CardAgent.get_agent_card() is get_card(CardAgent)  # type: ignore[attr-defined]


# ══════════════════════════════════════════════════════════════════════════════
# Skills
# ══════════════════════════════════════════════════════════════════════════════

class TestAgentDecoratorSkills:
    def test_skills_as_dicts(self):
        @agent(
            name="SkilledAgent",
            description="Has skills.",
            url="http://localhost:8003",
            skills=[
                {"id": "s1", "name": "Skill 1", "description": "First skill"},
                {"id": "s2", "name": "Skill 2", "description": "Second skill"},
            ],
        )
        class SkilledAgent:
            async def run(self, task: Task) -> str:
                return ""

        card = get_card(SkilledAgent)
        assert len(card.skills) == 2
        assert card.has_skill("s1") is True
        assert card.has_skill("s2") is True

    def test_skills_as_agentskill_objects(self):
        skill = AgentSkill(id="search", name="Search", description="Searches.")

        @agent(
            name="AgentWithSkillObj",
            description="Uses AgentSkill directly.",
            url="http://localhost:8004",
            skills=[skill],
        )
        class AgentWithSkillObj:
            async def run(self, task: Task) -> str:
                return ""

        card = get_card(AgentWithSkillObj)
        assert card.skills[0].id == "search"

    def test_no_skills_is_valid(self):
        @agent(name="NoSkillAgent", description="Bare agent.", url="http://localhost:8005")
        class NoSkillAgent:
            async def run(self, task: Task) -> str:
                return ""

        assert get_card(NoSkillAgent).skills == []


# ══════════════════════════════════════════════════════════════════════════════
# Capabilities
# ══════════════════════════════════════════════════════════════════════════════

class TestAgentDecoratorCapabilities:
    def test_streaming_flag(self):
        @agent(name="StreamAgent", description="Streams.", url="http://localhost:8006",
               streaming=True)
        class StreamAgent:
            async def run(self, task: Task) -> str:
                return ""

        assert get_card(StreamAgent).capabilities.streaming is True

    def test_push_notifications_flag(self):
        @agent(name="PushAgent", description="Pushes.", url="http://localhost:8007",
               push_notifications=True)
        class PushAgent:
            async def run(self, task: Task) -> str:
                return ""

        assert get_card(PushAgent).capabilities.push_notifications is True

    def test_defaults_are_false(self):
        @agent(name="DefaultCapsAgent", description="D.", url="http://localhost:8008")
        class DefaultCapsAgent:
            async def run(self, task: Task) -> str:
                return ""

        caps = get_card(DefaultCapsAgent).capabilities
        assert caps.streaming is False
        assert caps.push_notifications is False

    def test_auth_scheme(self):
        @agent(name="SecureAgent", description="Secure.", url="http://localhost:8009",
               auth_scheme=AuthScheme.JWT)
        class SecureAgent:
            async def run(self, task: Task) -> str:
                return ""

        assert get_card(SecureAgent).authentication.scheme == AuthScheme.JWT


# ══════════════════════════════════════════════════════════════════════════════
# Error cases
# ══════════════════════════════════════════════════════════════════════════════

class TestAgentDecoratorErrors:
    def test_missing_async_run_raises(self):
        with pytest.raises(TypeError, match="async def run"):
            @agent(name="BadAgent", description="No run.", url="http://localhost:8010")
            class BadAgent:
                def run(self, task: Task) -> str:  # sync, not async
                    return ""

    def test_no_run_at_all_raises(self):
        with pytest.raises(TypeError, match="async def run"):
            @agent(name="NoRunAgent", description="No run.", url="http://localhost:8011")
            class NoRunAgent:
                pass

    def test_get_card_on_undecorated_class_raises(self):
        class PlainClass:
            async def run(self, task: Task) -> str:
                return ""

        with pytest.raises(TypeError, match="not decorated with @agent"):
            get_card(PlainClass)

    def test_decorator_fallback_description_from_classname(self):
        # No description, no docstring — uses "A2A agent: ClassName"
        @agent(url="http://localhost:8012")
        class NameOnlyAgent:
            async def run(self, task: Task) -> str:
                return ""

        card = get_card(NameOnlyAgent)
        assert "NameOnlyAgent" in card.description


# ══════════════════════════════════════════════════════════════════════════════
# Class still works normally after decoration
# ══════════════════════════════════════════════════════════════════════════════

class TestDecoratedClassBehaviour:
    def test_class_is_still_instantiable(self):
        @agent(name="LiveAgent", description="Works normally.", url="http://localhost:8013")
        class LiveAgent:
            def __init__(self):
                self.ready = True

            async def run(self, task: Task) -> str:
                return "live"

        instance = LiveAgent()
        assert instance.ready is True

    @pytest.mark.asyncio
    async def test_async_run_still_executes(self):
        @agent(name="AsyncAgent", description="Async run.", url="http://localhost:8014")
        class AsyncAgent:
            async def run(self, task: Task) -> str:
                return "async result"

        instance = AsyncAgent()
        msg = await instance.run(None)   # type: ignore[arg-type]
        assert msg == "async result"