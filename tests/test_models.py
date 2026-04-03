"""
tests/test_models.py

Tests for AgentCard, AgentSkill, Task, Message, Artifact, and Part models.
Run with:  uv run pytest tests/test_models.py -v
"""

import pytest
from pydantic import ValidationError

from nexus_a2a import (
    AgentCard,
    AgentSkill,
    AgentCapabilities,
    Artifact,
    Message,
    MessageRole,
    Part,
    PartType,
    Task,
    TaskState,
)


# ══════════════════════════════════════════════════════════════════════════════
# AgentSkill
# ══════════════════════════════════════════════════════════════════════════════

class TestAgentSkill:
    def test_valid_skill(self):
        skill = AgentSkill(
            id="search",
            name="Web search",
            description="Searches the web.",
        )
        assert skill.id == "search"
        assert skill.tags == []
        assert skill.examples == []

    def test_strips_whitespace_from_tags(self):
        skill = AgentSkill(
            id="s1", name="S", description="D",
            tags=["  ai  ", "", "  ml"],
        )
        assert skill.tags == ["ai", "ml"]  # empty string dropped, whitespace stripped

    def test_id_too_long_raises(self):
        with pytest.raises(ValidationError):
            AgentSkill(id="x" * 65, name="N", description="D")


# ══════════════════════════════════════════════════════════════════════════════
# AgentCard
# ══════════════════════════════════════════════════════════════════════════════

class TestAgentCard:
    def _make_card(self, **kwargs) -> AgentCard:
        defaults = dict(
            name="TestAgent",
            description="A test agent.",
            url="http://localhost:8000",
        )
        return AgentCard(**{**defaults, **kwargs})

    def test_minimal_card(self):
        card = self._make_card()
        assert card.name == "TestAgent"
        assert card.version == "0.1.0"
        assert card.capabilities.streaming is False

    def test_has_skill_true(self):
        skill = AgentSkill(id="search", name="Search", description="Searches.")
        card = self._make_card(skills=[skill])
        assert card.has_skill("search") is True

    def test_has_skill_false(self):
        card = self._make_card()
        assert card.has_skill("nonexistent") is False

    def test_skill_ids(self):
        skills = [
            AgentSkill(id="s1", name="S1", description="D"),
            AgentSkill(id="s2", name="S2", description="D"),
        ]
        card = self._make_card(skills=skills)
        assert card.skill_ids() == ["s1", "s2"]

    def test_to_well_known_dict_is_serialisable(self):
        import json
        card = self._make_card()
        d = card.to_well_known_dict()
        # Must be JSON-serialisable without errors
        json.dumps(d)
        assert d["name"] == "TestAgent"

    def test_extra_fields_ignored(self):
        # Simulates parsing an external agent's card that has unknown keys
        card = AgentCard(
            name="X", description="D", url="http://localhost:9000",
            unknown_future_field="should be ignored",  # type: ignore[call-arg]
        )
        assert card.name == "X"

    def test_invalid_url_raises(self):
        with pytest.raises(ValidationError):
            self._make_card(url="not-a-url")


# ══════════════════════════════════════════════════════════════════════════════
# Part & Message
# ══════════════════════════════════════════════════════════════════════════════

class TestPart:
    def test_text_part(self):
        p = Part(type=PartType.TEXT, content="hello")
        assert p.content == "hello"

    def test_json_part(self):
        p = Part(type=PartType.JSON, content={"key": "value"})
        assert p.content["key"] == "value"

    def test_none_content_raises(self):
        with pytest.raises(ValidationError):
            Part(type=PartType.TEXT, content=None)


class TestMessage:
    def test_user_text_shortcut(self):
        msg = Message.user_text("Hello")
        assert msg.role == MessageRole.USER
        assert msg.text() == "Hello"

    def test_agent_text_shortcut(self):
        msg = Message.agent_text("World")
        assert msg.role == MessageRole.AGENT
        assert msg.text() == "World"

    def test_text_joins_multiple_parts(self):
        msg = Message(
            role=MessageRole.USER,
            parts=[
                Part(type=PartType.TEXT, content="Hello"),
                Part(type=PartType.JSON, content={"x": 1}),  # not included in text()
                Part(type=PartType.TEXT, content="World"),
            ],
        )
        assert msg.text() == "Hello World"

    def test_empty_parts_raises(self):
        with pytest.raises(ValidationError):
            Message(role=MessageRole.USER, parts=[])


# ══════════════════════════════════════════════════════════════════════════════
# Task
# ══════════════════════════════════════════════════════════════════════════════

class TestTask:
    def _make_task(self) -> Task:
        return Task.create(
            initial_message=Message.user_text("Do something"),
            skill_id="test_skill",
        )

    def test_create_starts_submitted(self):
        task = self._make_task()
        assert task.state == TaskState.SUBMITTED
        assert len(task.history) == 1

    def test_valid_transition_submitted_to_working(self):
        task = self._make_task()
        task.transition(TaskState.WORKING)
        assert task.state == TaskState.WORKING

    def test_valid_transition_working_to_completed(self):
        task = self._make_task()
        task.transition(TaskState.WORKING)
        task.transition(TaskState.COMPLETED)
        assert task.is_done() is True

    def test_invalid_transition_raises(self):
        task = self._make_task()
        with pytest.raises(ValueError, match="Cannot transition"):
            task.transition(TaskState.COMPLETED)  # SUBMITTED → COMPLETED not allowed

    def test_failed_without_error_raises(self):
        task = self._make_task()
        task.transition(TaskState.WORKING)
        with pytest.raises(ValueError, match="error message"):
            task.transition(TaskState.FAILED)   # must pass error=

    def test_failed_with_error(self):
        task = self._make_task()
        task.transition(TaskState.WORKING)
        task.transition(TaskState.FAILED, error="Something went wrong")
        assert task.error == "Something went wrong"
        assert task.is_done() is True

    def test_terminal_state_blocks_further_transitions(self):
        task = self._make_task()
        task.transition(TaskState.WORKING)
        task.transition(TaskState.COMPLETED)
        with pytest.raises(ValueError):
            task.transition(TaskState.FAILED, error="too late")

    def test_add_message(self):
        task = self._make_task()
        task.add_message(Message.agent_text("Working on it..."))
        assert len(task.history) == 2
        assert task.latest_message().role == MessageRole.AGENT  # type: ignore[union-attr]

    def test_add_artifact(self):
        task = self._make_task()
        art = Artifact(
            name="result",
            parts=[Part(type=PartType.TEXT, content="Final answer")],
        )
        task.add_artifact(art)
        assert len(task.artifacts) == 1
        assert task.artifacts[0].name == "result"

    def test_updated_at_changes_on_transition(self):
        task = self._make_task()
        before = task.updated_at
        task.transition(TaskState.WORKING)
        assert task.updated_at >= before