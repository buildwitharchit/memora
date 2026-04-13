# Tests for core.context_builder -- LLM prompt assembly.
# Pure logic tests, no mocks needed.

from core.context_builder import ContextBuilder, SYSTEM_PROMPT
from core.episodic_memory import EpisodicEntry


class TestContextBuilder:
    """Verify context message ordering and optional tier injection."""

    def test_system_prompt_always_first(self):
        cb = ContextBuilder()
        messages = cb.build(
            sensory_messages=[{"role": "user", "content": "hi"}],
            semantic_summaries=[],
            episodic_entries=[],
        )
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == SYSTEM_PROMPT

    def test_sensory_messages_at_end(self):
        cb = ContextBuilder()
        messages = cb.build(
            sensory_messages=[
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "second"},
            ],
            semantic_summaries=[],
            episodic_entries=[],
        )
        # System prompt + 2 sensory messages.
        assert len(messages) == 3
        assert messages[-2]["content"] == "first"
        assert messages[-1]["content"] == "second"

    def test_semantic_summaries_injected(self):
        cb = ContextBuilder()
        summaries = [
            {"document": "Previously discussed recursion."},
            {"document": "Covered binary search."},
        ]
        messages = cb.build(
            sensory_messages=[{"role": "user", "content": "hi"}],
            semantic_summaries=summaries,
            episodic_entries=[],
        )
        # System + semantic block + user message.
        assert len(messages) == 3
        assert "[Relevant past context]" in messages[1]["content"]
        assert "recursion" in messages[1]["content"]
        assert "binary search" in messages[1]["content"]

    def test_episodic_entries_injected(self):
        cb = ContextBuilder()
        entries = [
            EpisodicEntry(
                id=1,
                timestamp="2025-04-13T10:00:00",
                user_id="u1",
                session_id="s1",
                intent="asked about trees",
                outcome="explained",
            ),
        ]
        messages = cb.build(
            sensory_messages=[{"role": "user", "content": "hi"}],
            semantic_summaries=[],
            episodic_entries=entries,
        )
        assert len(messages) == 3
        assert "[Recent interaction history]" in messages[1]["content"]
        assert "asked about trees" in messages[1]["content"]

    def test_full_context_ordering(self):
        """All tiers present: system -> semantic -> episodic -> sensory."""
        cb = ContextBuilder()
        messages = cb.build(
            sensory_messages=[{"role": "user", "content": "question"}],
            semantic_summaries=[{"document": "past summary"}],
            episodic_entries=[
                EpisodicEntry(
                    id=1,
                    timestamp="ts",
                    user_id="u",
                    session_id="s",
                    intent="intent",
                    outcome="outcome",
                )
            ],
        )
        # System + semantic + episodic + 1 sensory = 4 messages.
        assert len(messages) == 4
        assert messages[0]["role"] == "system"
        assert "past context" in messages[1]["content"].lower()
        assert "interaction history" in messages[2]["content"].lower()
        assert messages[3]["content"] == "question"

    def test_empty_tiers_produce_minimal_context(self):
        cb = ContextBuilder()
        messages = cb.build(
            sensory_messages=[],
            semantic_summaries=[],
            episodic_entries=[],
        )
        # Only the system prompt.
        assert len(messages) == 1

    def test_custom_system_prompt(self):
        cb = ContextBuilder(system_prompt="Custom prompt here.")
        messages = cb.build([], [], [])
        assert messages[0]["content"] == "Custom prompt here."
