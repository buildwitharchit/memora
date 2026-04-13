# Tests for core.episodic_memory -- Tier 4 SQLite event log.
# Uses in-memory SQLite (via tmp path) for isolation.

import os
import tempfile

import pytest

from core.episodic_memory import EpisodicMemory, EpisodicEntry


@pytest.fixture
def episodic_memory():
    """Create an EpisodicMemory backed by a temp directory (auto-cleaned)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mem = EpisodicMemory(store_path=tmpdir)
        yield mem
        mem.close()


class TestEpisodicMemory:
    """Verify SQLite-backed episodic logging."""

    def test_add_entry_returns_id(self, episodic_memory):
        row_id = episodic_memory.add_entry(
            user_id="u1", session_id="s1", intent="ask about trees", outcome="explained"
        )
        assert isinstance(row_id, int)
        assert row_id >= 1

    def test_count_reflects_inserts(self, episodic_memory):
        assert episodic_memory.count() == 0
        episodic_memory.add_entry("u1", "s1", "intent1", "outcome1")
        episodic_memory.add_entry("u1", "s1", "intent2", "outcome2")
        assert episodic_memory.count() == 2

    def test_get_recent_returns_newest_first(self, episodic_memory):
        episodic_memory.add_entry("u1", "s1", "first", "o1")
        episodic_memory.add_entry("u1", "s1", "second", "o2")
        episodic_memory.add_entry("u1", "s1", "third", "o3")

        recent = episodic_memory.get_recent("s1", limit=2)
        assert len(recent) == 2
        assert recent[0].intent == "third"
        assert recent[1].intent == "second"

    def test_get_recent_filters_by_session(self, episodic_memory):
        episodic_memory.add_entry("u1", "s1", "in s1", "o1")
        episodic_memory.add_entry("u1", "s2", "in s2", "o2")

        recent = episodic_memory.get_recent("s1", limit=10)
        assert len(recent) == 1
        assert recent[0].intent == "in s1"

    def test_get_all_no_filters(self, episodic_memory):
        episodic_memory.add_entry("u1", "s1", "intent1", "o1")
        episodic_memory.add_entry("u2", "s2", "intent2", "o2")

        all_entries = episodic_memory.get_all()
        assert len(all_entries) == 2

    def test_get_all_filter_by_user_id(self, episodic_memory):
        episodic_memory.add_entry("u1", "s1", "intent1", "o1")
        episodic_memory.add_entry("u2", "s1", "intent2", "o2")

        entries = episodic_memory.get_all(user_id="u1")
        assert len(entries) == 1
        assert entries[0].user_id == "u1"

    def test_get_all_filter_by_session_id(self, episodic_memory):
        episodic_memory.add_entry("u1", "s1", "intent1", "o1")
        episodic_memory.add_entry("u1", "s2", "intent2", "o2")

        entries = episodic_memory.get_all(session_id="s2")
        assert len(entries) == 1
        assert entries[0].session_id == "s2"

    def test_get_all_filter_by_intent_pattern(self, episodic_memory):
        episodic_memory.add_entry("u1", "s1", "asked about recursion", "o1")
        episodic_memory.add_entry("u1", "s1", "asked about sorting", "o2")
        episodic_memory.add_entry("u1", "s1", "greeted the bot", "o3")

        entries = episodic_memory.get_all(intent_pattern="asked about")
        assert len(entries) == 2

    def test_get_all_respects_limit(self, episodic_memory):
        for i in range(10):
            episodic_memory.add_entry("u1", "s1", f"intent{i}", f"o{i}")

        entries = episodic_memory.get_all(limit=3)
        assert len(entries) == 3

    def test_reset_clears_all(self, episodic_memory):
        episodic_memory.add_entry("u1", "s1", "intent", "outcome")
        assert episodic_memory.count() == 1
        episodic_memory.reset()
        assert episodic_memory.count() == 0

    def test_entry_has_timestamp(self, episodic_memory):
        episodic_memory.add_entry("u1", "s1", "intent", "outcome")
        entries = episodic_memory.get_recent("s1", limit=1)
        assert entries[0].timestamp is not None
        assert len(entries[0].timestamp) > 0

    def test_episodic_entry_model(self):
        entry = EpisodicEntry(
            id=1,
            timestamp="2025-04-13T10:00:00+00:00",
            user_id="u1",
            session_id="s1",
            intent="test",
            outcome="pass",
        )
        assert entry.intent == "test"
        assert entry.id == 1
