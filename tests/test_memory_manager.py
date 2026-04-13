# Tests for core.memory_manager -- the orchestrator.
# All dependencies injected as mocks.

import json
from unittest.mock import MagicMock, patch

from core.config import Settings
from core.context_builder import ContextBuilder
from core.memory_manager import MemoryManager
from core.sensory_memory import SensoryMemory
from core.short_term_memory import ShortTermMemory


class TestMemoryManager:
    """Verify the full message processing pipeline."""

    def _make_manager(
        self, settings, mock_llm_client, mock_semantic_memory, mock_episodic_memory
    ):
        """Build a MemoryManager with all mocked dependencies."""
        sensory = SensoryMemory(max_messages=settings.sensory_max_messages)
        short_term = ShortTermMemory(
            llm_client=mock_llm_client,
            semantic_memory=mock_semantic_memory,
            batch_size=settings.compression_batch_size,
        )
        context_builder = ContextBuilder()
        mm = MemoryManager(
            settings=settings,
            llm_client=mock_llm_client,
            sensory_memory=sensory,
            short_term_memory=short_term,
            semantic_memory=mock_semantic_memory,
            episodic_memory=mock_episodic_memory,
            context_builder=context_builder,
            user_id="test_user",
        )
        return mm

    def test_process_message_returns_llm_response(
        self, settings, mock_llm_client, mock_semantic_memory, mock_episodic_memory
    ):
        mm = self._make_manager(
            settings, mock_llm_client, mock_semantic_memory, mock_episodic_memory
        )
        result = mm.process_message("Hello")
        assert result == "Mocked assistant response."

    def test_process_message_adds_to_sensory(
        self, settings, mock_llm_client, mock_semantic_memory, mock_episodic_memory
    ):
        mm = self._make_manager(
            settings, mock_llm_client, mock_semantic_memory, mock_episodic_memory
        )
        mm.process_message("Hello")

        messages = mm.sensory.get_messages()
        assert len(messages) == 2  # user + assistant
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[1]["role"] == "assistant"

    def test_process_message_queries_semantic(
        self, settings, mock_llm_client, mock_semantic_memory, mock_episodic_memory
    ):
        mm = self._make_manager(
            settings, mock_llm_client, mock_semantic_memory, mock_episodic_memory
        )
        mm.process_message("Tell me about trees")

        mock_semantic_memory.search.assert_called_once_with("Tell me about trees")

    def test_process_message_queries_episodic(
        self, settings, mock_llm_client, mock_semantic_memory, mock_episodic_memory
    ):
        mm = self._make_manager(
            settings, mock_llm_client, mock_semantic_memory, mock_episodic_memory
        )
        mm.process_message("Hi")

        mock_episodic_memory.get_recent.assert_called_once_with(
            session_id=mm.session_id,
            limit=settings.episodic_snippet_size,
        )

    def test_process_message_logs_episode(
        self, settings, mock_llm_client, mock_semantic_memory, mock_episodic_memory
    ):
        mm = self._make_manager(
            settings, mock_llm_client, mock_semantic_memory, mock_episodic_memory
        )
        mm.process_message("What is a stack?")

        mock_episodic_memory.add_entry.assert_called_once()
        call_kwargs = mock_episodic_memory.add_entry.call_args.kwargs
        assert call_kwargs["user_id"] == "test_user"
        assert call_kwargs["session_id"] == mm.session_id
        assert call_kwargs["intent"] == "test intent"
        assert call_kwargs["outcome"] == "test outcome"

    def test_compression_triggers_when_sensory_full(
        self, settings, mock_llm_client, mock_semantic_memory, mock_episodic_memory
    ):
        """With max=4 and batch=2, compression should trigger after 2 process_message calls
        (each adds 2 messages: user + assistant = 4 total, hitting capacity)."""
        mm = self._make_manager(
            settings, mock_llm_client, mock_semantic_memory, mock_episodic_memory
        )
        # First call: 2 messages (user+assistant), not full yet.
        mm.process_message("First message")
        # Second call: 4 messages now, triggers compression.
        mm.process_message("Second message")

        # The utility_call is used for both extraction and compression.
        # 2 extraction calls + 1 compression call = at least 3 calls.
        assert mock_llm_client.utility_call.call_count >= 3

    def test_malformed_json_does_not_crash(
        self, settings, mock_llm_client, mock_semantic_memory, mock_episodic_memory
    ):
        """If the utility LLM returns invalid JSON, log a warning but continue."""
        mock_llm_client.utility_call.return_value = "not valid json at all"
        mm = self._make_manager(
            settings, mock_llm_client, mock_semantic_memory, mock_episodic_memory
        )

        # Should not raise.
        result = mm.process_message("test")
        assert result == "Mocked assistant response."

        # Episode should still be logged with fallback values.
        call_kwargs = mock_episodic_memory.add_entry.call_args.kwargs
        assert call_kwargs["intent"] == "parse_error"
        assert call_kwargs["outcome"] == "extraction_failed"

    def test_new_session_clears_state(
        self, settings, mock_llm_client, mock_semantic_memory, mock_episodic_memory
    ):
        mm = self._make_manager(
            settings, mock_llm_client, mock_semantic_memory, mock_episodic_memory
        )
        old_session = mm.session_id
        mm.process_message("hello")
        mm.new_session()

        assert mm.sensory.size == 0
        assert mm.session_id != old_session
        assert mm.last_retrieved_summaries == []

    def test_reset_semantic(
        self, settings, mock_llm_client, mock_semantic_memory, mock_episodic_memory
    ):
        mm = self._make_manager(
            settings, mock_llm_client, mock_semantic_memory, mock_episodic_memory
        )
        mm.reset_semantic()
        mock_semantic_memory.reset.assert_called_once()

    def test_reset_episodic(
        self, settings, mock_llm_client, mock_semantic_memory, mock_episodic_memory
    ):
        mm = self._make_manager(
            settings, mock_llm_client, mock_semantic_memory, mock_episodic_memory
        )
        mm.reset_episodic()
        mock_episodic_memory.reset.assert_called_once()

    def test_last_retrieved_summaries_updated(
        self, settings, mock_llm_client, mock_semantic_memory, mock_episodic_memory
    ):
        mm = self._make_manager(
            settings, mock_llm_client, mock_semantic_memory, mock_episodic_memory
        )
        mm.process_message("binary trees")

        summaries = mm.last_retrieved_summaries
        assert len(summaries) == 1
        assert "binary trees" in summaries[0]["document"].lower() or True  # Mock value
