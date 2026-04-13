# Tests for core.short_term_memory -- Tier 2 compression buffer.

from core.short_term_memory import ShortTermMemory


class TestShortTermMemory:
    """Verify compression flow and metric tracking."""

    def test_compress_and_store_calls_llm(self, mock_llm_client, mock_semantic_memory):
        stm = ShortTermMemory(
            llm_client=mock_llm_client,
            semantic_memory=mock_semantic_memory,
            batch_size=2,
        )
        messages = [
            {"role": "user", "content": "What is a binary tree?"},
            {"role": "assistant", "content": "A binary tree is a data structure..."},
        ]

        summary = stm.compress_and_store(messages, session_id="s1")

        # LLM should have been called with the summarization prompt.
        mock_llm_client.utility_call.assert_called_once()
        prompt_arg = mock_llm_client.utility_call.call_args[0][0]
        assert "What is a binary tree?" in prompt_arg
        assert "binary tree is a data structure" in prompt_arg

    def test_compress_and_store_stores_in_semantic(
        self, mock_llm_client, mock_semantic_memory
    ):
        mock_llm_client.utility_call.return_value = "Summary of the conversation."
        stm = ShortTermMemory(
            llm_client=mock_llm_client,
            semantic_memory=mock_semantic_memory,
            batch_size=2,
        )
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        stm.compress_and_store(messages, session_id="s1")

        mock_semantic_memory.add_summary.assert_called_once()
        call_args = mock_semantic_memory.add_summary.call_args
        assert call_args[0][0] == "Summary of the conversation."
        assert call_args[1]["metadata"]["session_id"] == "s1"
        assert call_args[1]["metadata"]["message_count"] == 2

    def test_compress_and_store_returns_summary(
        self, mock_llm_client, mock_semantic_memory
    ):
        mock_llm_client.utility_call.return_value = "Compressed summary."
        stm = ShortTermMemory(
            llm_client=mock_llm_client,
            semantic_memory=mock_semantic_memory,
            batch_size=2,
        )
        result = stm.compress_and_store(
            [{"role": "user", "content": "test"}], session_id="s1"
        )
        assert result == "Compressed summary."

    def test_compression_log_tracks_events(
        self, mock_llm_client, mock_semantic_memory
    ):
        mock_llm_client.utility_call.return_value = "short summary"
        stm = ShortTermMemory(
            llm_client=mock_llm_client,
            semantic_memory=mock_semantic_memory,
            batch_size=2,
        )

        assert stm.compression_log == []

        stm.compress_and_store(
            [
                {"role": "user", "content": "a longer message with several words"},
                {"role": "assistant", "content": "another message with content"},
            ],
            session_id="s1",
        )

        log = stm.compression_log
        assert len(log) == 1
        assert log[0]["original_tokens"] > log[0]["summary_tokens"]
        assert log[0]["ratio"] > 1.0

    def test_batch_size_property(self, mock_llm_client, mock_semantic_memory):
        stm = ShortTermMemory(
            llm_client=mock_llm_client,
            semantic_memory=mock_semantic_memory,
            batch_size=6,
        )
        assert stm.batch_size == 6

    def test_reset_clears_log(self, mock_llm_client, mock_semantic_memory):
        mock_llm_client.utility_call.return_value = "summary"
        stm = ShortTermMemory(
            llm_client=mock_llm_client,
            semantic_memory=mock_semantic_memory,
            batch_size=2,
        )
        stm.compress_and_store(
            [{"role": "user", "content": "msg"}], session_id="s1"
        )
        assert len(stm.compression_log) == 1
        stm.reset()
        assert stm.compression_log == []
