# Shared pytest fixtures for the Memora test suite.
# All fixtures use injected fakes -- no real API calls or model downloads in tests.

import pytest
from unittest.mock import MagicMock

from core.config import Settings
from core.llm_client import LLMClient


@pytest.fixture
def settings():
    """Minimal Settings instance for testing (small buffer sizes, temp store path)."""
    return Settings(
        openrouter_api_key="test-key-not-real",
        primary_model="test/primary-model",
        utility_model="test/utility-model",
        sensory_max_messages=4,
        compression_batch_size=2,
        semantic_top_k=2,
        episodic_snippet_size=3,
        embedding_model="all-MiniLM-L6-v2",
        store_path="/tmp/memora_test",
    )


@pytest.fixture
def mock_llm_client():
    """Fake LLMClient: chat() returns a fixed string, utility_call() returns valid intent JSON."""
    client = MagicMock(spec=LLMClient)
    client.chat.return_value = "Mocked assistant response."
    client.utility_call.return_value = '{"intent": "test intent", "outcome": "test outcome"}'
    return client


@pytest.fixture
def mock_semantic_memory():
    """Fake SemanticMemory for tests that don't need real ChromaDB."""
    mock = MagicMock()
    mock.search.return_value = [
        {
            "id": "fake-id-1",
            "document": "Previously discussed binary trees and DFS traversal.",
            "distance": 0.25,
            "metadata": {"session_id": "session-001"},
        }
    ]
    mock.count.return_value = 1
    mock.add_summary.return_value = "fake-doc-id"
    mock.get_all.return_value = []
    return mock


@pytest.fixture
def mock_episodic_memory():
    """Fake EpisodicMemory for tests that don't need real SQLite."""
    mock = MagicMock()
    mock.get_recent.return_value = []
    mock.count.return_value = 0
    mock.add_entry.return_value = 1
    mock.get_all.return_value = []
    return mock
