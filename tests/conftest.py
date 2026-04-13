# Shared pytest fixtures for the Memora test suite.
# All fixtures use injected fakes — no real API calls or model downloads in tests.
import pytest
from unittest.mock import MagicMock
from core.llm_client import LLMClient


@pytest.fixture
def mock_llm_client():
    """Fake LLMClient: chat() returns a fixed string, utility_call() returns valid intent JSON."""
    client = MagicMock(spec=LLMClient)
    client.chat.return_value = "Mocked assistant response."
    client.utility_call.return_value = '{"intent": "test intent", "outcome": "test outcome"}'
    return client
