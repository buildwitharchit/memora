# Tests for core.llm_client -- OpenRouter wrapper.

from unittest.mock import patch, MagicMock
from core.llm_client import LLMClient
from core.config import Settings


class TestLLMClient:
    """Verify LLMClient correctly delegates to the OpenAI SDK."""

    def _make_client(self, settings):
        """Build an LLMClient with a patched OpenAI constructor."""
        with patch("core.llm_client.OpenAI") as MockOpenAI:
            mock_openai_instance = MagicMock()
            MockOpenAI.return_value = mock_openai_instance
            client = LLMClient(settings)
            # Verify OpenRouter base URL and key were passed.
            MockOpenAI.assert_called_once_with(
                base_url="https://openrouter.ai/api/v1",
                api_key=settings.openrouter_api_key,
            )
        return client, mock_openai_instance

    def test_chat_sends_messages_to_primary_model(self, settings):
        client, mock_oai = self._make_client(settings)

        # Arrange: fake completion response.
        mock_choice = MagicMock()
        mock_choice.message.content = "Hello from the model."
        mock_oai.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

        messages = [{"role": "user", "content": "Hi"}]
        result = client.chat(messages)

        assert result == "Hello from the model."
        mock_oai.chat.completions.create.assert_called_once_with(
            model=settings.primary_model,
            messages=messages,
        )

    def test_utility_call_wraps_prompt_as_user_message(self, settings):
        client, mock_oai = self._make_client(settings)

        mock_choice = MagicMock()
        mock_choice.message.content = '{"intent": "ask", "outcome": "answered"}'
        mock_oai.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

        result = client.utility_call("Summarize this.")

        assert result == '{"intent": "ask", "outcome": "answered"}'
        mock_oai.chat.completions.create.assert_called_once_with(
            model=settings.utility_model,
            messages=[{"role": "user", "content": "Summarize this."}],
        )

    def test_chat_uses_primary_model(self, settings):
        client, mock_oai = self._make_client(settings)
        mock_choice = MagicMock()
        mock_choice.message.content = "ok"
        mock_oai.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

        client.chat([{"role": "user", "content": "test"}])

        call_kwargs = mock_oai.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "test/primary-model"

    def test_utility_call_uses_utility_model(self, settings):
        client, mock_oai = self._make_client(settings)
        mock_choice = MagicMock()
        mock_choice.message.content = "ok"
        mock_oai.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

        client.utility_call("extract this")

        call_kwargs = mock_oai.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "test/utility-model"
