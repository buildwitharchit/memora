"""
LLM client for Memora.

Wraps the OpenAI Python SDK pointed at OpenRouter to provide two call patterns:
  - chat(): multi-turn conversation with the primary model
  - utility_call(): single-turn extraction/summarization with the utility model
"""

from openai import OpenAI

from core.config import Settings


class LLMClient:
    """Thin wrapper around OpenRouter's chat completions API."""

    def __init__(self, settings: Settings) -> None:
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openrouter_api_key,
        )
        self._primary_model = settings.primary_model
        self._utility_model = settings.utility_model

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Send a multi-turn conversation to the primary model.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.

        Returns:
            The assistant's response content as a plain string.
        """
        response = self._client.chat.completions.create(
            model=self._primary_model,
            messages=messages,
        )
        return response.choices[0].message.content

    def utility_call(self, prompt: str) -> str:
        """Single-turn call to the utility model for summarization or extraction.

        Args:
            prompt: The full prompt text (sent as a single user message).

        Returns:
            The model's response content as a plain string.
        """
        response = self._client.chat.completions.create(
            model=self._utility_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
