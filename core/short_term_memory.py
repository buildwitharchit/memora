"""
Tier 2 -- Short-Term Memory (Compression Buffer).

When the sensory buffer (Tier 1) overflows, the oldest messages are sent here.
This tier summarizes them via a utility LLM call and stores the compressed
summary in semantic memory (Tier 3).

Tracks compression metrics (original vs. summary token counts) for the UI.
"""

from loguru import logger

from core.llm_client import LLMClient
from core.semantic_memory import SemanticMemory

SUMMARIZATION_PROMPT = """Summarize the following conversation messages into a concise paragraph.
Preserve: key facts discussed, decisions made, questions asked, and anything the user struggled with.
Do not add information not present in the messages.

Messages:
{messages_text}

Summary:"""


class ShortTermMemory:
    """Tier 2: Compresses batches of overflow messages from Tier 1 and stores in Tier 3."""

    def __init__(
        self,
        llm_client: LLMClient,
        semantic_memory: SemanticMemory,
        batch_size: int = 4,
    ) -> None:
        self._llm = llm_client
        self._semantic = semantic_memory
        self._batch_size = batch_size
        # Running log of compression events for the memory inspector UI.
        self._compression_log: list[dict] = []

    def compress_and_store(
        self, messages: list[dict[str, str]], session_id: str
    ) -> str:
        """Summarize a batch of messages via LLM, store the result in semantic memory.

        Args:
            messages: The raw messages to compress (list of role/content dicts).
            session_id: Current session identifier for metadata tagging.

        Returns:
            The generated summary text.
        """
        messages_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages
        )
        original_tokens = len(messages_text.split())

        prompt = SUMMARIZATION_PROMPT.format(messages_text=messages_text)
        summary = self._llm.utility_call(prompt)

        summary_tokens = len(summary.split())
        ratio = original_tokens / max(summary_tokens, 1)

        self._compression_log.append(
            {
                "original_tokens": original_tokens,
                "summary_tokens": summary_tokens,
                "ratio": round(ratio, 2),
            }
        )

        self._semantic.add_summary(
            summary,
            metadata={
                "session_id": session_id,
                "message_count": len(messages),
            },
        )

        logger.info(
            "Compressed {} messages ({} -> {} tokens, ratio={:.2f})",
            len(messages),
            original_tokens,
            summary_tokens,
            ratio,
        )
        return summary

    @property
    def batch_size(self) -> int:
        """Number of messages compressed per batch."""
        return self._batch_size

    @property
    def compression_log(self) -> list[dict]:
        """List of compression events: {original_tokens, summary_tokens, ratio}."""
        return list(self._compression_log)

    def reset(self) -> None:
        """Clear the compression log (does not affect Tier 3 storage)."""
        self._compression_log.clear()
