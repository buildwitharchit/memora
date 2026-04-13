"""
Context Builder.

Assembles the final list of messages sent to the LLM by combining:
  1. System prompt (always first)
  2. Relevant past summaries from Tier 3 (semantic memory)
  3. Recent episodic entries from Tier 4
  4. Current conversation messages from Tier 1 (sensory memory)
"""

from core.episodic_memory import EpisodicEntry

SYSTEM_PROMPT = (
    "You are Memora, a helpful AI assistant with a multi-tier memory system. "
    "You have access to your conversation history, compressed summaries of past "
    "conversations, and structured logs of previous interactions. Use this context "
    "to provide personalized, informed responses that reference relevant past "
    "discussions when appropriate."
)


class ContextBuilder:
    """Assembles the full LLM prompt from all memory tiers."""

    def __init__(self, system_prompt: str = SYSTEM_PROMPT) -> None:
        self._system_prompt = system_prompt

    def build(
        self,
        sensory_messages: list[dict[str, str]],
        semantic_summaries: list[dict],
        episodic_entries: list[EpisodicEntry],
    ) -> list[dict[str, str]]:
        """Build the ordered message list for the LLM.

        Args:
            sensory_messages: Current conversation (Tier 1).
            semantic_summaries: Retrieved summaries from ChromaDB (Tier 3).
                Each dict must have a "document" key.
            episodic_entries: Recent structured log entries (Tier 4).

        Returns:
            List of role/content message dicts ready for the LLM.
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt}
        ]

        # Inject relevant past summaries (Tier 3).
        if semantic_summaries:
            text = "\n---\n".join(s["document"] for s in semantic_summaries)
            messages.append(
                {"role": "system", "content": f"[Relevant past context]\n{text}"}
            )

        # Inject recent episodic history (Tier 4).
        if episodic_entries:
            lines = [
                f"- [{e.timestamp}] Intent: {e.intent} | Outcome: {e.outcome}"
                for e in episodic_entries
            ]
            text = "\n".join(lines)
            messages.append(
                {"role": "system", "content": f"[Recent interaction history]\n{text}"}
            )

        # Append the live conversation (Tier 1).
        messages.extend(sensory_messages)

        return messages
