"""
Memory Manager -- the central orchestrator.

Wires all four memory tiers together and implements the full message flow:
  1. Add user message to Tier 1 (sensory)
  2. Query Tier 3 (semantic) for relevant past summaries
  3. Query Tier 4 (episodic) for recent interaction log
  4. Build context and send to LLM
  5. Add assistant response to Tier 1
  6. Extract intent/outcome and log to Tier 4
  7. If Tier 1 is full, pop oldest batch and compress via Tier 2

All dependencies are injectable for testing. When not provided, real
implementations are constructed from settings.
"""

import json
import uuid

from loguru import logger

from core.config import Settings, get_settings
from core.context_builder import ContextBuilder
from core.episodic_memory import EpisodicMemory
from core.llm_client import LLMClient
from core.semantic_memory import SemanticMemory
from core.sensory_memory import SensoryMemory
from core.short_term_memory import ShortTermMemory

EXTRACTION_PROMPT = """Analyze this conversation exchange and extract the user's intent and the outcome.
Return ONLY valid JSON with no additional text: {{"intent": "...", "outcome": "..."}}

User message: {user_message}
Assistant response: {assistant_response}

JSON:"""


class MemoryManager:
    """Orchestrates all four memory tiers and the LLM client."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm_client: LLMClient | None = None,
        sensory_memory: SensoryMemory | None = None,
        short_term_memory: ShortTermMemory | None = None,
        semantic_memory: SemanticMemory | None = None,
        episodic_memory: EpisodicMemory | None = None,
        context_builder: ContextBuilder | None = None,
        user_id: str = "default_user",
    ) -> None:
        self._settings = settings or get_settings()
        self._user_id = user_id
        self._session_id = str(uuid.uuid4())

        # Build real implementations for any deps not injected.
        self._semantic = semantic_memory or SemanticMemory(
            store_path=self._settings.store_path,
            embedding_model_name=self._settings.embedding_model,
            top_k=self._settings.semantic_top_k,
        )
        self._episodic = episodic_memory or EpisodicMemory(
            store_path=self._settings.store_path,
        )
        self._llm = llm_client or LLMClient(self._settings)
        self._sensory = sensory_memory or SensoryMemory(
            max_messages=self._settings.sensory_max_messages,
        )
        self._short_term = short_term_memory or ShortTermMemory(
            llm_client=self._llm,
            semantic_memory=self._semantic,
            batch_size=self._settings.compression_batch_size,
        )
        self._context_builder = context_builder or ContextBuilder()

        # Tracks the last set of retrieved summaries (for the memory inspector UI).
        self._last_retrieved_summaries: list[dict] = []
        self._last_episodic_entries = []

    def process_message(self, user_input: str) -> str:
        """Full message-processing pipeline.

        Args:
            user_input: The user's new message text.

        Returns:
            The assistant's response string.
        """
        # Step 1: Add user message to sensory buffer.
        self._sensory.add({"role": "user", "content": user_input})

        # Step 2: Retrieve relevant past summaries from semantic memory.
        self._last_retrieved_summaries = self._semantic.search(user_input)

        # Step 3: Retrieve recent episodic entries.
        self._last_episodic_entries = self._episodic.get_recent(
            session_id=self._session_id,
            limit=self._settings.episodic_snippet_size,
        )

        # Step 4: Build context and call the LLM.
        context = self._context_builder.build(
            sensory_messages=self._sensory.get_messages(),
            semantic_summaries=self._last_retrieved_summaries,
            episodic_entries=self._last_episodic_entries,
        )
        response = self._llm.chat(context)

        # Step 5: Add assistant response to sensory buffer.
        self._sensory.add({"role": "assistant", "content": response})

        # Step 6: Extract intent/outcome and log to episodic memory.
        self._extract_and_log_episode(user_input, response)

        # Step 7: If sensory buffer is full, compress oldest batch.
        if self._sensory.is_full():
            oldest = self._sensory.pop_oldest(self._short_term.batch_size)
            if oldest:
                self._short_term.compress_and_store(oldest, self._session_id)

        return response

    def _extract_and_log_episode(self, user_input: str, response: str) -> None:
        """Use the utility LLM to extract intent/outcome, then log to episodic memory.

        Handles malformed JSON gracefully -- logs a warning but does not crash.
        """
        try:
            prompt = EXTRACTION_PROMPT.format(
                user_message=user_input, assistant_response=response
            )
            raw = self._llm.utility_call(prompt)
            data = json.loads(raw)
            intent = data.get("intent", "unknown")
            outcome = data.get("outcome", "unknown")
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Failed to extract intent/outcome: {}", exc)
            intent = "parse_error"
            outcome = "extraction_failed"

        self._episodic.add_entry(
            user_id=self._user_id,
            session_id=self._session_id,
            intent=intent,
            outcome=outcome,
        )

    # -- Property accessors for UI components --

    @property
    def sensory(self) -> SensoryMemory:
        return self._sensory

    @property
    def short_term(self) -> ShortTermMemory:
        return self._short_term

    @property
    def semantic(self) -> SemanticMemory:
        return self._semantic

    @property
    def episodic(self) -> EpisodicMemory:
        return self._episodic

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def user_id(self) -> str:
        return self._user_id

    @property
    def last_retrieved_summaries(self) -> list[dict]:
        """Summaries retrieved during the most recent process_message call."""
        return self._last_retrieved_summaries

    @property
    def last_episodic_entries(self) -> list:
        """Episodic entries retrieved during the most recent process_message call."""
        return self._last_episodic_entries

    # -- Reset methods --

    def reset_short_term(self) -> None:
        """Clear the compression log."""
        self._short_term.reset()

    def reset_semantic(self) -> None:
        """Delete all stored summaries from ChromaDB."""
        self._semantic.reset()

    def reset_episodic(self) -> None:
        """Delete all episodic log entries from SQLite."""
        self._episodic.reset()

    def new_session(self) -> None:
        """Start a fresh session: clear the sensory buffer and generate a new session ID."""
        self._sensory.clear()
        self._short_term.reset()
        self._session_id = str(uuid.uuid4())
        self._last_retrieved_summaries = []
        self._last_episodic_entries = []
