"""
Centralized configuration for Memora.

Loads settings from environment variables (with .env file support via python-dotenv).
Uses a singleton pattern so the Settings object is built once and reused.
"""

import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class Settings(BaseModel):
    """All configurable parameters for the Memora system."""

    # Required -- OpenRouter API key for LLM calls.
    openrouter_api_key: str

    # LLM model identifiers (OpenRouter format).
    primary_model: str = "google/gemma-3-12b-it"
    utility_model: str = "google/gemma-3-12b-it"

    # Tier 1 -- Sensory memory: max messages before compression triggers.
    sensory_max_messages: int = 8

    # Tier 2 -- Short-term memory: how many messages to compress per batch.
    compression_batch_size: int = 4

    # Tier 3 -- Semantic memory: number of similar summaries to retrieve.
    semantic_top_k: int = 3

    # Tier 4 -- Episodic memory: recent entries injected into context.
    episodic_snippet_size: int = 5

    # Sentence-transformer model used for embedding summaries.
    embedding_model: str = "all-MiniLM-L6-v2"

    # Root directory for persistent storage (ChromaDB, SQLite).
    store_path: str = "./store"


# Module-level singleton.
_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the global Settings instance, building it from env vars on first call."""
    global _settings
    if _settings is None:
        _settings = Settings(
            openrouter_api_key=os.environ["OPENROUTER_API_KEY"],
            primary_model=os.environ.get("PRIMARY_MODEL", "google/gemma-3-12b-it"),
            utility_model=os.environ.get("UTILITY_MODEL", "google/gemma-3-12b-it"),
            sensory_max_messages=int(os.environ.get("SENSORY_MAX_MESSAGES", "8")),
            compression_batch_size=int(os.environ.get("COMPRESSION_BATCH_SIZE", "4")),
            semantic_top_k=int(os.environ.get("SEMANTIC_TOP_K", "3")),
            episodic_snippet_size=int(os.environ.get("EPISODIC_SNIPPET_SIZE", "5")),
            embedding_model=os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            store_path=os.environ.get("STORE_PATH", "./store"),
        )
    return _settings
