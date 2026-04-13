# Tests for core.config -- Settings construction and defaults.

import os
import pytest
from core.config import Settings


class TestSettings:
    """Verify Settings validates required fields and applies defaults."""

    def test_required_api_key(self, settings):
        assert settings.openrouter_api_key == "test-key-not-real"

    def test_default_models(self):
        s = Settings(openrouter_api_key="k")
        assert s.primary_model == "google/gemma-3-12b-it"
        assert s.utility_model == "google/gemma-3-12b-it"

    def test_default_memory_tuning(self):
        s = Settings(openrouter_api_key="k")
        assert s.sensory_max_messages == 8
        assert s.compression_batch_size == 4
        assert s.semantic_top_k == 3
        assert s.episodic_snippet_size == 5

    def test_default_paths(self):
        s = Settings(openrouter_api_key="k")
        assert s.embedding_model == "all-MiniLM-L6-v2"
        assert s.store_path == "./store"

    def test_missing_api_key_raises(self):
        with pytest.raises(Exception):
            Settings()  # openrouter_api_key is required

    def test_custom_values_applied(self):
        s = Settings(
            openrouter_api_key="k",
            sensory_max_messages=16,
            compression_batch_size=8,
            store_path="/custom/path",
        )
        assert s.sensory_max_messages == 16
        assert s.compression_batch_size == 8
        assert s.store_path == "/custom/path"
