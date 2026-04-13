# Tests for core.semantic_memory -- Tier 3 ChromaDB vector store.
# All external deps (ChromaDB, SentenceTransformer) are patched.

from unittest.mock import patch, MagicMock

from core.semantic_memory import SemanticMemory


class TestSemanticMemory:
    """Verify SemanticMemory delegates correctly to ChromaDB and sentence-transformers."""

    def _make_memory(self):
        """Build a SemanticMemory with patched external dependencies."""
        with (
            patch("core.semantic_memory.SentenceTransformer") as MockST,
            patch("core.semantic_memory.chromadb") as mock_chromadb,
        ):
            mock_embedder = MagicMock()
            # Return a fake 384-dim vector.
            mock_embedder.encode.return_value = MagicMock(
                tolist=MagicMock(return_value=[0.1] * 384)
            )
            MockST.return_value = mock_embedder

            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chromadb.PersistentClient.return_value = mock_client

            mem = SemanticMemory(
                store_path="/tmp/test",
                embedding_model_name="test-model",
                top_k=3,
            )

        return mem, mock_collection, mock_embedder

    def test_add_summary_calls_collection_add(self):
        mem, collection, embedder = self._make_memory()
        doc_id = mem.add_summary("Test summary", metadata={"session_id": "s1"})

        assert isinstance(doc_id, str)
        embedder.encode.assert_called_once_with("Test summary")
        collection.add.assert_called_once()
        call_kwargs = collection.add.call_args.kwargs
        assert call_kwargs["documents"] == ["Test summary"]
        assert call_kwargs["metadatas"] == [{"session_id": "s1"}]

    def test_add_summary_default_metadata(self):
        mem, collection, _ = self._make_memory()
        mem.add_summary("No metadata")

        call_kwargs = collection.add.call_args.kwargs
        assert call_kwargs["metadatas"] == [{}]

    def test_search_returns_formatted_results(self):
        mem, collection, embedder = self._make_memory()
        collection.count.return_value = 2
        collection.query.return_value = {
            "ids": [["id1", "id2"]],
            "documents": [["summary 1", "summary 2"]],
            "distances": [[0.1, 0.3]],
            "metadatas": [[{"session_id": "s1"}, {"session_id": "s2"}]],
        }

        results = mem.search("binary trees")

        assert len(results) == 2
        assert results[0]["id"] == "id1"
        assert results[0]["document"] == "summary 1"
        assert results[0]["distance"] == 0.1
        assert results[1]["metadata"] == {"session_id": "s2"}

    def test_search_empty_collection_returns_empty(self):
        mem, collection, _ = self._make_memory()
        collection.count.return_value = 0

        results = mem.search("anything")

        assert results == []
        collection.query.assert_not_called()

    def test_search_custom_top_k(self):
        mem, collection, _ = self._make_memory()
        collection.count.return_value = 10
        collection.query.return_value = {
            "ids": [["id1"]],
            "documents": [["doc1"]],
            "distances": [[0.1]],
            "metadatas": [[{}]],
        }

        mem.search("query", top_k=1)

        call_kwargs = collection.query.call_args.kwargs
        assert call_kwargs["n_results"] == 1

    def test_count_delegates(self):
        mem, collection, _ = self._make_memory()
        collection.count.return_value = 42
        assert mem.count() == 42

    def test_get_all_empty(self):
        mem, collection, _ = self._make_memory()
        collection.count.return_value = 0
        assert mem.get_all() == []

    def test_get_all_returns_entries(self):
        mem, collection, _ = self._make_memory()
        collection.count.return_value = 1
        collection.get.return_value = {
            "ids": ["id1"],
            "documents": ["summary text"],
            "metadatas": [{"session_id": "s1"}],
        }

        results = mem.get_all()
        assert len(results) == 1
        assert results[0]["document"] == "summary text"

    def test_reset_deletes_and_recreates(self):
        mem, collection, _ = self._make_memory()
        mem.reset()

        mem._client.delete_collection.assert_called_once_with(
            SemanticMemory.COLLECTION_NAME
        )
        # get_or_create_collection called once in __init__ and once in reset.
        assert mem._client.get_or_create_collection.call_count == 2
