"""
Tier 3 -- Semantic Memory.

Stores compressed conversation summaries as vectors in ChromaDB and retrieves
the most relevant ones via cosine similarity search. Embeddings are generated
by sentence-transformers (default: all-MiniLM-L6-v2).

Persistence: ChromaDB stores its data under <store_path>/chroma_db/.
"""

import uuid
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


class SemanticMemory:
    """Tier 3: Vector store of compressed conversation summaries."""

    COLLECTION_NAME = "conversation_summaries"

    def __init__(
        self,
        store_path: str,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 3,
    ) -> None:
        self._top_k = top_k
        self._embedder = SentenceTransformer(embedding_model_name)
        db_path = str(Path(store_path) / "chroma_db")
        self._client = chromadb.PersistentClient(path=db_path)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_summary(self, summary: str, metadata: dict | None = None) -> str:
        """Embed and store a summary. Returns the generated document ID."""
        doc_id = str(uuid.uuid4())
        embedding = self._embedder.encode(summary).tolist()
        self._collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[summary],
            metadatas=[metadata or {}],
        )
        return doc_id

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """Return the top-K most similar summaries to the query.

        Each result dict contains: id, document, distance, metadata.
        """
        k = top_k or self._top_k
        if self._collection.count() == 0:
            return []
        embedding = self._embedder.encode(query).tolist()
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(k, self._collection.count()),
        )
        return [
            {
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "distance": results["distances"][0][i],
                "metadata": results["metadatas"][0][i],
            }
            for i in range(len(results["ids"][0]))
        ]

    def count(self) -> int:
        """Number of stored summaries."""
        return self._collection.count()

    def get_all(self) -> list[dict]:
        """Return every stored summary (for browsing UI). No distance scores."""
        if self._collection.count() == 0:
            return []
        results = self._collection.get()
        return [
            {
                "id": results["ids"][i],
                "document": results["documents"][i],
                "metadata": results["metadatas"][i],
            }
            for i in range(len(results["ids"]))
        ]

    def reset(self) -> None:
        """Delete all stored summaries and recreate the collection."""
        self._client.delete_collection(self.COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
