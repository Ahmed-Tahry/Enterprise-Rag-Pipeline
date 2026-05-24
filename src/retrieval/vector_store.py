"""
src/retrieval/vector_store.py
==============================
FAISS-backed vector store for dense retrieval.

Why FAISS?
  - Battle-tested at Meta scale (billions of vectors)
  - IndexFlatIP: exact search (best quality, feasible for < 1M docs)
  - IndexIVFFlat: approximate search (needed for > 1M docs)
  - Zero infrastructure: runs in-process, no external service

For very large scale (> 10M docs): consider Qdrant, Milvus, or Weaviate.
"""

import os
import pickle
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from loguru import logger

from src.ingestion.document_loader import Document


@dataclass
class SearchResult:
    document: Document
    score: float
    rank: int


class FAISSVectorStore:
    """
    FAISS vector store with persistence.
    
    Stores:
      - FAISS index (embeddings)
      - Document list (text + metadata) — pickled alongside index
    
    Index type selection:
      - n < 100k:   IndexFlatIP (exact, fast enough)
      - n > 100k:   IndexIVFFlat (approximate, requires training)
    """

    def __init__(self, embedding_dim: int, index_type: str = "flat"):
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            raise ImportError("Install FAISS: pip install faiss-cpu")

        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = self._create_index(index_type, embedding_dim)
        self.documents: List[Document] = []

    def _create_index(self, index_type: str, dim: int):
        if index_type == "flat":
            # Exact inner product search (= cosine on normalised vectors)
            return self.faiss.IndexFlatIP(dim)
        elif index_type == "ivf":
            # Approximate — 8× faster at 99% recall for large corpora
            quantiser = self.faiss.IndexFlatIP(dim)
            n_lists = 100  # tune: sqrt(n_docs) is a good heuristic
            index = self.faiss.IndexIVFFlat(quantiser, dim, n_lists, self.faiss.METRIC_INNER_PRODUCT)
            return index
        else:
            raise ValueError(f"Unknown index type: {index_type}")

    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        """
        Add documents and their embeddings to the store.
        
        Args:
            documents:  List of Document objects
            embeddings: (N, dim) float32 array of normalised embeddings
        """
        assert len(documents) == len(embeddings), \
            f"Mismatch: {len(documents)} docs vs {len(embeddings)} embeddings"
        assert embeddings.dtype == np.float32, "Embeddings must be float32"

        if self.index_type == "ivf" and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(embeddings)

        self.index.add(embeddings)
        self.documents.extend(documents)
        logger.info(f"Added {len(documents)} documents. Total: {len(self.documents)}")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> List[SearchResult]:
        """
        Search for the top-k most similar documents.
        
        Returns SearchResult objects sorted by descending similarity score.
        """
        if len(self.documents) == 0:
            return []

        query = query_embedding.astype(np.float32).reshape(1, -1)
        k = min(top_k, len(self.documents))

        scores, indices = self.index.search(query, k)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx >= 0:  # FAISS returns -1 for empty slots
                results.append(SearchResult(
                    document=self.documents[idx],
                    score=float(score),
                    rank=rank + 1,
                ))

        return results

    def save(self, path: str):
        """Persist index and documents to disk."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        self.faiss.write_index(self.index, str(save_path / "index.faiss"))

        with open(save_path / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)

        logger.info(f"Vector store saved to {save_path} ({len(self.documents)} docs)")

    @classmethod
    def load(cls, path: str, embedding_dim: int) -> "FAISSVectorStore":
        """Load a previously saved vector store."""
        try:
            import faiss
        except ImportError:
            raise ImportError("Install FAISS: pip install faiss-cpu")

        load_path = Path(path)
        index = faiss.read_index(str(load_path / "index.faiss"))

        with open(load_path / "documents.pkl", "rb") as f:
            documents = pickle.load(f)

        store = cls.__new__(cls)
        store.faiss = faiss
        store.embedding_dim = embedding_dim
        store.index_type = "loaded"
        store.index = index
        store.documents = documents

        logger.info(f"Vector store loaded from {load_path} ({len(documents)} docs)")
        return store

    def __len__(self):
        return len(self.documents)
