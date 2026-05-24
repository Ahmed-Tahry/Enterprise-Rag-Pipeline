"""
src/ingestion/embedder.py
=========================
Embedding pipeline — converts text chunks into dense vector representations.

Supports:
  - HuggingFace sentence-transformers (local, free, production-grade)
  - OpenAI text-embedding-3-small / large (best quality, costs money)

Production tip: BAAI/bge-base-en-v1.5 and BAAI/bge-large-en-v1.5 are the
best open-source English embeddings as of 2024, outperforming older models
like all-MiniLM-L6-v2 on retrieval benchmarks (MTEB).
"""

import os
import numpy as np
from typing import List, Optional
from loguru import logger
from tqdm import tqdm
from src.ingestion.document_loader import Document


class HuggingFaceEmbedder:
    """
    Local embedding using sentence-transformers.
    Runs entirely on your hardware — no API costs.
    
    Recommended models (MTEB leaderboard top performers):
      - BAAI/bge-base-en-v1.5    (768-dim, 109M params, fast)
      - BAAI/bge-large-en-v1.5   (1024-dim, 335M params, best quality)
      - BAAI/bge-m3              (multilingual, 570M params)
    """

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded. Dimension: {self.dimension}")

    def embed_documents(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Embed a list of document texts. Returns (N, dim) float32 array."""
        # BGE models work best with a query prefix — for documents, no prefix needed
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,  # L2 normalise → cosine = dot product
            show_progress_bar=len(texts) > 100,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.
        BGE models use a special instruction prefix for queries:
        'Represent this sentence for searching relevant passages: {query}'
        """
        # BGE instruction prefix improves retrieval quality
        if "bge" in self.model_name.lower():
            query = f"Represent this sentence for searching relevant passages: {query}"
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )
        return embedding[0].astype(np.float32)


class OpenAIEmbedder:
    """
    OpenAI embedding API.
    
    text-embedding-3-small: 1536-dim, fast, cheap ($0.02/1M tokens)
    text-embedding-3-large: 3072-dim, best quality ($0.13/1M tokens)
    
    Both support Matryoshka representation — you can truncate dimensions
    for faster retrieval with minor quality loss.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        dimensions: Optional[int] = None,
    ):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self.model = model
        self.dimensions = dimensions
        self.dimension = dimensions or (1536 if "small" in model else 3072)
        logger.info(f"OpenAI embedder initialised: {model}, dim={self.dimension}")

    def embed_documents(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Embed documents in batches to respect API rate limits."""
        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch = texts[i: i + batch_size]
            kwargs = {"input": batch, "model": self.model}
            if self.dimensions:
                kwargs["dimensions"] = self.dimensions

            response = self.client.embeddings.create(**kwargs)
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        embeddings = np.array(all_embeddings, dtype=np.float32)
        # L2 normalise
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-10)

    def embed_query(self, query: str) -> np.ndarray:
        kwargs = {"input": [query], "model": self.model}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions
        response = self.client.embeddings.create(**kwargs)
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        return embedding / max(np.linalg.norm(embedding), 1e-10)


def get_embedder(provider: str = "huggingface", **kwargs):
    """
    Factory: get an embedder by provider name.
    
    Usage:
        embedder = get_embedder("huggingface", model_name="BAAI/bge-base-en-v1.5")
        embedder = get_embedder("openai", model="text-embedding-3-small")
    """
    providers = {
        "huggingface": HuggingFaceEmbedder,
        "openai":      OpenAIEmbedder,
    }
    if provider not in providers:
        raise ValueError(f"Unknown embedding provider: '{provider}'. Choose from: {list(providers.keys())}")
    return providers[provider](**kwargs)
