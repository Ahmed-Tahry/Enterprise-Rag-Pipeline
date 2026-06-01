"""
src/retrieval/hybrid_retriever.py
==================================
Hybrid retrieval: BM25 (sparse) + Dense (embeddings) fused via RRF.

Why hybrid?
  Dense alone fails on:  exact keywords, product codes, names, rare terms
  BM25 alone fails on:   paraphrases, synonyms, semantic similarity

  Hybrid gets both. Reciprocal Rank Fusion (RRF) combines ranked lists
  without needing calibrated scores — robust across query types.

RRF formula:
  score(doc) = Σ 1 / (k + rank_i)
  where k=60 (canonical default, Cormack 2009)
"""

import math
import re
from collections import defaultdict
from typing import List, Dict, Optional
from loguru import logger

from src.ingestion.document_loader import Document
from src.retrieval.vector_store import FAISSVectorStore, SearchResult


# ---------------------------------------------------------------------------
# Standalone RRF utility (shared by retriever and RAGChain)
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    result_lists: List[List[SearchResult]],
    top_k: int,
    rrf_k: int = 60,
) -> List[SearchResult]:
    """Combine multiple ranked lists using Reciprocal Rank Fusion."""
    doc_scores: Dict[str, float] = defaultdict(float)
    doc_map: Dict[str, Document] = {}
    for results in result_lists:
        for rank, result in enumerate(results, 1):
            doc_id = result.document.page_content
            doc_scores[doc_id] += 1.0 / (rrf_k + rank)
            doc_map[doc_id] = result.document
    sorted_ids = sorted(doc_scores, key=lambda x: doc_scores[x], reverse=True)[:top_k]
    return [
        SearchResult(document=doc_map[doc_id], score=doc_scores[doc_id], rank=rank + 1)
        for rank, doc_id in enumerate(sorted_ids)
    ]


# ---------------------------------------------------------------------------
# BM25 Retriever
# ---------------------------------------------------------------------------


class BM25Retriever:
    """
    BM25 sparse retrieval — the gold standard keyword-based retrieval.

    BM25 advantages over TF-IDF:
      - Term saturation: repeated words have diminishing returns
      - Length normalisation: long docs aren't unfairly boosted

    k1=1.5 (term saturation), b=0.75 (length normalisation) are
    empirically validated defaults across many benchmarks.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[Document] = []
        self.corpus_tokens: List[List[str]] = []
        self.idf: Dict[str, float] = {}
        self.avg_doc_len: float = 0.0
        self.N: int = 0

    def fit(self, documents: List[Document]):
        """Build the BM25 index from documents."""
        self.documents = documents
        self.N = len(documents)
        self.corpus_tokens = [self._tokenise(doc.page_content) for doc in documents]

        total_len = sum(len(t) for t in self.corpus_tokens)
        self.avg_doc_len = total_len / max(self.N, 1)

        # Document frequency for each term
        df: Dict[str, int] = defaultdict(int)
        for tokens in self.corpus_tokens:
            for term in set(tokens):
                df[term] += 1

        # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        for term, freq in df.items():
            self.idf[term] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1)

        logger.info(f"BM25 index built: {self.N} docs, {len(self.idf)} unique terms")

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Return top-k documents by BM25 score."""
        if not self.documents:
            return []

        query_tokens = self._tokenise(query)
        scores = [0.0] * self.N

        for term in query_tokens:
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for i, tokens in enumerate(self.corpus_tokens):
                tf = tokens.count(term)
                if tf == 0:
                    continue
                dl = len(tokens)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * dl / self.avg_doc_len
                )
                scores[i] += idf * (numerator / denominator)

        # Get top-k indices by score
        top_indices = sorted(range(self.N), key=lambda i: scores[i], reverse=True)[
            :top_k
        ]

        return [
            SearchResult(document=self.documents[i], score=scores[i], rank=rank + 1)
            for rank, i in enumerate(top_indices)
            if scores[i] > 0
        ]

    def _tokenise(self, text: str) -> List[str]:
        """Lowercase + alphanumeric tokenisation."""
        return re.findall(r"\b[a-z0-9]+\b", text.lower())


# ---------------------------------------------------------------------------
# Cross-Encoder Reranker
# ---------------------------------------------------------------------------


class CrossEncoderReranker:
    """
    Cross-encoder reranker: takes (query, document) pairs and scores them jointly.

    Why rerank?
    Bi-encoders (like sentence-transformers) encode query and document separately.
    This is fast but misses fine-grained interaction signals.

    Cross-encoders read (query + document) together → much higher quality scores.
    Too slow to run on the full corpus → run on top-20 candidates from initial retrieval.

    Classic 2-stage pipeline:
      Retrieval (top 20) → Reranker (top 5) → LLM
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        logger.info(f"Loading reranker: {model_name}")
        self.model = CrossEncoder(model_name)
        logger.info("Reranker loaded")

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        Re-score and rerank results using cross-encoder.

        Returns top_k results sorted by reranker score (descending).
        """
        if not results:
            return []

        pairs = [(query, r.document.page_content) for r in results]
        scores = self.model.predict(pairs)

        reranked = sorted(
            zip(scores, results),
            key=lambda x: x[0],
            reverse=True,
        )

        return [
            SearchResult(
                document=r.document,
                score=float(score),
                rank=rank + 1,
            )
            for rank, (score, r) in enumerate(reranked[:top_k])
        ]


# ---------------------------------------------------------------------------
# Hybrid Retriever (BM25 + Dense + RRF)
# ---------------------------------------------------------------------------


class HybridRetriever:
    """
    Combines BM25 and dense retrieval via Reciprocal Rank Fusion.

    Pipeline:
      Query
        ├── BM25 → top_k results (ranked by BM25 score)
        ├── Dense → top_k results (ranked by cosine similarity)
        └── RRF fusion → merged ranking
              └── [optional] Cross-encoder reranker → final top_k
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        bm25: BM25Retriever,
        embedder,
        reranker: Optional[CrossEncoderReranker] = None,
        rrf_k: int = 60,
    ):
        self.vector_store = vector_store
        self.bm25 = bm25
        self.embedder = embedder
        self.reranker = reranker
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 5,
        retrieval_k: int = 20,
    ) -> List[SearchResult]:
        """
        Full hybrid retrieval pipeline.

        Args:
            query:       Natural language query
            top_k:       Final number of results to return
            retrieval_k: Candidates retrieved before reranking

        Returns:
            List of SearchResult, sorted by relevance.
        """
        # 1. Dense retrieval
        query_emb = self.embedder.embed_query(query)
        dense_results = self.vector_store.search(query_emb, top_k=retrieval_k)

        # 2. BM25 retrieval
        bm25_results = self.bm25.search(query, top_k=retrieval_k)

        # 3. RRF fusion
        fused_results = self._reciprocal_rank_fusion(
            [dense_results, bm25_results],
            top_k=retrieval_k,
        )

        # 4. Cross-encoder reranking (if available)
        if self.reranker and fused_results:
            final_results = self.reranker.rerank(query, fused_results, top_k=top_k)
        else:
            final_results = fused_results[:top_k]
            for rank, r in enumerate(final_results):
                r.rank = rank + 1

        return final_results

    def search_multi(
        self,
        queries: List[str],
        top_k: int = 5,
        retrieval_k: int = 20,
    ) -> List[SearchResult]:
        """
        Search with multiple queries and fuse results via RRF + single reranker pass.

        Designed for query rewriting: each rewritten query densifies different
        semantic aspects, BM25 runs per-query for lexical diversity, and a single
        reranker pass at the end avoids unnecessary computation.
        """
        if len(queries) <= 1:
            return self.search(
                queries[0] if queries else "", top_k=top_k, retrieval_k=retrieval_k
            )

        all_results: List[List[SearchResult]] = []
        for q in queries:
            q_emb = self.embedder.embed_query(q)
            dense = self.vector_store.search(q_emb, top_k=retrieval_k)
            bm25 = self.bm25.search(q, top_k=retrieval_k)
            fused = reciprocal_rank_fusion([dense, bm25], top_k=retrieval_k)
            all_results.append(fused)

        fused = reciprocal_rank_fusion(all_results, top_k=retrieval_k)

        if self.reranker and fused:
            final = self.reranker.rerank(queries[0], fused, top_k=top_k)
        else:
            final = fused[:top_k]
            for rank, r in enumerate(final):
                r.rank = rank + 1

        return final

    def _reciprocal_rank_fusion(
        self,
        result_lists: List[List[SearchResult]],
        top_k: int,
    ) -> List[SearchResult]:
        return reciprocal_rank_fusion(result_lists, top_k, rrf_k=self.rrf_k)
