import re
from typing import List

from src.retrieval.vector_store import SearchResult
from src.ingestion.document_loader import Document


class ContextCompressor:
    """
    Extract only query-relevant sentences from retrieved chunks.

    Reduces prompt noise, token cost, and hallucination risk by dropping
    irrelevant boilerplate (headers, navigation, repeated phrases) while
    preserving the information needed to answer the query.
    """

    def __init__(self, embedder, max_chars: int = 2000, min_sentences: int = 1):
        self.embedder = embedder
        self.max_chars = max_chars
        self.min_sentences = min_sentences

    def compress(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        if not results:
            return results

        query_emb = self.embedder.embed_query(query)
        compressed = []
        char_budget = self.max_chars

        for result in results:
            sentences = self._split_sentences(result.document.page_content)
            if not sentences:
                continue
            if len(sentences) <= self.min_sentences:
                chunk = sentences[0]
                compressed.append(
                    SearchResult(
                        document=Document(
                            page_content=chunk,
                            metadata=dict(result.document.metadata),
                        ),
                        score=result.score,
                        rank=result.rank,
                    )
                )
                char_budget -= len(chunk)
                continue

            sent_embs = self.embedder.embed_documents(sentences)
            sims = sent_embs @ query_emb
            scored = list(zip(sentences, sims, range(len(sentences))))

            kept_sentences = []
            used = min(self.min_sentences, len(scored))
            top_by_score = sorted(scored, key=lambda x: x[1], reverse=True)
            kept_indices = {idx for _, _, idx in top_by_score[:used]}
            for s, sim, idx in scored:
                if idx in kept_indices:
                    kept_sentences.append((s, sim, idx))
                elif char_budget > 0 and sim > 0.3:
                    if len(s) <= char_budget:
                        kept_indices.add(idx)
                        kept_sentences.append((s, sim, idx))
                        char_budget -= len(s)

            kept_sentences.sort(key=lambda x: x[2])
            compressed_text = " ".join(s[0] for s in kept_sentences)
            if compressed_text.strip():
                compressed.append(
                    SearchResult(
                        document=Document(
                            page_content=compressed_text,
                            metadata=dict(result.document.metadata),
                        ),
                        score=result.score,
                        rank=result.rank,
                    )
                )

        return compressed[: len(results)]

    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
