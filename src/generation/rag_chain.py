"""
src/generation/rag_chain.py
============================
Core RAG generation chain — takes a query, retrieves context, generates answer.

Key design decisions:
  1. Prompt places context BEFORE the question (better attention to context)
  2. Explicit instruction to cite sources reduces hallucination
  3. Explicit "I don't know" instruction prevents confident wrong answers
  4. Confidence scoring on the answer for downstream filtering
"""

import os
import re
import json
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.retrieval.hybrid_retriever import HybridRetriever, SearchResult


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class RAGResponse:
    """Full response from the RAG pipeline."""
    query:            str
    answer:           str
    sources:          List[Dict[str, Any]]   # [{filename, page, score, excerpt}, ...]
    retrieved_chunks: List[SearchResult]
    latency_ms:       float
    model:            str
    has_answer:       bool = True            # False if model said "I don't know"

    def to_dict(self) -> Dict:
        return {
            "query":    self.query,
            "answer":   self.answer,
            "sources":  self.sources,
            "latency_ms": round(self.latency_ms, 1),
            "model":    self.model,
            "has_answer": self.has_answer,
        }


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a precise, helpful assistant that answers questions based ONLY on the provided context.

Rules:
1. Answer using ONLY the information in the context below.
2. If the context does not contain enough information to answer, respond with exactly: "I don't have enough information to answer this question."
3. For every factual claim, cite the source using [Source N].
4. Be concise and accurate. Do not add information not present in the context.
5. If multiple sources support the same point, cite all of them."""

CONTEXT_TEMPLATE = """---
CONTEXT:
{context_blocks}
---

QUESTION: {question}

ANSWER:"""

def build_context_block(results: List[SearchResult]) -> str:
    """Format retrieved chunks into a numbered context block."""
    blocks = []
    for i, result in enumerate(results, 1):
        meta = result.result.metadata if hasattr(result, 'result') else result.document.metadata
        doc  = result.document
        source_info = f"Source {i} | {meta.get('filename', 'Unknown')} | score: {result.score:.3f}"
        if meta.get('page'):
            source_info += f" | page {meta['page']}"
        if meta.get('section_path'):
            source_info += f" | section: {meta['section_path']}"
        blocks.append(f"[{source_info}]\n{doc.page_content}")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# LLM Clients
# ---------------------------------------------------------------------------

class OpenAILLM:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def generate(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        return response.choices[0].message.content.strip()

    def generate_stream(self, system: str, user: str):
        """Stream tokens from the LLM one at a time."""
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content


class AnthropicLLM:
    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def generate(self, system: str, user: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text.strip()

    def generate_stream(self, system: str, user: str):
        """Stream tokens from the LLM one at a time."""
        with self.client.messages.stream(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        ) as stream:
            for text in stream.text_stream:
                yield text


class GeminiLLM:
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(
        self,
        model: str = "gemini-2.0-flash-lite",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        import httpx
        self.http = httpx
        self.api_key = api_key or os.environ["GOOGLE_API_KEY"]
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _build_body(self, system: str, user: str) -> dict:
        return {
            "system_instruction": {
                "parts": [{"text": system}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user}],
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
            },
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def generate(self, system: str, user: str) -> str:
        response = self.http.post(
            f"{self.BASE_URL}/{self.model}:generateContent",
            headers={"x-goog-api-key": self.api_key},
            json=self._build_body(system, user),
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts).strip()

    def generate_stream(self, system: str, user: str):
        with self.http.stream(
            "POST",
            f"{self.BASE_URL}/{self.model}:streamGenerateContent?alt=sse",
            headers={"x-goog-api-key": self.api_key},
            json=self._build_body(system, user),
            timeout=120,
        ) as resp:
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                try:
                    data = json.loads(line[6:])
                    for candidate in data.get("candidates", []):
                        parts = candidate.get("content", {}).get("parts", [])
                        for part in parts:
                            text = part.get("text", "")
                            if text:
                                yield text
                except json.JSONDecodeError:
                    continue


# ---------------------------------------------------------------------------
# RAG Chain
# ---------------------------------------------------------------------------

class RAGChain:
    """
    End-to-end RAG pipeline:
      Query → Retrieve → Build Prompt → Generate → Parse Response

    Usage:
        chain = RAGChain(retriever=hybrid_retriever, llm=openai_llm)
        response = chain.query("What is the refund policy?")
        print(response.answer)
        print(response.sources)
    """

    NO_ANSWER_PHRASES = [
        "i don't have enough information",
        "the context does not",
        "not mentioned in the",
        "no information available",
        "cannot be found in",
    ]

    def __init__(
        self,
        retriever: HybridRetriever,
        llm,
        top_k: int = 5,
    ):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k

    def query(self, question: str, top_k: Optional[int] = None) -> RAGResponse:
        """
        Run the full RAG pipeline for a question.
        
        Args:
            question: The user's question
            top_k:    Override default number of retrieved chunks
        
        Returns:
            RAGResponse with answer, sources, and latency
        """
        k = top_k or self.top_k
        t0 = time.perf_counter()

        # 1. Retrieve
        results = self.retriever.search(question, top_k=k)
        if not results:
            return RAGResponse(
                query=question,
                answer="No relevant documents found in the knowledge base.",
                sources=[],
                retrieved_chunks=[],
                latency_ms=(time.perf_counter() - t0) * 1000,
                model=str(type(self.llm).__name__),
                has_answer=False,
            )

        # 2. Build prompt
        context = build_context_block(results)
        user_prompt = CONTEXT_TEMPLATE.format(
            context_blocks=context,
            question=question,
        )

        # 3. Generate
        answer = self.llm.generate(SYSTEM_PROMPT, user_prompt)
        latency_ms = (time.perf_counter() - t0) * 1000

        # 4. Parse sources from answer
        sources = self._extract_sources(answer, results)

        # 5. Detect no-answer responses
        has_answer = not any(
            phrase in answer.lower() for phrase in self.NO_ANSWER_PHRASES
        )

        logger.info(
            f"Query answered in {latency_ms:.0f}ms | "
            f"{len(results)} chunks retrieved | has_answer={has_answer}"
        )

        return RAGResponse(
            query=question,
            answer=answer,
            sources=sources,
            retrieved_chunks=results,
            latency_ms=latency_ms,
            model=getattr(self.llm, 'model', type(self.llm).__name__),
            has_answer=has_answer,
        )

    def query_stream(self, question: str, top_k: Optional[int] = None):
        """
        Stream the RAG pipeline response token by token.
        
        Yields dicts with keys:
          - type: "retrieval" | "token" | "sources" | "error"
          - content: varies by type
        """
        k = top_k or self.top_k
        t0 = time.perf_counter()

        results = self.retriever.search(question, top_k=k)
        if not results:
            yield {"type": "error", "content": "No relevant documents found in the knowledge base."}
            return

        yield {"type": "retrieval", "content": len(results)}

        context = build_context_block(results)
        user_prompt = CONTEXT_TEMPLATE.format(
            context_blocks=context,
            question=question,
        )

        if not hasattr(self.llm, "generate_stream"):
            answer = self.llm.generate(SYSTEM_PROMPT, user_prompt)
            yield {"type": "token", "content": answer}
        else:
            full_answer = []
            try:
                for token in self.llm.generate_stream(SYSTEM_PROMPT, user_prompt):
                    full_answer.append(token)
                    yield {"type": "token", "content": token}
            except Exception as e:
                logger.error(f"Streaming failed, falling back to sync: {e}")
                answer = self.llm.generate(SYSTEM_PROMPT, user_prompt)
                yield {"type": "token", "content": answer}
                full_answer = [answer]

            answer = "".join(full_answer)

        latency_ms = (time.perf_counter() - t0) * 1000
        sources = self._extract_sources(answer, results)
        has_answer = not any(
            phrase in answer.lower() for phrase in self.NO_ANSWER_PHRASES
        )

        yield {
            "type": "sources",
            "content": {
                "sources": sources,
                "has_answer": has_answer,
                "latency_ms": round(latency_ms, 1),
                "model": getattr(self.llm, 'model', type(self.llm).__name__),
            },
        }

    def _extract_sources(
        self,
        answer: str,
        results: List[SearchResult],
    ) -> List[Dict[str, Any]]:
        """
        Extract which sources were cited in the answer.
        Builds a clean source list with metadata for display.
        """
        cited_indices = set(
            int(m) - 1
            for m in re.findall(r'\[Source (\d+)\]', answer)
        )

        sources = []
        for i, result in enumerate(results):
            meta = result.document.metadata
            source = {
                "rank":     i + 1,
                "filename": meta.get("filename", "Unknown"),
                "score":    round(result.score, 4),
                "excerpt":  result.document.page_content[:200] + "...",
                "cited":    i in cited_indices,
            }
            if meta.get("page"):
                source["page"] = meta["page"]
            if meta.get("section_path"):
                source["section"] = meta["section_path"]
            sources.append(source)

        return sources
