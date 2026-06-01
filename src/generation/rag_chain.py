import os
import re
import json
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.retrieval.hybrid_retriever import (
    HybridRetriever,
    SearchResult,
)
from src.generation.query_rewriter import QueryRewriter
from src.generation.context_compressor import ContextCompressor
from src.generation.chat_history import ChatHistory


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------


@dataclass
class RAGResponse:
    """Full response from the RAG pipeline."""

    query: str
    answer: str
    sources: List[Dict[str, Any]]
    retrieved_chunks: List[SearchResult]
    latency_ms: float
    model: str
    has_answer: bool = True
    confidence: Optional[float] = None

    def to_dict(self) -> Dict:
        d = {
            "query": self.query,
            "answer": self.answer,
            "sources": self.sources,
            "latency_ms": round(self.latency_ms, 1),
            "model": self.model,
            "has_answer": self.has_answer,
        }
        if self.confidence is not None:
            d["confidence"] = self.confidence
        return d


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

STRUCTURED_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + """

Return your answer in JSON format:
{
  "answer": "your answer with [Source N] citations",
  "citations": [{"source_index": 1, "text": "specific claim"}, ...],
  "confidence": 0.95
}"""
)

CONTEXT_TEMPLATE = """---
CONTEXT:
{context_blocks}
---

{history_block}QUESTION: {question}

ANSWER:"""


def build_context_block(results: List[SearchResult]) -> str:
    blocks = []
    for i, result in enumerate(results, 1):
        meta = (
            result.result.metadata
            if hasattr(result, "result")
            else result.document.metadata
        )
        doc = result.document
        source_info = f"Source {i} | {meta.get('filename', 'Unknown')} | score: {result.score:.3f}"
        if meta.get("page"):
            source_info += f" | page {meta['page']}"
        if meta.get("section_path"):
            source_info += f" | section: {meta['section_path']}"
        blocks.append(f"[{source_info}]\n{doc.page_content}")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# LLM Clients (unchanged)
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
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content.strip()

    def generate_stream(self, system: str, user: str):
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
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

        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ["ANTHROPIC_API_KEY"]
        )
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
            "system_instruction": {"parts": [{"text": system}]},
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
        data = response.json()
        error = data.get("error")
        if error:
            logger.error(f"Gemini API error: {error.get('message', error)}")
            raise Exception(f"Gemini API error: {error.get('message', error)}")
        candidates = data.get("candidates", [])
        if not candidates:
            logger.error(f"Gemini returned empty candidates. Response: {data}")
            raise Exception(
                "Gemini returned no candidates — check your model name and API key."
            )
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
                    error = data.get("error")
                    if error:
                        logger.error(
                            f"Gemini streaming error: {error.get('message', error)}"
                        )
                        raise Exception(
                            f"Gemini API error: {error.get('message', error)}"
                        )
                    for candidate in data.get("candidates", []):
                        parts = candidate.get("content", {}).get("parts", [])
                        for part in parts:
                            text = part.get("text", "")
                            if text:
                                yield text
                except json.JSONDecodeError:
                    continue


# ---------------------------------------------------------------------------
# RAG Chain (enhanced)
# ---------------------------------------------------------------------------


class RAGChain:
    """
    End-to-end RAG pipeline with:
      - Query Rewriting (Multi-Query + HyDE)   — expands recall
      - Adaptive Retrieval                      — widens search if low confidence
      - Context Compression                     — removes prompt noise
      - Structured JSON Output                  — machine-parseable answers
      - Conversational Memory                   — multi-turn support

    Usage:
        chain = RAGChain(retriever=retriever, llm=llm)
        response = chain.query("What is the refund policy?")
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
        query_rewriter: Optional[QueryRewriter] = None,
        context_compressor: Optional[ContextCompressor] = None,
        structured_output: bool = True,
        retrieval_multiplier: int = 4,
    ):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.query_rewriter = query_rewriter
        self.context_compressor = context_compressor
        self.structured_output = structured_output
        self.retrieval_multiplier = retrieval_multiplier

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        chat_history: Optional[ChatHistory] = None,
    ) -> RAGResponse:
        k = top_k or self.top_k
        retrieval_k = k * self.retrieval_multiplier
        t0 = time.perf_counter()

        # 1. Query Rewriting (Multi-Query + HyDE)
        if self.query_rewriter:
            queries = self.query_rewriter.rewrite(question)
        else:
            queries = [question]

        # 2. Multi-query retrieval + fusion
        if len(queries) > 1:
            results = self.retriever.search_multi(
                queries, top_k=k, retrieval_k=retrieval_k
            )
        else:
            results = self.retriever.search(question, top_k=k, retrieval_k=retrieval_k)

        # 3. Adaptive Retrieval — widen search if too few results returned
        if results and len(results) < k:
            logger.info(
                f"Only {len(results)} results (requested {k}), widening search..."
            )
            wider = self.retriever.search(
                question, top_k=k, retrieval_k=retrieval_k * 2
            )
            if len(wider) > len(results):
                results = wider

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

        # 4. Context Compression
        final_results = results[:k]
        if self.context_compressor:
            compressed = self.context_compressor.compress(question, final_results)
            if compressed:
                final_results = compressed

        # 5. Build prompt with conversation history
        history_context = chat_history.to_context() if chat_history else ""
        history_block = (
            f"CONVERSATION HISTORY:\n{history_context}\n\n" if history_context else ""
        )
        context_blocks = build_context_block(final_results)
        user_prompt = CONTEXT_TEMPLATE.format(
            context_blocks=context_blocks,
            history_block=history_block,
            question=question,
        )
        system_prompt = (
            STRUCTURED_SYSTEM_PROMPT if self.structured_output else SYSTEM_PROMPT
        )

        # 6. Generate
        answer = self.llm.generate(system_prompt, user_prompt)
        latency_ms = (time.perf_counter() - t0) * 1000

        # 7. Parse structured output
        if self.structured_output:
            parsed_answer, cited_indices, confidence = self._parse_structured_answer(
                answer
            )
        else:
            parsed_answer = answer
            cited_indices = self._extract_cited_indices(answer)
            confidence = None

        # 8. Build sources
        sources = self._build_sources(final_results, cited_indices)

        # 9. Detect no-answer
        has_answer = not any(
            phrase in parsed_answer.lower() for phrase in self.NO_ANSWER_PHRASES
        )

        logger.info(
            f"Query answered in {latency_ms:.0f}ms | "
            f"{len(final_results)} chunks | has_answer={has_answer}"
        )

        return RAGResponse(
            query=question,
            answer=parsed_answer,
            sources=sources,
            retrieved_chunks=final_results,
            latency_ms=latency_ms,
            model=getattr(self.llm, "model", type(self.llm).__name__),
            has_answer=has_answer,
            confidence=confidence,
        )

    def query_stream(
        self,
        question: str,
        top_k: Optional[int] = None,
        chat_history: Optional[ChatHistory] = None,
    ):
        k = top_k or self.top_k
        retrieval_k = k * self.retrieval_multiplier
        t0 = time.perf_counter()

        # 1. Query Rewriting
        if self.query_rewriter:
            queries = self.query_rewriter.rewrite(question)
        else:
            queries = [question]

        # 2. Multi-query retrieval
        if len(queries) > 1:
            results = self.retriever.search_multi(
                queries, top_k=k, retrieval_k=retrieval_k
            )
        else:
            results = self.retriever.search(question, top_k=k, retrieval_k=retrieval_k)

        # 3. Adaptive
        if results and len(results) < k:
            wider = self.retriever.search(
                question, top_k=k, retrieval_k=retrieval_k * 2
            )
            if len(wider) > len(results):
                results = wider

        if not results:
            yield {
                "type": "error",
                "content": "No relevant documents found in the knowledge base.",
            }
            return

        yield {"type": "retrieval", "content": len(results)}

        # 4. Context compression
        final_results = results[:k]
        if self.context_compressor:
            compressed = self.context_compressor.compress(question, final_results)
            if compressed:
                final_results = compressed

        # 5. Build prompt
        history_context = chat_history.to_context() if chat_history else ""
        history_block = (
            f"CONVERSATION HISTORY:\n{history_context}\n\n" if history_context else ""
        )
        context_blocks = build_context_block(final_results)
        user_prompt = CONTEXT_TEMPLATE.format(
            context_blocks=context_blocks,
            history_block=history_block,
            question=question,
        )
        system_prompt = (
            STRUCTURED_SYSTEM_PROMPT if self.structured_output else SYSTEM_PROMPT
        )

        # 6. Generate (streaming)
        if not hasattr(self.llm, "generate_stream"):
            answer = self.llm.generate(system_prompt, user_prompt)
            yield {"type": "token", "content": answer}
        else:
            full_answer = []
            try:
                for token in self.llm.generate_stream(system_prompt, user_prompt):
                    full_answer.append(token)
                    yield {"type": "token", "content": token}
            except Exception as e:
                logger.error(f"Streaming failed, falling back to sync: {e}")
                answer = self.llm.generate(system_prompt, user_prompt)
                yield {"type": "token", "content": answer}
                full_answer = [answer]

            answer = "".join(full_answer)

        latency_ms = (time.perf_counter() - t0) * 1000

        # 7. Parse structured output
        if self.structured_output:
            parsed_answer, cited_indices, confidence = self._parse_structured_answer(
                answer
            )
        else:
            parsed_answer = answer
            cited_indices = self._extract_cited_indices(answer)
            confidence = None

        sources = self._build_sources(final_results, cited_indices)
        has_answer = not any(
            phrase in parsed_answer.lower() for phrase in self.NO_ANSWER_PHRASES
        )

        yield {
            "type": "sources",
            "content": {
                "sources": sources,
                "has_answer": has_answer,
                "latency_ms": round(latency_ms, 1),
                "model": getattr(self.llm, "model", type(self.llm).__name__),
                "confidence": confidence,
            },
        }

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _extract_cited_indices(self, answer: str) -> set:
        return set(int(m) - 1 for m in re.findall(r"\[Source (\d+)\]", answer))

    def _build_sources(
        self,
        results: List[SearchResult],
        cited_indices: set,
    ) -> List[Dict[str, Any]]:
        sources = []
        for i, result in enumerate(results):
            meta = result.document.metadata
            source = {
                "rank": i + 1,
                "filename": meta.get("filename", "Unknown"),
                "score": round(result.score, 4),
                "excerpt": result.document.page_content[:200] + "...",
                "cited": i in cited_indices,
            }
            if meta.get("page"):
                source["page"] = meta["page"]
            if meta.get("section_path"):
                source["section"] = meta["section_path"]
            sources.append(source)
        return sources

    def _parse_structured_answer(self, answer_text: str) -> tuple:
        """
        Parse JSON-structured output from the LLM.
        Falls back to regex-based citation extraction on failure.
        Returns (answer_text, cited_indices_set, confidence_or_None).
        """
        start = answer_text.find("{")
        if start >= 0:
            try:
                decoder = json.JSONDecoder()
                obj, _ = decoder.raw_decode(answer_text, start)
                answer = obj.get("answer", answer_text)
                citations = obj.get("citations", [])
                cited_indices = {
                    c["source_index"] - 1 for c in citations if "source_index" in c
                }
                confidence = obj.get("confidence", None)
                return answer, cited_indices, confidence
            except (json.JSONDecodeError, Exception):
                pass
        cited_indices = self._extract_cited_indices(answer_text)
        return answer_text, cited_indices, None
