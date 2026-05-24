"""
src/evaluation/hallucination_detector.py
=========================================
Real-time hallucination detection layer.

Runs AFTER generation to flag answers that contain claims
not supported by the retrieved context.

Two detection methods:
  1. Embedding similarity (fast, ~10ms, no API cost)
  2. LLM-based NLI verification (accurate, ~200ms, costs tokens)

Production pattern: use embedding method in the hot path (every request),
LLM method in async quality monitoring (sampled 5% of requests).
"""

import re
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
from loguru import logger


@dataclass
class HallucinationResult:
    """Result of hallucination detection on a single answer."""
    is_hallucination:  bool
    confidence:        float       # 0-1, probability of hallucination
    method:            str
    flagged_claims:    List[str]   # Specific sentences that look hallucinated
    faithfulness_score: float      # Overall faithfulness (1 - hallucination risk)

    @property
    def risk_level(self) -> str:
        if self.faithfulness_score >= 0.75:
            return "LOW"
        elif self.faithfulness_score >= 0.50:
            return "MEDIUM"
        else:
            return "HIGH"

    def __str__(self):
        status = "⚠️  HALLUCINATION RISK" if self.is_hallucination else "✅ GROUNDED"
        return (
            f"{status} | Risk: {self.risk_level} | "
            f"Faithfulness: {self.faithfulness_score:.3f} | "
            f"Method: {self.method}"
        )


class EmbeddingHallucinationDetector:
    """
    Fast hallucination detection using embedding similarity.
    
    Logic:
      1. Split answer into individual claims (sentences)
      2. Embed each claim and all context chunks
      3. For each claim, find max cosine similarity to any context chunk
      4. Claims with low similarity → potentially hallucinated
      5. Aggregate → faithfulness score
    
    Threshold tuning:
      claim_threshold: min cosine similarity for a claim to be "grounded"
                       Typical: 0.45-0.60 depending on embedding model
    """

    def __init__(self, embedder, claim_threshold: float = 0.45):
        self.embedder = embedder
        self.claim_threshold = claim_threshold

    def detect(self, answer: str, contexts: List[str]) -> HallucinationResult:
        """
        Detect hallucinations in an answer given the retrieved contexts.
        
        Args:
            answer:   Generated answer text
            contexts: List of retrieved context strings
        
        Returns:
            HallucinationResult with faithfulness score and flagged claims
        """
        # Split into atomic claims (sentences)
        claims = self._extract_claims(answer)

        if not claims or not contexts:
            return HallucinationResult(
                is_hallucination  = False,
                confidence        = 0.0,
                method            = "embedding",
                flagged_claims    = [],
                faithfulness_score = 1.0,
            )

        # Embed
        claim_embs = self.embedder.embed_documents(claims)
        ctx_embs   = self.embedder.embed_documents(contexts)

        # Score each claim
        claim_scores = []
        flagged_claims = []

        for claim, claim_emb in zip(claims, claim_embs):
            max_sim = float(np.max(ctx_embs @ claim_emb))
            claim_scores.append(max_sim)
            if max_sim < self.claim_threshold:
                flagged_claims.append(claim)

        faithfulness_score = float(np.mean(claim_scores)) if claim_scores else 1.0
        is_hallucination   = faithfulness_score < self.claim_threshold

        return HallucinationResult(
            is_hallucination  = is_hallucination,
            confidence        = max(0.0, 1.0 - faithfulness_score),
            method            = "embedding",
            flagged_claims    = flagged_claims,
            faithfulness_score = faithfulness_score,
        )

    def _extract_claims(self, text: str) -> List[str]:
        """Split text into individual factual claims."""
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Filter: skip citations like "[Source 1]" and short fragments
        claims = []
        for s in sentences:
            clean = re.sub(r'\[Source \d+\]', '', s).strip()
            if len(clean) > 20:
                claims.append(clean)
        return claims


class LLMHallucinationDetector:
    """
    LLM-based hallucination detection using Natural Language Inference.
    
    More accurate than embedding-based but requires an LLM call.
    Use for: audit logging, quality sampling, high-stakes answers.
    
    Method:
      For each claim in the answer, ask the LLM:
      "Is this claim directly supported by the provided context?"
      → SUPPORTED / NOT_SUPPORTED / PARTIALLY_SUPPORTED
    """

    VERIFICATION_PROMPT = """You are a fact-checker. Given a CONTEXT and a CLAIM, determine if the claim is supported by the context.

CONTEXT:
{context}

CLAIM: {claim}

Answer with exactly one of: SUPPORTED, NOT_SUPPORTED, or PARTIALLY_SUPPORTED
Then briefly explain why (one sentence).

Format: VERDICT: <verdict>
REASON: <reason>"""

    def __init__(self, llm):
        self.llm = llm

    def detect(self, answer: str, contexts: List[str]) -> HallucinationResult:
        """Verify each claim in the answer against the contexts."""
        claims = self._extract_claims(answer)
        if not claims or not contexts:
            return HallucinationResult(
                is_hallucination=False, confidence=0.0,
                method="llm_nli", flagged_claims=[], faithfulness_score=1.0,
            )

        combined_context = "\n\n".join(contexts[:3])  # Use top 3 contexts
        verdicts = []
        flagged = []

        for claim in claims:
            prompt = self.VERIFICATION_PROMPT.format(
                context=combined_context[:2000],
                claim=claim,
            )
            try:
                response = self.llm.generate(
                    "You are a precise fact-checker. Only respond with the verdict and reason.",
                    prompt,
                )
                verdict = self._parse_verdict(response)
                verdicts.append(verdict)
                if verdict == "NOT_SUPPORTED":
                    flagged.append(claim)
            except Exception as e:
                logger.warning(f"LLM verification failed for claim: {e}")
                verdicts.append("PARTIALLY_SUPPORTED")

        # Score: SUPPORTED=1.0, PARTIALLY=0.5, NOT_SUPPORTED=0.0
        score_map = {"SUPPORTED": 1.0, "PARTIALLY_SUPPORTED": 0.5, "NOT_SUPPORTED": 0.0}
        scores = [score_map.get(v, 0.5) for v in verdicts]
        faithfulness = float(np.mean(scores)) if scores else 1.0

        return HallucinationResult(
            is_hallucination  = faithfulness < 0.5,
            confidence        = 1.0 - faithfulness,
            method            = "llm_nli",
            flagged_claims    = flagged,
            faithfulness_score = faithfulness,
        )

    def _extract_claims(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [re.sub(r'\[Source \d+\]', '', s).strip() for s in sentences if len(s.strip()) > 20]

    def _parse_verdict(self, response: str) -> str:
        for verdict in ["NOT_SUPPORTED", "PARTIALLY_SUPPORTED", "SUPPORTED"]:
            if verdict in response.upper():
                return verdict
        return "PARTIALLY_SUPPORTED"
