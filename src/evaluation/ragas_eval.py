"""
src/evaluation/ragas_eval.py
=============================
RAGAS-based evaluation pipeline for the RAG system.

RAGAS Metrics:
  Faithfulness:      Are all answer claims supported by the retrieved context?
                     → Detects hallucinations
  Answer Relevancy:  Does the answer actually address the question?
                     → Detects off-topic answers
  Context Precision: Of the retrieved chunks, how many were actually relevant?
                     → Measures retrieval precision
  Context Recall:    Does the retrieved context contain all info needed?
                     → Requires ground truth answers

Why RAGAS matters:
  You can't improve what you can't measure. RAGAS gives you a continuous
  quality signal to track across model versions, retrieval configs, and
  document updates — without expensive human evaluation every time.
"""

import json
import time
import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger

from src.generation.rag_chain import RAGChain, RAGResponse


@dataclass
class EvalSample:
    """A single evaluation example."""
    question:         str
    ground_truth:     Optional[str] = None    # Reference answer (for recall)
    generated_answer: Optional[str] = None    # Populated after running RAG
    contexts:         List[str] = field(default_factory=list)  # Retrieved chunks
    scores:           Dict[str, float] = field(default_factory=dict)


@dataclass
class EvalReport:
    """Aggregated evaluation results."""
    n_samples:         int
    faithfulness:      float
    answer_relevancy:  float
    context_precision: float
    context_recall:    Optional[float]
    ragas_score:       float
    latency_p50_ms:    float
    latency_p95_ms:    float
    individual:        List[EvalSample] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "n_samples":         self.n_samples,
            "faithfulness":      round(self.faithfulness, 4),
            "answer_relevancy":  round(self.answer_relevancy, 4),
            "context_precision": round(self.context_precision, 4),
            "context_recall":    round(self.context_recall, 4) if self.context_recall else None,
            "ragas_score":       round(self.ragas_score, 4),
            "latency_p50_ms":    round(self.latency_p50_ms, 1),
            "latency_p95_ms":    round(self.latency_p95_ms, 1),
        }

    def __str__(self) -> str:
        lines = [
            "=" * 50,
            "  RAG EVALUATION REPORT",
            "=" * 50,
            f"  Samples evaluated:  {self.n_samples}",
            f"  Faithfulness:       {self.faithfulness:.3f}  (hallucination check)",
            f"  Answer Relevancy:   {self.answer_relevancy:.3f}  (on-topic check)",
            f"  Context Precision:  {self.context_precision:.3f}  (retrieval precision)",
        ]
        if self.context_recall is not None:
            lines.append(f"  Context Recall:     {self.context_recall:.3f}  (retrieval recall)")
        lines += [
            f"  ─────────────────────────────────────────",
            f"  RAGAS Score:        {self.ragas_score:.3f}",
            f"  Latency P50:        {self.latency_p50_ms:.0f}ms",
            f"  Latency P95:        {self.latency_p95_ms:.0f}ms",
            "=" * 50,
        ]
        return "\n".join(lines)


class RAGASEvaluator:
    """
    Evaluates a RAG pipeline using RAGAS metrics.

    Uses embedding similarity as a proxy for LLM-judged metrics.
    For production: use the official ragas library with an LLM judge.
    """

    def __init__(self, embedder):
        self.embedder = embedder

    def faithfulness(self, answer: str, contexts: List[str]) -> float:
        """
        Does the answer stay within the bounds of the retrieved context?
        
        Method: Split answer into claims (sentences). For each claim,
        measure max cosine similarity to any context chunk.
        Faithfulness = mean similarity across all claims.
        
        Low score → model is hallucinating beyond the context.
        """
        import re
        claims = [s.strip() for s in re.split(r'[.!?]', answer) if len(s.strip()) > 20]
        if not claims:
            return 1.0

        claim_embs = self.embedder.embed_documents(claims)
        ctx_embs   = self.embedder.embed_documents(contexts)

        # For each claim, find max cosine similarity to any context
        scores = []
        for claim_emb in claim_embs:
            sims = ctx_embs @ claim_emb
            scores.append(float(np.max(sims)))

        return float(np.mean(scores))

    def answer_relevancy(self, answer: str, question: str) -> float:
        """
        Does the answer actually address the question?
        
        Method: Embed both answer and question, compute cosine similarity.
        High score → answer is semantically aligned with the question.
        Low score → answer went off-topic.
        """
        q_emb = self.embedder.embed_query(question)
        a_emb = self.embedder.embed_documents([answer])[0]
        return float(np.dot(q_emb, a_emb))

    def context_precision(self, contexts: List[str], question: str) -> float:
        """
        Of the retrieved contexts, how many are actually relevant?
        
        Method: Cosine similarity of each context to the question.
        Average across all retrieved contexts.
        High score → all retrieved chunks are useful.
        Low score → retrieval is noisy (irrelevant chunks contaminate the prompt).
        """
        if not contexts:
            return 0.0
        q_emb = self.embedder.embed_query(question)
        ctx_embs = self.embedder.embed_documents(contexts)
        sims = ctx_embs @ q_emb
        return float(np.mean(sims))

    def context_recall(self, contexts: List[str], ground_truth: str) -> float:
        """
        Does the context contain all information needed to answer?
        
        Method: Cosine similarity between ground truth answer and best context chunk.
        Requires a reference answer — use when you have labelled eval data.
        """
        if not ground_truth or not contexts:
            return 0.0
        gt_emb   = self.embedder.embed_documents([ground_truth])[0]
        ctx_embs = self.embedder.embed_documents(contexts)
        sims = ctx_embs @ gt_emb
        return float(np.max(sims))

    def evaluate_sample(self, sample: EvalSample) -> EvalSample:
        """Run all applicable metrics on a single eval sample."""
        if not sample.generated_answer or not sample.contexts:
            return sample

        sample.scores["faithfulness"] = self.faithfulness(
            sample.generated_answer, sample.contexts
        )
        sample.scores["answer_relevancy"] = self.answer_relevancy(
            sample.generated_answer, sample.question
        )
        sample.scores["context_precision"] = self.context_precision(
            sample.contexts, sample.question
        )
        if sample.ground_truth:
            sample.scores["context_recall"] = self.context_recall(
                sample.contexts, sample.ground_truth
            )

        return sample

    def evaluate_dataset(
        self,
        rag_chain: RAGChain,
        questions: List[str],
        ground_truths: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> EvalReport:
        """
        Evaluate the RAG pipeline on a list of questions.
        
        Args:
            rag_chain:     The RAG pipeline to evaluate
            questions:     List of evaluation questions
            ground_truths: Optional reference answers for context_recall
            save_path:     Optional path to save detailed results JSON
        
        Returns:
            EvalReport with aggregate metrics
        """
        gt_list = ground_truths or [None] * len(questions)
        samples = []
        latencies = []

        logger.info(f"Evaluating {len(questions)} questions...")

        for i, (question, gt) in enumerate(zip(questions, gt_list)):
            logger.info(f"  [{i+1}/{len(questions)}] {question[:60]}...")

            # Run RAG
            t0 = time.perf_counter()
            response: RAGResponse = rag_chain.query(question)
            latency_ms = (time.perf_counter() - t0) * 1000
            latencies.append(latency_ms)

            # Build eval sample
            sample = EvalSample(
                question         = question,
                ground_truth     = gt,
                generated_answer = response.answer,
                contexts         = [r.document.page_content for r in response.retrieved_chunks],
            )

            # Score
            sample = self.evaluate_sample(sample)
            samples.append(sample)

        # Aggregate
        metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        agg = {}
        for m in metrics:
            vals = [s.scores[m] for s in samples if m in s.scores]
            agg[m] = float(np.mean(vals)) if vals else None

        core_metrics = [v for k, v in agg.items() if v is not None and k != "context_recall"]
        ragas_score  = float(np.mean(core_metrics)) if core_metrics else 0.0

        report = EvalReport(
            n_samples         = len(samples),
            faithfulness      = agg.get("faithfulness", 0.0),
            answer_relevancy  = agg.get("answer_relevancy", 0.0),
            context_precision = agg.get("context_precision", 0.0),
            context_recall    = agg.get("context_recall"),
            ragas_score       = ragas_score,
            latency_p50_ms    = float(np.percentile(latencies, 50)),
            latency_p95_ms    = float(np.percentile(latencies, 95)),
            individual        = samples,
        )

        logger.info(str(report))

        if save_path:
            self._save_report(report, save_path)

        return report

    def _save_report(self, report: EvalReport, path: str):
        """Save detailed evaluation results to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = report.to_dict()
        data["individual"] = [
            {
                "question": s.question,
                "answer":   s.generated_answer,
                "scores":   s.scores,
                "contexts": [c[:100] + "..." for c in s.contexts],
            }
            for s in report.individual
        ]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Eval report saved to {path}")
