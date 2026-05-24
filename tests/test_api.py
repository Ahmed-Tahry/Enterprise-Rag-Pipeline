"""
tests/test_api.py — FastAPI endpoint tests using TestClient.
All tests local, no external services or API keys.
"""

import sys, os, pytest, threading, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from src.api.main import app
from src.ingestion.document_loader import Document
from src.retrieval.vector_store import FAISSVectorStore
from src.retrieval.hybrid_retriever import BM25Retriever
from src.generation.rag_chain import RAGChain


class MockEmbedder:
    dimension = 4

    def embed_documents(self, texts):
        rng = np.random.RandomState(42)
        embs = rng.randn(len(texts), self.dimension).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
        return embs

    def embed_query(self, query):
        return self.embed_documents([query])[0]


class MockLLM:
    def __init__(self):
        self.model = "mock-llm"

    def generate(self, system, user):
        return "This is a test answer based on the provided context. [Source 1]"


class MockRetriever:
    def search(self, query, top_k=5):
        doc = Document(
            page_content="Test document content about company policy.",
            metadata={"source": "test.txt", "filename": "test.txt", "chunk_index": 0},
        )
        from src.retrieval.vector_store import SearchResult
        return [SearchResult(document=doc, score=0.95, rank=1)]


class MockRAGChain:
    def query(self, question, top_k=5):
        from src.generation.rag_chain import RAGResponse
        import time
        return RAGResponse(
            query=question,
            answer="This is a test answer based on the provided context. [Source 1]",
            sources=[{"rank": 1, "filename": "test.txt", "score": 0.95, "cited": True, "excerpt": "Test document..."}],
            retrieved_chunks=MockRetriever().search(question),
            latency_ms=10.0,
            model="mock-llm",
            has_answer=True,
        )


class MockEvaluator:
    def evaluate_dataset(self, rag_chain, questions, ground_truths=None, save_path=None):
        from src.evaluation.ragas_eval import EvalReport
        return EvalReport(
            n_samples=len(questions),
            faithfulness=0.85,
            answer_relevancy=0.90,
            context_precision=0.80,
            context_recall=None,
            ragas_score=0.85,
            latency_p50_ms=100.0,
            latency_p95_ms=200.0,
        )


class MockHallucDetector:
    def detect(self, answer, contexts):
        from src.evaluation.hallucination_detector import HallucinationResult
        return HallucinationResult(
            is_hallucination=False,
            confidence=0.1,
            method="embedding",
            flagged_claims=[],
            faithfulness_score=0.89,
        )


@pytest.fixture(autouse=True)
def setup_pipeline():
    vs = FAISSVectorStore(embedding_dim=4)
    doc = Document(
        page_content="Test document content about company policy.",
        metadata={"source": "test.txt", "filename": "test.txt"},
    )
    emb = np.random.randn(1, 4).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    vs.add_documents([doc], emb)

    bm25 = BM25Retriever()
    bm25.fit([doc])

    app.state.pipeline = {
        "embedder":       MockEmbedder(),
        "vector_store":   vs,
        "bm25":           bm25,
        "retriever":      MockRetriever(),
        "rag_chain":      MockRAGChain(),
        "evaluator":      MockEvaluator(),
        "halluc_detector": MockHallucDetector(),
        "loader":         None,
        "chunker":        None,
        "stats":          {"documents_indexed": 1, "queries_handled": 0, "total_latency_ms": 0.0},
        "_lock":          threading.Lock(),
    }
    yield
    app.state.pipeline = None


client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_stats():
    resp = client.get("/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["documents_indexed"] == 1
    assert data["llm_model"]


def test_query_basic():
    resp = client.post("/query", json={"question": "What is the company policy?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["has_answer"] is True
    assert "test answer" in data["answer"].lower()
    assert len(data["sources"]) > 0
    assert "hallucination" in data


def test_query_no_hallucination():
    resp = client.post("/query", json={"question": "What is the policy?", "detect_hallucination": False})
    assert resp.status_code == 200
    assert resp.json()["hallucination"] is None


def test_query_short_question():
    resp = client.post("/query", json={"question": "Hi"})
    assert resp.status_code == 422  # min_length=3 validation


def test_query_empty_index(setup_pipeline):
    app.state.pipeline["vector_store"] = FAISSVectorStore(embedding_dim=4)
    app.state.pipeline["stats"]["documents_indexed"] = 0
    resp = client.post("/query", json={"question": "What is the policy?"})
    assert resp.status_code == 400
    assert "No documents indexed" in resp.json()["detail"]


def test_evaluate_basic():
    resp = client.post("/evaluate", json={"questions": ["What is the policy?", "How does leave work?"]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_samples"] == 2
    assert data["faithfulness"] == 0.85
    assert "ragas_score" in data


def test_evaluate_too_many_questions():
    resp = client.post("/evaluate", json={"questions": ["q"] * 101})
    assert resp.status_code == 400


def test_evaluate_empty_index(setup_pipeline):
    app.state.pipeline["vector_store"] = FAISSVectorStore(embedding_dim=4)
    resp = client.post("/evaluate", json={"questions": ["What is the policy?"]})
    assert resp.status_code == 400


def test_clear_index():
    resp = client.delete("/index")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "cleared"
    assert "chunks_removed" in data
