"""
src/api/main.py
================
FastAPI REST API for the Enterprise RAG system.

Endpoints:
  POST /ingest          — Upload and index documents
  POST /query           — Query the RAG pipeline
  GET  /health          — Health check
  GET  /stats           — System statistics
  GET  /files           — Indexed files list
  POST /evaluate        — Run RAGAS evaluation on a test set
  DELETE /index         — Clear the vector store
"""

import os
from dotenv import load_dotenv
import json
import time
import tempfile
import asyncio
import threading


from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger
from werkzeug.utils import secure_filename

# ── Pipeline stored in app.state (thread-safe via FastAPI lifecycle) ─────
load_dotenv()
PROVIDER_KEY_MAP = {
    "gemini": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def validate_env():
    """Check required env vars at startup. Fail fast with a clear message."""
    provider = os.getenv("LLM_PROVIDER", "gemini")
    required_key = PROVIDER_KEY_MAP.get(provider, "GOOGLE_API_KEY")
    if not os.getenv(required_key):
        logger.error(
            f"Missing {required_key} for LLM_PROVIDER={provider}. "
            f"Set it in your .env file."
        )
        raise SystemExit(f"Missing {required_key}")


def init_pipeline():
    """Initialise pipeline components."""
    from src.ingestion.embedder import get_embedder
    from src.retrieval.vector_store import FAISSVectorStore
    from src.retrieval.hybrid_retriever import (
        BM25Retriever,
        HybridRetriever,
        CrossEncoderReranker,
    )
    from src.generation.rag_chain import RAGChain, OpenAILLM
    from src.evaluation.ragas_eval import RAGASEvaluator
    from src.evaluation.hallucination_detector import EmbeddingHallucinationDetector
    from src.ingestion.document_loader import DocumentLoader
    from src.ingestion.chunker import RecursiveCharacterChunker, ChunkConfig

    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "huggingface")
    embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

    embedder = get_embedder(embedding_provider, model_name=embedding_model)

    chunk_cfg = ChunkConfig(
        chunk_size=int(os.getenv("CHUNK_SIZE", 512)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 64)),
    )

    vector_store = FAISSVectorStore(embedding_dim=embedder.dimension)
    bm25 = BM25Retriever()

    use_reranker = os.getenv("USE_RERANKER", "true").lower() == "true"
    reranker = None
    if use_reranker:
        try:
            reranker = CrossEncoderReranker(
                os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            )
        except Exception as e:
            logger.warning(f"Reranker failed to load: {e}")

    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25=bm25,
        embedder=embedder,
        reranker=reranker,
    )

    llm_provider = os.getenv("LLM_PROVIDER", "gemini")
    llm_model = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")
    llm_kwargs = {
        "model": llm_model,
        "temperature": float(os.getenv("LLM_TEMPERATURE", 0.1)),
        "max_tokens": int(os.getenv("LLM_MAX_TOKENS", 1024)),
    }
    if llm_provider == "openai":
        from src.generation.rag_chain import OpenAILLM

        llm = OpenAILLM(**llm_kwargs)
    elif llm_provider == "anthropic":
        from src.generation.rag_chain import AnthropicLLM

        llm = AnthropicLLM(**llm_kwargs)
    else:
        from src.generation.rag_chain import GeminiLLM

        llm = GeminiLLM(**llm_kwargs)

    rag_chain = RAGChain(retriever=retriever, llm=llm)
    evaluator = RAGASEvaluator(embedder)
    halluc_detector = EmbeddingHallucinationDetector(embedder)

    return {
        "embedder": embedder,
        "vector_store": vector_store,
        "bm25": bm25,
        "retriever": retriever,
        "rag_chain": rag_chain,
        "evaluator": evaluator,
        "halluc_detector": halluc_detector,
        "loader": DocumentLoader(),
        "chunker": RecursiveCharacterChunker(chunk_cfg),
        "stats": {
            "documents_indexed": 0,
            "queries_handled": 0,
            "total_latency_ms": 0.0,
            "indexed_files": [],
        },
        "_lock": threading.Lock(),
    }


def get_pipeline(req: Request):
    """Access pipeline from app.state (set during lifespan)."""
    return req.app.state.pipeline


# ── Pydantic models ──────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    question: str = Field(
        ..., min_length=3, max_length=2000, description="The question to answer"
    )
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")
    detect_hallucination: bool = Field(True, description="Run hallucination detection")


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[dict]
    has_answer: bool
    latency_ms: float
    model: str
    hallucination: Optional[dict] = None


class EvalRequest(BaseModel):
    questions: List[str]
    ground_truths: Optional[List[str]] = None


class IndexedFile(BaseModel):
    filename: str
    pages: int
    chunks: int
    size_bytes: int
    indexed_at: float


class FilesResponse(BaseModel):
    files: List[IndexedFile]


class IngestResponse(BaseModel):
    status: str
    files_indexed: int
    chunks_added: int
    latency_ms: float
    chunk_strategy: str = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 64


# ── App ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Enterprise RAG API...")
    validate_env()
    app.state.pipeline = init_pipeline()
    n_docs = app.state.pipeline["stats"]["documents_indexed"]
    n_chunks = len(app.state.pipeline["vector_store"])
    logger.info(f"Pipeline ready. Documents: {n_docs}, Chunks in index: {n_chunks}")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Enterprise RAG API",
    description="Production-grade Retrieval-Augmented Generation with RAGAS evaluation",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/stats")
async def stats(req: Request):
    p = get_pipeline(req)
    return {
        "documents_indexed": p["stats"]["documents_indexed"],
        "chunks_in_index": len(p["vector_store"]),
        "queries_handled": p["stats"]["queries_handled"],
        "avg_latency_ms": (
            p["stats"]["total_latency_ms"] / max(p["stats"]["queries_handled"], 1)
        ),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5"),
        "llm_model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
    }


@app.get("/files", response_model=FilesResponse)
async def indexed_files(req: Request):
    p = get_pipeline(req)
    with p["_lock"]:
        files = list(p["stats"]["indexed_files"])
    files.sort(key=lambda f: f.get("indexed_at", 0), reverse=True)
    return {"files": files}


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(
    req: Request,
    files: List[UploadFile] = File(...),
    chunk_strategy: str = "recursive",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
):
    """
    Upload and index one or more documents.

    Supports: PDF, DOCX, HTML, TXT, MD

    Chunk strategies: recursive (default), markdown, semantic
    """
    from src.ingestion.chunker import ChunkConfig, get_chunker

    p = get_pipeline(req)
    t0 = time.perf_counter()

    filenames = [f.filename for f in files]
    logger.info(
        f"Ingesting {len(files)} files: {filenames} | strategy={chunk_strategy}"
    )

    chunk_cfg = ChunkConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunker = get_chunker(chunk_strategy, config=chunk_cfg)

    all_chunks = []
    n_files = 0
    file_entries = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for upload in files:
            suffix = Path(upload.filename).suffix.lower()
            if suffix not in [".pdf", ".docx", ".doc", ".html", ".htm", ".txt", ".md"]:
                logger.warning(
                    f"Unsupported file type skipped: {upload.filename} ({suffix})"
                )
                raise HTTPException(400, f"Unsupported file type: {suffix}")

            safe_name = secure_filename(upload.filename)
            if not safe_name:
                logger.warning(f"Invalid filename rejected: {upload.filename}")
                raise HTTPException(400, f"Invalid filename: {upload.filename}")

            tmp_path = Path(tmpdir) / safe_name
            content = await upload.read()
            with open(tmp_path, "wb") as f:
                f.write(content)

            try:
                docs = p["loader"].load(str(tmp_path))
                chunks = chunker.split_documents(docs)
                all_chunks.extend(chunks)
                file_entries.append(
                    {
                        "filename": safe_name,
                        "pages": len(docs),
                        "chunks": len(chunks),
                        "size_bytes": len(content),
                        "indexed_at": time.time(),
                    }
                )
                n_files += 1
                logger.info(
                    f"Loaded {safe_name}: {len(docs)} pages, {len(chunks)} chunks"
                )
            except Exception as e:
                logger.error(f"Failed to load {safe_name}: {e}")
                raise HTTPException(500, f"Failed to process {safe_name}: {str(e)}")

    if not all_chunks:
        raise HTTPException(400, "No text could be extracted from the uploaded files.")

    texts = [c.page_content for c in all_chunks]
    embeddings = await asyncio.to_thread(p["embedder"].embed_documents, texts)

    def _index():
        with p["_lock"]:
            p["vector_store"].add_documents(all_chunks, embeddings)
            p["bm25"].fit(p["vector_store"].documents)
            p["stats"]["documents_indexed"] += n_files
            p["stats"]["indexed_files"].extend(file_entries)

    await asyncio.to_thread(_index)

    latency_ms = (time.perf_counter() - t0) * 1000

    return IngestResponse(
        status="success",
        files_indexed=n_files,
        chunks_added=len(all_chunks),
        latency_ms=round(latency_ms, 1),
        chunk_strategy=chunk_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


@app.post("/query", response_model=QueryResponse)
async def query(req: Request, body: QueryRequest):
    """
    Query the RAG pipeline.

    Returns the answer, cited sources, and optionally a hallucination check.
    """
    p = get_pipeline(req)

    if len(p["vector_store"]) == 0:
        logger.warning(f"Query rejected (empty index): '{body.question[:80]}'")
        raise HTTPException(400, "No documents indexed yet. Use POST /ingest first.")

    logger.info(f"Query: '{body.question[:120]}' | top_k={body.top_k}")

    def _do_query():
        with p["_lock"]:
            response = p["rag_chain"].query(body.question, top_k=body.top_k)

            hallucination_result = None
            if body.detect_hallucination and response.retrieved_chunks:
                contexts = [r.document.page_content for r in response.retrieved_chunks]
                h_result = p["halluc_detector"].detect(response.answer, contexts)
                hallucination_result = {
                    "risk_level": h_result.risk_level,
                    "faithfulness_score": round(h_result.faithfulness_score, 3),
                    "is_hallucination": h_result.is_hallucination,
                    "flagged_claims": h_result.flagged_claims[:3],
                }

            p["stats"]["queries_handled"] += 1
            p["stats"]["total_latency_ms"] += response.latency_ms

        return response, hallucination_result

    response, hallucination_result = await asyncio.to_thread(_do_query)

    return QueryResponse(
        question=response.query,
        answer=response.answer,
        sources=response.sources,
        has_answer=response.has_answer,
        latency_ms=round(response.latency_ms, 1),
        model=response.model,
        hallucination=hallucination_result,
    )


@app.post("/query/stream")
async def query_stream(req: Request, body: QueryRequest):
    """
    Stream the RAG pipeline response token by token (SSE format).

    Events:
      data: {"type": "retrieval", "content": <chunk_count>}
      data: {"type": "token", "content": "<token>"}
      data: {"type": "sources", "content": {...}}
    """
    p = get_pipeline(req)

    if len(p["vector_store"]) == 0:
        raise HTTPException(400, "No documents indexed yet. Use POST /ingest first.")

    logger.info(f"Streaming query: '{body.question[:120]}' | top_k={body.top_k}")

    async def event_stream():
        def _collect():
            return list(
                p["rag_chain"].query_stream(body.question, top_k=body.top_k)
            )

        try:
            events = await asyncio.to_thread(_collect)
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            return

        for event in events:
            yield f"data: {json.dumps(event)}\n\n"

        def _update_stats():
            with p["_lock"]:
                p["stats"]["queries_handled"] += 1

        await asyncio.to_thread(_update_stats)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/evaluate")
async def evaluate(req: Request, body: EvalRequest):
    """
    Run RAGAS evaluation on a test set.
    Results saved to data/eval_report.json.
    """
    p = get_pipeline(req)

    if len(p["vector_store"]) == 0:
        logger.warning("Evaluate rejected (empty index)")
        raise HTTPException(400, "No documents indexed. Ingest documents first.")

    if len(body.questions) > 100:
        logger.warning(
            f"Evaluate rejected: {len(body.questions)} questions exceeds max 100"
        )
        raise HTTPException(400, "Max 100 questions per evaluation run.")

    logger.info(f"Evaluating {len(body.questions)} questions")

    def _evaluate():
        return p["evaluator"].evaluate_dataset(
            rag_chain=p["rag_chain"],
            questions=body.questions,
            ground_truths=body.ground_truths,
            save_path="data/eval_report.json",
        )

    report = await asyncio.to_thread(_evaluate)
    return report.to_dict()


@app.delete("/index")
async def clear_index(req: Request):
    """Clear the vector store and BM25 index."""
    from src.retrieval.vector_store import FAISSVectorStore
    from src.retrieval.hybrid_retriever import BM25Retriever

    p = get_pipeline(req)
    dim = p["embedder"].dimension

    with p["_lock"]:
        was_docs = len(p["vector_store"])
        p["vector_store"] = FAISSVectorStore(dim)
        p["bm25"] = BM25Retriever()
        p["retriever"].vector_store = p["vector_store"]
        p["retriever"].bm25 = p["bm25"]
        p["stats"]["documents_indexed"] = 0
        p["stats"]["indexed_files"] = []

    logger.info(f"Index cleared ({was_docs} chunks removed).")
    return {"status": "cleared", "chunks_removed": was_docs}
