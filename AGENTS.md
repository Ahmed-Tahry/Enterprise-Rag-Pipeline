# AGENTS.md — Enterprise RAG Pipeline

## Commands (make-driven, use these)

| Command | What it does |
|---------|-------------|
| `make install` | `pip install -r requirements.txt` |
| `make run-api` | `uvicorn src.api.main:app --reload --port 8000` |
| `make run-ui` | `streamlit run ui/app.py` |
| `make ingest` | `python scripts/ingest.py --dir ./data/sample_docs --save ./data/vectorstore` |
| `make test` | `pytest tests/ -v --tb=short` |
| `make lint` | `ruff check src/ tests/` |
| `make format` | `ruff format src/ tests/` |
| `make docker-up` | `docker-compose up --build -d` |

Run lint → format → test before committing.

## Entrypoints

- **API**: `src/api/main.py` — FastAPI app. Pipeline initialized once in `lifespan` handler via `init_pipeline()`, stored in `app.state.pipeline`. Endpoints: `POST /ingest`, `POST /query`, `POST /evaluate`, `GET /health`, `GET /stats`, `DELETE /index`.
- **UI**: `ui/app.py` — Streamlit, hits API at `http://localhost:8000`.
- **CLI ingest**: `scripts/ingest.py` — indexes from a directory, saves FAISS + BM25 to disk.

## Project structure

Single-package Python project (no `pyproject.toml`). Packages under `src/`:
- `src/ingestion/` — `document_loader.py`, `chunker.py`, `embedder.py`
- `src/retrieval/` — `vector_store.py` (FAISS), `hybrid_retriever.py` (BM25 + dense + RRF + reranker)
- `src/generation/` — `rag_chain.py` (RAGChain with OpenAILLM / AnthropicLLM)
- `src/evaluation/` — `ragas_eval.py`, `hallucination_detector.py`
- `src/api/` — `main.py` (FastAPI)

## Testing

- Single file: `tests/test_pipeline.py` (contains two test classes: ingestion and retrieval).
- Tests modify `sys.path` directly (no package install needed). Run from repo root.
- All tests are local (no API keys, no external services). Uses numpy random embeddings for FAISS tests.

## Framework quirks

- **Path hack**: `scripts/ingest.py` and `tests/test_pipeline.py` both use `sys.path.insert(0, ...)` to add project root. Do not move them.
- **`.env` required**: `python-dotenv` loads `.env` automatically. Missing `OPENAI_API_KEY` will crash the API at `/query`.
- **Docker**: Single `Dockerfile` runs both API (background `uvicorn`) and UI (foreground Streamlit) via `render_start.sh`.
- **Pipeline state**: Dict of components held in `app.state.pipeline` with `threading.Lock()` for thread safety. Never access component instances directly outside the lock.
- **Vector store persistence**: `data/vectorstore/` — FAISS index + pickled documents. `.gitignore`d.
- **BGE query prefix**: `embedder.py` prepends `"Represent this sentence for searching relevant passages: "` to queries when using BGE models (line 61).

## Config (all via `.env`)

Key vars: `OPENAI_API_KEY`, `EMBEDDING_PROVIDER` (huggingface|openai), `LLM_MODEL` (default gpt-4o-mini), `CHUNK_SIZE` (512), `RETRIEVAL_TOP_K` (20), `RERANK_TOP_K` (5), `USE_HYBRID`, `USE_RERANKER`.

## Style

- Ruff for lint + format (same config, no `pyproject.toml` — runs on `src/` and `tests/`).
- Logging via `loguru`, not `logging`.
- Type hints on all public functions.
