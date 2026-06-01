# AGENTS.md — Enterprise RAG Pipeline

## Commands (make-driven)

| Command | Action |
|---------|--------|
| `make install` | `pip install -r requirements.txt` |
| `make run-api` | `uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload` |
| `make run-ui` | `streamlit run ui/app.py` |
| `make ingest` | `python scripts/ingest.py --dir ./data/sample_docs --save ./data/vectorstore` |
| `make test` | `pytest tests/ -v --tb=short` |
| `make lint` | `ruff check src/ tests/` |
| `make format` | `ruff format src/ tests/` |
| `make docker-up` | `docker-compose up --build -d` |
| `make docker-down` | `docker-compose down` |

Run `make lint && make format && make test` before committing.

## Entrypoints

- **API**: `src/api/main.py` — FastAPI app. Pipeline initialized once in `lifespan` via `init_pipeline()`, stored in `app.state.pipeline` with `threading.Lock()`. Endpoints: `POST /ingest`, `POST /query`, `POST /query/stream` (SSE), `POST /evaluate`, `GET /health`, `GET /stats`, `GET /files`, `DELETE /index`.
- **UI**: `ui/app.py` — Streamlit, hits API at `http://localhost:8000`.
- **CLI ingest**: `scripts/ingest.py` — loads directory, chunks, embeds, saves FAISS + pickled BM25.

## Testing

Two test files (both use `sys.path.insert(0, ...)` — run from repo root):
- `tests/test_pipeline.py` — ingestion, retrieval, hallucination detector (uses numpy random embeddings)
- `tests/test_api.py` — FastAPI TestClient with `autouse` fixture that mocks every component + sets `app.state.pipeline`; no API keys needed.

## Framework quirks

- **`.env` required**: `python-dotenv` loads `.env` automatically. `LLM_PROVIDER` selects which API key is mandatory (`gemini` → `GOOGLE_API_KEY`, `openai` → `OPENAI_API_KEY`, `anthropic` → `ANTHROPIC_API_KEY`). Missing key crashes startup.
- **Path hack**: `scripts/ingest.py` and both test files use `sys.path.insert(0, ...)` to add project root. Do not move or restructure them.
- **Pipeline state**: Dict in `app.state.pipeline` with `threading.Lock()`. Never access components outside the lock. API wraps CPU-bound work (embedding, indexing) in `asyncio.to_thread`.
- **Docker**: Single `Dockerfile` runs both API (background `uvicorn`) and UI (foreground Streamlit) via `render_start.sh`.
- **Vector store persistence**: `data/vectorstore/` — FAISS index + pickled BM25. `.gitignore`d.
- **BGE query prefix**: `embedder.py:61` — BGE models get `"Represent this sentence for searching relevant passages: "` prepended to queries. Documents are embedded without prefix.
- **Enhanced RAG path**: `rag_chain.py` now runs query rewriting (Multi-Query + HyDE), adaptive retrieval, context compression, structured JSON parsing, and optional conversation memory (`conversation_id`).
- **Hallucination risk thresholds**: LOW ≥ 0.75, MEDIUM 0.50–0.75, HIGH < 0.50.
- **RRF k=60** (canonical default). Default embedding dimension = 768 (BGE-base).
- **Render deploy**: `render.yaml` expects `OPENAI_API_KEY` env var.

## Config (`.env`)

Key vars: `LLM_PROVIDER` (gemini\|openai\|anthropic), `GOOGLE_API_KEY`, `EMBEDDING_PROVIDER` (huggingface\|openai), `EMBEDDING_MODEL` (default `BAAI/bge-base-en-v1.5`), `LLM_MODEL` (default `gemini-2.0-flash-lite`), `LLM_TEMPERATURE` (0.0), `CHUNK_SIZE` (512), `CHUNK_OVERLAP` (64), `RETRIEVAL_TOP_K` (20), `RERANK_TOP_K` (5), `USE_HYBRID`, `USE_RERANKER`, `RERANKER_MODEL`, `VECTOR_STORE_PATH`.

## Style

- Ruff for lint + format (runs on `src/` `tests/`). No `pyproject.toml`.
- Logging via `loguru`, not `logging`.
- Type hints on all public functions.
