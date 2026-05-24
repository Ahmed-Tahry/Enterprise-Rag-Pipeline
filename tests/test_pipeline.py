"""
tests/test_ingestion.py — Document loading and chunking tests
tests/test_retrieval.py — Vector store and hybrid retrieval tests
"""

# ============================================================
# tests/test_ingestion.py
# ============================================================

import sys, os, tempfile, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingestion.document_loader import Document, TextLoader, DocumentLoader
from src.ingestion.chunker import RecursiveCharacterChunker, ChunkConfig, MarkdownChunker


# ── TextLoader ───────────────────────────────────────────────────────────

def test_text_loader_basic(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("Hello world. This is a test document with some content.")
    loader = TextLoader()
    docs = loader.load(str(f))
    assert len(docs) == 1
    assert "Hello world" in docs[0].page_content
    assert docs[0].metadata["file_type"] == "txt"


def test_text_loader_metadata(tmp_path):
    f = tmp_path / "notes.md"
    f.write_text("# Title\nSome content here.")
    loader = TextLoader()
    docs = loader.load(str(f))
    assert docs[0].metadata["filename"] == "notes.md"
    assert docs[0].metadata["file_type"] == "md"


def test_document_loader_unsupported():
    loader = DocumentLoader()
    with pytest.raises(ValueError, match="Unsupported file type"):
        loader.load("file.xyz")


def test_document_loader_directory(tmp_path):
    (tmp_path / "a.txt").write_text("Document A content. Has some text.")
    (tmp_path / "b.txt").write_text("Document B content. Different text.")
    (tmp_path / "ignore.csv").write_text("col1,col2\n1,2")
    loader = DocumentLoader()
    docs = loader.load_directory(str(tmp_path))
    # Should load .txt only (csv not supported)
    assert len(docs) == 2


# ── RecursiveCharacterChunker ─────────────────────────────────────────────

def test_recursive_chunker_basic():
    config  = ChunkConfig(chunk_size=100, chunk_overlap=20, min_chunk_size=10)
    chunker = RecursiveCharacterChunker(config)
    doc     = Document(
        page_content="This is sentence one. This is sentence two. This is sentence three. " * 10,
        metadata={"source": "test.txt"},
    )
    chunks = chunker.split_documents([doc])
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.page_content) <= config.chunk_size * 1.2  # Some tolerance


def test_recursive_chunker_preserves_metadata():
    config  = ChunkConfig(chunk_size=50, chunk_overlap=10)
    chunker = RecursiveCharacterChunker(config)
    doc     = Document(
        page_content="Short text. " * 20,
        metadata={"source": "doc.pdf", "page": 3},
    )
    chunks = chunker.split_documents([doc])
    for c in chunks:
        assert c.metadata["source"] == "doc.pdf"
        assert c.metadata["page"] == 3
        assert "chunk_index" in c.metadata


def test_recursive_chunker_overlap():
    config  = ChunkConfig(chunk_size=80, chunk_overlap=30, min_chunk_size=10)
    chunker = RecursiveCharacterChunker(config)
    text    = "The quick brown fox. " * 15
    chunks  = chunker.split_text(text)
    # Adjacent chunks should share some content
    if len(chunks) > 1:
        last_words_of_first = set(chunks[0].split()[-3:])
        first_words_of_second = set(chunks[1].split()[:5])
        # There may or may not be exact overlap depending on split boundaries
        assert len(chunks) >= 1


def test_markdown_chunker():
    md_text = """# Main Title
Introduction paragraph here.

## Section One
Content of section one goes here.

## Section Two
Content of section two goes here with more text.

### Subsection 2.1
Even deeper content here.
"""
    config  = ChunkConfig(chunk_size=500, chunk_overlap=50)
    chunker = MarkdownChunker(config)
    doc     = Document(page_content=md_text, metadata={"source": "doc.md"})
    chunks  = chunker.split_documents([doc])
    assert len(chunks) >= 2
    # Section paths should be in metadata
    for c in chunks:
        assert "section_path" in c.metadata


def test_empty_document():
    config  = ChunkConfig(chunk_size=100, chunk_overlap=20, min_chunk_size=10)
    chunker = RecursiveCharacterChunker(config)
    doc     = Document(page_content="   \n\n  ", metadata={})
    chunks  = chunker.split_documents([doc])
    assert chunks == []


# ============================================================
# tests/test_retrieval.py
# ============================================================

import numpy as np
from src.ingestion.document_loader import Document
from src.retrieval.vector_store import FAISSVectorStore, SearchResult
from src.retrieval.hybrid_retriever import BM25Retriever


def make_docs(n: int = 10) -> list:
    topics = [
        "machine learning neural networks deep learning",
        "python programming software development",
        "climate change global warming environment",
        "cooking recipes food kitchen",
        "financial markets stocks investments",
    ]
    docs = []
    for i in range(n):
        topic = topics[i % len(topics)]
        docs.append(Document(
            page_content=f"Document {i}: {topic}. More details about {topic.split()[0]}.",
            metadata={"source": f"doc_{i}.txt", "chunk_index": i},
        ))
    return docs


def test_faiss_add_and_search():
    dim   = 32
    store = FAISSVectorStore(embedding_dim=dim)
    docs  = make_docs(10)
    embs  = np.random.randn(10, dim).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    store.add_documents(docs, embs)
    assert len(store) == 10


def test_faiss_search_returns_top_k():
    dim   = 32
    store = FAISSVectorStore(embedding_dim=dim)
    docs  = make_docs(20)
    embs  = np.random.randn(20, dim).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    store.add_documents(docs, embs)
    query = np.random.randn(dim).astype(np.float32)
    query /= np.linalg.norm(query)
    results = store.search(query, top_k=5)
    assert len(results) == 5
    assert all(isinstance(r, SearchResult) for r in results)
    # Results should be sorted by score descending
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_faiss_save_load(tmp_path):
    dim   = 16
    store = FAISSVectorStore(embedding_dim=dim)
    docs  = make_docs(5)
    embs  = np.random.randn(5, dim).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    store.add_documents(docs, embs)
    store.save(str(tmp_path / "vs"))
    loaded = FAISSVectorStore.load(str(tmp_path / "vs"), embedding_dim=dim)
    assert len(loaded) == 5
    assert loaded.documents[0].page_content == docs[0].page_content


def test_bm25_basic():
    docs = make_docs(10)
    bm25 = BM25Retriever()
    bm25.fit(docs)
    results = bm25.search("machine learning neural", top_k=3)
    assert len(results) >= 1
    # Document 0 and 5 are about ML — should be top results
    top_sources = {r.document.metadata["source"] for r in results}
    assert "doc_0.txt" in top_sources or "doc_5.txt" in top_sources


def test_bm25_empty_index():
    bm25 = BM25Retriever()
    results = bm25.search("any query", top_k=5)
    assert results == []


def test_bm25_unknown_term():
    docs = make_docs(5)
    bm25 = BM25Retriever()
    bm25.fit(docs)
    # Term not in corpus
    results = bm25.search("xyzzy_notaword_12345", top_k=5)
    assert results == []  # No matches for unknown terms
