"""
scripts/ingest.py
==================
CLI script to index documents from a directory.

Usage:
    python scripts/ingest.py --dir ./data/docs
    python scripts/ingest.py --dir ./data/docs --save ./data/vectorstore
    python scripts/ingest.py --dir ./data/docs --chunk-size 512 --overlap 64
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Index documents into the RAG vector store")
    parser.add_argument("--dir",        required=True,  help="Directory containing documents to index")
    parser.add_argument("--save",       default="./data/vectorstore", help="Path to save the vector store")
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size in characters")
    parser.add_argument("--overlap",    type=int, default=64,  help="Chunk overlap in characters")
    parser.add_argument("--strategy",   default="recursive", choices=["recursive", "markdown", "semantic"])
    parser.add_argument("--embedder",   default="huggingface", choices=["huggingface", "openai"])
    parser.add_argument("--model",      default="BAAI/bge-base-en-v1.5", help="Embedding model name")
    args = parser.parse_args()

    from src.ingestion.document_loader import DocumentLoader
    from src.ingestion.chunker import ChunkConfig, get_chunker
    from src.ingestion.embedder import get_embedder
    from src.retrieval.vector_store import FAISSVectorStore
    from src.retrieval.hybrid_retriever import BM25Retriever

    # 1. Load documents
    logger.info(f"Loading documents from: {args.dir}")
    loader = DocumentLoader()
    docs   = loader.load_directory(args.dir)
    logger.info(f"Loaded {len(docs)} document pages")

    # 2. Chunk
    config  = ChunkConfig(chunk_size=args.chunk_size, chunk_overlap=args.overlap)
    chunker = get_chunker(args.strategy, config=config)
    chunks  = chunker.split_documents(docs)
    logger.info(f"Chunked into {len(chunks)} chunks")

    # 3. Embed
    logger.info(f"Embedding with {args.embedder}/{args.model}...")
    embedder   = get_embedder(args.embedder, model_name=args.model)
    texts      = [c.page_content for c in chunks]
    embeddings = embedder.embed_documents(texts)
    logger.info(f"Embedded {len(embeddings)} chunks (dim={embedder.dimension})")

    # 4. Index
    vector_store = FAISSVectorStore(embedding_dim=embedder.dimension)
    vector_store.add_documents(chunks, embeddings)
    vector_store.save(args.save)

    bm25 = BM25Retriever()
    bm25.fit(chunks)
    import pickle
    with open(Path(args.save) / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    logger.success(
        f"\n✅ Indexing complete!\n"
        f"   Documents: {len(docs)}\n"
        f"   Chunks:    {len(chunks)}\n"
        f"   Saved to:  {args.save}"
    )


if __name__ == "__main__":
    main()
