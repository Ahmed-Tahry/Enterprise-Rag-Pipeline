"""
src/ingestion/chunker.py
========================
Smart document chunking strategies.

Why chunking matters:
  - Too large: retrieval is imprecise, context is diluted
  - Too small: lacks context for meaningful answers
  - Bad splits: cut sentences mid-thought, breaking coherence

Strategies implemented:
  1. RecursiveCharacterChunker  — default, sentence-aware
  2. SemanticChunker            — splits on semantic similarity breaks
  3. MarkdownChunker            — preserves markdown header hierarchy
"""

import re
from typing import List, Optional
from dataclasses import dataclass, field
from src.ingestion.document_loader import Document
from loguru import logger


@dataclass
class ChunkConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 50       # Discard chunks smaller than this
    add_start_index: bool = True   # Track where chunk starts in original doc


class RecursiveCharacterChunker:
    """
    Splits text recursively on natural separators:
    paragraph breaks → sentences → words → characters.

    This is the production default — works well across all document types.
    Sentence-aware: won't cut mid-sentence unless absolutely necessary.
    """

    SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]

    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()

    def split_documents(self, documents: List[Document]) -> List[Document]:
        chunks = []
        for doc in documents:
            doc_chunks = self.split_text(doc.page_content)
            for i, chunk_text in enumerate(doc_chunks):
                new_meta = dict(doc.metadata)
                new_meta["chunk_index"] = i
                new_meta["total_chunks"] = len(doc_chunks)
                chunks.append(Document(page_content=chunk_text, metadata=new_meta))
        logger.info(f"Split {len(documents)} docs → {len(chunks)} chunks")
        return chunks

    def split_text(self, text: str) -> List[str]:
        return self._split_recursive(text, self.SEPARATORS)

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        if not separators:
            return self._merge_splits([text], "")

        separator = separators[0]
        remaining = separators[1:]

        if separator == "":
            splits = list(text)
        else:
            splits = text.split(separator)

        # Re-attach separator to splits (keep punctuation with sentence)
        if separator in {". ", "? ", "! "}:
            splits = [s + separator if i < len(splits) - 1 else s
                      for i, s in enumerate(splits)]

        good_splits, final_chunks = [], []

        for split in splits:
            if len(split) <= self.config.chunk_size:
                good_splits.append(split)
            else:
                if good_splits:
                    final_chunks.extend(self._merge_splits(good_splits, separator))
                    good_splits = []
                sub_chunks = self._split_recursive(split, remaining)
                final_chunks.extend(sub_chunks)

        if good_splits:
            final_chunks.extend(self._merge_splits(good_splits, separator))

        return final_chunks

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge small splits into chunks of target size with overlap."""
        chunks = []
        current_doc = []
        current_len = 0

        for split in splits:
            split_len = len(split)

            if current_len + split_len > self.config.chunk_size and current_doc:
                chunk_text = separator.join(current_doc).strip()
                if len(chunk_text) >= self.config.min_chunk_size:
                    chunks.append(chunk_text)

                # Keep overlap: remove from front until we're within overlap budget
                while current_doc and current_len > self.config.chunk_overlap:
                    removed = current_doc.pop(0)
                    current_len -= len(removed)

            current_doc.append(split)
            current_len += split_len

        if current_doc:
            chunk_text = separator.join(current_doc).strip()
            if len(chunk_text) >= self.config.min_chunk_size:
                chunks.append(chunk_text)

        return chunks


class MarkdownChunker:
    """
    Chunks Markdown documents by header hierarchy.

    Preserves document structure:
    - H1 → top-level section
    - H2 → subsection (included with H1 context)
    - H3+ → sub-subsection

    Each chunk carries its full header breadcrumb as metadata context,
    which significantly improves retrieval relevance.
    """

    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()
        self.text_chunker = RecursiveCharacterChunker(config)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        chunks = []
        for doc in documents:
            chunks.extend(self._split_markdown(doc))
        logger.info(f"Markdown chunked {len(documents)} docs → {len(chunks)} chunks")
        return chunks

    def _split_markdown(self, doc: Document) -> List[Document]:
        text = doc.page_content
        lines = text.split("\n")

        sections = []
        current_headers = {}  # level → header text
        current_content = []

        for line in lines:
            header_match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if header_match:
                # Save previous section
                if current_content:
                    sections.append({
                        "headers": dict(current_headers),
                        "content": "\n".join(current_content),
                    })
                    current_content = []

                level = len(header_match.group(1))
                title = header_match.group(2)
                current_headers[level] = title
                # Remove headers deeper than current
                for l in list(current_headers.keys()):
                    if l > level:
                        del current_headers[l]
            else:
                current_content.append(line)

        if current_content:
            sections.append({
                "headers": dict(current_headers),
                "content": "\n".join(current_content),
            })

        chunks = []
        for section in sections:
            # Build breadcrumb: "Section > Subsection > Sub-subsection"
            breadcrumb = " > ".join(
                section["headers"].get(l, "")
                for l in sorted(section["headers"].keys())
                if section["headers"].get(l)
            )
            content = section["content"].strip()
            if not content:
                continue

            # Prepend breadcrumb to content for better retrieval
            full_content = f"[Section: {breadcrumb}]\n\n{content}" if breadcrumb else content

            # Further split if section is too large
            if len(full_content) > self.config.chunk_size:
                sub_chunks = self.text_chunker.split_text(full_content)
                for i, sub in enumerate(sub_chunks):
                    meta = dict(doc.metadata)
                    meta.update({"section_path": breadcrumb, "sub_chunk": i})
                    chunks.append(Document(page_content=sub, metadata=meta))
            else:
                meta = dict(doc.metadata)
                meta["section_path"] = breadcrumb
                chunks.append(Document(page_content=full_content, metadata=meta))

        return chunks


class SemanticChunker:
    """
    Splits text at semantic boundaries using embedding similarity.

    Algorithm:
    1. Split text into sentences
    2. Embed each sentence
    3. Find sentences where cosine similarity with the NEXT sentence drops sharply
    4. Those are semantic boundaries → split there

    Produces more coherent chunks than fixed-size splitting,
    at the cost of requiring an embedding model during ingestion.
    """

    def __init__(
        self,
        embedding_model=None,
        breakpoint_threshold: float = 0.5,
        config: Optional[ChunkConfig] = None,
    ):
        self.model = embedding_model
        self.threshold = breakpoint_threshold
        self.config = config or ChunkConfig()

    def split_documents(self, documents: List[Document]) -> List[Document]:
        if self.model is None:
            logger.warning("No embedding model provided for SemanticChunker. Falling back to RecursiveCharacterChunker.")
            return RecursiveCharacterChunker(self.config).split_documents(documents)

        chunks = []
        for doc in documents:
            chunks.extend(self._split_semantic(doc))
        logger.info(f"Semantic chunked {len(documents)} docs → {len(chunks)} chunks")
        return chunks

    def _split_semantic(self, doc: Document) -> List[Document]:
        import numpy as np

        # Sentence tokenisation
        sentences = re.split(r"(?<=[.!?])\s+", doc.page_content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if len(sentences) <= 2:
            return [doc]

        # Embed all sentences
        embeddings = self.model.encode(sentences, normalize_embeddings=True)

        # Find cosine similarity between adjacent sentences
        similarities = [
            float(embeddings[i] @ embeddings[i + 1])
            for i in range(len(embeddings) - 1)
        ]

        # Identify breakpoints: where similarity drops below threshold
        mean_sim = float(np.mean(similarities))
        breakpoints = [
            i + 1 for i, sim in enumerate(similarities)
            if sim < (mean_sim * self.threshold)
        ]

        # Build chunks from breakpoints
        chunk_texts = []
        start = 0
        for bp in breakpoints:
            chunk_texts.append(" ".join(sentences[start:bp]))
            start = bp
        chunk_texts.append(" ".join(sentences[start:]))

        result = []
        for i, chunk_text in enumerate(chunk_texts):
            if len(chunk_text.strip()) >= self.config.min_chunk_size:
                meta = dict(doc.metadata)
                meta["chunk_index"] = i
                result.append(Document(page_content=chunk_text.strip(), metadata=meta))

        return result


def get_chunker(strategy: str = "recursive", config: Optional[ChunkConfig] = None, **kwargs):
    """Factory function to get a chunker by strategy name."""
    strategies = {
        "recursive": RecursiveCharacterChunker,
        "markdown":  MarkdownChunker,
        "semantic":  SemanticChunker,
    }
    if strategy not in strategies:
        raise ValueError(f"Unknown chunking strategy: '{strategy}'. Choose from: {list(strategies.keys())}")
    return strategies[strategy](config=config, **kwargs)
