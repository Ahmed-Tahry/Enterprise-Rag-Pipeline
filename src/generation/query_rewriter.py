import re
from typing import List

from loguru import logger


REWRITE_SYSTEM_PROMPT = (
    "You are a query rewriting assistant that helps improve search quality."
)

REWRITE_USER_PROMPT = """Original question: {question}

Generate {num_rewrites} alternative phrasings that capture the same intent using different words and sentence structures. Then generate a short hypothetical document passage that would perfectly answer the question.

Format:
ALTERNATIVE 1: ...
ALTERNATIVE 2: ...
ALTERNATIVE 3: ...
HYPOTHETICAL_DOC: ..."""


class QueryRewriter:
    """
    Generate multiple query variations (paraphrases + HyDE) to improve retrieval recall.

    Multi-Query: catches synonyms, reformulations, and underspecified questions.
    HyDE (Hypothetical Document Embeddings): embeds a "perfect answer" which is
    semantically closer to real documents than the question itself.
    """

    def __init__(self, llm, num_rewrites: int = 3):
        self.llm = llm
        self.num_rewrites = num_rewrites

    def rewrite(self, question: str) -> List[str]:
        prompt = REWRITE_USER_PROMPT.format(
            question=question, num_rewrites=self.num_rewrites
        )
        try:
            response = self.llm.generate(REWRITE_SYSTEM_PROMPT, prompt)
            queries = self._parse(response)
            if not queries:
                logger.warning(
                    f"Query rewriting produced no alternatives for '{question[:60]}...'"
                )
        except Exception as e:
            logger.warning(f"Query rewriting failed: {e}")
            queries = []

        all_queries = [question] + queries

        seen = set()
        unique = []
        for q in all_queries:
            normalized = q.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                unique.append(q)
        return unique

    def _parse(self, response: str) -> List[str]:
        queries = []
        for line in response.split("\n"):
            line = line.strip()
            alt_match = re.match(r"ALTERNATIVE\s+\d+:\s*(.+)", line, re.IGNORECASE)
            hyde_match = re.match(r"HYPOTHETICAL_DOC:\s*(.+)", line, re.IGNORECASE)
            if alt_match:
                queries.append(alt_match.group(1).strip())
            elif hyde_match:
                queries.append(hyde_match.group(1).strip())
        return queries
