"""
retrieval/bm25_retriever.py
===========================
Sparse BM25 retriever over historical email corpus.
Uses rank-bm25 library with whitespace tokenisation.
"""

from __future__ import annotations
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.types import HistoricalEmail


def _tokenize(text: str) -> list[str]:
    """Lowercase + split on non-alphanumeric chars."""
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever:
    """
    Wraps rank_bm25.BM25Okapi with the HistoricalEmail corpus.
    Returns (email, bm25_score) pairs ranked by relevance.
    """

    def __init__(self, emails: list[HistoricalEmail]):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("rank-bm25 not installed. Run: pip install rank-bm25")

        self._emails = emails
        corpus = [_tokenize(e.to_retrieval_text()) for e in emails]
        self._bm25 = BM25Okapi(corpus)
        print(f"[BM25Retriever] Indexed {len(emails)} emails")

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[HistoricalEmail, float]]:
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        k = min(top_k, len(self._emails))
        import numpy as np
        top_idx = np.argsort(scores)[::-1][:k]
        results = [(self._emails[i], float(scores[i])) for i in top_idx]
        # Filter zero-score results (no token overlap at all)
        return [(e, s) for e, s in results if s > 0]
