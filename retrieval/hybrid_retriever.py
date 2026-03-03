"""
retrieval/hybrid_retriever.py
==============================
Hybrid retriever: BM25 (sparse) + Qwen Embedding (dense) fused via RRF,
then optionally re-ranked by 通义千问 gte-rerank.

Architecture:
  query
    ├─ BM25Retriever  → sparse_results (email, bm25_score)
    ├─ NumpyVectorStore.search → dense_results (email, cosine_score)
    ├─ RRF fusion → unified ranked list
    └─ QwenReranker (optional, live API only) → final top-k
"""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.base import BaseRetriever
from core.config import get_settings, get_api_key
from core.types import CampaignBrief, HistoricalEmail, RetrievalResult
from retrieval.bm25_retriever import BM25Retriever
from retrieval.vector_store import NumpyVectorStore, QwenEmbedding


# ──────────────────────────────────────────────
# Qwen Reranker
# ──────────────────────────────────────────────

class QwenReranker:
    """
    通义千问 gte-rerank via DashScope text-rerank endpoint.
    Falls back to score pass-through when API key is absent.
    """

    ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"

    def __init__(self):
        self.settings = get_settings()
        self._api_key = None
        self._live = False
        try:
            self._api_key = get_api_key("qwen")
            self._live = True
        except EnvironmentError:
            print("[QwenReranker] DASHSCOPE_API_KEY not set — skipping rerank step")

    def rerank(
        self,
        query: str,
        candidates: list[HistoricalEmail],
        top_n: int = 3,
    ) -> list[tuple[HistoricalEmail, float]]:
        if not self._live or not candidates:
            return [(e, 1.0 / (i + 1)) for i, e in enumerate(candidates[:top_n])]

        import urllib.request, urllib.error
        docs = [e.to_retrieval_text() for e in candidates]
        payload = json.dumps({
            "model": self.settings.rerank.model,
            "input": {
                "query": query,
                "documents": docs,
            },
            "parameters": {
                "top_n": top_n,
                "return_documents": False,
            },
        }).encode()
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        try:
            req = urllib.request.Request(
                self.ENDPOINT, data=payload, headers=headers, method="POST"
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
            ranked = result["output"]["results"]
            ranked.sort(key=lambda x: x["index"])
            # Map index back to original candidates list
            out = []
            for item in sorted(ranked, key=lambda x: x["relevance_score"], reverse=True):
                out.append((candidates[item["index"]], float(item["relevance_score"])))
            return out[:top_n]
        except Exception as e:
            print(f"[QwenReranker] API error: {e} — falling back to RRF order")
            return [(e, 1.0 / (i + 1)) for i, e in enumerate(candidates[:top_n])]


# ──────────────────────────────────────────────
# RRF fusion
# ──────────────────────────────────────────────

def _rrf_fuse(
    dense_ranked: list[tuple[HistoricalEmail, float]],
    sparse_ranked: list[tuple[HistoricalEmail, float]],
    rrf_k: int = 60,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> list[RetrievalResult]:
    """
    Reciprocal Rank Fusion.
    score(d) = dense_weight * 1/(k + rank_dense) + sparse_weight * 1/(k + rank_sparse)
    """
    # Build rank maps keyed by email_id
    dense_rank = {e.email_id: (rank, score) for rank, (e, score) in enumerate(dense_ranked, 1)}
    sparse_rank = {e.email_id: (rank, score) for rank, (e, score) in enumerate(sparse_ranked, 1)}

    all_ids = set(dense_rank) | set(sparse_rank)
    # Build lookup from email_id → HistoricalEmail
    id_to_email: dict[str, HistoricalEmail] = {}
    for e, _ in dense_ranked + sparse_ranked:
        id_to_email[e.email_id] = e

    results: list[RetrievalResult] = []
    for eid in all_ids:
        d_rank, d_score = dense_rank.get(eid, (len(dense_ranked) + rrf_k, 0.0))
        s_rank, s_score = sparse_rank.get(eid, (len(sparse_ranked) + rrf_k, 0.0))
        rrf = (dense_weight / (rrf_k + d_rank)) + (sparse_weight / (rrf_k + s_rank))
        results.append(RetrievalResult(
            email=id_to_email[eid],
            dense_score=d_score,
            sparse_score=s_score,
            rrf_score=rrf,
        ))

    results.sort(key=lambda r: r.rrf_score, reverse=True)
    for i, r in enumerate(results):
        r.rank = i + 1
    return results


# ──────────────────────────────────────────────
# HybridRetriever
# ──────────────────────────────────────────────

class HybridRetriever(BaseRetriever):
    """
    Combines BM25 + Qwen dense retrieval via RRF, then applies Qwen reranker.

    Usage:
        retriever = HybridRetriever(emails, vector_store)
        results = retriever.retrieve_from_brief(brief, top_k=5)
    """

    def __init__(
        self,
        emails: list[HistoricalEmail],
        vector_store: NumpyVectorStore,
    ):
        self.settings = get_settings()
        self._emails = emails
        self._bm25 = BM25Retriever(emails)
        self._vector_store = vector_store
        self._embedder = QwenEmbedding()
        self._reranker = QwenReranker()

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        cfg = self.settings.retrieval

        # 1. Dense retrieval
        q_vec = self._embedder.embed_query(query)
        dense_raw = self._vector_store.search(q_vec, top_k=cfg.top_k)

        # 2. Sparse retrieval
        sparse_raw = self._bm25.retrieve(query, top_k=cfg.top_k)

        # 3. RRF fusion
        fused = _rrf_fuse(
            dense_ranked=dense_raw,
            sparse_ranked=sparse_raw,
            rrf_k=cfg.rrf_k,
            dense_weight=cfg.dense_weight,
            sparse_weight=cfg.bm25_weight,
        )

        # 4. Qwen rerank on top candidates
        pre_rerank = [r.email for r in fused[: cfg.top_k]]
        reranked = self._reranker.rerank(query, pre_rerank, top_n=top_k)

        # Rebuild RetrievalResult list from reranked order
        rerank_scores = {e.email_id: score for e, score in reranked}
        rrf_map = {r.email.email_id: r for r in fused}

        final: list[RetrievalResult] = []
        for e, _ in reranked:
            rr = rrf_map.get(e.email_id)
            if rr:
                # Annotate with rerank score in rrf_score field
                rr.rrf_score = rerank_scores.get(e.email_id, rr.rrf_score)
                final.append(rr)

        for i, r in enumerate(final):
            r.rank = i + 1

        return final

    def retrieve_from_brief(
        self, brief: CampaignBrief, top_k: int = 5
    ) -> list[RetrievalResult]:
        query = self.build_query(brief)
        return self.retrieve(query, top_k=top_k)
