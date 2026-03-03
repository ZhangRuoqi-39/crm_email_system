"""
retrieval/vector_store.py
=========================
Dense vector store using numpy cosine similarity.
Calls 通义千问 text-embedding-v3 via DashScope API for embeddings.

Falls back gracefully to deterministic random vectors when DASHSCOPE_API_KEY
is not set, so the UI can still be demonstrated without live credentials.
"""

from __future__ import annotations
import json
import os
import sys
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.base import BaseEmbedding, BaseVectorStore
from core.config import get_settings, get_api_key
from core.types import HistoricalEmail


# ──────────────────────────────────────────────
# Qwen Embedding Client
# ──────────────────────────────────────────────

class QwenEmbedding(BaseEmbedding):
    """
    通义千问 text-embedding-v3 via DashScope HTTP API.
    
    Correct DashScope v1 request format:
      POST https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding
      {
        "model": "text-embedding-v3",
        "input": { "texts": ["text1", "text2"] },
        "parameters": { "text_type": "document" }   # or "query"
      }
    """

    ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
    EMBED_DIM = 1024
    # DashScope text-embedding-v3 max texts per request
    BATCH_SIZE = 10

    def __init__(self):
        self.settings = get_settings()
        self._api_key: Optional[str] = None
        self._live = False
        try:
            self._api_key = get_api_key("qwen")
            self._live = True
            print(f"[QwenEmbedding] Live mode — model: {self.settings.embedding.model}")
        except EnvironmentError:
            print("[QwenEmbedding] DASHSCOPE_API_KEY not set — using random fallback embeddings")

    def _call_api(self, texts: list[str], text_type: str = "document") -> list[list[float]]:
        """
        Call DashScope embedding API for a batch of texts.
        text_type: "document" for indexing, "query" for search queries.
        """
        import urllib.request
        import urllib.error

        payload = json.dumps({
            "model": self.settings.embedding.model,
            "input": {
                "texts": texts
            },
            "parameters": {
                "text_type": text_type
            }
        }, ensure_ascii=False).encode("utf-8")

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json;charset=utf-8",
            "X-DashScope-Async": "disable",
        }

        req = urllib.request.Request(
            self.ENDPOINT,
            data=payload,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"DashScope API error {e.code}: {e.reason}\n"
                f"Response body: {body}\n"
                f"Request payload sample: model={self.settings.embedding.model}, "
                f"texts[0]={texts[0][:80] if texts else 'empty'}"
            ) from e

        embeddings = result["output"]["embeddings"]
        # Sort by text_index to preserve original order
        embeddings.sort(key=lambda x: x["text_index"])
        return [e["embedding"] for e in embeddings]

    def _random_embed(self, texts: list[str]) -> list[list[float]]:
        """Deterministic fallback: hash-seeded unit vectors."""
        vecs = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.standard_normal(self.EMBED_DIM).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-9
            vecs.append(v.tolist())
        return vecs

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents (for indexing)."""
        if not texts:
            return []
        if not self._live:
            return self._random_embed(texts)

        results = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i: i + self.BATCH_SIZE]
            batch_vecs = self._call_api(batch, text_type="document")
            results.extend(batch_vecs)
        return results

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""
        if not self._live:
            return self._random_embed([text])[0]
        return self._call_api([text], text_type="query")[0]


# ──────────────────────────────────────────────
# Numpy-based Vector Store
# ──────────────────────────────────────────────

class NumpyVectorStore(BaseVectorStore):
    """
    In-memory cosine-similarity vector store backed by numpy.
    Persists to disk as a pickle for reuse across sessions.
    """

    def __init__(self):
        self._emails: list[HistoricalEmail] = []
        self._matrix: Optional[np.ndarray] = None   # shape (N, D)

    def add(self, emails: list[HistoricalEmail], embeddings: list[list[float]]) -> None:
        self._emails.extend(emails)
        new_vecs = np.array(embeddings, dtype=np.float32)
        # L2-normalise for cosine similarity via dot product
        norms = np.linalg.norm(new_vecs, axis=1, keepdims=True) + 1e-9
        new_vecs = new_vecs / norms
        if self._matrix is None:
            self._matrix = new_vecs
        else:
            self._matrix = np.vstack([self._matrix, new_vecs])
        print(f"[NumpyVectorStore] {len(self._emails)} vectors indexed (dim={self._matrix.shape[1]})")

    def search(self, query_embedding: list[float], top_k: int) -> list[tuple[HistoricalEmail, float]]:
        if self._matrix is None or len(self._emails) == 0:
            return []
        q = np.array(query_embedding, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)
        scores = self._matrix @ q
        k = min(top_k, len(self._emails))
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [(self._emails[i], float(scores[i])) for i in top_idx]

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"emails": self._emails, "matrix": self._matrix}, f)
        print(f"[NumpyVectorStore] Saved to {path}")

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._emails = data["emails"]
        self._matrix = data["matrix"]
        print(f"[NumpyVectorStore] Loaded {len(self._emails)} vectors from {path}")

    def __len__(self) -> int:
        return len(self._emails)


# ──────────────────────────────────────────────
# Build / load vector index
# ──────────────────────────────────────────────

def build_vector_index(
    emails: list[HistoricalEmail],
    index_path: str = "data/processed/vector_index.pkl",
    force_rebuild: bool = False,
) -> NumpyVectorStore:
    """
    Build (or load from cache) a dense vector index over historical emails.
    Called by the pipeline and Streamlit UI on startup.
    """
    store = NumpyVectorStore()
    cache = Path(index_path)

    if cache.exists() and not force_rebuild:
        store.load(index_path)
        if len(store) == len(emails):
            print(f"[build_vector_index] Loaded cached index ({len(store)} vectors)")
            return store
        print("[build_vector_index] Cache size mismatch — rebuilding")

    embedder = QwenEmbedding()
    texts = [e.to_retrieval_text() for e in emails]
    print(f"[build_vector_index] Embedding {len(texts)} emails via {embedder.settings.embedding.model}...")
    vectors = embedder.embed(texts)
    store.add(emails, vectors)
    store.save(index_path)
    return store