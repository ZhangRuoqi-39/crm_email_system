"""
core/config.py
==============
Loads config.yaml and exposes a typed Settings object.
All modules call get_settings() — never read yaml directly elsewhere.

Providers supported:
  LLM:       deepseek | anthropic | openai
  Embedding: qwen | openai | sentence-transformers
  Rerank:    qwen | cohere
"""

from __future__ import annotations
import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    provider: str = "deepseek"
    model: str = "deepseek-chat"
    temperature: float = 0.7
    max_tokens: int = 2048


class EmbeddingConfig(BaseModel):
    provider: str = "qwen"
    model: str = "text-embedding-v3"
    batch_size: int = 32


class RerankConfig(BaseModel):
    provider: str = "qwen"
    model: str = "gte-rerank"
    top_n: int = 3


class RetrievalConfig(BaseModel):
    top_k: int = 10
    rerank_top_k: int = 3
    bm25_weight: float = 0.4
    dense_weight: float = 0.6
    rrf_k: int = 60


class GenerationConfig(BaseModel):
    num_variants: int = 3
    variant_styles: list[str] = Field(default_factory=lambda: ["urgency", "storytelling", "direct_value"])
    max_subject_length: int = 60
    max_body_length: int = 400


class ScoringConfig(BaseModel):
    weights: dict[str, float] = Field(
        default_factory=lambda: {"relevance": 0.30, "tone": 0.25, "compliance": 0.25, "creativity": 0.20}
    )
    compliance_threshold: float = 0.70
    banned_words: list[str] = Field(default_factory=list)


class DataConfig(BaseModel):
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    templates_dir: str = "data/templates"
    historical_emails: str = "data/processed/historical_emails.json"
    campaign_briefs: str = "data/processed/campaign_briefs.json"
    faiss_index: str = "data/processed/faiss_index"


class EvaluationConfig(BaseModel):
    golden_set_path: str = "data/processed/golden_test_set.json"
    baseline_open_rate: float = 0.21
    baseline_ctr: float = 0.030
    uplift_simulation_runs: int = 100


class Settings(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


@lru_cache(maxsize=1)
def get_settings(config_path: str = "config.yaml") -> Settings:
    """Load and cache settings from config.yaml."""
    path = Path(config_path)
    if not path.exists():
        path = Path(__file__).parent.parent / "config.yaml"
    if not path.exists():
        print("[config] Warning: config.yaml not found, using defaults.")
        return Settings()
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Settings(**raw)


def get_api_key(provider: str) -> str:
    """Fetch API key from environment. Raises EnvironmentError if missing."""
    key_map = {
        "deepseek":  "DEEPSEEK_API_KEY",
        "qwen":      "DASHSCOPE_API_KEY",   # 通义千问用DashScope key
        "anthropic": "ANTHROPIC_API_KEY",
        "openai":    "OPENAI_API_KEY",
        "azure":     "AZURE_OPENAI_API_KEY",
    }
    env_var = key_map.get(provider)
    if not env_var:
        raise ValueError(f"Unknown provider: {provider}")
    key = os.environ.get(env_var)
    if not key:
        raise EnvironmentError(
            f"Missing API key for '{provider}'. Run: export {env_var}='your-key-here'"
        )
    return key
