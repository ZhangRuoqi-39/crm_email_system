"""
core/base.py
============
Abstract base classes for every pluggable component.
Inspired by DEV_SPEC: clean interfaces allow swapping any component with zero
changes to the rest of the pipeline.
"""

from abc import ABC, abstractmethod
from typing import Any
from core.types import (
    CampaignBrief, EmailVariant, HistoricalEmail,
    QualityScore, RetrievalResult,
)


class BaseLoader(ABC):
    @abstractmethod
    def load(self, source_path: str) -> list[HistoricalEmail]: ...

    def validate(self, emails: list[HistoricalEmail]) -> list[HistoricalEmail]:
        return [e for e in emails if e.subject_line and e.body_text]


class BaseTransform(ABC):
    @abstractmethod
    def transform(self, emails: list[HistoricalEmail]) -> list[HistoricalEmail]: ...


class BaseEmbedding(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...


class BaseVectorStore(ABC):
    @abstractmethod
    def add(self, emails: list[HistoricalEmail], embeddings: list[list[float]]) -> None: ...

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int) -> list[tuple[HistoricalEmail, float]]: ...

    @abstractmethod
    def save(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> None: ...


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]: ...

    def build_query(self, brief: CampaignBrief) -> str:
        return (
            f"{brief.game} {brief.campaign_type.value} "
            f"{brief.target_segment.value} {brief.tone_style.value} "
            f"{brief.context}"
        )


class BaseGenerator(ABC):
    @abstractmethod
    def generate(
        self,
        brief: CampaignBrief,
        retrieved_context: list[HistoricalEmail],
        num_variants: int = 3,
    ) -> list[EmailVariant]: ...


class BaseScorer(ABC):
    @abstractmethod
    def score(self, variant: EmailVariant, brief: CampaignBrief) -> QualityScore: ...

    def score_batch(self, variants: list[EmailVariant], brief: CampaignBrief) -> list[QualityScore]:
        return [self.score(v, brief) for v in variants]


class BaseGuardrail(ABC):
    @abstractmethod
    def check(self, variant: EmailVariant, brief: CampaignBrief) -> tuple[bool, list[str]]: ...


class BaseTemplateRenderer(ABC):
    @abstractmethod
    def render(self, variant: EmailVariant, brief: CampaignBrief) -> str: ...


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, test_cases: list[dict[str, Any]]) -> dict[str, Any]: ...
