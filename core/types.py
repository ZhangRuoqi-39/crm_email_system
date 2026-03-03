"""
core/types.py
=============
Central type definitions for the CRM Email Generation System.
All modules import from here — never define data shapes elsewhere.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4


class CampaignType(str, Enum):
    REACTIVATION   = "reactivation"
    EVENT_PROMO    = "event_promotion"
    SEASON_PASS    = "season_pass"
    VIP_UPGRADE    = "vip_upgrade"
    NEW_CONTENT    = "new_content_drop"
    RETENTION      = "retention"


class ToneStyle(str, Enum):
    URGENCY        = "urgency"
    STORYTELLING   = "storytelling"
    DIRECT_VALUE   = "direct_value"


class GameTitle(str, Enum):
    HONOR_OF_KINGS  = "Honor of Kings"
    PUBG_MOBILE     = "PUBG Mobile"
    LOL_MOBILE      = "League of Legends Mobile"
    VALORANT_MOBILE = "Valorant Mobile"
    GENERIC         = "Generic Game"


class PlayerSegment(str, Enum):
    LAPSED_VIP      = "Lapsed VIP Players"
    LAPSED_CASUAL   = "Lapsed Casual Players"
    ACTIVE_WHALE    = "Active Whale Players"
    NEW_PLAYERS     = "New Players"
    AT_RISK         = "At-Risk Players"


@dataclass
class CampaignBrief:
    """Structured input provided by the CRM team via the UI."""
    game: str
    campaign_type: CampaignType
    target_segment: PlayerSegment
    tone_style: ToneStyle
    kpi: str
    context: str
    brand_keywords: list[str] = field(default_factory=list)
    forbidden_keywords: list[str] = field(default_factory=list)
    brief_id: str = field(default_factory=lambda: str(uuid4())[:8])
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "brief_id": self.brief_id,
            "game": self.game,
            "campaign_type": self.campaign_type.value,
            "target_segment": self.target_segment.value,
            "tone_style": self.tone_style.value,
            "kpi": self.kpi,
            "context": self.context,
            "brand_keywords": self.brand_keywords,
            "forbidden_keywords": self.forbidden_keywords,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class HistoricalEmail:
    """A past email campaign record with performance metrics, stored in the knowledge base."""
    email_id: str
    game: str
    campaign_type: str
    target_segment: str
    subject_line: str
    preheader: str
    body_text: str
    cta_text: str
    tone_style: str
    open_rate: float
    ctr: float
    unsubscribe_rate: float
    conversion_rate: float
    sent_at: Optional[str] = None
    send_volume: int = 0
    performance_score: float = 0.0

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    def to_retrieval_text(self) -> str:
        """Text that gets embedded into the vector store for semantic search."""
        return (
            f"Game: {self.game}\n"
            f"Campaign: {self.campaign_type}\n"
            f"Segment: {self.target_segment}\n"
            f"Tone: {self.tone_style}\n"
            f"Subject: {self.subject_line}\n"
            f"Body: {self.body_text}\n"
            f"CTA: {self.cta_text}\n"
            f"Performance: open_rate={self.open_rate:.1%} ctr={self.ctr:.1%}"
        )


@dataclass
class QualityScore:
    """4-dimensional quality score for a generated email. All values 0.0–1.0."""
    relevance: float
    tone: float
    compliance: float
    creativity: float
    overall: float = 0.0
    passed_guardrails: bool = True
    guardrail_flags: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.overall = (
            self.relevance  * 0.30 +
            self.tone       * 0.25 +
            self.compliance * 0.25 +
            self.creativity * 0.20
        )

    def to_dict(self) -> dict:
        return {
            "relevance":         round(self.relevance, 3),
            "tone":              round(self.tone, 3),
            "compliance":        round(self.compliance, 3),
            "creativity":        round(self.creativity, 3),
            "overall":           round(self.overall, 3),
            "passed_guardrails": self.passed_guardrails,
            "guardrail_flags":   self.guardrail_flags,
        }


@dataclass
class EmailVariant:
    """A single AI-generated email variant."""
    variant_id: str = field(default_factory=lambda: str(uuid4())[:8])
    tone_style: ToneStyle = ToneStyle.URGENCY
    subject_line: str = ""
    preheader: str = ""
    body_text: str = ""
    cta_text: str = ""
    html_content: str = ""
    quality_score: Optional[QualityScore] = None

    def to_dict(self) -> dict:
        d = {
            "variant_id": self.variant_id,
            "tone_style": self.tone_style.value,
            "subject_line": self.subject_line,
            "preheader": self.preheader,
            "body_text": self.body_text,
            "cta_text": self.cta_text,
            "html_content": self.html_content,
        }
        if self.quality_score:
            d["quality_score"] = self.quality_score.to_dict()
        return d


@dataclass
class RetrievalResult:
    """A single result from hybrid search, carrying the email and its fusion score."""
    email: HistoricalEmail
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rrf_score: float = 0.0
    rank: int = 0


@dataclass
class GenerationResult:
    """The final output of one full pipeline run."""
    result_id: str = field(default_factory=lambda: str(uuid4())[:8])
    brief: Optional[CampaignBrief] = None
    retrieved_emails: list[HistoricalEmail] = field(default_factory=list)
    variants: list[EmailVariant] = field(default_factory=list)
    best_variant_id: Optional[str] = None
    pipeline_metadata: dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.utcnow)

    def get_best_variant(self) -> Optional[EmailVariant]:
        if not self.variants:
            return None
        return max(
            self.variants,
            key=lambda v: v.quality_score.overall if v.quality_score else 0.0
        )

    def to_dict(self) -> dict:
        return {
            "result_id": self.result_id,
            "brief": self.brief.to_dict() if self.brief else None,
            "variants": [v.to_dict() for v in self.variants],
            "best_variant_id": self.best_variant_id,
            "pipeline_metadata": self.pipeline_metadata,
            "generated_at": self.generated_at.isoformat(),
        }
