"""
scoring/quality_scorer.py
==========================
LLM-as-judge quality scorer: rates each email variant on 4 dimensions.
Uses DeepSeek (same API key) for scoring. Falls back to heuristic scoring.
"""

from __future__ import annotations
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base import BaseScorer
from core.config import get_settings, get_api_key
from core.types import CampaignBrief, EmailVariant, QualityScore

DEEPSEEK_BASE_URL = "https://api.deepseek.com"

SCORER_SYSTEM = """You are a senior CRM email quality analyst for a global gaming company.
Score the provided email on 4 dimensions, each from 0.0 to 1.0.
Respond ONLY with a valid JSON object — no markdown, no explanation outside JSON."""

SCORER_PROMPT = """
EMAIL TO SCORE:
  Subject:  {subject}
  Preheader:{preheader}
  Body:     {body}
  CTA:      {cta}

CAMPAIGN CONTEXT:
  Game: {game} | Type: {campaign_type} | Segment: {segment} | KPI: {kpi}
  Tone requested: {tone}

Score on these 4 dimensions (0.0–1.0):
- relevance:   Does content match campaign type, segment, and KPI goal?
- tone:        Does writing style match the requested tone ({tone})?
- compliance:  Is content professional, non-misleading, and brand-safe?
- creativity:  Is the copy original, engaging, and memorable?

Respond with ONLY this JSON:
{{
  "relevance": 0.0,
  "tone": 0.0,
  "compliance": 0.0,
  "creativity": 0.0,
  "rationale": "One sentence explaining the scores."
}}
"""


def _heuristic_score(variant: EmailVariant, brief: CampaignBrief) -> QualityScore:
    """Rule-based fallback scorer when LLM is unavailable."""
    subject = variant.subject_line
    body = variant.body_text
    tone = variant.tone_style.value

    # Relevance: check game/segment name appears in content
    game_mentioned = brief.game.lower().split()[0] in (subject + body).lower()
    relevance = 0.75 + (0.15 if game_mentioned else 0.0)

    # Tone match: check tone-specific signal words
    tone_signals = {
        "urgency": ["expires", "tonight", "now", "last chance", "limited", "hurry", "countdown"],
        "storytelling": ["story", "legend", "chapter", "journey", "remember", "world", "awaits"],
        "direct_value": ["free", "get", "bundle", "coins", "no purchase", "%", "instantly"],
    }
    signals = tone_signals.get(tone, [])
    hits = sum(1 for s in signals if s in body.lower())
    tone_score = min(1.0, 0.50 + hits * 0.08)

    # Compliance: penalise banned words
    banned = brief.forbidden_keywords + get_settings().scoring.banned_words
    violations = sum(1 for w in banned if w.lower() in (subject + body).lower())
    compliance = max(0.0, 1.0 - violations * 0.25)

    # Creativity: length and structure variety
    has_bullets = "\n-" in body or "•" in body
    word_count = len(body.split())
    creativity = 0.60 + (0.15 if has_bullets else 0.0) + min(0.20, word_count / 300)

    return QualityScore(
        relevance=round(min(1.0, relevance), 3),
        tone=round(tone_score, 3),
        compliance=round(compliance, 3),
        creativity=round(min(1.0, creativity), 3),
    )


class LLMQualityScorer(BaseScorer):
    """
    Scores email variants using DeepSeek as judge.
    Gracefully degrades to heuristic scorer when API key is absent.
    """

    def __init__(self):
        self.settings = get_settings()
        self._client = None
        self._live = False
        try:
            api_key = get_api_key("deepseek")
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
            self._live = True
        except (EnvironmentError, ImportError):
            print("[LLMQualityScorer] Falling back to heuristic scorer")

    def score(self, variant: EmailVariant, brief: CampaignBrief) -> QualityScore:
        if not self._live:
            return _heuristic_score(variant, brief)

        prompt = SCORER_PROMPT.format(
            subject=variant.subject_line,
            preheader=variant.preheader,
            body=variant.body_text[:400],
            cta=variant.cta_text,
            game=brief.game,
            campaign_type=brief.campaign_type.value,
            segment=brief.target_segment.value,
            kpi=brief.kpi,
            tone=variant.tone_style.value,
        )
        try:
            response = self._client.chat.completions.create(
                model=self.settings.llm.model,
                temperature=0.1,   # Low temp for consistent scoring
                max_tokens=256,
                messages=[
                    {"role": "system", "content": SCORER_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            raw = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)
            return QualityScore(
                relevance=float(data.get("relevance", 0.7)),
                tone=float(data.get("tone", 0.7)),
                compliance=float(data.get("compliance", 0.8)),
                creativity=float(data.get("creativity", 0.6)),
            )
        except Exception as e:
            print(f"[LLMQualityScorer] LLM scoring failed: {e} — using heuristic")
            return _heuristic_score(variant, brief)
