"""
generation/email_generator.py
==============================
LLM-based email generator calling DeepSeek API.
Produces 3 variants (urgency / storytelling / direct_value) per campaign brief.

DeepSeek is OpenAI-compatible, so we use the openai SDK pointed at DeepSeek's base URL.
Falls back to deterministic mock output when DEEPSEEK_API_KEY is not set.
"""

from __future__ import annotations
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base import BaseGenerator
from core.config import get_settings, get_api_key
from core.types import (
    CampaignBrief, EmailVariant, HistoricalEmail, ToneStyle,
)
from generation.prompt_templates import build_generation_prompt


# ──────────────────────────────────────────────
# DeepSeek client (OpenAI-compatible)
# ──────────────────────────────────────────────

DEEPSEEK_BASE_URL = "https://api.deepseek.com"

TONE_ENUM_MAP = {
    "urgency":      ToneStyle.URGENCY,
    "storytelling": ToneStyle.STORYTELLING,
    "direct_value": ToneStyle.DIRECT_VALUE,
}


def _clean_json(text: str) -> str:
    """Strip markdown fences if model wraps response in ```json ... ```."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


# ──────────────────────────────────────────────
# Mock fallback (no API key)
# ──────────────────────────────────────────────

MOCK_VARIANTS = {
    "urgency": {
        "subject_line": "⚡ {game}: Your exclusive offer expires tonight",
        "preheader": "48 hours left — your personalised rewards are ready",
        "body_text": (
            "Hey {segment},\n\n"
            "Time is running out. Your exclusive comeback package for {game} expires in 48 hours.\n\n"
            "What's waiting for you:\n"
            "- 500 free in-game coins (no purchase needed)\n"
            "- Limited-edition returning player skin\n"
            "- 2x XP boost for your first 5 matches\n\n"
            "The countdown has started. Your squad is already back in action."
        ),
        "cta_text": "Claim My Rewards Now",
        "copywriter_notes": "Opens with scarcity signal, itemises value stack, closes with social proof.",
    },
    "storytelling": {
        "subject_line": "Your {game} legend isn't over",
        "preheader": "The world changed while you were away — come see what's new",
        "body_text": (
            "Dear {segment},\n\n"
            "Every great champion has a comeback story.\n\n"
            "Since you last logged in, {game} has evolved — new maps, new rivalries, new legends rising. "
            "But your achievements? Your rank? Your history? Those are still yours.\n\n"
            "We've kept something special waiting for you. Not because we had to — because you earned it.\n\n"
            "This is your second chapter. The story doesn't write itself."
        ),
        "cta_text": "Continue My Story",
        "copywriter_notes": "Narrative arc positions player as protagonist, uses their past investment as hook.",
    },
    "direct_value": {
        "subject_line": "Free {game} rewards — no catch",
        "preheader": "Log in this week and grab what's yours",
        "body_text": (
            "Hi {segment},\n\n"
            "Quick update — we've got a comeback bundle ready in {game}.\n\n"
            "Here's exactly what you get:\n"
            "- 500 free coins (credited instantly on login)\n"
            "- Exclusive returning player skin (7-day claim window)\n"
            "- 2x XP for your first 5 games this week\n\n"
            "No purchase required. Valid until Sunday 23:59."
        ),
        "cta_text": "Get My Free Bundle",
        "copywriter_notes": "Leads with benefit, uses specificity (numbers, deadlines) to build credibility.",
    },
}


def _make_mock_variant(brief: CampaignBrief, tone: str) -> EmailVariant:
    template = MOCK_VARIANTS.get(tone, MOCK_VARIANTS["direct_value"])
    fmt = {"game": brief.game, "segment": brief.target_segment.value}
    return EmailVariant(
        tone_style=TONE_ENUM_MAP.get(tone, ToneStyle.DIRECT_VALUE),
        subject_line=template["subject_line"].format(**fmt),
        preheader=template["preheader"],
        body_text=template["body_text"].format(**fmt),
        cta_text=template["cta_text"],
    )


# ──────────────────────────────────────────────
# DeepSeek Generator
# ──────────────────────────────────────────────

class DeepSeekEmailGenerator(BaseGenerator):
    """
    Generates 3 email variants (urgency / storytelling / direct_value)
    using DeepSeek Chat via OpenAI-compatible API.

    Falls back to deterministic mock variants when DEEPSEEK_API_KEY is absent.
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
            print(f"[DeepSeekEmailGenerator] Live mode — model: {self.settings.llm.model}")
        except (EnvironmentError, ImportError) as e:
            print(f"[DeepSeekEmailGenerator] Fallback mock mode ({e})")

    def _call_llm(self, system: str, user: str) -> dict:
        """Single LLM call, returns parsed JSON dict."""
        response = self._client.chat.completions.create(
            model=self.settings.llm.model,
            temperature=self.settings.llm.temperature,
            max_tokens=self.settings.llm.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            response_format={"type": "json_object"},  # DeepSeek supports JSON mode
        )
        raw = response.choices[0].message.content
        return json.loads(_clean_json(raw))

    def _parse_variant(self, data: dict, tone: str) -> EmailVariant:
        return EmailVariant(
            tone_style=TONE_ENUM_MAP.get(tone, ToneStyle.DIRECT_VALUE),
            subject_line=data.get("subject_line", "")[:60],
            preheader=data.get("preheader", ""),
            body_text=data.get("body_text", ""),
            cta_text=data.get("cta_text", ""),
        )

    def generate(
        self,
        brief: CampaignBrief,
        retrieved_context: list[HistoricalEmail],
        num_variants: int = 3,
    ) -> list[EmailVariant]:
        tones = self.settings.generation.variant_styles[:num_variants]
        variants = []

        for tone in tones:
            try:
                if self._live:
                    system, user = build_generation_prompt(brief, retrieved_context, tone_override=tone)
                    data = self._call_llm(system, user)
                    variant = self._parse_variant(data, tone)
                    time.sleep(0.3)  # polite rate limiting
                else:
                    variant = _make_mock_variant(brief, tone)
                variants.append(variant)
                print(f"[DeepSeekEmailGenerator] Generated variant: {tone} → '{variant.subject_line[:50]}'")
            except Exception as e:
                print(f"[DeepSeekEmailGenerator] Error on tone '{tone}': {e} — using mock")
                variants.append(_make_mock_variant(brief, tone))

        return variants
