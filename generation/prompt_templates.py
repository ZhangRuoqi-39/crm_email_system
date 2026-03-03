"""
generation/prompt_templates.py
================================
Prompt template library for CRM email generation.
Each (campaign_type, tone_style) combination has a tailored system + user prompt.
"""

from __future__ import annotations
from core.types import CampaignBrief, HistoricalEmail


# ──────────────────────────────────────────────
# System prompt (shared, injected with brand context)
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert CRM email copywriter for Tencent's IEGG (Interactive Entertainment Global Games) division.
You write high-performance marketing emails for global live-service games including Honor of Kings, PUBG Mobile, League of Legends Mobile, and Valorant Mobile.

Your emails must:
- Be concise, compelling, and immediately readable on mobile
- Match the specified tone style exactly
- Include a clear, action-oriented CTA
- Respect brand voice: energetic but not aggressive, personal but not creepy
- Never use: guaranteed returns, misleading urgency, or manipulative language
- Subject lines must be under 60 characters

Respond ONLY with a valid JSON object — no preamble, no markdown fences.
"""

# ──────────────────────────────────────────────
# Tone-specific instruction blocks
# ──────────────────────────────────────────────

TONE_INSTRUCTIONS = {
    "urgency": (
        "TONE: Urgency — create FOMO with real deadlines and scarcity. "
        "Use power words (expires, limited, last chance, tonight only). "
        "Every sentence should push toward immediate action."
    ),
    "storytelling": (
        "TONE: Storytelling — draw the player into a narrative. "
        "Open with an evocative scene or the player's journey. "
        "Build emotional connection before revealing the offer. "
        "Make them feel this email was written just for them."
    ),
    "direct_value": (
        "TONE: Direct Value — lead with the concrete benefit immediately. "
        "Use bullet points for clarity. Be factual and specific (numbers, percentages). "
        "No fluff — respect the reader's time."
    ),
}

# ──────────────────────────────────────────────
# Campaign-type context blocks
# ──────────────────────────────────────────────

CAMPAIGN_CONTEXT = {
    "reactivation": (
        "CAMPAIGN: Win-back lapsed players. Acknowledge their absence warmly, "
        "not accusatorially. Highlight what's new or what they're missing. "
        "Offer a tangible return reward."
    ),
    "event_promotion": (
        "CAMPAIGN: Drive participation in a limited-time event. "
        "Emphasise exclusivity and time-sensitivity. "
        "Spell out exactly what they'll gain by participating."
    ),
    "season_pass": (
        "CAMPAIGN: Convert or retain Season Pass subscribers. "
        "Show the full value stack (cosmetics, currency, XP). "
        "Address the FOMO of missing out on time-limited season rewards."
    ),
    "vip_upgrade": (
        "CAMPAIGN: Invite high-value players to VIP tier. "
        "Make them feel selected, not sold to. "
        "Lead with exclusivity and status recognition."
    ),
    "new_content_drop": (
        "CAMPAIGN: Announce new game content. "
        "Generate excitement and curiosity. "
        "Early-access framing works well here."
    ),
    "retention": (
        "CAMPAIGN: Keep active players engaged. "
        "Remind them of ongoing rewards and streak benefits. "
        "Keep it brief — they're already playing."
    ),
}


# ──────────────────────────────────────────────
# Context formatter (historical email examples)
# ──────────────────────────────────────────────

def _format_examples(emails: list[HistoricalEmail], max_examples: int = 3) -> str:
    if not emails:
        return "No historical examples available."
    lines = ["TOP-PERFORMING HISTORICAL EMAILS FOR REFERENCE:"]
    for i, e in enumerate(emails[:max_examples], 1):
        lines.append(
            f"\nExample {i} (open_rate={e.open_rate:.1%}, CTR={e.ctr:.2%}):\n"
            f"  Subject:  {e.subject_line}\n"
            f"  Preheader:{e.preheader}\n"
            f"  Body:     {e.body_text[:200]}{'...' if len(e.body_text) > 200 else ''}\n"
            f"  CTA:      {e.cta_text}"
        )
    return "\n".join(lines)


# ──────────────────────────────────────────────
# Main prompt builder
# ──────────────────────────────────────────────

def build_generation_prompt(
    brief: CampaignBrief,
    retrieved_emails: list[HistoricalEmail],
    tone_override: str | None = None,
) -> tuple[str, str]:
    """
    Returns (system_prompt, user_prompt) for a single email variant.

    Args:
        brief: The campaign brief from the UI.
        retrieved_emails: Top-k historical emails for RAG context.
        tone_override: If provided, overrides brief.tone_style (used for A/B variants).
    """
    tone = tone_override or brief.tone_style.value
    campaign = brief.campaign_type.value

    tone_block = TONE_INSTRUCTIONS.get(tone, TONE_INSTRUCTIONS["direct_value"])
    campaign_block = CAMPAIGN_CONTEXT.get(campaign, "")
    examples_block = _format_examples(retrieved_emails)

    brand_kw = ", ".join(brief.brand_keywords) if brief.brand_keywords else "none specified"
    forbidden_kw = ", ".join(brief.forbidden_keywords) if brief.forbidden_keywords else "none"

    user_prompt = f"""
GAME: {brief.game}
CAMPAIGN TYPE: {campaign}
TARGET SEGMENT: {brief.target_segment.value}
KPI FOCUS: {brief.kpi}
CAMPAIGN CONTEXT: {brief.context}
BRAND KEYWORDS TO USE: {brand_kw}
FORBIDDEN WORDS/PHRASES: {forbidden_kw}

{tone_block}

{campaign_block}

{examples_block}

---
Generate ONE email for this campaign. Respond with ONLY a JSON object in this exact schema:
{{
  "subject_line": "...",          // max 60 chars, compelling, no clickbait
  "preheader": "...",             // 40-90 chars, complements subject
  "body_text": "...",             // 3-5 short paragraphs, use \\n for line breaks
  "cta_text": "...",              // 3-6 words, action verb first
  "tone_used": "{tone}",
  "copywriter_notes": "..."       // 1 sentence: what strategy this copy uses
}}
""".strip()

    return SYSTEM_PROMPT, user_prompt
