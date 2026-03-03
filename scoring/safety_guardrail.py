"""
scoring/safety_guardrail.py
============================
Rule-based safety guardrail: checks for policy violations before email is approved.
"""

from __future__ import annotations
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base import BaseGuardrail
from core.config import get_settings
from core.types import CampaignBrief, EmailVariant


# Globally banned phrases (regulatory / brand policy)
GLOBAL_BANNED = [
    "guaranteed",
    "100% free",
    "act now or else",
    "click here immediately",
    "risk-free",
    "no risk",
    "you must",
    "you have to",
    "limited to first",   # misleading scarcity
]

# Patterns that indicate over-promising
OVERPROMISE_PATTERNS = [
    r"\b\d{3,}\s*%\s*(off|bonus|extra)\b",   # "500% bonus" etc.
    r"\bguaranteed?\s+\w+\b",
    r"\b(win|earn|make)\s+\$\d+",
]

# Spam trigger words (high unsubscribe risk)
SPAM_TRIGGERS = [
    "!!!",
    "FREE!!!",
    "WINNER",
    "CONGRATULATIONS",
    "ACT IMMEDIATELY",
    "URGENT URGENT",
]


class SafetyGuardrail(BaseGuardrail):
    """
    Checks an email variant against:
    1. Global banned phrases
    2. Campaign-specific forbidden keywords (from brief)
    3. Over-promise patterns
    4. Spam trigger language
    5. Subject line length
    6. CTA quality (must have an action verb)
    """

    def __init__(self):
        self.settings = get_settings()
        self._extra_banned = [w.lower() for w in self.settings.scoring.banned_words]

    def check(self, variant: EmailVariant, brief: CampaignBrief) -> tuple[bool, list[str]]:
        """
        Returns:
            (passed: bool, flags: list[str])
            passed=True means no violations found.
        """
        flags = []
        full_text = (
            variant.subject_line + " " +
            variant.preheader + " " +
            variant.body_text + " " +
            variant.cta_text
        ).lower()

        # 1. Global banned phrases
        for phrase in GLOBAL_BANNED:
            if phrase.lower() in full_text:
                flags.append(f"Banned phrase detected: '{phrase}'")

        # 2. Campaign-specific forbidden keywords
        for kw in brief.forbidden_keywords:
            if kw.lower() in full_text:
                flags.append(f"Brief-forbidden keyword detected: '{kw}'")

        # 3. Config banned words
        for kw in self._extra_banned:
            if kw in full_text:
                flags.append(f"Config-banned word detected: '{kw}'")

        # 4. Over-promise patterns
        for pattern in OVERPROMISE_PATTERNS:
            if re.search(pattern, full_text, re.IGNORECASE):
                flags.append(f"Over-promise pattern detected: {pattern}")

        # 5. Spam trigger language
        for trigger in SPAM_TRIGGERS:
            if trigger.lower() in full_text:
                flags.append(f"Spam trigger language: '{trigger}'")

        # 6. Subject line length
        if len(variant.subject_line) > 60:
            flags.append(
                f"Subject line too long: {len(variant.subject_line)} chars (max 60)"
            )

        # 7. Empty fields
        if not variant.subject_line.strip():
            flags.append("Missing subject line")
        if not variant.body_text.strip():
            flags.append("Missing body text")
        if not variant.cta_text.strip():
            flags.append("Missing CTA")

        # 8. CTA must start with a verb (basic check)
        cta_first_word = variant.cta_text.strip().split()[0].lower() if variant.cta_text.strip() else ""
        action_verbs = {"claim", "get", "play", "join", "unlock", "buy", "start",
                        "continue", "accept", "enter", "collect", "download", "see", "explore"}
        if cta_first_word and cta_first_word not in action_verbs:
            flags.append(f"CTA may lack action verb (first word: '{cta_first_word}')")

        passed = len(flags) == 0
        return passed, flags
