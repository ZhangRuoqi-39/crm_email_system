"""
data/mock_generator.py
======================
Generates a realistic synthetic dataset of historical gaming CRM emails
with simulated performance metrics.

Why synthetic data?
    No public dataset of Tencent/gaming CRM emails exists. Industry-standard
    practice for AI product prototyping is "domain-adapted synthetic corpus"
    calibrated to real Mailchimp gaming benchmarks
    (avg open rate 21%, avg CTR 3.0%).

Usage:
    python data/mock_generator.py
    # Writes data/processed/historical_emails.json (80 emails)
    # Writes data/raw/historical_emails.csv
    # Writes data/processed/golden_test_set.json (10 test cases)
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.types import HistoricalEmail

random.seed(42)


# ─────────────────────────────────────────────
# Email content templates per (campaign_type, tone_style)
# ─────────────────────────────────────────────

TEMPLATES = {
    ("reactivation", "urgency"): [
        {
            "subject": "{game}: Your account misses you — claim bonus before midnight",
            "preheader": "Your rewards are expiring. Log in now.",
            "body": (
                "Hey {segment},\n\n"
                "It's been a while since you last played {game} — and we've saved something special for you.\n\n"
                "Your COMEBACK PACKAGE expires in 24 hours:\n"
                "- 500 free in-game coins\n"
                "- Limited comeback skin (today only)\n"
                "- Double XP for your first 3 matches\n\n"
                "Don't let your hard-earned progress fade away. Your squad is waiting."
            ),
            "cta": "Claim My Comeback Rewards",
        },
        {
            "subject": "Last chance: {game} exclusive offer ends tonight",
            "preheader": "48 hours left — your personalised rewards are ready",
            "body": (
                "Dear {segment},\n\n"
                "We don't want you to miss out. As one of our valued players, "
                "we've reserved an exclusive offer just for you in {game}.\n\n"
                "This offer expires in 48 hours:\n"
                "- Premium currency bundle (30% extra)\n"
                "- Exclusive reactivation badge\n"
                "- Early access to upcoming season content\n\n"
                "The battlefield needs you back."
            ),
            "cta": "Unlock My Offer Now",
        },
    ],
    ("reactivation", "storytelling"): [
        {
            "subject": "The {game} world evolved. Your legacy awaits.",
            "preheader": "A personal invitation back to greatness",
            "body": (
                "Dear {segment},\n\n"
                "Every great story has a second chapter.\n\n"
                "Since you last logged into {game}, our world has grown — "
                "new maps, new heroes, new rivalries. But some things remain constant: "
                "your achievements, your rank, your history.\n\n"
                "We're inviting you back with a package designed to honour "
                "the player you already are. Your account holds memories that matter.\n\n"
                "This is not just a game. It's your story."
            ),
            "cta": "Continue My Story",
        },
    ],
    ("reactivation", "direct_value"): [
        {
            "subject": "Free rewards for coming back to {game}",
            "preheader": "Log in this week and grab what's yours",
            "body": (
                "Hey {segment},\n\n"
                "Quick update — we've got a comeback bundle ready for you in {game}.\n\n"
                "Here's what you get when you return:\n"
                "- 500 free coins (no purchase needed)\n"
                "- Exclusive returning player skin (7-day window)\n"
                "- 2x XP boost for your first 5 games\n\n"
                "No strings attached. Just log in and it's yours. Valid this week only."
            ),
            "cta": "Get My Free Rewards",
        },
    ],
    ("event_promotion", "urgency"): [
        {
            "subject": "{game} Festival Event — starts tomorrow!",
            "preheader": "Limited-time rewards you can only get this week",
            "body": (
                "Hi {segment},\n\n"
                "The {game} Festival Event kicks off TOMORROW — "
                "and we've packed it with exclusive rewards you can only earn during the event window.\n\n"
                "Event exclusives:\n"
                "- Festival-themed hero skin (limited to event only)\n"
                "- 3x event currency per match\n"
                "- Special festival missions with epic rewards\n\n"
                "The event closes in 7 days. Don't miss your chance."
            ),
            "cta": "Join the Festival Now",
        },
    ],
    ("event_promotion", "storytelling"): [
        {
            "subject": "A new chapter unfolds in {game} — will you be part of it?",
            "preheader": "The anniversary event is here",
            "body": (
                "Dear {segment},\n\n"
                "Every year, the {game} universe marks its anniversary with something extraordinary.\n\n"
                "For seven days, the world of {game} transforms. "
                "Legends are reborn. Rivalries are settled. History is made.\n\n"
                "We invite you to be part of this moment — not just as a player, "
                "but as part of the story that makes {game} what it is.\n\n"
                "The anniversary event begins now. Your chapter starts today."
            ),
            "cta": "Enter the Anniversary Event",
        },
    ],
    ("event_promotion", "direct_value"): [
        {
            "subject": "{game} new season — here is everything that changed",
            "preheader": "New heroes, maps, and ranked rewards this season",
            "body": (
                "Hi {segment},\n\n"
                "Season update is live in {game}. Here's what's new:\n\n"
                "- 2 new heroes added to the roster\n"
                "- Ranked season reset — fresh start for everyone\n"
                "- New battle pass with 50 reward tiers\n"
                "- Improved matchmaking system\n\n"
                "Log in now to claim your season launch rewards before they expire."
            ),
            "cta": "See What's New",
        },
    ],
    ("season_pass", "urgency"): [
        {
            "subject": "{game} Season Pass — only 3 days left to buy",
            "preheader": "Don't lose access to premium rewards",
            "body": (
                "Hi {segment},\n\n"
                "The current {game} Season Pass closes in just 3 days — "
                "after that, all premium rewards become permanently unavailable.\n\n"
                "What you'll miss if you don't upgrade:\n"
                "- 12 exclusive cosmetic items\n"
                "- Premium missions with 3x currency\n"
                "- Season-ending legendary skin\n\n"
                "Thousands of players have already unlocked these rewards. "
                "The clock is ticking."
            ),
            "cta": "Upgrade to Season Pass",
        },
    ],
    ("season_pass", "direct_value"): [
        {
            "subject": "{game} Season Pass is 20% off this weekend",
            "preheader": "Best value bundle of the season",
            "body": (
                "Hi {segment},\n\n"
                "This weekend only, the {game} Season Pass is 20% off.\n\n"
                "What you get:\n"
                "- 50-tier reward track with guaranteed items at every 5 tiers\n"
                "- Exclusive battle pass hero skin\n"
                "- Daily bonus missions for extra currency\n"
                "- Priority matchmaking queue\n\n"
                "Offer valid until Sunday 23:59. Discount applied automatically."
            ),
            "cta": "Buy Season Pass (20% Off)",
        },
    ],
    ("vip_upgrade", "storytelling"): [
        {
            "subject": "You've earned it — your VIP invitation for {game}",
            "preheader": "Exclusive access for our most dedicated players",
            "body": (
                "Dear {segment},\n\n"
                "Not everyone gets this email.\n\n"
                "Based on your dedication to {game} — your playtime, your achievements, "
                "your contribution to our community — you've been identified as one of our "
                "most valued players.\n\n"
                "We'd like to invite you to join the {game} VIP Programme.\n\n"
                "As a VIP member, you'll receive:\n"
                "- Monthly exclusive cosmetic drop\n"
                "- Priority customer support\n"
                "- Early access to new content\n"
                "- Personal account manager\n\n"
                "This invitation is personal. And it's only available to you."
            ),
            "cta": "Accept My VIP Invitation",
        },
    ],
    ("retention", "direct_value"): [
        {
            "subject": "Your {game} weekly rewards are ready to collect",
            "preheader": "Don't let your login streak expire",
            "body": (
                "Hi {segment},\n\n"
                "Your weekly rewards are ready in {game}.\n\n"
                "Log in before midnight to collect:\n"
                "- Daily login bonus\n"
                "- Weekly mission completion rewards\n"
                "- Streak bonus — keep your streak alive!\n\n"
                "Keep your streak going to unlock bigger rewards each week."
            ),
            "cta": "Collect My Rewards",
        },
    ],
    ("new_content_drop", "urgency"): [
        {
            "subject": "NEW: {game} just dropped something major",
            "preheader": "Be among the first to experience it",
            "body": (
                "Hey {segment},\n\n"
                "We just dropped the biggest content update in {game} history — "
                "and early access is live RIGHT NOW for existing players.\n\n"
                "What's new:\n"
                "- Brand new map: The Forgotten Realm\n"
                "- New legendary hero: Shadow Empress\n"
                "- New ranked mode: Team Conquest\n\n"
                "First 10,000 players to log in get an exclusive early-access badge."
            ),
            "cta": "Play New Content Now",
        },
    ],
}

# ─────────────────────────────────────────────
# Performance model
# Calibrated to Mailchimp gaming benchmarks:
# avg open_rate=0.21, avg CTR=0.030
# ─────────────────────────────────────────────

PERFORMANCE_PARAMS = {
    ("reactivation",    "urgency"):      (0.28, 0.04, 0.048, 0.010),
    ("reactivation",    "storytelling"): (0.24, 0.05, 0.038, 0.009),
    ("reactivation",    "direct_value"): (0.22, 0.04, 0.042, 0.008),
    ("event_promotion", "urgency"):      (0.31, 0.05, 0.055, 0.012),
    ("event_promotion", "storytelling"): (0.26, 0.05, 0.040, 0.010),
    ("event_promotion", "direct_value"): (0.25, 0.04, 0.045, 0.010),
    ("season_pass",     "urgency"):      (0.29, 0.05, 0.052, 0.011),
    ("season_pass",     "direct_value"): (0.23, 0.04, 0.040, 0.009),
    ("vip_upgrade",     "storytelling"): (0.35, 0.06, 0.062, 0.013),
    ("retention",       "direct_value"): (0.20, 0.03, 0.032, 0.007),
    ("new_content_drop","urgency"):      (0.33, 0.05, 0.058, 0.012),
}

GAMES    = ["Honor of Kings", "PUBG Mobile", "League of Legends Mobile", "Valorant Mobile"]
SEGMENTS = ["Lapsed VIP Players", "Lapsed Casual Players", "Active Whale Players",
            "At-Risk Players", "New Players"]


def _sample_performance(campaign_type: str, tone_style: str) -> dict:
    key = (campaign_type, tone_style)
    or_mean, or_std, ctr_mean, ctr_std = PERFORMANCE_PARAMS.get(key, (0.21, 0.04, 0.030, 0.008))
    open_rate = max(0.05, min(0.60, random.gauss(or_mean, or_std)))
    ctr       = max(0.005, min(0.15, random.gauss(ctr_mean, ctr_std)))
    unsub     = max(0.001, min(0.02, random.gauss(0.005, 0.002)))
    conv      = max(0.001, min(0.05, ctr * random.gauss(0.25, 0.05)))
    return {
        "open_rate":        round(open_rate, 4),
        "ctr":              round(ctr, 4),
        "unsubscribe_rate": round(unsub, 4),
        "conversion_rate":  round(conv, 4),
        "send_volume":      random.randint(5000, 500000),
    }


def _compute_performance_score(open_rate: float, ctr: float, unsub: float) -> float:
    """Composite score normalised to 0-1. Rewards open rate and CTR, penalises unsubs."""
    score = (open_rate * 0.35) + (ctr * 5.0 * 0.45) - (unsub * 10.0 * 0.20)
    return round(max(0.0, min(1.0, score)), 4)


def generate_mock_emails(n: int = 80) -> list:
    """Generate n synthetic historical HistoricalEmail objects."""
    emails = []
    combos = list(TEMPLATES.keys())

    for i in range(n):
        combo = combos[i % len(combos)]
        campaign_type, tone_style = combo
        template = random.choice(TEMPLATES[combo])

        game    = random.choice(GAMES)
        segment = random.choice(SEGMENTS)

        subject   = template["subject"].format(game=game, segment=segment)
        preheader = template["preheader"]
        body      = template["body"].format(game=game, segment=segment,
                                             day=random.randint(1, 7),
                                             streak=random.randint(1, 30))
        cta       = template["cta"]

        perf      = _sample_performance(campaign_type, tone_style)
        perf_score = _compute_performance_score(
            perf["open_rate"], perf["ctr"], perf["unsubscribe_rate"]
        )

        email = HistoricalEmail(
            email_id=f"email_{i+1:04d}",
            game=game,
            campaign_type=campaign_type,
            target_segment=segment,
            subject_line=subject,
            preheader=preheader,
            body_text=body,
            cta_text=cta,
            tone_style=tone_style,
            open_rate=perf["open_rate"],
            ctr=perf["ctr"],
            unsubscribe_rate=perf["unsubscribe_rate"],
            conversion_rate=perf["conversion_rate"],
            send_volume=perf["send_volume"],
            sent_at=f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
            performance_score=perf_score,
        )
        emails.append(email)

    return emails


def save_to_json(emails: list, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([e.to_dict() for e in emails], f, indent=2, ensure_ascii=False)
    print(f"[mock_generator] Saved {len(emails)} emails to {path}")


def save_to_csv(emails: list, path: str) -> None:
    import csv
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(emails[0].to_dict().keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([e.to_dict() for e in emails])
    print(f"[mock_generator] Saved {len(emails)} emails to {path}")


def generate_golden_test_set(emails: list, n: int = 10) -> list:
    """Golden test set for offline evaluation — top-N emails as expected retrievals."""
    top = sorted(emails, key=lambda e: e.performance_score, reverse=True)[:20]
    return [
        {
            "test_id": f"tc_{i+1:03d}",
            "query": f"{e.game} {e.campaign_type} {e.target_segment}",
            "expected_email_ids": [e.email_id],
            "expected_game": e.game,
            "expected_campaign_type": e.campaign_type,
            "min_open_rate": round(e.open_rate * 0.8, 4),
            "min_ctr": round(e.ctr * 0.8, 4),
        }
        for i, e in enumerate(top[:n])
    ]


if __name__ == "__main__":
    print("[mock_generator] Generating synthetic CRM email dataset...")

    emails = generate_mock_emails(n=80)
    save_to_json(emails, "data/processed/historical_emails.json")
    save_to_csv(emails,  "data/raw/historical_emails.csv")

    golden = generate_golden_test_set(emails, n=10)
    golden_path = "data/processed/golden_test_set.json"
    Path(golden_path).parent.mkdir(parents=True, exist_ok=True)
    with open(golden_path, "w") as f:
        json.dump({"test_cases": golden}, f, indent=2)
    print(f"[mock_generator] Saved {len(golden)} golden test cases to {golden_path}")

    open_rates = [e.open_rate for e in emails]
    ctrs = [e.ctr for e in emails]
    print(f"\n[mock_generator] Dataset stats:")
    print(f"  Total emails   : {len(emails)}")
    print(f"  Avg open rate  : {sum(open_rates)/len(open_rates):.1%}")
    print(f"  Avg CTR        : {sum(ctrs)/len(ctrs):.2%}")
    print(f"  Games          : {sorted(set(e.game for e in emails))}")
    print(f"  Campaign types : {sorted(set(e.campaign_type for e in emails))}")
