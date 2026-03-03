"""
ingestion/pipeline.py
=====================
The data ingestion pipeline: Load -> Validate -> Transform -> Save.

This is Stage 1 of the system. It reads historical email data from JSON/CSV,
computes enrichments (normalised performance score, retrieval text),
and writes the processed output ready for Day 2 vector indexing.

Usage:
    from ingestion.pipeline import IngestionPipeline
    pipeline = IngestionPipeline()
    emails = pipeline.run("data/processed/historical_emails.json")
"""

import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base import BaseLoader, BaseTransform
from core.config import get_settings
from core.types import HistoricalEmail


class JSONEmailLoader(BaseLoader):
    """Loads HistoricalEmail records from a JSON file."""

    def load(self, source_path: str) -> list:
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {source_path}\n"
                "Run 'python data/mock_generator.py' first."
            )
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)

        emails = []
        for r in records:
            try:
                email = HistoricalEmail(
                    email_id=r["email_id"],
                    game=r["game"],
                    campaign_type=r["campaign_type"],
                    target_segment=r["target_segment"],
                    subject_line=r["subject_line"],
                    preheader=r.get("preheader", ""),
                    body_text=r["body_text"],
                    cta_text=r["cta_text"],
                    tone_style=r["tone_style"],
                    open_rate=float(r["open_rate"]),
                    ctr=float(r["ctr"]),
                    unsubscribe_rate=float(r["unsubscribe_rate"]),
                    conversion_rate=float(r["conversion_rate"]),
                    sent_at=r.get("sent_at"),
                    send_volume=int(r.get("send_volume", 0)),
                    performance_score=float(r.get("performance_score", 0.0)),
                )
                emails.append(email)
            except (KeyError, ValueError) as e:
                print(f"[JSONEmailLoader] Skipping malformed record: {e}")

        print(f"[JSONEmailLoader] Loaded {len(emails)} emails from {source_path}")
        return emails


class CSVEmailLoader(BaseLoader):
    """Loads HistoricalEmail records from a CSV file (e.g. Kaggle dataset)."""

    def load(self, source_path: str) -> list:
        import csv
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {source_path}")

        emails = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                try:
                    email = HistoricalEmail(
                        email_id=row.get("email_id", f"csv_{i+1:04d}"),
                        game=row.get("game", "Generic Game"),
                        campaign_type=row.get("campaign_type", "retention"),
                        target_segment=row.get("target_segment", "General"),
                        subject_line=row["subject_line"],
                        preheader=row.get("preheader", ""),
                        body_text=row["body_text"],
                        cta_text=row.get("cta_text", "Learn More"),
                        tone_style=row.get("tone_style", "direct_value"),
                        open_rate=float(row.get("open_rate", 0.21)),
                        ctr=float(row.get("ctr", 0.03)),
                        unsubscribe_rate=float(row.get("unsubscribe_rate", 0.005)),
                        conversion_rate=float(row.get("conversion_rate", 0.01)),
                        sent_at=row.get("sent_at"),
                        send_volume=int(row.get("send_volume", 0)),
                        performance_score=float(row.get("performance_score", 0.0)),
                    )
                    emails.append(email)
                except (KeyError, ValueError) as e:
                    print(f"[CSVEmailLoader] Skipping row {i+1}: {e}")

        print(f"[CSVEmailLoader] Loaded {len(emails)} emails from {source_path}")
        return emails


class PerformanceScoreTransform(BaseTransform):
    """
    Recomputes performance_score using a consistent formula.
    Formula: open_rate*0.35 + (ctr*5)*0.45 - (unsub*10)*0.20  (normalised to [0,1])
    """

    def transform(self, emails: list) -> list:
        for email in emails:
            score = (
                email.open_rate        * 0.35 +
                email.ctr * 5.0        * 0.45 -
                email.unsubscribe_rate * 10.0 * 0.20
            )
            email.performance_score = round(max(0.0, min(1.0, score)), 4)
        print(f"[PerformanceScoreTransform] Recomputed {len(emails)} scores")
        return emails


class TextNormalisationTransform(BaseTransform):
    """Strips whitespace, normalises line endings, truncates oversized bodies."""

    MAX_BODY_LENGTH = 2000

    def transform(self, emails: list) -> list:
        truncated = 0
        for email in emails:
            email.subject_line = email.subject_line.strip()
            email.preheader    = email.preheader.strip()
            email.cta_text     = email.cta_text.strip()
            email.body_text    = " ".join(email.body_text.split())
            if len(email.body_text) > self.MAX_BODY_LENGTH:
                email.body_text = email.body_text[:self.MAX_BODY_LENGTH] + "..."
                truncated += 1
        if truncated:
            print(f"[TextNormalisationTransform] Truncated {truncated} oversized bodies")
        print(f"[TextNormalisationTransform] Normalised {len(emails)} emails")
        return emails


class MetadataEnrichmentTransform(BaseTransform):
    """Logs corpus statistics useful for observability."""

    def transform(self, emails: list) -> list:
        if not emails:
            return emails
        avg_subj = sum(len(e.subject_line) for e in emails) / len(emails)
        avg_body = sum(len(e.body_text.split()) for e in emails) / len(emails)
        print(
            f"[MetadataEnrichmentTransform] "
            f"avg_subject={avg_subj:.0f} chars | avg_body={avg_body:.0f} words"
        )
        return emails


class IngestionPipeline:
    """
    Orchestrates: Load -> Validate -> Transform -> Return.
    The transforms list is pluggable.
    """

    def __init__(self, loader=None, transforms=None):
        self.loader = loader or JSONEmailLoader()
        self.transforms = transforms or [
            PerformanceScoreTransform(),
            TextNormalisationTransform(),
            MetadataEnrichmentTransform(),
        ]
        self.settings = get_settings()

    def run(self, source_path=None) -> list:
        path = source_path or self.settings.data.historical_emails
        print(f"\n[IngestionPipeline] Starting ingestion from: {path}")

        emails = self.loader.load(path)
        before = len(emails)
        emails = self.loader.validate(emails)
        if len(emails) < before:
            print(f"[IngestionPipeline] Validation removed {before - len(emails)} records")

        for transform in self.transforms:
            emails = transform.transform(emails)

        print(f"\n[IngestionPipeline] Done. {len(emails)} emails ready for indexing.")
        self._print_summary(emails)
        return emails

    def _print_summary(self, emails: list) -> None:
        if not emails:
            return
        avg_open  = sum(e.open_rate for e in emails) / len(emails)
        avg_ctr   = sum(e.ctr for e in emails) / len(emails)
        avg_score = sum(e.performance_score for e in emails) / len(emails)
        print(f"  Avg open rate  : {avg_open:.1%}")
        print(f"  Avg CTR        : {avg_ctr:.2%}")
        print(f"  Avg perf score : {avg_score:.3f}")
        print(f"  Campaign types : {sorted(set(e.campaign_type for e in emails))}")


def load_emails_for_indexing(source_path=None) -> list:
    """One-line helper used by retrieval module on Day 2."""
    return IngestionPipeline().run(source_path)


if __name__ == "__main__":
    pipeline = IngestionPipeline()
    emails = pipeline.run()
    print("\nTop 3 by performance score:")
    for e in sorted(emails, key=lambda x: x.performance_score, reverse=True)[:3]:
        print(f"  [{e.email_id}] score={e.performance_score} | {e.subject_line[:55]}")
