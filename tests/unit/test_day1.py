"""
tests/unit/test_day1.py
=======================
Unit tests for Day 1 components: types, config, and ingestion pipeline.
Run: pytest tests/unit/test_day1.py -v
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.types import (
    CampaignBrief, CampaignType, ToneStyle, PlayerSegment,
    HistoricalEmail, EmailVariant, QualityScore, GenerationResult,
)
from core.config import get_settings


# ─────────────────────────────────────────────
# Types tests
# ─────────────────────────────────────────────

class TestQualityScore:
    def test_overall_computed_on_init(self):
        qs = QualityScore(relevance=1.0, tone=1.0, compliance=1.0, creativity=1.0)
        assert qs.overall == pytest.approx(1.0, abs=0.001)

    def test_overall_weighted_correctly(self):
        qs = QualityScore(relevance=1.0, tone=0.0, compliance=0.0, creativity=0.0)
        assert qs.overall == pytest.approx(0.30, abs=0.001)

    def test_overall_zero_when_all_zero(self):
        qs = QualityScore(relevance=0.0, tone=0.0, compliance=0.0, creativity=0.0)
        assert qs.overall == pytest.approx(0.0)

    def test_to_dict_has_all_keys(self):
        qs = QualityScore(relevance=0.8, tone=0.7, compliance=0.9, creativity=0.6)
        d = qs.to_dict()
        for key in ["relevance", "tone", "compliance", "creativity", "overall",
                    "passed_guardrails", "guardrail_flags"]:
            assert key in d


class TestHistoricalEmail:
    def _make_email(self):
        return HistoricalEmail(
            email_id="test_001",
            game="Honor of Kings",
            campaign_type="reactivation",
            target_segment="Lapsed VIP Players",
            subject_line="Come back to Honor of Kings",
            preheader="Your rewards await",
            body_text="We miss you. Here are your rewards.",
            cta_text="Claim Now",
            tone_style="urgency",
            open_rate=0.28,
            ctr=0.045,
            unsubscribe_rate=0.003,
            conversion_rate=0.012,
        )

    def test_retrieval_text_contains_key_fields(self):
        email = self._make_email()
        text = email.to_retrieval_text()
        assert "Honor of Kings" in text
        assert "reactivation" in text
        assert "Lapsed VIP Players" in text
        assert "Come back to Honor of Kings" in text

    def test_to_dict_roundtrip(self):
        email = self._make_email()
        d = email.to_dict()
        assert d["email_id"] == "test_001"
        assert d["open_rate"] == 0.28
        assert d["game"] == "Honor of Kings"


class TestCampaignBrief:
    def test_brief_creates_with_uuid(self):
        brief = CampaignBrief(
            game="PUBG Mobile",
            campaign_type=CampaignType.REACTIVATION,
            target_segment=PlayerSegment.LAPSED_VIP,
            tone_style=ToneStyle.URGENCY,
            kpi="CTR",
            context="Players inactive 30-90 days",
        )
        assert brief.brief_id is not None
        assert len(brief.brief_id) == 8

    def test_to_dict_serialises_enums(self):
        brief = CampaignBrief(
            game="PUBG Mobile",
            campaign_type=CampaignType.REACTIVATION,
            target_segment=PlayerSegment.LAPSED_VIP,
            tone_style=ToneStyle.URGENCY,
            kpi="CTR",
            context="Test context",
        )
        d = brief.to_dict()
        assert d["campaign_type"] == "reactivation"
        assert d["target_segment"] == "Lapsed VIP Players"
        assert d["tone_style"] == "urgency"


class TestGenerationResult:
    def test_get_best_variant_returns_highest_overall(self):
        from core.types import EmailVariant, ToneStyle
        v1 = EmailVariant(tone_style=ToneStyle.URGENCY)
        v1.quality_score = QualityScore(0.9, 0.8, 0.9, 0.7)
        v2 = EmailVariant(tone_style=ToneStyle.STORYTELLING)
        v2.quality_score = QualityScore(0.6, 0.6, 0.6, 0.6)

        result = GenerationResult(variants=[v1, v2])
        best = result.get_best_variant()
        assert best.quality_score.overall > 0.8

    def test_get_best_variant_empty(self):
        result = GenerationResult(variants=[])
        assert result.get_best_variant() is None


# ─────────────────────────────────────────────
# Config tests
# ─────────────────────────────────────────────

class TestConfig:
    def test_settings_loads_defaults(self):
        settings = get_settings.__wrapped__("nonexistent_config.yaml")
        assert settings.llm.temperature == 0.7
        assert settings.retrieval.rrf_k == 60

    def test_settings_llm_provider(self):
        settings = get_settings.__wrapped__("config.yaml")
        assert settings.llm.provider in ("deepseek", "anthropic", "openai", "azure")


# ─────────────────────────────────────────────
# Mock Generator tests
# ─────────────────────────────────────────────

class TestMockGenerator:
    def test_generate_returns_correct_count(self):
        from data.mock_generator import generate_mock_emails
        emails = generate_mock_emails(n=10)
        assert len(emails) == 10

    def test_emails_have_valid_performance_metrics(self):
        from data.mock_generator import generate_mock_emails
        emails = generate_mock_emails(n=20)
        for e in emails:
            assert 0.0 <= e.open_rate <= 1.0
            assert 0.0 <= e.ctr <= 1.0
            assert 0.0 <= e.unsubscribe_rate <= 1.0
            assert e.subject_line != ""
            assert e.body_text != ""

    def test_performance_score_in_range(self):
        from data.mock_generator import generate_mock_emails
        emails = generate_mock_emails(n=20)
        for e in emails:
            assert 0.0 <= e.performance_score <= 1.0

    def test_golden_test_set_format(self):
        from data.mock_generator import generate_mock_emails, generate_golden_test_set
        emails = generate_mock_emails(n=20)
        golden = generate_golden_test_set(emails, n=5)
        assert len(golden) == 5
        for tc in golden:
            assert "test_id" in tc
            assert "query" in tc
            assert "expected_email_ids" in tc


# ─────────────────────────────────────────────
# Ingestion Pipeline tests
# ─────────────────────────────────────────────

class TestIngestionPipeline:
    def _write_temp_json(self, emails):
        from data.mock_generator import generate_mock_emails
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump([e.to_dict() for e in emails], tmp)
        tmp.close()
        return tmp.name

    def test_pipeline_runs_end_to_end(self):
        from data.mock_generator import generate_mock_emails
        from ingestion.pipeline import IngestionPipeline

        emails = generate_mock_emails(n=10)
        tmp_path = self._write_temp_json(emails)

        pipeline = IngestionPipeline()
        result = pipeline.run(tmp_path)

        assert len(result) == 10
        Path(tmp_path).unlink()

    def test_performance_scores_recomputed(self):
        from data.mock_generator import generate_mock_emails
        from ingestion.pipeline import IngestionPipeline

        emails = generate_mock_emails(n=5)
        tmp_path = self._write_temp_json(emails)

        pipeline = IngestionPipeline()
        result = pipeline.run(tmp_path)

        for e in result:
            assert 0.0 <= e.performance_score <= 1.0

        Path(tmp_path).unlink()

    def test_pipeline_skips_malformed_records(self):
        bad_records = [
            {"email_id": "bad_001"},  # missing required fields
            {
                "email_id": "good_001", "game": "Test Game",
                "campaign_type": "reactivation", "target_segment": "All",
                "subject_line": "Test subject", "preheader": "", "body_text": "Test body",
                "cta_text": "Click", "tone_style": "urgency",
                "open_rate": 0.25, "ctr": 0.04, "unsubscribe_rate": 0.005,
                "conversion_rate": 0.01, "performance_score": 0.3
            }
        ]
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(bad_records, tmp)
        tmp.close()

        from ingestion.pipeline import IngestionPipeline
        pipeline = IngestionPipeline()
        result = pipeline.run(tmp.name)

        assert len(result) == 1
        assert result[0].email_id == "good_001"

        Path(tmp.name).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
