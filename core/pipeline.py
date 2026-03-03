"""
core/pipeline.py
================
Orchestrates the full end-to-end generation run:
  Brief → Retrieve → Generate → Score → Guardrail → Render → UpliftEstimate → Result
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import get_settings
from core.types import CampaignBrief, GenerationResult, EmailVariant


class CRMEmailPipeline:
    """Lazy-loads all components on first run to keep import time fast."""

    def __init__(self):
        self.settings = get_settings()
        self._emails = None
        self._retriever = None
        self._generator = None
        self._scorer = None
        self._guardrail = None
        self._renderer = None
        self._uplift = None
        self._initialized = False

    def _init_components(self):
        if self._initialized:
            return
        print("[Pipeline] Initialising components...")

        from ingestion.pipeline import IngestionPipeline
        from retrieval.vector_store import build_vector_index
        from retrieval.hybrid_retriever import HybridRetriever
        from generation.email_generator import DeepSeekEmailGenerator
        from scoring.quality_scorer import LLMQualityScorer
        from scoring.safety_guardrail import SafetyGuardrail
        from templating.html_renderer import HTMLEmailRenderer
        from evaluation.uplift_estimator import UpliftEstimator

        self._emails = IngestionPipeline().run()
        vector_store = build_vector_index(self._emails)
        self._retriever = HybridRetriever(self._emails, vector_store)
        self._generator = DeepSeekEmailGenerator()
        self._scorer = LLMQualityScorer()
        self._guardrail = SafetyGuardrail()
        self._renderer = HTMLEmailRenderer()
        self._uplift = UpliftEstimator()
        self._initialized = True
        print("[Pipeline] Ready.")

    def run(self, brief: CampaignBrief, num_variants: int = 3) -> GenerationResult:
        self._init_components()
        t0 = time.time()
        result = GenerationResult(brief=brief)

        # 1. Retrieve
        print(f"[Pipeline] Retrieving historical context for: {brief.game} / {brief.campaign_type.value}")
        retrieved = self._retriever.retrieve_from_brief(brief, top_k=3)
        result.retrieved_emails = [r.email for r in retrieved]
        result.pipeline_metadata["retrieval_scores"] = [
            {"email_id": r.email.email_id, "rrf": round(r.rrf_score, 4), "rank": r.rank}
            for r in retrieved
        ]

        # 2. Generate
        print(f"[Pipeline] Generating {num_variants} variants via DeepSeek...")
        variants = self._generator.generate(brief, result.retrieved_emails, num_variants)

        # 3. Score + Guardrail + Render
        for v in variants:
            v.quality_score = self._scorer.score(v, brief)
            passed, flags = self._guardrail.check(v, brief)
            v.quality_score.passed_guardrails = passed
            v.quality_score.guardrail_flags = flags
            v.html_content = self._renderer.render(v, brief, show_scores=True)

        result.variants = variants
        best = result.get_best_variant()
        if best:
            result.best_variant_id = best.variant_id

        result.pipeline_metadata["latency_s"] = round(time.time() - t0, 2)
        result.pipeline_metadata["num_retrieved"] = len(result.retrieved_emails)
        result.pipeline_metadata["num_variants"] = len(variants)

        print(
            f"[Pipeline] Done in {result.pipeline_metadata['latency_s']}s. "
            f"Best variant: {result.best_variant_id} "
            f"(score={best.quality_score.overall:.2f})" if best else ""
        )
        return result

    @property
    def uplift_estimator(self):
        self._init_components()
        return self._uplift

    @property
    def historical_emails(self):
        self._init_components()
        return self._emails
