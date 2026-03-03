"""
evaluation/uplift_estimator.py
================================
Monte Carlo uplift simulation + efficiency savings estimator.
No external dependencies — pure numpy + math.
"""

from __future__ import annotations
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import get_settings
from core.types import EmailVariant, QualityScore


@dataclass
class UpliftReport:
    """Results of one uplift simulation run."""
    baseline_open_rate: float
    baseline_ctr: float
    predicted_open_rate: float
    predicted_ctr: float
    open_rate_uplift_pct: float       # e.g. 12.5 means +12.5%
    ctr_uplift_pct: float
    open_rate_ci_low: float
    open_rate_ci_high: float
    ctr_ci_low: float
    ctr_ci_high: float
    simulation_runs: int
    quality_score_used: float

    # Efficiency
    manual_minutes_per_email: float = 45.0
    ai_minutes_per_email: float = 3.0
    monthly_email_volume: int = 200
    time_saved_hours_per_month: float = 0.0
    fte_equivalent_saved: float = 0.0

    def __post_init__(self):
        saved_mins = (self.manual_minutes_per_email - self.ai_minutes_per_email) * self.monthly_email_volume
        self.time_saved_hours_per_month = round(saved_mins / 60, 1)
        self.fte_equivalent_saved = round(self.time_saved_hours_per_month / 160, 2)  # 160 working hrs/month


class UpliftEstimator:
    """
    Estimates email performance uplift using quality score as a proxy signal.

    Method:
    - Quality score (0-1) is mapped to a multiplicative uplift factor
    - 95% confidence intervals computed via bootstrap simulation
    - Baseline calibrated to Mailchimp gaming benchmarks
    """

    def __init__(self):
        self.settings = get_settings()

    def _score_to_uplift_factor(self, quality_overall: float) -> float:
        """
        Maps quality score to expected lift multiplier.
        At quality=0.5 (average): +0% lift (factor = 1.0)
        At quality=0.8 (good):    +15% lift
        At quality=1.0 (perfect): +25% lift
        At quality=0.3 (poor):    -10% lift
        """
        centered = quality_overall - 0.5      # range: -0.5 to +0.5
        factor = 1.0 + (centered * 0.50)      # linear: maps to 0.75–1.25
        return max(0.70, min(1.35, factor))

    def estimate(
        self,
        variant: EmailVariant,
        monthly_email_volume: int = 200,
        manual_minutes: float = 45.0,
    ) -> UpliftReport:
        cfg = self.settings.evaluation
        qs = variant.quality_score

        if qs is None:
            # No score yet — use neutral estimate
            quality_overall = 0.65
        else:
            quality_overall = qs.overall

        factor = self._score_to_uplift_factor(quality_overall)

        # Point estimates
        pred_open = cfg.baseline_open_rate * factor
        pred_ctr = cfg.baseline_ctr * factor

        # Monte Carlo simulation for confidence intervals
        rng = np.random.default_rng(42)
        n = cfg.uplift_simulation_runs

        # Add noise: quality score uncertainty (std=0.05), campaign variance (std=0.03)
        quality_samples = np.clip(
            rng.normal(quality_overall, 0.05, n), 0.0, 1.0
        )
        factors = 1.0 + (quality_samples - 0.5) * 0.50
        campaign_noise = rng.normal(1.0, 0.03, n)
        open_samples = cfg.baseline_open_rate * factors * campaign_noise
        ctr_samples = cfg.baseline_ctr * factors * campaign_noise

        return UpliftReport(
            baseline_open_rate=cfg.baseline_open_rate,
            baseline_ctr=cfg.baseline_ctr,
            predicted_open_rate=round(pred_open, 4),
            predicted_ctr=round(pred_ctr, 4),
            open_rate_uplift_pct=round((pred_open / cfg.baseline_open_rate - 1) * 100, 1),
            ctr_uplift_pct=round((pred_ctr / cfg.baseline_ctr - 1) * 100, 1),
            open_rate_ci_low=round(float(np.percentile(open_samples, 2.5)), 4),
            open_rate_ci_high=round(float(np.percentile(open_samples, 97.5)), 4),
            ctr_ci_low=round(float(np.percentile(ctr_samples, 2.5)), 4),
            ctr_ci_high=round(float(np.percentile(ctr_samples, 97.5)), 4),
            simulation_runs=n,
            quality_score_used=round(quality_overall, 3),
            manual_minutes_per_email=manual_minutes,
            monthly_email_volume=monthly_email_volume,
        )

    def compare_variants(self, variants: list[EmailVariant]) -> list[tuple[EmailVariant, UpliftReport]]:
        """Run uplift estimation for each variant and return sorted by predicted open rate."""
        results = [(v, self.estimate(v)) for v in variants]
        results.sort(key=lambda x: x[1].predicted_open_rate, reverse=True)
        return results
