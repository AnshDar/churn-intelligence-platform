"""Tests for the monitoring metrics (no MLflow / DB dependency)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.monitoring import (
    business_metrics,
    inference_metrics,
    population_stability_index,
    score_drift,
)


def _log_df() -> pd.DataFrame:
    return pd.DataFrame({
        "batch_id": ["b0"] * 3 + ["b1"] * 3,
        "churn_probability": [0.1, 0.2, 0.3, 0.8, 0.85, 0.9],
        "risk_segment": ["LOW", "LOW", "MEDIUM", "HIGH", "HIGH", "HIGH"],
        "intervention_code": ["none", "none", "engagement_nudge",
                              "loyalty_discount", "retention_call", "escalate_cs"],
        "revenue_at_risk": [600, 700, 800, 1200, 1300, 1400],
        "expected_revenue_saved": [0, 0, 50, 300, 280, 260],
        "intervention_cost": [0, 0, 10, 240, 25, 80],
        "latency_ms": [10, 20, 30, 40, 50, 60],
    })


def test_psi_zero_for_identical_distributions():
    x = np.random.default_rng(0).normal(size=1000)
    assert population_stability_index(x, x.copy()) < 1e-6


def test_psi_large_for_shifted_distributions():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 1000)
    cur = rng.normal(3, 1, 1000)  # big mean shift
    assert population_stability_index(ref, cur) > 0.2


def test_score_drift_flags_shift():
    df = _log_df()  # b0 low probs, b1 high probs -> significant drift
    result = score_drift(df)
    assert result["drift_detected"] is True
    assert result["reference_batch"] == "b0"


def test_business_metrics():
    biz = business_metrics(_log_df())
    assert biz["total_customers"] == 6
    assert biz["high_risk_count"] == 3
    assert biz["at_risk_count"] == 4  # 3 HIGH + 1 MEDIUM
    assert biz["interventions_recommended"] == 4  # non-"none"
    assert biz["intervention_mix"]["none"] == 2


def test_inference_metrics():
    inf = inference_metrics(_log_df())
    assert inf["predictions"] == 6
    assert inf["avg_latency_ms"] == 35.0
    assert inf["throughput_per_sec"] > 0
