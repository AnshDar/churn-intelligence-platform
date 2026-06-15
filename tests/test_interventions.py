"""Tests for the intervention recommendation engine."""
from __future__ import annotations

import pandas as pd

from src import config
from src.interventions import recommend, recommend_batch, risk_segment


def test_risk_segment_bands():
    assert risk_segment(0.95) == "HIGH"
    assert risk_segment(0.55) == "MEDIUM"
    assert risk_segment(0.10) == "LOW"


def test_high_value_customer_gets_loyalty_discount():
    customer = {
        "MonthlyCharges": 100.0, "tenure": 20, "Contract": "Month-to-month",
        "support_tickets": 0, "usage_trend_3m": 0.0,
    }
    rec = recommend(0.85, customer)
    assert rec.intervention_code == "loyalty_discount"
    assert rec.risk_segment == "HIGH"
    assert rec.revenue_at_risk == 100.0 * 12
    # Discount cost is 20% of exposure.
    assert rec.intervention_cost == 0.20 * rec.revenue_at_risk
    assert rec.roi > 0


def test_new_struggling_customer_gets_onboarding():
    customer = {
        "MonthlyCharges": 40.0, "tenure": 3, "Contract": "Month-to-month",
        "support_tickets": 0, "usage_trend_3m": -0.2,
    }
    rec = recommend(0.80, customer)
    assert rec.intervention_code == "onboarding_support"


def test_low_risk_gets_no_action():
    customer = {"MonthlyCharges": 50.0, "tenure": 40, "Contract": "Two year",
                "support_tickets": 0, "usage_trend_3m": 0.1}
    rec = recommend(0.10, customer)
    assert rec.intervention_code == "none"
    assert rec.intervention_cost == 0.0


def test_batch_is_ranked_by_net_benefit():
    df = pd.DataFrame([
        {config.ID_COL: "A", "MonthlyCharges": 100.0, "tenure": 5,
         "Contract": "Month-to-month", "support_tickets": 5, "usage_trend_3m": -0.3},
        {config.ID_COL: "B", "MonthlyCharges": 20.0, "tenure": 60,
         "Contract": "Two year", "support_tickets": 0, "usage_trend_3m": 0.2},
    ])
    out = recommend_batch(df, [0.9, 0.05])
    assert list(out["priority_rank"]) == [1, 2]
    # Net benefit must be sorted descending.
    assert out["net_benefit"].is_monotonic_decreasing
