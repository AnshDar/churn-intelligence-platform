"""Tests for the real-time scoring service.

Builds the service from a freshly trained (non-tuned) model so the test does
not depend on on-disk artifacts.
"""
from __future__ import annotations

import pytest

from src.data_prep import prepare_data
from src.inference import ScoringService
from src.model import train_model


@pytest.fixture(scope="module")
def service() -> ScoringService:
    data = prepare_data(save=False)
    model = train_model(data, tune=False)
    return ScoringService(model=model, preprocessor=data.preprocessor, threshold=0.5)


def test_score_one_contract(service):
    result = service.score_one({
        "customerID": "X-1", "tenure": 3, "MonthlyCharges": 95.0,
        "Contract": "Month-to-month", "support_tickets": 5, "usage_trend_3m": -0.3,
        "InternetService": "Fiber optic", "TechSupport": "No",
    })
    expected_keys = {
        "customer_id", "churn_probability", "risk_segment", "top_risk_factors",
        "intervention", "revenue_at_risk", "roi", "net_benefit", "latency_ms",
    }
    assert expected_keys <= set(result)
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert len(result["top_risk_factors"]) == 3
    assert result["latency_ms"] >= 0


def test_partial_payload_uses_defaults(service):
    # Only a couple of fields supplied — defaults should fill the rest.
    result = service.score_one({"tenure": 60, "MonthlyCharges": 20.0, "Contract": "Two year"})
    assert result["customer_id"] == "unknown"
    assert result["risk_segment"] in {"HIGH", "MEDIUM", "LOW"}


def test_high_risk_profile_scores_higher_than_low(service):
    high = service.score_one({
        "tenure": 1, "MonthlyCharges": 110.0, "Contract": "Month-to-month",
        "support_tickets": 6, "usage_trend_3m": -0.4, "InternetService": "Fiber optic",
        "TechSupport": "No",
    })
    low = service.score_one({
        "tenure": 68, "MonthlyCharges": 25.0, "Contract": "Two year",
        "support_tickets": 0, "usage_trend_3m": 0.2, "TechSupport": "Yes",
        "Partner": "Yes", "Dependents": "Yes",
    })
    assert high["churn_probability"] > low["churn_probability"]
