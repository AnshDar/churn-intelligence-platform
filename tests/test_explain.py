"""Tests for the SHAP explainability layer.

Trains a small, non-tuned model so the test is independent of pre-existing
artifacts (and fast enough for CI).
"""
from __future__ import annotations

from src.data_prep import prepare_data
from src.explain import ChurnExplainer, prettify
from src.model import train_model


def test_prettify_labels():
    assert prettify("MonthlyCharges") == "Monthly charges"
    assert prettify("Contract_Month-to-month") == "Contract: Month-to-month"


def test_top_reasons_shape_and_signs():
    data = prepare_data(save=False)
    model = train_model(data, tune=False)
    explainer = ChurnExplainer(model, data.feature_names)

    reasons = explainer.top_reasons(data.X_test[0], k=3, toward_churn=True)
    assert len(reasons) == 3
    # Risk-increasing factors are sorted by descending contribution.
    contribs = [r.contribution for r in reasons]
    assert contribs == sorted(contribs, reverse=True)
    # Each factor carries a readable label and its encoded source name.
    for r in reasons:
        assert r.factor and r.raw_feature in data.feature_names


def test_narrative_mentions_probability():
    data = prepare_data(save=False)
    model = train_model(data, tune=False)
    explainer = ChurnExplainer(model, data.feature_names)
    text = explainer.narrative(data.X_test[0], churn_probability=0.83)
    assert "83%" in text
