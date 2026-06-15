"""Fast smoke tests for the Phase 1 data + model pipeline.

Kept deliberately quick (small sample, no hyperparameter search) so they are
safe to run in CI on every push.
"""
from __future__ import annotations

import numpy as np

from src import config
from src.data_prep import engineer_features, prepare_data
from src.generate_data import generate_churn_data
from src.model import best_threshold, evaluate_model, train_model


def test_generator_schema_and_rate():
    df = generate_churn_data(n=1_000)
    # All declared raw columns must be present.
    for col in [config.ID_COL, config.TARGET_COL, *config.NUMERIC_FEATURES, *config.CATEGORICAL_FEATURES]:
        assert col in df.columns, f"missing column {col}"
    rate = (df[config.TARGET_COL] == "Yes").mean()
    assert 0.15 < rate < 0.40, f"churn rate {rate:.2f} outside realistic band"


def test_engineer_features_adds_columns():
    df = generate_churn_data(n=500)
    df[config.TARGET_COL] = (df[config.TARGET_COL] == "Yes").astype(int)
    out = engineer_features(df)
    for col in config.ENGINEERED_FEATURES:
        assert col in out.columns
    # Spend velocity must be finite even for zero-tenure customers.
    assert np.isfinite(out["charges_per_tenure"]).all()


def test_pipeline_trains_and_scores():
    data = prepare_data(save=False)
    model = train_model(data, tune=False)  # fast: skip the search
    thr = best_threshold(model, data.X_train, data.y_train)
    metrics = evaluate_model(model, data, threshold=thr)
    # A non-tuned model on this signal should still clear a sane AUC floor.
    assert metrics["auc"] > 0.80
    assert 0.0 <= metrics["precision"] <= 1.0
    assert len(data.feature_names) == data.X_train.shape[1]
