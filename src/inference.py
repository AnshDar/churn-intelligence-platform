"""Real-time scoring pipeline (Phase 3).

A :class:`ScoringService` bundles the trained model, preprocessor, SHAP
explainer, and intervention engine behind a single ``score_one`` /
``score_batch`` interface. Each call:

    preprocess -> predict -> explain (SHAP) -> recommend intervention -> return

The service is stateless per call and reusable across the simulation (Phase 3),
the dashboard (Phase 5), and the API (Phase 6).
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import joblib
import pandas as pd

from . import config, interventions
from .data_prep import engineer_features, feature_names_from
from .explain import ChurnExplainer

# Sensible defaults so partial customer payloads (e.g. from the API) still score.
_DEFAULTS: dict = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "tenure": 0,
    "MonthlyCharges": 0.0,
    "support_tickets": 0,
    "usage_trend_3m": 0.0,
}


def _load_threshold() -> float:
    """Read the F1-optimal decision threshold saved by Phase 1."""
    try:
        with open(config.METRICS_PATH, encoding="utf-8") as fh:
            return float(json.load(fh).get("threshold", 0.5))
    except (FileNotFoundError, ValueError, KeyError):
        return 0.5


class ScoringService:
    """Loads artifacts once and scores customers in real time."""

    def __init__(self, model=None, preprocessor=None, threshold: float | None = None):
        self.model = model or joblib.load(config.MODEL_PATH)
        self.preprocessor = preprocessor or joblib.load(config.PREPROCESSOR_PATH)
        self.feature_names = feature_names_from(self.preprocessor)
        self.explainer = ChurnExplainer(self.model, self.feature_names)
        self.threshold = threshold if threshold is not None else _load_threshold()
        self.model_version = config.MODEL_VERSION

    # -- framing ----------------------------------------------------------- #
    def _prepare_frame(self, customer_data: dict) -> pd.DataFrame:
        """Build a one-row engineered DataFrame from a raw customer payload."""
        record = {**_DEFAULTS, **customer_data}
        # TotalCharges defaults to tenure * monthly if not supplied.
        if customer_data.get("TotalCharges") in (None, ""):
            record["TotalCharges"] = float(record["tenure"]) * float(record["MonthlyCharges"])
        df = pd.DataFrame([record])
        return engineer_features(df)

    # -- scoring ----------------------------------------------------------- #
    def score_one(self, customer_data: dict, *, explain: bool = True,
                  batch_id: str = "adhoc", timestamp: datetime | None = None) -> dict:
        """Score a single customer; return a record ready for DB/API use."""
        start = time.perf_counter()

        df = self._prepare_frame(customer_data)
        feature_cols = (
            config.NUMERIC_FEATURES + config.ENGINEERED_FEATURES + config.CATEGORICAL_FEATURES
        )
        X = self.preprocessor.transform(df[feature_cols])
        proba = float(self.model.predict_proba(X)[0, 1])
        prediction = int(proba >= self.threshold)

        factors = (
            [r.as_dict() for r in self.explainer.top_reasons(X[0], k=3)]
            if explain else []
        )
        rec = interventions.recommend(proba, df.iloc[0])
        latency_ms = (time.perf_counter() - start) * 1000.0

        customer_id = (
            customer_data.get(config.ID_COL)
            or customer_data.get("customer_id")
            or "unknown"
        )
        return {
            "timestamp": timestamp or datetime.now(timezone.utc),
            "batch_id": batch_id,
            "customer_id": customer_id,
            "churn_probability": round(proba, 4),
            "churn_prediction": prediction,
            "risk_segment": rec.risk_segment,
            "top_risk_factors": factors,
            "intervention_code": rec.intervention_code,
            "intervention": rec.intervention,
            "revenue_at_risk": round(rec.revenue_at_risk, 2),
            "intervention_cost": round(rec.intervention_cost, 2),
            "expected_revenue_saved": round(rec.expected_revenue_saved, 2),
            "roi": round(rec.roi, 2),
            "net_benefit": round(rec.net_benefit, 2),
            "model_version": self.model_version,
            "latency_ms": round(latency_ms, 2),
        }

    def score_batch(self, df: pd.DataFrame, *, batch_id: str = "adhoc",
                    timestamp: datetime | None = None, explain: bool = True) -> list[dict]:
        """Score every row of a raw-schema DataFrame."""
        return [
            self.score_one(row.to_dict(), explain=explain, batch_id=batch_id, timestamp=timestamp)
            for _, row in df.iterrows()
        ]


# Module-level lazy singleton (avoids reloading artifacts on every call).
_service: ScoringService | None = None


def get_service() -> ScoringService:
    """Return a process-wide :class:`ScoringService`, creating it on first use."""
    global _service
    if _service is None:
        _service = ScoringService()
    return _service


def real_time_score(customer_data: dict, *, log: bool = False) -> dict:
    """Convenience entry point: score one customer, optionally logging to SQLite."""
    result = get_service().score_one(customer_data)
    if log:
        from .database import insert_scores

        insert_scores([result])
    return result
