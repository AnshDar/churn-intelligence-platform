"""Tests for the SQLite scoring-history layer."""
from __future__ import annotations

from datetime import datetime, timezone

from src.database import fetch_all, get_engine, init_db, insert_scores, summary_stats


def _record(cid: str, segment: str, prob: float) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc),
        "batch_id": "test",
        "customer_id": cid,
        "churn_probability": prob,
        "churn_prediction": int(prob >= 0.5),
        "risk_segment": segment,
        "top_risk_factors": [{"factor": "Tenure", "contribution": 0.5}],
        "intervention_code": "retention_call",
        "intervention": "Proactive retention call",
        "revenue_at_risk": 1200.0,
        "intervention_cost": 25.0,
        "expected_revenue_saved": 300.0,
        "roi": 12.0,
        "net_benefit": 275.0,
        "model_version": "v-test",
        "latency_ms": 5.0,
    }


def test_insert_and_fetch(tmp_path):
    engine = init_db(get_engine(db_path=tmp_path / "t.db"))
    n = insert_scores([_record("A", "HIGH", 0.9), _record("B", "LOW", 0.1)], engine=engine)
    assert n == 2
    df = fetch_all(engine=engine)
    assert len(df) == 2
    # JSON risk factors round-trip as a string column.
    assert df["top_risk_factors"].str.contains("Tenure").all()


def test_summary_stats(tmp_path):
    engine = init_db(get_engine(db_path=tmp_path / "t.db"))
    insert_scores(
        [_record("A", "HIGH", 0.9), _record("B", "MEDIUM", 0.5), _record("C", "LOW", 0.1)],
        engine=engine,
    )
    stats = summary_stats(engine=engine)
    assert stats["total_customers"] == 3
    assert stats["at_risk_count"] == 2  # HIGH + MEDIUM
    assert stats["at_risk_percentage"] == round(200 / 3, 2)
    assert stats["total_revenue_at_risk"] == 3600.0
