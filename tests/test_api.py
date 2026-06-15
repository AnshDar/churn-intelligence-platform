"""Tests for the FastAPI service (Phase 6)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.database import fetch_all


@pytest.fixture(scope="module")
def client():
    # `with` triggers the lifespan (loads the scoring service once).
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_score_returns_full_response(client):
    payload = {"customer_data": {
        "customerID": "T-1", "tenure": 2, "MonthlyCharges": 95.0,
        "Contract": "Month-to-month", "support_tickets": 4, "usage_trend_3m": -0.25,
        "InternetService": "Fiber optic", "TechSupport": "No",
    }}
    r = client.post("/score", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["customer_id"] == "T-1"
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_risk"] in {"HIGH", "MEDIUM", "LOW"}
    assert len(body["top_risk_factors"]) == 3
    assert {"factor", "contribution"} <= set(body["top_risk_factors"][0])


def test_score_rejects_invalid_input(client):
    # Negative tenure violates the ge=0 constraint -> 422.
    r = client.post("/score", json={"customer_data": {"tenure": -5}})
    assert r.status_code == 422


def test_partial_payload_scores(client):
    r = client.post("/score", json={"customer_data": {"MonthlyCharges": 30.0, "tenure": 50}})
    assert r.status_code == 200
    assert r.json()["customer_id"] == "unknown"


def test_model_performance(client):
    r = client.get("/model/performance")
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["auc"] <= 1.0
    assert {"precision", "recall", "f1", "model_version", "last_updated"} <= set(body)


@pytest.mark.skipif(fetch_all().empty, reason="no scoring data — run the simulation first")
def test_dashboard_data(client):
    r = client.get("/dashboard-data")
    assert r.status_code == 200
    body = r.json()
    assert body["total_customers"] > 0
    assert len(body["risk_distribution"]) == 3


def test_docs_available(client):
    assert client.get("/openapi.json").status_code == 200
