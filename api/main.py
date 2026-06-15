"""FastAPI service for the Churn Intelligence Platform (Phase 6).

Endpoints
---------
* ``POST /score``              — score one customer (predict + SHAP + intervention)
* ``GET  /model/performance``  — current model metrics + drift status
* ``GET  /dashboard-data``     — aggregate business KPIs for the scoring history

Plus ``GET /health`` and the auto-generated docs at ``/docs``.

Run::

    uvicorn api.main:app --reload
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

# Make the `src` package importable when launched via uvicorn from anywhere.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from api.schemas import (  # noqa: E402
    DashboardResponse,
    PerformanceResponse,
    ScoreRequest,
    ScoreResponse,
)
from src import config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("churn-api")

# Holds objects loaded once at startup.
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the scoring service and baseline performance once on startup."""
    from src.inference import ScoringService
    from src.monitoring import model_performance

    logger.info("Loading scoring service…")
    state["service"] = ScoringService()
    try:
        state["performance"] = model_performance()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not compute performance at startup: %s", exc)
        state["performance"] = {}
    state["model_mtime"] = (
        datetime.fromtimestamp(config.MODEL_PATH.stat().st_mtime, tz=timezone.utc)
        if config.MODEL_PATH.exists() else datetime.now(timezone.utc)
    )
    logger.info("Startup complete. Model version %s", state["service"].model_version)
    yield
    state.clear()


app = FastAPI(
    title="Churn Intelligence Platform API",
    description="Predict customer churn, explain it with SHAP, and recommend "
                "ROI-ranked retention interventions.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health", tags=["meta"])
def health() -> dict:
    """Liveness probe."""
    return {"status": "ok", "model_version": state.get("service").model_version
            if state.get("service") else None}


@app.get("/", tags=["meta"])
def root() -> dict:
    return {"service": "churn-intelligence", "docs": "/docs",
            "endpoints": ["/score", "/model/performance", "/dashboard-data"]}


@app.post("/score", response_model=ScoreResponse, tags=["scoring"])
def score_customer(request: ScoreRequest, log: bool = False) -> ScoreResponse:
    """Score a single customer.

    Set ``?log=true`` to persist the result to the SQLite scoring history.
    """
    service = state["service"]
    payload = request.customer_data.model_dump(exclude_none=True)
    try:
        result = service.score_one(payload)
    except Exception as exc:
        logger.exception("Scoring failed")
        raise HTTPException(status_code=400, detail=f"Scoring failed: {exc}") from exc

    if log:
        from src.database import insert_scores

        insert_scores([result])

    return ScoreResponse(
        customer_id=str(result["customer_id"]),
        churn_probability=result["churn_probability"],
        churn_risk=result["risk_segment"],
        top_risk_factors=[
            {"factor": f["factor"], "contribution": f["contribution"]}
            for f in result["top_risk_factors"]
        ],
        intervention=result["intervention"],
        revenue_at_risk=result["revenue_at_risk"],
        intervention_cost=result["intervention_cost"],
        expected_revenue_saved=result["expected_revenue_saved"],
        roi=result["roi"],
        net_benefit=result["net_benefit"],
        model_version=result["model_version"],
        latency_ms=result["latency_ms"],
    )


@app.get("/model/performance", response_model=PerformanceResponse, tags=["model"])
def model_performance_endpoint() -> PerformanceResponse:
    """Return the production model's evaluation metrics and drift status."""
    perf = state.get("performance") or {}
    if not perf:
        raise HTTPException(status_code=503, detail="Performance metrics unavailable")
    return PerformanceResponse(
        auc=perf.get("current_auc", perf.get("baseline_auc", 0.0)),
        precision=perf.get("precision", 0.0),
        recall=perf.get("recall", 0.0),
        f1=perf.get("f1", 0.0),
        model_version=perf.get("model_version", state["service"].model_version),
        drift_detected=bool(perf.get("drift_detected", False)),
        last_updated=state["model_mtime"].date().isoformat(),
    )


@app.get("/dashboard-data", response_model=DashboardResponse, tags=["dashboard"])
def dashboard_data() -> DashboardResponse:
    """Aggregate business KPIs over the full scoring history."""
    import json

    from src.database import fetch_all
    from src.monitoring import business_metrics

    df = fetch_all()
    biz = business_metrics(df)
    if biz.get("total_customers", 0) == 0:
        raise HTTPException(status_code=404, detail="No scoring history yet — run the simulation.")

    risk_dist = [
        {"segment": seg, "count": int((df["risk_segment"] == seg).sum())}
        for seg in ["HIGH", "MEDIUM", "LOW"]
    ]

    # Global top risk factors from the baseline feature-importance report.
    top_factors: list[str] = []
    try:
        with open(config.METRICS_PATH, encoding="utf-8") as fh:
            from src.explain import prettify

            top_factors = [prettify(name) for name, _ in json.load(fh).get("top_features", [])[:5]]
    except (FileNotFoundError, KeyError):
        pass

    return DashboardResponse(
        total_customers=biz["total_customers"],
        at_risk_count=biz["at_risk_count"],
        at_risk_percentage=biz["at_risk_percentage"],
        total_revenue_at_risk=biz["total_revenue_at_risk"],
        expected_revenue_saved=biz["expected_revenue_saved"],
        top_risk_factors=top_factors,
        risk_distribution=risk_dist,
        intervention_mix=biz["intervention_mix"],
    )
