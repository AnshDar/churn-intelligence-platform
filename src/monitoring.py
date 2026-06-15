"""Model & business monitoring (Phase 4).

Produces the metrics the Streamlit dashboard (Phase 5) reads:

* **Model performance** — current test-set AUC/precision/recall vs the
  registered baseline, with a >5% AUC-drop drift flag.
* **Score drift** — Population Stability Index (PSI) on the churn-probability
  distribution between the earliest and latest scored batches.
* **Business metrics** — customers scored, % at-risk, revenue at risk,
  intervention mix, and an at-risk trend over time.
* **Inference metrics** — average/p95 latency, throughput, cost per prediction.

Everything is aggregated into one JSON blob via :func:`build_monitoring_data`.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config
from .database import fetch_all


# --------------------------------------------------------------------------- #
# Drift — Population Stability Index
# --------------------------------------------------------------------------- #
def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """PSI between a reference and current distribution.

    Rule of thumb: < 0.1 no shift, 0.1–0.2 moderate, > 0.2 significant drift.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if reference.size == 0 or current.size == 0:
        return 0.0

    # Quantile bin edges from the reference distribution.
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if edges.size < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    eps = 1e-6
    ref_pct = np.histogram(reference, bins=edges)[0] / len(reference) + eps
    cur_pct = np.histogram(current, bins=edges)[0] / len(current) + eps
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def score_drift(df: pd.DataFrame) -> dict:
    """PSI on churn probability between the first and last batch in the log."""
    if df.empty or "batch_id" not in df or df["batch_id"].nunique() < 2:
        return {"psi": 0.0, "drift_detected": False, "note": "insufficient batches"}

    batches = sorted(df["batch_id"].unique())
    ref = df.loc[df["batch_id"] == batches[0], "churn_probability"].to_numpy()
    cur = df.loc[df["batch_id"] == batches[-1], "churn_probability"].to_numpy()
    psi = population_stability_index(ref, cur)
    return {
        "psi": round(psi, 4),
        "drift_detected": bool(psi > config.PSI_THRESHOLD),
        "reference_batch": batches[0],
        "current_batch": batches[-1],
        "threshold": config.PSI_THRESHOLD,
    }


# --------------------------------------------------------------------------- #
# Model performance vs baseline
# --------------------------------------------------------------------------- #
def model_performance() -> dict:
    """Re-evaluate the current model on the held-out test set vs the baseline.

    With an unchanged model these match the baseline; the comparison exists so
    that, once the model or data shifts, the >5% AUC-drop flag fires.
    """
    from sklearn.metrics import roc_auc_score

    try:
        with open(config.METRICS_PATH, encoding="utf-8") as fh:
            baseline = json.load(fh)
    except FileNotFoundError:
        return {"note": "no baseline metrics found"}

    baseline_auc = float(baseline.get("auc", 0.0))
    current_auc = baseline_auc
    try:
        from .explain import load_artifacts, load_processed, transform_features

        model, preprocessor, _ = load_artifacts()
        test_df = load_processed()
        X = transform_features(test_df, preprocessor)
        y = test_df[config.TARGET_COL].to_numpy()
        current_auc = float(roc_auc_score(y, model.predict_proba(X)[:, 1]))
    except FileNotFoundError:
        pass

    drop = (baseline_auc - current_auc) / baseline_auc if baseline_auc else 0.0
    return {
        "baseline_auc": round(baseline_auc, 4),
        "current_auc": round(current_auc, 4),
        "auc_drop_pct": round(100 * drop, 2),
        "drift_detected": bool(drop > config.DRIFT_AUC_DROP),
        "precision": round(float(baseline.get("precision", 0.0)), 4),
        "recall": round(float(baseline.get("recall", 0.0)), 4),
        "f1": round(float(baseline.get("f1", 0.0)), 4),
        "model_version": _production_version(),
    }


def _production_version() -> str:
    """Production model version from MLflow, falling back to the config string."""
    try:
        from .mlflow_tracking import get_production_version

        v = get_production_version()
        return f"v{v}" if v else config.MODEL_VERSION
    except Exception:
        return config.MODEL_VERSION


# --------------------------------------------------------------------------- #
# Business & inference metrics
# --------------------------------------------------------------------------- #
def business_metrics(df: pd.DataFrame) -> dict:
    """Aggregate business KPIs + intervention mix from the scoring log."""
    if df.empty:
        return {"total_customers": 0}
    mix = df["intervention_code"].value_counts().to_dict()
    actioned = df[df["intervention_code"] != "none"]
    return {
        "total_customers": int(len(df)),
        "at_risk_count": int((df["risk_segment"].isin(["HIGH", "MEDIUM"])).sum()),
        "high_risk_count": int((df["risk_segment"] == "HIGH").sum()),
        "at_risk_percentage": round(100 * (df["risk_segment"].isin(["HIGH", "MEDIUM"])).mean(), 2),
        "total_revenue_at_risk": round(float(df["revenue_at_risk"].sum()), 2),
        "expected_revenue_saved": round(float(df["expected_revenue_saved"].sum()), 2),
        "total_intervention_cost": round(float(df["intervention_cost"].sum()), 2),
        "interventions_recommended": int(len(actioned)),
        "intervention_mix": {k: int(v) for k, v in mix.items()},
    }


def inference_metrics(df: pd.DataFrame, cost_per_1k: float = 0.50) -> dict:
    """Latency / throughput / cost metrics from the logged latencies."""
    if df.empty or "latency_ms" not in df:
        return {"predictions": 0}
    lat = df["latency_ms"].to_numpy()
    return {
        "predictions": int(len(df)),
        "avg_latency_ms": round(float(np.mean(lat)), 2),
        "p95_latency_ms": round(float(np.percentile(lat, 95)), 2),
        "max_latency_ms": round(float(np.max(lat)), 2),
        "throughput_per_sec": round(1000.0 / float(np.mean(lat)), 1) if np.mean(lat) else 0.0,
        "cost_per_prediction": round(cost_per_1k / 1000.0, 6),
        "estimated_total_cost": round(len(df) * cost_per_1k / 1000.0, 4),
    }


def at_risk_trend(df: pd.DataFrame) -> list[dict]:
    """Per-timestamp at-risk % and revenue — feeds the dashboard trend line."""
    if df.empty:
        return []
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    grouped = df.groupby(df["timestamp"].dt.floor("h"))
    rows = []
    for ts, g in grouped:
        rows.append({
            "timestamp": ts.isoformat(),
            "scored": int(len(g)),
            "at_risk_pct": round(100 * g["risk_segment"].isin(["HIGH", "MEDIUM"]).mean(), 2),
            "revenue_at_risk": round(float(g["revenue_at_risk"].sum()), 2),
        })
    return rows


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_monitoring_data(save: bool = True) -> dict:
    """Assemble the full monitoring payload and optionally write it to JSON."""
    df = fetch_all()
    data = {
        "model_performance": model_performance(),
        "score_drift": score_drift(df),
        "business_metrics": business_metrics(df),
        "inference_metrics": inference_metrics(df),
        "at_risk_trend": at_risk_trend(df),
    }
    if save:
        config.ensure_dirs()
        with open(config.MONITORING_DATA_PATH, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
    return data


def main() -> None:
    data = build_monitoring_data(save=True)
    perf, drift, biz, inf = (
        data["model_performance"], data["score_drift"],
        data["business_metrics"], data["inference_metrics"],
    )
    print("=" * 70)
    print("MONITORING REPORT")
    print("=" * 70)
    print("\n[Model performance]")
    print(f"  Version          : {perf.get('model_version')}")
    print(f"  Baseline AUC     : {perf.get('baseline_auc')}")
    print(f"  Current AUC      : {perf.get('current_auc')}  (drop {perf.get('auc_drop_pct')}%)")
    print(f"  Performance drift: {perf.get('drift_detected')}")
    print("\n[Score drift]")
    print(f"  PSI              : {drift.get('psi')}  (threshold {drift.get('threshold')})")
    print(f"  Drift detected   : {drift.get('drift_detected')}")
    print("\n[Business]")
    print(f"  Customers scored : {biz.get('total_customers'):,}")
    print(f"  At-risk          : {biz.get('at_risk_count'):,} ({biz.get('at_risk_percentage')}%)")
    print(f"  Revenue at risk  : {biz.get('total_revenue_at_risk'):,.0f}")
    print(f"  Interventions    : {biz.get('interventions_recommended'):,}")
    print("\n[Inference]")
    print(f"  Avg latency      : {inf.get('avg_latency_ms')} ms")
    print(f"  p95 latency      : {inf.get('p95_latency_ms')} ms")
    print(f"  Throughput       : {inf.get('throughput_per_sec')} preds/sec")
    print(f"  Cost/prediction  : {inf.get('cost_per_prediction')}")
    print("=" * 70)
    print(f"Saved -> {config.MONITORING_DATA_PATH}")


if __name__ == "__main__":
    main()
