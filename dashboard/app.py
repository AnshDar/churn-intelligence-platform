"""Streamlit dashboard for the Churn Intelligence Platform (Phase 5).

Six sections (tabs):
  1. Executive Summary    — headline KPIs + at-risk split
  2. Risk Distribution    — probability histogram, segment mix, at-risk trend
  3. At-Risk Customers    — interactive, filterable table with SHAP + intervention
  4. Explainability       — global SHAP importance + per-customer drivers
  5. Model Performance    — AUC/precision/recall/F1, confusion matrix, versions
  6. Business Metrics     — interventions, ROI scenario, cost-benefit

Run from the project root::

    streamlit run dashboard/app.py

Reads from the SQLite scoring DB, the monitoring JSON, the baseline metrics, and
the SHAP summary image produced by earlier phases — so run the pipeline +
simulation first (see README).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make the `src` package importable when run via `streamlit run`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.database import fetch_all  # noqa: E402
from src.monitoring import build_monitoring_data  # noqa: E402

# --------------------------------------------------------------------------- #
# Page config & theme
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Churn Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

SEGMENT_COLORS = {"HIGH": "#d62728", "MEDIUM": "#ff7f0e", "LOW": "#2ca02c"}


# --------------------------------------------------------------------------- #
# Data loading (cached)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_scores() -> pd.DataFrame:
    df = fetch_all()
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["top_factors_str"] = df["top_risk_factors"].apply(_factors_to_str)
    return df


@st.cache_data(show_spinner=False)
def load_monitoring() -> dict:
    return build_monitoring_data(save=False)


@st.cache_data(show_spinner=False)
def load_baseline_metrics() -> dict:
    try:
        with open(config.METRICS_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return {}


def _factors_to_str(raw: str) -> str:
    try:
        factors = json.loads(raw) if isinstance(raw, str) else (raw or [])
        return " · ".join(f"{f['factor']} ({f['contribution']:+.2f})" for f in factors)
    except (json.JSONDecodeError, TypeError, KeyError):
        return ""


def _mlflow_versions() -> list[str]:
    try:
        import mlflow

        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        client = mlflow.MlflowClient()
        versions = client.search_model_versions(f"name='{config.REGISTERED_MODEL_NAME}'")
        return [f"v{v.version}" for v in sorted(versions, key=lambda x: int(x.version), reverse=True)]
    except Exception:
        return [config.MODEL_VERSION]


def fmt_money(x: float) -> str:
    return f"${x:,.0f}"


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
scores = load_scores()

st.title("📊 Real-Time Customer Churn Intelligence Platform")

if scores.empty:
    st.warning(
        "No scoring data found. Generate some by running the simulation:\n\n"
        "```\npython -m src.simulate --batches 10 --batch-size 50 --reset\n```"
    )
    st.stop()

monitoring = load_monitoring()
baseline = load_baseline_metrics()

# --------------------------------------------------------------------------- #
# Sidebar filters (applied to the at-risk table & distributions)
# --------------------------------------------------------------------------- #
st.sidebar.header("Filters")
segments = st.sidebar.multiselect(
    "Risk segment", options=["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM", "LOW"]
)
prob_range = st.sidebar.slider("Churn probability", 0.0, 1.0, (0.0, 1.0), 0.05)
if st.sidebar.button("🔄 Refresh data"):
    st.cache_data.clear()
    st.rerun()
st.sidebar.caption(f"Model version: **{monitoring['model_performance'].get('model_version')}**")
st.sidebar.caption(f"Records: **{len(scores):,}**")

filtered = scores[
    scores["risk_segment"].isin(segments)
    & scores["churn_probability"].between(*prob_range)
]

# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Executive Summary", "Risk Distribution", "At-Risk Customers",
    "Explainability", "Model Performance", "Business Metrics",
])

# ---- 1. Executive Summary -------------------------------------------------- #
with tab1:
    st.subheader("Executive Summary")
    biz = monitoring["business_metrics"]
    top_intervention = max(
        ((k, v) for k, v in biz["intervention_mix"].items() if k != "none"),
        key=lambda kv: kv[1], default=("—", 0),
    )[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers Monitored", f"{biz['total_customers']:,}")
    c2.metric("At-Risk", f"{biz['at_risk_count']:,}", f"{biz['at_risk_percentage']}%")
    c3.metric("Revenue at Risk", fmt_money(biz["total_revenue_at_risk"]))
    c4.metric("Top Intervention", top_intervention.replace("_", " ").title())

    c5, c6 = st.columns([1, 1])
    with c5:
        seg_counts = scores["risk_segment"].value_counts()
        fig = px.pie(
            names=seg_counts.index, values=seg_counts.values, hole=0.45,
            color=seg_counts.index, color_discrete_map=SEGMENT_COLORS,
            title="Risk Segment Split",
        )
        st.plotly_chart(fig, width="stretch")
    with c6:
        st.markdown("#### Expected impact")
        st.metric("Expected Revenue Saved", fmt_money(biz["expected_revenue_saved"]))
        st.metric("Total Intervention Cost", fmt_money(biz["total_intervention_cost"]))
        net = biz["expected_revenue_saved"] - biz["total_intervention_cost"]
        st.metric("Net Benefit", fmt_money(net))

# ---- 2. Risk Distribution -------------------------------------------------- #
with tab2:
    st.subheader("Risk Distribution")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(
            scores, x="churn_probability", nbins=30, title="Churn Probability Distribution",
            color_discrete_sequence=["#1f77b4"],
        )
        fig.add_vline(x=0.7, line_dash="dash", line_color="#d62728", annotation_text="HIGH")
        fig.add_vline(x=0.4, line_dash="dash", line_color="#ff7f0e", annotation_text="MEDIUM")
        st.plotly_chart(fig, width="stretch")
    with c2:
        seg_counts = scores["risk_segment"].value_counts()
        fig = px.bar(
            x=seg_counts.index, y=seg_counts.values, color=seg_counts.index,
            color_discrete_map=SEGMENT_COLORS, title="Customers per Risk Segment",
            labels={"x": "Segment", "y": "Customers"},
        )
        st.plotly_chart(fig, width="stretch")

    trend = pd.DataFrame(monitoring["at_risk_trend"])
    if not trend.empty:
        trend["timestamp"] = pd.to_datetime(trend["timestamp"])
        fig = px.line(
            trend, x="timestamp", y="at_risk_pct", markers=True,
            title="% At-Risk Over Time", labels={"at_risk_pct": "% at risk"},
        )
        st.plotly_chart(fig, width="stretch")

# ---- 3. At-Risk Customers -------------------------------------------------- #
with tab3:
    st.subheader("At-Risk Customers")
    st.caption(f"{len(filtered):,} customers match the current filters.")
    table = filtered.sort_values("churn_probability", ascending=False)[[
        "customer_id", "churn_probability", "risk_segment", "top_factors_str",
        "intervention", "revenue_at_risk", "intervention_cost", "roi", "net_benefit",
    ]].rename(columns={
        "customer_id": "Customer", "churn_probability": "Churn Prob",
        "risk_segment": "Segment", "top_factors_str": "Top Risk Factors (SHAP)",
        "intervention": "Intervention", "revenue_at_risk": "Revenue at Risk",
        "intervention_cost": "Cost", "roi": "ROI", "net_benefit": "Net Benefit",
    })
    st.dataframe(
        table, width="stretch", hide_index=True,
        column_config={
            "Churn Prob": st.column_config.ProgressColumn(
                "Churn Prob", min_value=0.0, max_value=1.0, format="%.2f"),
            "Revenue at Risk": st.column_config.NumberColumn(format="$%.0f"),
            "Cost": st.column_config.NumberColumn(format="$%.0f"),
            "ROI": st.column_config.NumberColumn(format="%.2fx"),
            "Net Benefit": st.column_config.NumberColumn(format="$%.0f"),
        },
    )
    st.download_button(
        "⬇️ Download filtered list (CSV)",
        table.to_csv(index=False).encode("utf-8"),
        "at_risk_customers.csv", "text/csv",
    )

# ---- 4. Explainability ----------------------------------------------------- #
with tab4:
    st.subheader("Explainability (SHAP)")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("#### Global Feature Importance")
        bar_path = config.REPORTS_DIR / "shap_summary_bar.png"
        if bar_path.exists():
            st.image(str(bar_path), width="stretch")
        else:
            st.info("Run `python -m src.explain` to generate SHAP summary plots.")
    with c2:
        st.markdown("#### Why is this customer at risk?")
        high = scores[scores["risk_segment"] == "HIGH"].sort_values(
            "churn_probability", ascending=False)
        pool = high if not high.empty else scores
        cust = st.selectbox("Select a customer", pool["customer_id"].tolist())
        row = scores[scores["customer_id"] == cust].iloc[0]
        st.metric("Churn probability", f"{row['churn_probability']:.0%}")
        st.write(f"**Recommended action:** {row['intervention']}")

        factors = json.loads(row["top_risk_factors"]) if isinstance(row["top_risk_factors"], str) else []
        if factors:
            fdf = pd.DataFrame(factors)
            fig = go.Figure(go.Bar(
                x=fdf["contribution"], y=fdf["factor"], orientation="h",
                marker_color=["#d62728" if c > 0 else "#2ca02c" for c in fdf["contribution"]],
            ))
            fig.update_layout(
                title="SHAP contribution (→ pushes toward churn)",
                xaxis_title="contribution", height=300, yaxis={"autorange": "reversed"},
            )
            st.plotly_chart(fig, width="stretch")

# ---- 5. Model Performance -------------------------------------------------- #
with tab5:
    st.subheader("Model Performance")
    perf = monitoring["model_performance"]
    versions = _mlflow_versions()
    st.selectbox("Model version", versions, index=0,
                 help="Versions registered in the MLflow model registry.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AUC-ROC", f"{perf.get('current_auc', baseline.get('auc', 0)):.3f}")
    c2.metric("Precision", f"{perf.get('precision', baseline.get('precision', 0)):.3f}")
    c3.metric("Recall", f"{perf.get('recall', baseline.get('recall', 0)):.3f}")
    c4.metric("F1", f"{perf.get('f1', baseline.get('f1', 0)):.3f}")

    drift = perf.get("drift_detected", False)
    st.markdown(
        f"**Drift status:** {'🔴 DRIFT DETECTED' if drift else '🟢 Healthy'} "
        f"(AUC drop {perf.get('auc_drop_pct', 0)}% vs baseline {perf.get('baseline_auc', 0)})"
    )
    psi = monitoring["score_drift"]
    st.markdown(
        f"**Score drift (PSI):** {psi.get('psi')} "
        f"({'🔴 drift' if psi.get('drift_detected') else '🟢 stable'}, threshold {psi.get('threshold')})"
    )

    cm = baseline.get("confusion_matrix")
    if cm:
        fig = px.imshow(
            cm, text_auto=True, color_continuous_scale="Blues",
            x=["Pred Retained", "Pred Churned"], y=["Actual Retained", "Actual Churned"],
            title="Confusion Matrix",
        )
        st.plotly_chart(fig, width="stretch")

# ---- 6. Business Metrics --------------------------------------------------- #
with tab6:
    st.subheader("Business Metrics & ROI")
    biz = monitoring["business_metrics"]
    actioned = scores[scores["intervention_code"] != "none"]
    success_rate = st.slider("Assumed intervention success rate", 0.0, 0.5, 0.20, 0.05)

    revenue_saved = actioned["revenue_at_risk"].sum() * success_rate
    total_cost = actioned["intervention_cost"].sum()
    roi = revenue_saved / total_cost if total_cost else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Interventions Recommended", f"{biz['interventions_recommended']:,}")
    c2.metric(f"Revenue Saved @ {success_rate:.0%}", fmt_money(revenue_saved))
    c3.metric("Program ROI", f"{roi:.2f}x")

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Revenue Saved", x=["Cost-Benefit"], y=[revenue_saved],
                         marker_color="#2ca02c"))
    fig.add_trace(go.Bar(name="Intervention Cost", x=["Cost-Benefit"], y=[total_cost],
                         marker_color="#d62728"))
    fig.update_layout(barmode="group", title="Cost vs Benefit", yaxis_title="$")
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### Intervention mix")
    mix = pd.Series(biz["intervention_mix"]).drop("none", errors="ignore").sort_values()
    if not mix.empty:
        fig = px.bar(x=mix.values, y=mix.index, orientation="h",
                     labels={"x": "Customers", "y": "Intervention"})
        st.plotly_chart(fig, width="stretch")

st.caption("Built with Streamlit · XGBoost · SHAP · MLflow · SQLite")
