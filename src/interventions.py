"""Intervention recommendation engine.

Turns a churn probability + customer attributes into a concrete, ROI-ranked
retention action. Three pieces, per the brief:

1. **Risk-based segmentation** — HIGH / MEDIUM / LOW, with rule-based action
   selection inside the HIGH band.
2. **Intervention scoring** — revenue at risk, intervention cost, expected
   revenue saved, and ROI.
3. **Priority ranking** — rank customers by expected net benefit so a finite
   retention budget goes to the highest-value saves first.

All monetary values are in the same currency units as ``MonthlyCharges``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from . import config

# --------------------------------------------------------------------------- #
# Economic assumptions (tunable knobs)
# --------------------------------------------------------------------------- #
HORIZON_MONTHS: int = 12          # revenue horizon used for "revenue at risk"
HIGH_RISK_THRESHOLD: float = 0.70
MEDIUM_RISK_THRESHOLD: float = 0.40
HIGH_VALUE_MONTHLY: float = 80.0  # "high-value customer" cut-off
HIGH_TICKETS: int = 3             # support-ticket count considered "high"
NEW_CUSTOMER_TENURE: int = 6      # months — "new customer" cut-off


@dataclass(frozen=True)
class InterventionType:
    """A retention play with its economics."""

    code: str
    label: str
    success_prob: float           # P(retain | intervention applied)
    cost_fn: str                  # how cost is derived ("fixed" or "pct_revenue")
    cost_param: float             # fixed amount, or fraction of revenue-at-risk


# Catalogue of plays. Costs are intentionally simple and transparent.
INTERVENTIONS: dict[str, InterventionType] = {
    "loyalty_discount": InterventionType(
        "loyalty_discount", "High-value customer at risk — offer loyalty discount",
        success_prob=0.35, cost_fn="pct_revenue", cost_param=0.20,
    ),
    "onboarding_support": InterventionType(
        "onboarding_support", "New customer struggling — offer onboarding support",
        success_prob=0.30, cost_fn="fixed", cost_param=50.0,
    ),
    "escalate_cs": InterventionType(
        "escalate_cs", "Escalate to customer success team",
        success_prob=0.28, cost_fn="fixed", cost_param=80.0,
    ),
    "retention_call": InterventionType(
        "retention_call", "Proactive retention call / generic offer",
        success_prob=0.22, cost_fn="fixed", cost_param=25.0,
    ),
    "engagement_nudge": InterventionType(
        "engagement_nudge", "Targeted engagement nudge (email / in-app)",
        success_prob=0.15, cost_fn="fixed", cost_param=10.0,
    ),
    "none": InterventionType(
        "none", "No action — low risk", success_prob=0.0, cost_fn="fixed", cost_param=0.0,
    ),
}


@dataclass
class Recommendation:
    """Scored intervention recommendation for one customer."""

    risk_segment: str
    intervention_code: str
    intervention: str
    revenue_at_risk: float
    intervention_cost: float
    expected_revenue_saved: float
    roi: float
    net_benefit: float

    def as_dict(self) -> dict:
        d = asdict(self)
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items()}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _get(customer, key, default=None):
    """Read a field from a dict or a pandas Series."""
    if isinstance(customer, pd.Series):
        return customer.get(key, default)
    return customer.get(key, default)


def _contract_months(customer) -> int:
    """Resolve committed contract months from either field."""
    if _get(customer, "contract_months") is not None:
        return int(_get(customer, "contract_months"))
    contract = _get(customer, "Contract", "Month-to-month")
    return config.CONTRACT_MONTHS.get(contract, 1)


def risk_segment(churn_probability: float) -> str:
    """Map a probability to HIGH / MEDIUM / LOW."""
    if churn_probability >= HIGH_RISK_THRESHOLD:
        return "HIGH"
    if churn_probability >= MEDIUM_RISK_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def _select_intervention(churn_probability: float, customer) -> InterventionType:
    """Rule-based intervention selection (mirrors the brief's pseudo-code)."""
    segment = risk_segment(churn_probability)

    if segment == "HIGH":
        monthly = float(_get(customer, "MonthlyCharges", 0.0))
        tenure = float(_get(customer, "tenure", 0.0))
        tickets = float(_get(customer, "support_tickets", 0.0))
        usage_trend = float(_get(customer, "usage_trend_3m", 0.0))
        months = _contract_months(customer)

        if monthly > HIGH_VALUE_MONTHLY and months < 12:
            return INTERVENTIONS["loyalty_discount"]
        if usage_trend < 0 and tenure < NEW_CUSTOMER_TENURE:
            return INTERVENTIONS["onboarding_support"]
        if tickets >= HIGH_TICKETS:
            return INTERVENTIONS["escalate_cs"]
        return INTERVENTIONS["retention_call"]

    if segment == "MEDIUM":
        return INTERVENTIONS["engagement_nudge"]
    return INTERVENTIONS["none"]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def recommend(churn_probability: float, customer) -> Recommendation:
    """Score a single customer and return a :class:`Recommendation`.

    Economics
    ---------
    * ``revenue_at_risk``        = MonthlyCharges × HORIZON_MONTHS (full exposure)
    * ``expected_revenue_saved`` = revenue_at_risk × churn_prob × success_prob
    * ``intervention_cost``      = fixed amount, or a % of revenue_at_risk
    * ``roi``                    = expected_revenue_saved / intervention_cost
    * ``net_benefit``            = expected_revenue_saved − intervention_cost
    """
    monthly = float(_get(customer, "MonthlyCharges", 0.0))
    revenue_at_risk = monthly * HORIZON_MONTHS

    play = _select_intervention(churn_probability, customer)

    if play.cost_fn == "pct_revenue":
        cost = play.cost_param * revenue_at_risk
    else:
        cost = play.cost_param

    expected_saved = revenue_at_risk * churn_probability * play.success_prob
    roi = (expected_saved / cost) if cost > 0 else 0.0
    net_benefit = expected_saved - cost

    return Recommendation(
        risk_segment=risk_segment(churn_probability),
        intervention_code=play.code,
        intervention=play.label,
        revenue_at_risk=revenue_at_risk,
        intervention_cost=cost,
        expected_revenue_saved=expected_saved,
        roi=roi,
        net_benefit=net_benefit,
    )


def recommend_batch(df: pd.DataFrame, probabilities) -> pd.DataFrame:
    """Score every row in ``df`` and return a ranked recommendations frame.

    The returned frame is sorted by ``net_benefit`` descending — the priority
    order in which a retention team should work the list.
    """
    records = []
    for (_, row), p in zip(df.iterrows(), probabilities):
        rec = recommend(float(p), row)
        record = {
            config.ID_COL: row.get(config.ID_COL),
            "churn_probability": round(float(p), 4),
            **rec.as_dict(),
        }
        records.append(record)
    out = pd.DataFrame(records)
    out = out.sort_values("net_benefit", ascending=False).reset_index(drop=True)
    out.insert(0, "priority_rank", out.index + 1)
    return out


def main() -> None:
    """Phase 2 intervention demo: score the test set, save sample recommendations."""
    from .explain import load_artifacts, load_processed, transform_features

    config.ensure_dirs()
    model, preprocessor, _ = load_artifacts()
    test_df = load_processed()
    X = transform_features(test_df, preprocessor)
    proba = model.predict_proba(X)[:, 1]

    recs = recommend_batch(test_df, proba)
    seg_counts = recs["risk_segment"].value_counts().to_dict()
    high = recs[recs["risk_segment"] == "HIGH"]

    print("=" * 70)
    print("INTERVENTION ENGINE — test-set summary")
    print("=" * 70)
    print(f"  Customers scored      : {len(recs):,}")
    print(f"  Risk segments         : {seg_counts}")
    print(f"  Total revenue at risk : {recs['revenue_at_risk'].sum():,.0f}")
    print(f"  HIGH-risk exposure    : {high['revenue_at_risk'].sum():,.0f}")
    print(f"  Expected revenue saved: {recs['expected_revenue_saved'].sum():,.0f}")
    print(f"  Total intervention cost: {recs['intervention_cost'].sum():,.0f}")
    print("\n  Intervention mix:")
    for code, cnt in recs["intervention_code"].value_counts().items():
        print(f"    {code:<20} {cnt}")
    print("\n  Top 5 priority customers:")
    cols = ["priority_rank", config.ID_COL, "churn_probability", "intervention",
            "revenue_at_risk", "intervention_cost", "roi", "net_benefit"]
    print(recs[cols].head(5).to_string(index=False))
    print("=" * 70)

    out_path = config.REPORTS_DIR / "intervention_recommendations.json"
    recs.head(100).to_json(out_path, orient="records", indent=2)
    print(f"[interventions] Saved top-100 recommendations -> {out_path}")


if __name__ == "__main__":
    main()
