"""Pydantic request/response models for the churn API (Phase 6).

Keeping the schemas in their own module keeps ``main.py`` focused on routing and
gives FastAPI clean models to render in the auto-generated ``/docs``.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #
class CustomerData(BaseModel):
    """Raw customer attributes. All optional — missing fields use model defaults.

    Extra keys are allowed so callers can pass the full Telco schema without the
    API rejecting unknown columns.
    """

    model_config = ConfigDict(extra="allow")

    customerID: str | None = Field(default=None, examples=["CUST-009123"])
    tenure: int | None = Field(default=None, ge=0, examples=[5])
    MonthlyCharges: float | None = Field(default=None, ge=0, examples=[95.5])
    TotalCharges: float | None = Field(default=None, ge=0, examples=[480.0])
    Contract: str | None = Field(default=None, examples=["Month-to-month"])
    InternetService: str | None = Field(default=None, examples=["Fiber optic"])
    TechSupport: str | None = Field(default=None, examples=["No"])
    PaymentMethod: str | None = Field(default=None, examples=["Electronic check"])
    support_tickets: int | None = Field(default=None, ge=0, examples=[4])
    usage_trend_3m: float | None = Field(default=None, examples=[-0.25])


class ScoreRequest(BaseModel):
    """Wrapper matching the brief's ``{"customer_data": {...}}`` shape."""

    customer_data: CustomerData


# --------------------------------------------------------------------------- #
# Responses
# --------------------------------------------------------------------------- #
class RiskFactorOut(BaseModel):
    factor: str
    contribution: float


class ScoreResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_risk: str = Field(description="HIGH / MEDIUM / LOW")
    top_risk_factors: list[RiskFactorOut]
    intervention: str
    revenue_at_risk: float
    intervention_cost: float
    expected_revenue_saved: float
    roi: float
    net_benefit: float
    model_version: str
    latency_ms: float


class PerformanceResponse(BaseModel):
    auc: float
    precision: float
    recall: float
    f1: float
    model_version: str
    drift_detected: bool
    last_updated: str


class RiskBucket(BaseModel):
    segment: str
    count: int


class DashboardResponse(BaseModel):
    total_customers: int
    at_risk_count: int
    at_risk_percentage: float
    total_revenue_at_risk: float
    expected_revenue_saved: float
    top_risk_factors: list[str]
    risk_distribution: list[RiskBucket]
    intervention_mix: dict[str, int]
