"""SQLite persistence for real-time scoring history (Phase 3).

Uses SQLAlchemy's ORM so the same code works against SQLite now and a heavier
database (Postgres) later with only a URL change. The ``customer_scores`` table
follows the brief's schema, plus a few columns the dashboard (Phase 5) and
monitoring (Phase 4) need: model version, expected saved/net benefit, the hard
churn prediction, and prediction latency.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import (
    Float,
    Integer,
    String,
    create_engine,
    func,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from . import config


class Base(DeclarativeBase):
    pass


class CustomerScore(Base):
    """One scoring event for one customer at one point in time."""

    __tablename__ = "customer_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(index=True)
    batch_id: Mapped[str] = mapped_column(String, index=True)
    customer_id: Mapped[str] = mapped_column(String, index=True)

    churn_probability: Mapped[float] = mapped_column(Float)
    churn_prediction: Mapped[int] = mapped_column(Integer)        # 0/1 at threshold
    risk_segment: Mapped[str] = mapped_column(String, index=True)
    top_risk_factors: Mapped[str] = mapped_column(String)         # JSON string

    intervention_code: Mapped[str] = mapped_column(String)
    intervention: Mapped[str] = mapped_column(String)
    revenue_at_risk: Mapped[float] = mapped_column(Float)
    intervention_cost: Mapped[float] = mapped_column(Float)
    expected_revenue_saved: Mapped[float] = mapped_column(Float)
    roi: Mapped[float] = mapped_column(Float)
    net_benefit: Mapped[float] = mapped_column(Float)

    model_version: Mapped[str] = mapped_column(String, index=True)
    latency_ms: Mapped[float] = mapped_column(Float)


# --------------------------------------------------------------------------- #
# Engine / session helpers
# --------------------------------------------------------------------------- #
def get_engine(db_path=None, echo: bool = False):
    """Create a SQLAlchemy engine for the SQLite scoring database."""
    db_path = db_path or config.DB_PATH
    config.ensure_dirs()
    return create_engine(f"sqlite:///{db_path}", echo=echo, future=True)


def init_db(engine=None):
    """Create tables if they do not exist. Returns the engine."""
    engine = engine or get_engine()
    Base.metadata.create_all(engine)
    return engine


# --------------------------------------------------------------------------- #
# Writes
# --------------------------------------------------------------------------- #
def _to_row(record: dict) -> CustomerScore:
    """Map a scoring-result dict to a :class:`CustomerScore` row."""
    factors = record.get("top_risk_factors", [])
    if not isinstance(factors, str):
        factors = json.dumps(factors)
    ts = record.get("timestamp") or datetime.now(timezone.utc)
    return CustomerScore(
        timestamp=ts,
        batch_id=record.get("batch_id", "adhoc"),
        customer_id=str(record["customer_id"]),
        churn_probability=float(record["churn_probability"]),
        churn_prediction=int(record.get("churn_prediction", 0)),
        risk_segment=record["risk_segment"],
        top_risk_factors=factors,
        intervention_code=record.get("intervention_code", "none"),
        intervention=record.get("intervention", ""),
        revenue_at_risk=float(record.get("revenue_at_risk", 0.0)),
        intervention_cost=float(record.get("intervention_cost", 0.0)),
        expected_revenue_saved=float(record.get("expected_revenue_saved", 0.0)),
        roi=float(record.get("roi", 0.0)),
        net_benefit=float(record.get("net_benefit", 0.0)),
        model_version=record.get("model_version", config.MODEL_VERSION),
        latency_ms=float(record.get("latency_ms", 0.0)),
    )


def insert_scores(records: list[dict], engine=None) -> int:
    """Bulk-insert scoring results. Returns the number of rows written."""
    engine = engine or init_db()
    with Session(engine) as session:
        rows = [_to_row(r) for r in records]
        session.add_all(rows)
        session.commit()
    return len(records)


# --------------------------------------------------------------------------- #
# Reads
# --------------------------------------------------------------------------- #
def fetch_all(engine=None) -> pd.DataFrame:
    """Return the entire scoring history as a DataFrame."""
    engine = engine or init_db()
    with engine.connect() as conn:
        return pd.read_sql_table("customer_scores", conn)


def fetch_recent(limit: int = 100, engine=None) -> pd.DataFrame:
    """Return the most recent ``limit`` scoring events."""
    engine = engine or init_db()
    stmt = select(CustomerScore).order_by(CustomerScore.timestamp.desc()).limit(limit)
    with engine.connect() as conn:
        return pd.read_sql(stmt, conn)


def summary_stats(engine=None) -> dict:
    """Aggregate business metrics over the whole scoring history."""
    engine = engine or init_db()
    with Session(engine) as session:
        total = session.scalar(select(func.count()).select_from(CustomerScore)) or 0
        if total == 0:
            return {"total_customers": 0}
        at_risk = session.scalar(
            select(func.count()).select_from(CustomerScore).where(
                CustomerScore.risk_segment.in_(["HIGH", "MEDIUM"])
            )
        )
        revenue = session.scalar(select(func.sum(CustomerScore.revenue_at_risk))) or 0.0
        saved = session.scalar(select(func.sum(CustomerScore.expected_revenue_saved))) or 0.0
        cost = session.scalar(select(func.sum(CustomerScore.intervention_cost))) or 0.0
    return {
        "total_customers": int(total),
        "at_risk_count": int(at_risk),
        "at_risk_percentage": round(100.0 * at_risk / total, 2),
        "total_revenue_at_risk": round(float(revenue), 2),
        "expected_revenue_saved": round(float(saved), 2),
        "total_intervention_cost": round(float(cost), 2),
    }


def reset_db(engine=None):
    """Drop and recreate all tables (used by the simulation's --reset)."""
    engine = engine or get_engine()
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    return engine
