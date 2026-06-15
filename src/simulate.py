"""Real-time scoring simulation (Phase 3).

Simulates new customer data arriving in hourly batches. For each batch we score
every customer (predict + SHAP + intervention) and append the results to the
SQLite database, so the dashboard and monitoring phases have a realistic
scoring history to read from.

Run::

    python -m src.simulate --batches 10 --batch-size 50 --reset
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

import pandas as pd

from . import config
from .database import fetch_recent, init_db, insert_scores, reset_db, summary_stats
from .generate_data import generate_churn_data
from .inference import ScoringService


def _new_customers(batch_size: int, seed: int, start_id: int) -> pd.DataFrame:
    """Generate a batch of fresh, unlabelled customers with unique IDs."""
    df = generate_churn_data(n=batch_size, seed=seed)
    df = df.drop(columns=[config.TARGET_COL])  # arriving customers are unlabelled
    df[config.ID_COL] = [f"CUST-{start_id + i:07d}" for i in range(batch_size)]
    return df


def run_simulation(n_batches: int = 10, batch_size: int = 50, reset: bool = False) -> dict:
    """Score ``n_batches`` hourly batches and log them to SQLite."""
    engine = reset_db() if reset else init_db()
    service = ScoringService()

    # Backfill timestamps so the last batch lands at "now".
    now = datetime.now(timezone.utc)
    base_time = now - timedelta(hours=n_batches - 1)

    total_scored = 0
    for b in range(n_batches):
        batch_id = f"batch-{b:03d}"
        ts = base_time + timedelta(hours=b)
        customers = _new_customers(batch_size, seed=1000 + b, start_id=100_000 + b * batch_size)

        results = service.score_batch(customers, batch_id=batch_id, timestamp=ts)
        insert_scores(results, engine=engine)
        total_scored += len(results)

        n_high = sum(r["risk_segment"] == "HIGH" for r in results)
        avg_latency = sum(r["latency_ms"] for r in results) / len(results)
        print(
            f"  {batch_id} @ {ts:%Y-%m-%d %H:%M} UTC | "
            f"scored {len(results):>3} | HIGH-risk {n_high:>2} | "
            f"avg latency {avg_latency:5.1f} ms"
        )

    stats = summary_stats(engine=engine)
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print(f"  Total customers scored : {total_scored:,}")
    print(f"  At-risk (HIGH+MEDIUM)  : {stats['at_risk_count']:,} ({stats['at_risk_percentage']}%)")
    print(f"  Total revenue at risk  : {stats['total_revenue_at_risk']:,.0f}")
    print(f"  Expected revenue saved : {stats['expected_revenue_saved']:,.0f}")
    print(f"  Total intervention cost: {stats['total_intervention_cost']:,.0f}")
    print(f"  Database               : {config.DB_PATH}")

    # Persist the first 100 logged rows as a sample artifact.
    sample = fetch_recent(limit=100, engine=engine)
    sample_path = config.REPORTS_DIR / "sample_scores.csv"
    sample.to_csv(sample_path, index=False)
    print(f"  Sample (100 rows)      : {sample_path}")
    print("=" * 70)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate real-time churn scoring.")
    parser.add_argument("--batches", type=int, default=10, help="number of hourly batches")
    parser.add_argument("--batch-size", type=int, default=50, help="customers per batch")
    parser.add_argument("--reset", action="store_true", help="wipe the DB before starting")
    args = parser.parse_args()
    run_simulation(n_batches=args.batches, batch_size=args.batch_size, reset=args.reset)


if __name__ == "__main__":
    main()
