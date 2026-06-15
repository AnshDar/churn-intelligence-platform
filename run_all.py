"""One-shot orchestrator: run the entire platform pipeline end to end.

Executes every phase in order so a fresh clone goes from nothing to a fully
populated model registry, scoring database, and dashboard data in a single
command::

    python run_all.py            # full run (with hyperparameter tuning)
    python run_all.py --fast     # skip tuning (quick smoke run)

After it finishes:
    streamlit run dashboard/app.py     # dashboard
    uvicorn api.main:app --reload      # API at /docs
"""
from __future__ import annotations

import argparse
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full churn platform pipeline.")
    parser.add_argument("--fast", action="store_true", help="skip XGBoost hyperparameter tuning")
    parser.add_argument("--batches", type=int, default=10, help="simulation batches")
    parser.add_argument("--batch-size", type=int, default=50, help="customers per batch")
    parser.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    args = parser.parse_args()

    start = time.perf_counter()

    def step(n: int, title: str) -> None:
        print(f"\n{'=' * 72}\n[{n}/7] {title}\n{'=' * 72}")

    # 1. Data + baseline model
    step(1, "Data preparation & baseline model")
    from src.model import run_pipeline
    run_pipeline(tune=not args.fast)

    # 2. Explainability
    step(2, "SHAP explainability artifacts")
    from src.explain import main as explain_main
    explain_main()

    # 3. Interventions (sample recommendations)
    step(3, "Intervention recommendations")
    from src.interventions import main as interventions_main
    interventions_main()

    # 4. Real-time scoring simulation -> SQLite
    step(4, "Real-time scoring simulation")
    from src.simulate import run_simulation
    run_simulation(n_batches=args.batches, batch_size=args.batch_size, reset=True)

    # 5. MLflow tracking + registry
    if not args.no_mlflow:
        step(5, "MLflow logging & model registry")
        try:
            from src.mlflow_tracking import log_existing
            log_existing()
        except Exception as exc:  # pragma: no cover
            print(f"[run_all] MLflow step skipped: {exc}")
    else:
        step(5, "MLflow logging & model registry (skipped)")

    # 6. Monitoring snapshot
    step(6, "Monitoring snapshot")
    from src.monitoring import build_monitoring_data
    build_monitoring_data(save=True)

    # 7. Done
    step(7, "Complete")
    elapsed = time.perf_counter() - start
    print(f"Pipeline finished in {elapsed:.1f}s.")
    print("\nNext:")
    print("  streamlit run dashboard/app.py     # dashboard")
    print("  uvicorn api.main:app --reload      # API docs at /docs")


if __name__ == "__main__":
    main()
