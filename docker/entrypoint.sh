#!/usr/bin/env bash
# Ensure model artifacts and a scoring history exist, then run the given command.
# Artifacts are normally baked at build time; this is a safety net for the case
# where /app/models or /app/data are mounted as empty volumes.
set -e

if [ ! -f models/model.pkl ]; then
    echo "[entrypoint] No model found — training (fast, no tuning)…"
    python -m src.generate_data
    python -c "from src.model import run_pipeline; run_pipeline(tune=False)"
    python -m src.explain || true
fi

if [ ! -f data/churn_scores.db ]; then
    echo "[entrypoint] No scoring history — seeding via simulation…"
    python -m src.simulate --batches 5 --batch-size 30 --reset || true
fi

echo "[entrypoint] Starting: $*"
exec "$@"
