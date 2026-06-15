"""MLflow experiment tracking & model registry (Phase 4).

Logs the XGBoost training run (params, metrics, the model artifact, and a data
snapshot) to a local SQLite-backed MLflow store, registers the model, and tags
the best version with the ``production`` alias. Inference can then load the
production model straight from the registry.

A SQLite tracking backend is used (rather than the default file store) because
the model *registry* requires a database backend.

Run::

    python -m src.mlflow_tracking          # log existing model.pkl + register
    python -m src.mlflow_tracking --train  # retrain from scratch, then log
"""
from __future__ import annotations

import argparse
import json

import joblib
import pandas as pd

from . import config


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
def setup_mlflow():
    """Point MLflow at the local SQLite store and ensure the experiment exists."""
    import mlflow

    config.ensure_dirs()
    config.MLFLOW_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    exp = mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT)
    if exp is None:
        mlflow.create_experiment(
            config.MLFLOW_EXPERIMENT,
            artifact_location=config.MLFLOW_ARTIFACT_DIR.as_uri(),
        )
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    return mlflow


# --------------------------------------------------------------------------- #
# Data snapshot
# --------------------------------------------------------------------------- #
def _data_snapshot() -> dict:
    """Lightweight dataset lineage info logged alongside the run."""
    snap: dict = {}
    try:
        train = pd.read_csv(config.PROCESSED_TRAIN_PATH)
        test = pd.read_csv(config.PROCESSED_TEST_PATH)
        snap = {
            "train_rows": len(train),
            "test_rows": len(test),
            "train_churn_rate": round(float(train[config.TARGET_COL].mean()), 4),
            "test_churn_rate": round(float(test[config.TARGET_COL].mean()), 4),
            "n_features_raw": len(
                config.NUMERIC_FEATURES + config.ENGINEERED_FEATURES + config.CATEGORICAL_FEATURES
            ),
        }
    except FileNotFoundError:
        pass
    return snap


_LOGGED_PARAMS = [
    "n_estimators", "max_depth", "learning_rate", "subsample",
    "colsample_bytree", "min_child_weight", "gamma", "scale_pos_weight",
]


# --------------------------------------------------------------------------- #
# Logging + registration
# --------------------------------------------------------------------------- #
def log_run(model, metrics: dict, register: bool = True, promote: bool = True) -> dict:
    """Log a training run to MLflow and (optionally) register + promote it.

    Returns a dict with ``run_id`` and, if registered, ``model_version``.
    """
    mlflow = setup_mlflow()

    params = {k: v for k, v in model.get_params().items() if k in _LOGGED_PARAMS}
    snapshot = _data_snapshot()
    metric_keys = ["accuracy", "precision", "recall", "f1", "auc", "threshold"]

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.log_params(params)
        if snapshot:
            mlflow.log_params({f"data_{k}": v for k, v in snapshot.items()})
        mlflow.log_metrics({k: float(metrics[k]) for k in metric_keys if k in metrics})
        mlflow.set_tags({
            "model_type": "XGBoost",
            "phase": "baseline",
            "target_auc": config.TARGET_AUC,
        })

        # Log a couple of human-readable artifacts if present.
        for artifact in (config.METRICS_PATH,
                         config.REPORTS_DIR / "roc_curve.png",
                         config.REPORTS_DIR / "confusion_matrix.png"):
            if artifact.exists():
                mlflow.log_artifact(str(artifact))

        # Log (and optionally register) the model.
        reg_name = config.REGISTERED_MODEL_NAME if register else None
        try:
            mlflow.sklearn.log_model(model, name="model", registered_model_name=reg_name)
        except TypeError:
            # Older MLflow uses artifact_path instead of name.
            mlflow.sklearn.log_model(model, artifact_path="model",
                                     registered_model_name=reg_name)

    result = {"run_id": run_id}
    print(f"[mlflow] Logged run {run_id} (AUC={metrics.get('auc'):.4f})")

    if register:
        version = _latest_version(mlflow)
        result["model_version"] = version
        print(f"[mlflow] Registered '{config.REGISTERED_MODEL_NAME}' version {version}")
        if promote:
            promote_to_production(version)
            result["alias"] = config.PRODUCTION_ALIAS
    return result


def _latest_version(mlflow) -> str:
    """Return the highest version number for the registered model."""
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{config.REGISTERED_MODEL_NAME}'")
    return str(max(int(v.version) for v in versions))


def promote_to_production(version: str) -> None:
    """Tag a model version with the ``production`` alias (and a stage tag)."""
    import mlflow

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(
        config.REGISTERED_MODEL_NAME, config.PRODUCTION_ALIAS, version
    )
    client.set_model_version_tag(
        config.REGISTERED_MODEL_NAME, version, "stage", "production"
    )
    print(f"[mlflow] '{config.REGISTERED_MODEL_NAME}' v{version} -> @{config.PRODUCTION_ALIAS}")


# --------------------------------------------------------------------------- #
# Loading from the registry
# --------------------------------------------------------------------------- #
def get_production_version() -> str | None:
    """Return the version number currently aliased ``production`` (or None)."""
    import mlflow

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()
    try:
        mv = client.get_model_version_by_alias(
            config.REGISTERED_MODEL_NAME, config.PRODUCTION_ALIAS
        )
        return mv.version
    except Exception:
        return None


def load_production_model():
    """Load the production-aliased model from the MLflow registry."""
    import mlflow

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    uri = f"models:/{config.REGISTERED_MODEL_NAME}@{config.PRODUCTION_ALIAS}"
    return mlflow.sklearn.load_model(uri)


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
def log_existing() -> dict:
    """Load the Phase 1 model.pkl + metrics and log/register them."""
    model = joblib.load(config.MODEL_PATH)
    with open(config.METRICS_PATH, encoding="utf-8") as fh:
        metrics = json.load(fh)
    return log_run(model, metrics, register=True, promote=True)


def train_and_log() -> dict:
    """Retrain from scratch (full pipeline) and log/register the result."""
    from .model import run_pipeline

    model, metrics = run_pipeline(tune=True)
    return log_run(model, metrics, register=True, promote=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="MLflow tracking for the churn model.")
    parser.add_argument("--train", action="store_true",
                        help="retrain from scratch instead of logging the saved model")
    args = parser.parse_args()

    result = train_and_log() if args.train else log_existing()

    prod = get_production_version()
    print("\n" + "=" * 70)
    print("MLFLOW SUMMARY")
    print("=" * 70)
    print(f"  Tracking URI      : {config.MLFLOW_TRACKING_URI}")
    print(f"  Experiment        : {config.MLFLOW_EXPERIMENT}")
    print(f"  Registered model  : {config.REGISTERED_MODEL_NAME}")
    print(f"  Run id            : {result['run_id']}")
    print(f"  Production version : {prod}")
    print(f"\n  View UI with: mlflow ui --backend-store-uri {config.MLFLOW_TRACKING_URI}")
    print("=" * 70)


if __name__ == "__main__":
    main()
