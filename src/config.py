"""Central configuration: paths, schema definitions, and modelling constants.

Keeping these in one place means every phase (data prep, training, inference,
API, dashboard) reads from a single source of truth instead of hard-coding
column names and file locations.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Project root = the directory that contains this `src/` package.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

# Canonical file locations used across phases.
RAW_DATA_PATH: Path = RAW_DATA_DIR / "telco_churn.csv"
DB_PATH: Path = DATA_DIR / "churn_scores.db"
PROCESSED_TRAIN_PATH: Path = PROCESSED_DATA_DIR / "train.csv"
PROCESSED_TEST_PATH: Path = PROCESSED_DATA_DIR / "test.csv"
PREPROCESSOR_PATH: Path = MODELS_DIR / "preprocessing.pkl"
MODEL_PATH: Path = MODELS_DIR / "model.pkl"
METRICS_PATH: Path = REPORTS_DIR / "baseline_metrics.json"

# --------------------------------------------------------------------------- #
# Schema — modelled on the Kaggle Telco Customer Churn dataset, plus a few
# engineered/usage columns the synthetic generator adds so later phases
# (interventions, lag features) have something realistic to work with.
# --------------------------------------------------------------------------- #
TARGET_COL: str = "Churn"
ID_COL: str = "customerID"

# Numerical features present in the raw data.
NUMERIC_FEATURES: list[str] = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "support_tickets",      # synthetic extension
    "usage_trend_3m",       # synthetic extension (lag / trend signal)
]

# Categorical features (one-hot encoded downstream).
CATEGORICAL_FEATURES: list[str] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

# Engineered interaction/lag features created in feature engineering.
ENGINEERED_FEATURES: list[str] = [
    "charges_per_tenure",        # TotalCharges / tenure  (spend velocity)
    "contract_x_charges",        # contract length (months) * MonthlyCharges
    "tenure_x_charges",          # tenure * MonthlyCharges (lifetime value proxy)
    "services_count",            # number of add-on services subscribed
]

# Map contract type -> minimum committed months (used for interactions and the
# Phase 2 intervention rules, e.g. "contract_months < 12").
CONTRACT_MONTHS: dict[str, int] = {
    "Month-to-month": 1,
    "One year": 12,
    "Two year": 24,
}

# --------------------------------------------------------------------------- #
# Modelling constants
# --------------------------------------------------------------------------- #
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20

# Model version string surfaced by the API / dashboard / score logs.
# Phase 4 (MLflow) will source this from the model registry instead.
MODEL_VERSION: str = "v1.0"

# --------------------------------------------------------------------------- #
# MLflow (Phase 4) — SQLite-backed tracking so the model registry works locally
# --------------------------------------------------------------------------- #
MLFLOW_DB_PATH: Path = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI: str = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
MLFLOW_ARTIFACT_DIR: Path = PROJECT_ROOT / "mlartifacts"
MLFLOW_EXPERIMENT: str = "churn_intelligence"
REGISTERED_MODEL_NAME: str = "churn_xgboost"
PRODUCTION_ALIAS: str = "production"

# Monitoring thresholds.
DRIFT_AUC_DROP: float = 0.05   # flag if AUC drops >5% vs the registered baseline
PSI_THRESHOLD: float = 0.20    # population stability index drift cut-off
MONITORING_DATA_PATH: Path = REPORTS_DIR / "monitoring_data.json"

# Target performance bar from the project brief.
TARGET_AUC: float = 0.85
TARGET_PRECISION: float = 0.75


def ensure_dirs() -> None:
    """Create all output directories if they do not already exist."""
    for d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
