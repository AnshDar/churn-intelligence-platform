"""Data preparation pipeline: load -> EDA -> feature engineering -> split.

Produces everything Phase 1's baseline model needs:

* a cleaned, feature-engineered dataset (written to ``data/processed/``)
* a fitted scikit-learn preprocessor (``models/preprocessing.pkl``)
* train/test matrices with human-readable feature names (for SHAP later)

The module is import-safe and also runnable as a script::

    python -m src.data_prep
"""
from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


# --------------------------------------------------------------------------- #
# Container for the prepared data
# --------------------------------------------------------------------------- #
@dataclass
class PreparedData:
    """Bundle returned by :func:`prepare_data`."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    preprocessor: ColumnTransformer
    train_df: pd.DataFrame          # engineered, pre-transform (incl. id + target)
    test_df: pd.DataFrame


# --------------------------------------------------------------------------- #
# Load & clean
# --------------------------------------------------------------------------- #
def load_raw_data(path=None) -> pd.DataFrame:
    """Load the raw churn CSV, cleaning the known Telco quirks.

    If the file is missing, a synthetic dataset is generated on the fly so the
    pipeline always runs end-to-end.
    """
    path = path or config.RAW_DATA_PATH
    if not path.exists():
        print(f"[data_prep] {path} not found — generating synthetic dataset.")
        from .generate_data import generate_churn_data

        config.ensure_dirs()
        df = generate_churn_data()
        df.to_csv(path, index=False)
    else:
        df = pd.read_csv(path)

    # Telco's TotalCharges has blank strings for brand-new customers.
    # (Checks cover both legacy `object` and pandas' newer `string` dtype.)
    if not pd.api.types.is_numeric_dtype(df["TotalCharges"]):
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Normalise the target to 0/1 if it arrives as Yes/No.
    if not pd.api.types.is_numeric_dtype(df[config.TARGET_COL]):
        df[config.TARGET_COL] = (
            df[config.TARGET_COL].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
        )

    return df


# --------------------------------------------------------------------------- #
# Exploratory Data Analysis
# --------------------------------------------------------------------------- #
def run_eda(df: pd.DataFrame) -> dict:
    """Print a compact EDA report and return key stats as a dict.

    Covers the brief's requirements: missing values, class imbalance, and
    feature distributions.
    """
    n_rows, n_cols = df.shape
    missing = df.isna().sum()
    missing = missing[missing > 0]

    churn_counts = df[config.TARGET_COL].value_counts().sort_index()
    churn_rate = df[config.TARGET_COL].mean()

    print("=" * 70)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 70)
    print(f"Shape: {n_rows:,} rows x {n_cols} columns")
    print("\n--- Missing values ---")
    if missing.empty:
        print("None")
    else:
        for col, cnt in missing.items():
            print(f"  {col}: {cnt} ({cnt / n_rows:.1%})")

    print("\n--- Class balance (target = Churn) ---")
    print(f"  Retained (0): {churn_counts.get(0, 0):,}")
    print(f"  Churned  (1): {churn_counts.get(1, 0):,}")
    print(f"  Churn rate  : {churn_rate:.1%}")
    if churn_rate < 0.35:
        print("  -> Imbalanced; will use scale_pos_weight in XGBoost.")

    print("\n--- Numeric feature summary ---")
    numeric_present = [c for c in config.NUMERIC_FEATURES if c in df.columns]
    with pd.option_context("display.width", 120):
        print(df[numeric_present].describe().round(2).T)
    print("=" * 70)

    return {
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "missing": {k: int(v) for k, v in missing.items()},
        "churn_rate": float(churn_rate),
    }


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction & lag-style features described in the brief.

    Creates:
      * ``charges_per_tenure``  — spend velocity (TotalCharges / tenure)
      * ``contract_x_charges``  — contract commitment months * MonthlyCharges
      * ``tenure_x_charges``    — tenure * MonthlyCharges (lifetime-value proxy)
      * ``services_count``      — number of add-on services subscribed
    """
    df = df.copy()

    # Impute the handful of missing TotalCharges (new customers) with 0 spend.
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Spend velocity — guard against divide-by-zero for 0-tenure customers.
    df["charges_per_tenure"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

    # Contract commitment in months, then the interaction with monthly spend.
    contract_months = df["Contract"].map(config.CONTRACT_MONTHS).fillna(1)
    df["contract_x_charges"] = contract_months * df["MonthlyCharges"]
    df["tenure_x_charges"] = df["tenure"] * df["MonthlyCharges"]

    # Count subscribed add-on services ("Yes" across the service columns).
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    present = [c for c in service_cols if c in df.columns]
    df["services_count"] = (df[present] == "Yes").sum(axis=1)

    return df


def build_preprocessor() -> ColumnTransformer:
    """Build the column transformer: scale numerics, one-hot encode categoricals."""
    numeric_features = config.NUMERIC_FEATURES + config.ENGINEERED_FEATURES
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                config.CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def feature_names_from(preprocessor: ColumnTransformer) -> list[str]:
    """Recover output feature names from a fitted ColumnTransformer.

    The order matches the transformed matrix's columns: scaled numerics first,
    then the one-hot encoded categorical columns. Used by SHAP (Phase 2) and the
    API (Phase 6) to label contributions.
    """
    numeric_features = config.NUMERIC_FEATURES + config.ENGINEERED_FEATURES
    ohe: OneHotEncoder = preprocessor.named_transformers_["cat"]
    cat_names = ohe.get_feature_names_out(config.CATEGORICAL_FEATURES).tolist()
    return numeric_features + cat_names


# Backwards-compatible private alias.
_feature_names = feature_names_from


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def prepare_data(save: bool = True) -> PreparedData:
    """Run the full prep pipeline and return a :class:`PreparedData` bundle."""
    config.ensure_dirs()

    df = load_raw_data()
    run_eda(df)
    df = engineer_features(df)

    feature_cols = (
        config.NUMERIC_FEATURES + config.ENGINEERED_FEATURES + config.CATEGORICAL_FEATURES
    )
    X = df[feature_cols]
    y = df[config.TARGET_COL].astype(int)

    # Stratified 80/20 split to preserve the churn ratio in both folds.
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)
    feature_names = _feature_names(preprocessor)

    # Re-attach id + target for the human-readable processed CSVs.
    train_df = df.loc[X_train_df.index].copy()
    test_df = df.loc[X_test_df.index].copy()

    if save:
        train_df.to_csv(config.PROCESSED_TRAIN_PATH, index=False)
        test_df.to_csv(config.PROCESSED_TEST_PATH, index=False)
        joblib.dump(preprocessor, config.PREPROCESSOR_PATH)
        print(f"\n[data_prep] Saved processed train -> {config.PROCESSED_TRAIN_PATH}")
        print(f"[data_prep] Saved processed test  -> {config.PROCESSED_TEST_PATH}")
        print(f"[data_prep] Saved preprocessor    -> {config.PREPROCESSOR_PATH}")

    print(
        f"[data_prep] Train: {X_train.shape}, Test: {X_test.shape}, "
        f"features after encoding: {len(feature_names)}"
    )

    return PreparedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train.to_numpy(),
        y_test=y_test.to_numpy(),
        feature_names=feature_names,
        preprocessor=preprocessor,
        train_df=train_df,
        test_df=test_df,
    )


if __name__ == "__main__":
    prepare_data()
