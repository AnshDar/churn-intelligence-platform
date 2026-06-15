"""SHAP explainability layer for the churn model.

Wraps a trained XGBoost model in a SHAP ``TreeExplainer`` and provides:

* global feature importance (bar + beeswarm summary plots)
* local explanations ("top 3 reasons this customer is at risk")
* a saved interactive force plot (HTML) for a single customer

Designed to be reused by the dashboard (Phase 5) and the API (Phase 6), so the
core ``ChurnExplainer`` works on a single preprocessed row and returns plain
dicts rather than rendering anything.
"""
from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from . import config
from .data_prep import feature_names_from


# --------------------------------------------------------------------------- #
# Human-readable feature labels
# --------------------------------------------------------------------------- #
_PRETTY_NUMERIC = {
    "tenure": "Tenure (months)",
    "MonthlyCharges": "Monthly charges",
    "TotalCharges": "Total charges",
    "support_tickets": "Support tickets",
    "usage_trend_3m": "3-month usage trend",
    "charges_per_tenure": "Spend velocity",
    "contract_x_charges": "Contract × charges",
    "tenure_x_charges": "Tenure × charges",
    "services_count": "Add-on services",
}


def prettify(feature: str) -> str:
    """Turn an encoded feature name into a readable label.

    ``Contract_Month-to-month`` -> ``Contract: Month-to-month``;
    ``MonthlyCharges`` -> ``Monthly charges``.
    """
    if feature in _PRETTY_NUMERIC:
        return _PRETTY_NUMERIC[feature]
    if "_" in feature:
        col, _, value = feature.partition("_")
        return f"{col}: {value}"
    return feature


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class RiskFactor:
    """A single SHAP contribution toward (or against) churn."""

    factor: str           # readable label
    raw_feature: str      # encoded feature name
    contribution: float   # signed SHAP value (positive => pushes toward churn)

    def as_dict(self) -> dict:
        return {
            "factor": self.factor,
            "raw_feature": self.raw_feature,
            "contribution": round(self.contribution, 4),
        }


# --------------------------------------------------------------------------- #
# Explainer
# --------------------------------------------------------------------------- #
class ChurnExplainer:
    """SHAP TreeExplainer wrapper around the trained XGBoost model."""

    def __init__(self, model, feature_names: list[str]):
        import shap

        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)

    # -- core -------------------------------------------------------------- #
    def shap_values(self, X: np.ndarray) -> np.ndarray:
        """Return a ``(n_samples, n_features)`` SHAP-value matrix."""
        values = self.explainer.shap_values(X)
        # Some SHAP/XGBoost combos return a list (per class); take positive class.
        if isinstance(values, list):
            values = values[1]
        return np.asarray(values)

    def top_reasons(self, x_row: np.ndarray, k: int = 3, toward_churn: bool = True) -> list[RiskFactor]:
        """Top-``k`` SHAP contributions for one preprocessed row.

        ``toward_churn=True`` ranks the factors *increasing* churn risk; set
        ``False`` to surface the strongest protective factors.
        """
        row = np.asarray(x_row).reshape(1, -1)
        contribs = self.shap_values(row)[0]
        order = np.argsort(contribs)[::-1] if toward_churn else np.argsort(contribs)
        factors: list[RiskFactor] = []
        for idx in order[:k]:
            name = self.feature_names[idx]
            factors.append(
                RiskFactor(factor=prettify(name), raw_feature=name, contribution=float(contribs[idx]))
            )
        return factors

    # -- global plots ------------------------------------------------------ #
    def save_summary_plots(self, X: np.ndarray, out_dir=None) -> dict:
        """Save bar + beeswarm SHAP summary plots; return their paths."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap

        out_dir = out_dir or config.REPORTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        X_df = pd.DataFrame(X, columns=self.feature_names)
        sv = self.shap_values(X)
        pretty_cols = [prettify(c) for c in self.feature_names]
        X_pretty = X_df.copy()
        X_pretty.columns = pretty_cols

        bar_path = out_dir / "shap_summary_bar.png"
        beeswarm_path = out_dir / "shap_summary_beeswarm.png"

        plt.figure()
        shap.summary_plot(sv, X_pretty, plot_type="bar", show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(bar_path, dpi=120, bbox_inches="tight")
        plt.close()

        plt.figure()
        shap.summary_plot(sv, X_pretty, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(beeswarm_path, dpi=120, bbox_inches="tight")
        plt.close()

        return {"bar": bar_path, "beeswarm": beeswarm_path}

    def save_force_plot(self, x_row: np.ndarray, out_path=None):
        """Save an interactive SHAP force plot (HTML) for one customer."""
        import shap

        out_path = out_path or (config.REPORTS_DIR / "force_plot.html")
        row = np.asarray(x_row).reshape(1, -1)
        sv = self.shap_values(row)
        base = self.explainer.expected_value
        if isinstance(base, (list, np.ndarray)):
            base = float(np.asarray(base).ravel()[-1])
        fp = shap.force_plot(
            base, sv[0], pd.DataFrame(row, columns=[prettify(c) for c in self.feature_names]).iloc[0],
        )
        shap.save_html(str(out_path), fp)
        return out_path

    # -- narrative --------------------------------------------------------- #
    def narrative(self, x_row: np.ndarray, churn_probability: float, k: int = 3) -> str:
        """Plain-English 'why is this customer at risk?' sentence."""
        reasons = self.top_reasons(x_row, k=k, toward_churn=True)
        bullet = "; ".join(r.factor for r in reasons)
        return (
            f"Estimated churn probability {churn_probability:.0%}. "
            f"Top drivers increasing risk: {bullet}."
        )


# --------------------------------------------------------------------------- #
# Convenience loaders
# --------------------------------------------------------------------------- #
def load_artifacts():
    """Load the trained model and fitted preprocessor from ``models/``."""
    model = joblib.load(config.MODEL_PATH)
    preprocessor = joblib.load(config.PREPROCESSOR_PATH)
    feature_names = feature_names_from(preprocessor)
    return model, preprocessor, feature_names


def load_processed(path=None) -> pd.DataFrame:
    """Load a processed (feature-engineered) CSV produced in Phase 1."""
    path = path or config.PROCESSED_TEST_PATH
    return pd.read_csv(path)


def transform_features(df: pd.DataFrame, preprocessor) -> np.ndarray:
    """Apply the fitted preprocessor to an engineered dataframe."""
    feature_cols = (
        config.NUMERIC_FEATURES + config.ENGINEERED_FEATURES + config.CATEGORICAL_FEATURES
    )
    return preprocessor.transform(df[feature_cols])


def build_explainer() -> tuple[ChurnExplainer, pd.DataFrame, np.ndarray]:
    """Load everything needed for explanations: explainer + test data + matrix."""
    model, preprocessor, feature_names = load_artifacts()
    test_df = load_processed()
    X_test = transform_features(test_df, preprocessor)
    explainer = ChurnExplainer(model, feature_names)
    return explainer, test_df, X_test


def main() -> None:
    """Generate the Phase 2 explainability artifacts."""
    config.ensure_dirs()
    explainer, test_df, X_test = build_explainer()

    paths = explainer.save_summary_plots(X_test)
    print(f"[explain] SHAP summary (bar)      -> {paths['bar']}")
    print(f"[explain] SHAP summary (beeswarm) -> {paths['beeswarm']}")

    # Force plot + narrative for the single highest-risk customer.
    proba = explainer.model.predict_proba(X_test)[:, 1]
    idx = int(np.argmax(proba))
    fp_path = explainer.save_force_plot(X_test[idx])
    print(f"[explain] Force plot              -> {fp_path}")
    print("\n[explain] Highest-risk customer:")
    print(f"  customerID: {test_df.iloc[idx][config.ID_COL]}")
    print(f"  {explainer.narrative(X_test[idx], float(proba[idx]))}")
    for r in explainer.top_reasons(X_test[idx]):
        print(f"    - {r.factor:<28} contribution={r.contribution:+.3f}")


if __name__ == "__main__":
    main()
