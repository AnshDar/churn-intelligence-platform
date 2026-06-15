"""Baseline model: tuned XGBoost classifier with full evaluation.

Run as a script to execute the whole Phase 1 flow (data prep -> tune -> train
-> evaluate -> save artifacts)::

    python -m src.model

Outputs:
  * ``models/model.pkl``            — trained XGBoost classifier
  * ``reports/baseline_metrics.json`` — accuracy/precision/recall/f1/auc + CM
  * ``reports/roc_curve.png`` & ``reports/confusion_matrix.png`` (if matplotlib)
"""
from __future__ import annotations

import json

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from . import config
from .data_prep import PreparedData, prepare_data


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def best_threshold(model: XGBClassifier, X: np.ndarray, y: np.ndarray) -> float:
    """Pick the probability threshold that maximises F1 on the given data.

    Fitted on the *training* set (never the test set) to avoid leakage.
    """
    proba = model.predict_proba(X)[:, 1]
    grid = np.linspace(0.1, 0.9, 81)
    f1s = [f1_score(y, (proba >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def train_model(data: PreparedData, tune: bool = True) -> XGBClassifier:
    """Train (optionally tuning) an XGBoost classifier on the prepared data.

    Note: we leave ``scale_pos_weight`` at 1.0 rather than the negatives/positives
    ratio. Aggressive reweighting inflates recall but tanks precision; instead we
    tune the *decision threshold* afterwards (:func:`best_threshold`) to balance
    precision and recall against the brief's targets.
    """
    base = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=1.0,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )

    if not tune:
        base.set_params(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.9)
        base.fit(data.X_train, data.y_train)
        return base

    # Randomized search over a sensible XGBoost grid, optimising AUC.
    param_dist = {
        "n_estimators": [200, 300, 400, 600],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.3],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=30,
        scoring="roc_auc",
        cv=cv,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    print("\n[model] Running randomized hyperparameter search (30 candidates x 5 folds)...")
    search.fit(data.X_train, data.y_train)
    print(f"[model] Best CV AUC: {search.best_score_:.4f}")
    print(f"[model] Best params: {search.best_params_}")
    return search.best_estimator_


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate_model(model: XGBClassifier, data: PreparedData, threshold: float = 0.5) -> dict:
    """Evaluate on the held-out test set and return a metrics dict."""
    y_true = data.y_test
    y_proba = model.predict_proba(data.X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_proba)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "threshold": float(threshold),
    }

    print("\n" + "=" * 70)
    print("TEST-SET EVALUATION")
    print("=" * 70)
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}  (target > {config.TARGET_PRECISION})")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1       : {metrics['f1']:.4f}")
    print(f"  AUC-ROC  : {metrics['auc']:.4f}  (target > {config.TARGET_AUC})")
    print(f"  Threshold: {threshold:.3f}  (F1-optimal, tuned on train)")
    print("\n  Confusion matrix [[TN, FP], [FN, TP]]:")
    print(f"    {metrics['confusion_matrix']}")
    print("\n" + classification_report(y_true, y_pred, target_names=["Retained", "Churned"]))

    auc_ok = metrics["auc"] > config.TARGET_AUC
    prec_ok = metrics["precision"] > config.TARGET_PRECISION
    print(f"  Targets met -> AUC: {'PASS' if auc_ok else 'below target'}, "
          f"Precision: {'PASS' if prec_ok else 'below target'}")
    print("=" * 70)

    return metrics


def top_feature_importance(model: XGBClassifier, feature_names: list[str], k: int = 15) -> list[tuple[str, float]]:
    """Return the top-``k`` features by XGBoost gain importance."""
    importances = model.feature_importances_
    ranked = sorted(zip(feature_names, importances), key=lambda t: t[1], reverse=True)
    print("\n--- Top feature importances ---")
    for name, imp in ranked[:k]:
        print(f"  {name:<35} {imp:.4f}")
    return [(n, float(i)) for n, i in ranked[:k]]


# --------------------------------------------------------------------------- #
# Plots (optional — only if matplotlib is available)
# --------------------------------------------------------------------------- #
def save_plots(model: XGBClassifier, data: PreparedData, metrics: dict) -> None:
    """Save ROC curve and confusion-matrix images to ``reports/``."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[model] matplotlib not installed — skipping plots.")
        return

    y_proba = model.predict_proba(data.X_test)[:, 1]
    fpr, tpr, _ = roc_curve(data.y_test, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {metrics['auc']:.3f}", color="#1f77b4")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Churn Model")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "roc_curve.png", dpi=120)
    plt.close(fig)

    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Retained", "Churned"])
    ax.set_yticks([0, 1], labels=["Retained", "Churned"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "confusion_matrix.png", dpi=120)
    plt.close(fig)
    print(f"[model] Saved ROC + confusion-matrix plots -> {config.REPORTS_DIR}")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_pipeline(tune: bool = True) -> tuple[XGBClassifier, dict]:
    """Full Phase 1 flow: prepare data, train, evaluate, and persist artifacts."""
    data = prepare_data(save=True)
    model = train_model(data, tune=tune)
    threshold = best_threshold(model, data.X_train, data.y_train)
    metrics = evaluate_model(model, data, threshold=threshold)
    metrics["top_features"] = top_feature_importance(model, data.feature_names)
    metrics["feature_names"] = data.feature_names
    save_plots(model, data, metrics)

    config.ensure_dirs()
    joblib.dump(model, config.MODEL_PATH)
    with open(config.METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\n[model] Saved model   -> {config.MODEL_PATH}")
    print(f"[model] Saved metrics -> {config.METRICS_PATH}")
    return model, metrics


if __name__ == "__main__":
    run_pipeline(tune=True)
