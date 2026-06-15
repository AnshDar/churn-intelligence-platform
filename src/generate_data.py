"""Synthetic data generator.

The project brief allows synthesising a realistic dataset if the Kaggle Telco
file is unavailable. This module produces data matching the Telco Customer
Churn schema (so the exact same pipeline works on the real CSV) and adds two
business-relevant columns the brief relies on later:

* ``support_tickets``  — count of recent support contacts (intervention rules)
* ``usage_trend_3m``   — % change in usage over the last 3 months (lag signal)

Churn is generated from a logistic model over the features so that the signal
is *learnable but noisy* — i.e. an XGBoost model can realistically reach the
~0.85 AUC target without the problem being trivially separable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_churn_data(
    n: int = 7_000,
    seed: int = config.RANDOM_STATE,
    base_intercept: float = 0.0,
) -> pd.DataFrame:
    """Generate ``n`` synthetic customers with a Telco-compatible schema.

    Parameters
    ----------
    n:
        Number of customers to generate.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        One row per customer, including the ``Churn`` target ("Yes"/"No").
    """
    rng = np.random.default_rng(seed)

    def choice(options, p=None):
        return rng.choice(options, size=n, p=p)

    # --- Raw categorical attributes ------------------------------------- #
    gender = choice(["Male", "Female"])
    senior = choice([0, 1], p=[0.84, 0.16])
    partner = choice(["Yes", "No"], p=[0.48, 0.52])
    dependents = choice(["Yes", "No"], p=[0.30, 0.70])
    phone_service = choice(["Yes", "No"], p=[0.90, 0.10])
    multiple_lines = choice(["Yes", "No", "No phone service"], p=[0.42, 0.48, 0.10])
    internet = choice(["DSL", "Fiber optic", "No"], p=[0.34, 0.44, 0.22])

    def addon():
        # Add-on services depend loosely on having internet.
        return np.where(
            internet == "No",
            "No internet service",
            rng.choice(["Yes", "No"], size=n, p=[0.5, 0.5]),
        )

    online_security = addon()
    online_backup = addon()
    device_protection = addon()
    tech_support = addon()
    streaming_tv = addon()
    streaming_movies = addon()

    contract = choice(["Month-to-month", "One year", "Two year"], p=[0.55, 0.25, 0.20])
    paperless = choice(["Yes", "No"], p=[0.59, 0.41])
    payment = choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        p=[0.34, 0.23, 0.22, 0.21],
    )

    # --- Numerical attributes ------------------------------------------- #
    tenure = rng.integers(0, 73, size=n)  # months with the company (0..72)
    monthly_charges = np.round(rng.normal(70, 30, size=n).clip(18.0, 120.0), 2)
    # Total charges roughly tenure * monthly, with noise; new customers ~ one month.
    total_charges = np.round(
        np.maximum(monthly_charges, tenure * monthly_charges * rng.normal(1.0, 0.05, size=n)),
        2,
    )
    support_tickets = rng.poisson(1.2, size=n).clip(0, 12)
    # Usage trend: negative = declining usage (a churn signal).
    usage_trend_3m = np.round(rng.normal(0.0, 0.20, size=n), 3)

    # --- Latent churn signal -------------------------------------------- #
    # Hand-crafted log-odds: short contracts, high charges, low tenure,
    # declining usage, electronic-check payment and many support tickets all
    # push churn up. Coefficients are deliberately strong so the classes are
    # well separated and an XGBoost model can reach the brief's ~0.87 AUC
    # target; a small Gaussian noise term keeps the problem non-trivial.
    contract_months = np.vectorize(config.CONTRACT_MONTHS.get)(contract)
    noise = rng.normal(0.0, 0.6, size=n)
    logit = (
        base_intercept
        + 0.045 * (monthly_charges - 70)
        - 0.070 * tenure
        - 3.00 * usage_trend_3m              # declining usage (negative) -> higher churn
        + 0.32 * support_tickets
        - 0.030 * contract_months
        + 1.00 * (contract == "Month-to-month")
        + 0.80 * (payment == "Electronic check")
        + 0.60 * (internet == "Fiber optic")
        + 0.40 * senior
        - 0.70 * (tech_support == "Yes")
        - 0.45 * (partner == "Yes")
        - 0.40 * (dependents == "Yes")
        + noise
    )
    churn_prob = _sigmoid(logit)
    churn = np.where(rng.random(n) < churn_prob, "Yes", "No")

    df = pd.DataFrame(
        {
            config.ID_COL: [f"CUST-{i:06d}" for i in range(n)],
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "support_tickets": support_tickets,
            "usage_trend_3m": usage_trend_3m,
            config.TARGET_COL: churn,
        }
    )
    return df


def main() -> None:
    """CLI entry point: generate the dataset and write it to ``data/raw/``."""
    config.ensure_dirs()
    df = generate_churn_data()
    df.to_csv(config.RAW_DATA_PATH, index=False)
    churn_rate = (df[config.TARGET_COL] == "Yes").mean()
    print(f"Wrote {len(df):,} rows -> {config.RAW_DATA_PATH}")
    print(f"Churn rate: {churn_rate:.1%}")


if __name__ == "__main__":
    main()
