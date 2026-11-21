"""Create the extra demo datasets in sample_data/ (no network access needed).

    python sample_data/make_datasets.py

- breast_cancer.csv : real Wisconsin diagnostic breast cancer data bundled with
                      scikit-learn (569 rows, binary, medical). Small and easy, so
                      Model 2 usually has too few errors -> exercises the
                      insufficient-signal fallback honestly.
- customer_tiers.csv: synthetic multiclass dataset (3,000 rows, 3 classes) with
                      categorical features, missing values, label noise and a
                      sub-population the model handles badly (heteroscedastic
                      noise), where per-record reliability signals matter.
- loans.csv         : already present (generate_loans.py).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

OUT = Path(__file__).resolve().parent


def make_breast_cancer() -> pd.DataFrame:
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    df["diagnosis"] = np.where(df.pop("target") == 1, "benign", "malignant")
    df.columns = [c.replace(" ", "_") for c in df.columns]
    return df


def make_customer_tiers(n: int = 3000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    region = rng.choice(["north", "south", "east", "west", "online"], n, p=[0.25, 0.25, 0.2, 0.2, 0.1])
    plan = rng.choice(["basic", "plus", "pro"], n, p=[0.5, 0.35, 0.15])
    tenure = np.clip(rng.gamma(2.0, 18.0, n), 0, 120).round(1)
    monthly_spend = np.clip(rng.lognormal(4.0, 0.6, n), 5, 900).round(2)
    support_tickets = rng.poisson(1.5, n)
    sessions_per_week = np.clip(rng.normal(6, 3, n), 0, None).round(1)
    discount = rng.choice([0, 5, 10, 20], n, p=[0.6, 0.2, 0.15, 0.05])

    score = (
        0.015 * tenure
        + 0.004 * monthly_spend
        + 0.12 * sessions_per_week
        - 0.25 * support_tickets
        + 0.6 * (plan == "plus") + 1.2 * (plan == "pro")
        + 0.03 * discount
    )
    # Online customers are much noisier: the same signals predict their tier poorly
    noise_scale = np.where(region == "online", 1.6, 0.45)
    score = score + rng.normal(0, noise_scale, n)
    q1, q2 = np.quantile(score, [0.45, 0.8])
    tier = np.where(score < q1, "bronze", np.where(score < q2, "silver", "gold"))

    # 3% label noise
    flip = rng.rand(n) < 0.03
    tier[flip] = rng.choice(["bronze", "silver", "gold"], flip.sum())

    df = pd.DataFrame({
        "region": region,
        "plan": plan,
        "tenure_months": tenure,
        "monthly_spend": monthly_spend,
        "support_tickets": support_tickets,
        "sessions_per_week": sessions_per_week,
        "discount_pct": discount,
        "tier": tier,
    })
    for col, rate in [("sessions_per_week", 0.06), ("monthly_spend", 0.03), ("region", 0.02)]:
        df.loc[rng.rand(n) < rate, col] = np.nan
    return df


if __name__ == "__main__":
    make_breast_cancer().to_csv(OUT / "breast_cancer.csv", index=False)
    make_customer_tiers().to_csv(OUT / "customer_tiers.csv", index=False)
    print("Wrote breast_cancer.csv and customer_tiers.csv to", OUT)
