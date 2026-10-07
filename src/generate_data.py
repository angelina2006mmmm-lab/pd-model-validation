\
from pathlib import Path
import numpy as np
import pandas as pd


RANDOM_STATE = 42
N_ROWS = 6000


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_dataset(n_rows=N_ROWS, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)

    annual_revenue = rng.lognormal(mean=15.2, sigma=0.9, size=n_rows)
    debt_ratio = np.clip(rng.beta(2.2, 3.8, size=n_rows), 0, 1)
    current_ratio = np.clip(rng.lognormal(mean=0.35, sigma=0.45, size=n_rows), 0.2, 6)
    years_in_business = np.clip(rng.gamma(shape=3.2, scale=4.0, size=n_rows), 0.2, 50)
    late_payments_12m = rng.poisson(lam=1.4, size=n_rows)
    utilization = np.clip(rng.beta(2.5, 2.0, size=n_rows), 0, 1)
    employees = np.maximum(1, rng.lognormal(mean=3.7, sigma=1.0, size=n_rows).astype(int))
    sector_risk = rng.integers(1, 6, size=n_rows)
    requested_limit = rng.lognormal(mean=13.8, sigma=0.9, size=n_rows)

    revenue_log = np.log1p(annual_revenue)
    limit_to_revenue = requested_limit / np.maximum(annual_revenue, 1)

    linear = (
        -2.9
        + 2.7 * debt_ratio
        - 0.45 * current_ratio
        - 0.035 * years_in_business
        + 0.28 * late_payments_12m
        + 1.65 * utilization
        + 0.24 * (sector_risk - 3)
        + 0.55 * np.clip(limit_to_revenue, 0, 4)
        - 0.10 * (revenue_log - revenue_log.mean())
    )
    pd_true = sigmoid(linear)
    default_12m = rng.binomial(1, pd_true)

    df = pd.DataFrame({
        "annual_revenue": annual_revenue.round(2),
        "debt_ratio": debt_ratio.round(4),
        "current_ratio": current_ratio.round(4),
        "years_in_business": years_in_business.round(2),
        "late_payments_12m": late_payments_12m,
        "utilization": utilization.round(4),
        "employees": employees,
        "sector_risk": sector_risk,
        "requested_limit": requested_limit.round(2),
        "default_12m": default_12m,
    })

    # Add a small amount of realistic missingness for validation / preprocessing.
    for col, frac in {
        "current_ratio": 0.015,
        "years_in_business": 0.01,
        "employees": 0.008,
    }.items():
        idx = rng.choice(df.index, size=max(1, int(len(df) * frac)), replace=False)
        df.loc[idx, col] = np.nan

    return df


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "credit_risk_synthetic.csv"
    df = generate_dataset()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")
    print(f"Default rate: {df['default_12m'].mean():.3f}")
