"""Generate a synthetic SMB-advertiser customer dataset with a churn label.

WHY SYNTHETIC: This portfolio project targets the small/mid-size-business (SMB)
advertising/subscription domain. Rather than depend on a Kaggle download (which
needs credentials and is Telco-flavored), we simulate a realistic customer base
whose churn is driven by a KNOWN logistic relationship. That lets us (a) ship a
fully reproducible repo, and (b) sanity-check that the model and SHAP recover
the relationships we baked in.

The generated columns intentionally map to concrete retention levers so the
next-best-action layer has something real to act on:
    - price sensitivity      -> discount offer
    - low engagement         -> onboarding / re-engagement outreach
    - high support burden    -> proactive support call
    - month-to-month billing -> annual-contract incentive
    - high value, no manager -> assign an account manager
"""

import argparse
import os

import numpy as np
import pandas as pd

# Categorical option sets (declared once so generation + docs stay in sync).
CONTRACT_TYPES = ["Month-to-month", "Annual", "Two-year"]
PAYMENT_METHODS = ["Credit card", "Bank transfer", "PayPal", "Manual invoice"]
PLAN_TIERS = ["Starter", "Growth", "Pro", "Enterprise"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function mapping a score to a probability."""
    return 1.0 / (1.0 + np.exp(-x))


def generate_customers(n: int = 6000, seed: int = 42) -> pd.DataFrame:
    """Build an n-row customer table with behavioral features and a churn label."""
    rng = np.random.default_rng(seed)

    # --- Account / behavioral features (drawn from plausible distributions) ---
    tenure_months = rng.integers(1, 73, size=n)                       # 1-72 months
    contract_type = rng.choice(CONTRACT_TYPES, size=n, p=[0.55, 0.30, 0.15])
    plan_tier = rng.choice(PLAN_TIERS, size=n, p=[0.35, 0.35, 0.20, 0.10])
    payment_method = rng.choice(PAYMENT_METHODS, size=n, p=[0.45, 0.25, 0.20, 0.10])

    # Monthly spend scales with plan tier (Starter cheap ... Enterprise pricey).
    tier_base = {"Starter": 120, "Growth": 380, "Pro": 900, "Enterprise": 2400}
    monthly_spend = np.array([tier_base[t] for t in plan_tier]) * rng.lognormal(0, 0.25, n)
    monthly_spend = monthly_spend.round(2)

    # Engagement signals.
    logins_per_week = np.clip(rng.normal(4.5, 2.2, n), 0, None).round(1)
    last_login_days = np.clip(rng.exponential(8, n).round(0), 0, 120).astype(int)
    active_campaigns = np.clip(rng.poisson(3, n), 0, None)

    # Support burden over the trailing 90 days.
    support_tickets_90d = rng.poisson(1.2, n)

    # Commercial context.
    discount_pct = np.clip(rng.normal(8, 6, n), 0, 40).round(1)
    price_increase_recent = rng.choice([0, 1], size=n, p=[0.75, 0.25])
    has_account_manager = rng.choice([0, 1], size=n, p=[0.7, 0.3])
    nps_score = np.clip(rng.normal(30, 35, n), -100, 100).round(0)

    # --- Known latent churn relationship (the "ground truth" we baked in) ---
    # Each term is a documented business driver of SMB churn.
    score = (
        -1.4                                                   # base log-odds (low churn baseline)
        - 0.030 * tenure_months                                # loyalty: longer tenure -> stickier
        + 0.045 * last_login_days                              # disengagement -> churn
        - 0.12 * logins_per_week                               # active usage -> retention
        + 0.55 * support_tickets_90d                           # friction -> churn
        + 0.9 * price_increase_recent                          # price shock -> churn
        - 0.020 * discount_pct                                 # discounts soften churn
        - 0.8 * has_account_manager                            # human touch -> retention
        - 0.012 * nps_score                                    # promoters stay
        + 0.0002 * monthly_spend                               # high spenders feel cost more
        - 0.20 * active_campaigns                              # product stickiness
    )
    # Month-to-month contracts churn far more than committed contracts.
    contract_effect = np.where(
        contract_type == "Month-to-month", 1.1,
        np.where(contract_type == "Annual", -0.4, -1.0),
    )
    score = score + contract_effect

    churn_prob = _sigmoid(score + rng.normal(0, 0.4, n))       # add irreducible noise
    churn = (rng.uniform(0, 1, n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customer_id": [f"SMB-{i:05d}" for i in range(n)],
        "tenure_months": tenure_months,
        "contract_type": contract_type,
        "plan_tier": plan_tier,
        "payment_method": payment_method,
        "monthly_spend": monthly_spend,
        "logins_per_week": logins_per_week,
        "last_login_days": last_login_days,
        "active_campaigns": active_campaigns,
        "support_tickets_90d": support_tickets_90d,
        "discount_pct": discount_pct,
        "price_increase_recent": price_increase_recent,
        "has_account_manager": has_account_manager,
        "nps_score": nps_score,
        "churn": churn,
    })

    # Inject a small amount of realistic missingness for the notebook to handle.
    missing_idx = rng.choice(n, size=int(0.02 * n), replace=False)
    df.loc[missing_idx, "nps_score"] = np.nan
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic SMB churn data.")
    parser.add_argument("--n", type=int, default=6000, help="number of customers")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--out", default="data/customers.csv", help="output CSV path")
    args = parser.parse_args()

    df = generate_customers(n=args.n, seed=args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows to {args.out} | churn rate = {df['churn'].mean():.1%}")


if __name__ == "__main__":
    main()
