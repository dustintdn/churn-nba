"""End-to-end model tests — catch serialization and scoring regressions.

Verifies the persisted artifact can be loaded and can score a raw customer
record (the exact path the API takes), and that training on a small sample
produces a usable pipeline. These guard the train -> serialize -> serve boundary.
"""

import joblib
import pandas as pd
import pytest

from src.train import (CATEGORICAL_FEATURES, MODEL_PATH, NUMERIC_FEATURES,
                       load_data, train)

ONE_RECORD = pd.DataFrame([{
    "tenure_months": 3, "monthly_spend": 420.5, "logins_per_week": 0.5,
    "last_login_days": 34, "active_campaigns": 1, "support_tickets_90d": 4,
    "discount_pct": 2.0, "price_increase_recent": 1, "has_account_manager": 0,
    "nps_score": -20, "contract_type": "Month-to-month", "plan_tier": "Growth",
    "payment_method": "Manual invoice",
}])


def test_saved_model_scores_one_record_end_to_end():
    model = joblib.load(MODEL_PATH)
    proba = model.predict_proba(ONE_RECORD[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    assert proba.shape == (1, 2)
    p_churn = float(proba[0, 1])
    assert 0.0 <= p_churn <= 1.0


def test_training_produces_usable_model(tmp_path):
    # Train in fast mode to a TEMP path so the committed artifact is never clobbered.
    metrics = train(fast=True,
                    model_path=str(tmp_path / "m.joblib"),
                    metrics_path=str(tmp_path / "metrics.json"))
    assert metrics["winner_metrics"]["pr_auc"] > metrics["base_churn_rate"]
    assert 0.0 <= metrics["winner_metrics"]["roc_auc"] <= 1.0
    assert metrics["winner"] in metrics["model_comparison"]
    assert (tmp_path / "m.joblib").exists()


def test_load_data_has_no_nulls_in_features():
    df = load_data()
    assert df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].isna().sum().sum() == 0
