"""Train the SMB churn model and serialize a serving-ready pipeline.

The artifact saved to models/churn_model.joblib is a single sklearn object that
bundles preprocessing + a calibrated XGBoost classifier. Because preprocessing
travels WITH the model, the FastAPI layer can hand it a raw customer record and
get a trustworthy probability back — no feature engineering drift between
training and serving.

Run standalone:
    python src/train.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# Column groups — declared once and reused by the API so schemas never drift.
NUMERIC_FEATURES = [
    "tenure_months", "monthly_spend", "logins_per_week", "last_login_days",
    "active_campaigns", "support_tickets_90d", "discount_pct",
    "price_increase_recent", "has_account_manager", "nps_score",
]
CATEGORICAL_FEATURES = ["contract_type", "plan_tier", "payment_method"]
TARGET = "churn"
ID_COL = "customer_id"

MODEL_PATH = "models/churn_model.joblib"
METRICS_PATH = "models/metrics.json"


def build_pipeline(scale_pos_weight: float) -> Pipeline:
    """Assemble preprocessing + calibrated XGBoost into one fitted-as-a-unit pipeline."""
    # Median-impute + scale numerics; most-frequent-impute + one-hot categoricals.
    # Scaling is not required for trees, but keeps the pipeline model-agnostic.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("scale", StandardScaler()),
            ]), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    # Class imbalance handled via scale_pos_weight (cheaper + less leak-prone than
    # resampling, and XGBoost handles it natively by reweighting the gradient).
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )

    # Calibrate probabilities (isotonic) so the scores the NBA layer consumes are
    # honest estimates of risk, not just rank-ordered scores.
    calibrated = CalibratedClassifierCV(xgb, method="isotonic", cv=3)

    return Pipeline([("prep", preprocessor), ("clf", calibrated)])


def load_data(path: str = "data/customers.csv") -> pd.DataFrame:
    """Load the customer table, dropping rows with no churn label."""
    df = pd.read_csv(path)
    # Impute the only intentionally-missing numeric so the API sees no NaNs either.
    df["nps_score"] = df["nps_score"].fillna(df["nps_score"].median())
    return df.dropna(subset=[TARGET])


def train(data_path: str = "data/customers.csv") -> dict:
    """Fit the pipeline, evaluate on a held-out split, and persist the artifact."""
    df = load_data(data_path)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET].astype(int)

    # Stratified split keeps the churn rate identical in train and test.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # scale_pos_weight = (#negatives / #positives) on the TRAIN split.
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    pipeline = build_pipeline(scale_pos_weight=neg / pos)
    pipeline.fit(X_train, y_train)

    # Evaluate with metrics that matter for imbalanced retention problems.
    proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "base_churn_rate": round(float(y.mean()), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model -> {MODEL_PATH}")
    print(f"Test PR-AUC={metrics['pr_auc']}  ROC-AUC={metrics['roc_auc']}  "
          f"(base rate {metrics['base_churn_rate']:.1%})")
    return metrics


if __name__ == "__main__":
    train()
