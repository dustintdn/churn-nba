"""Train the SMB churn model and serialize a serving-ready pipeline.

The artifact saved to models/churn_model.joblib is a single sklearn object that
bundles preprocessing + a calibrated classifier. Because preprocessing travels
WITH the model, the FastAPI layer can hand it a raw customer record and get a
trustworthy probability back — no feature-engineering drift between training and
serving.

Model selection (not just "use XGBoost"):
    1. Fit a logistic-regression BASELINE so any tree model has to earn its keep.
    2. Run a light RandomizedSearchCV over XGBoost hyperparameters (PR-AUC scored).
    3. Compare baseline / default-XGB / tuned-XGB on a held-out split.
    4. Calibrate and persist the winner. The comparison is logged to metrics.json.

Run standalone:
    python src/train.py            # full search (default)
    python src/train.py --fast     # skip the search, use known-good params (CI / tests)
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
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

# Known-good XGBoost params (the historical default; also the --fast fallback).
DEFAULT_XGB_PARAMS = dict(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
)


def build_preprocessor() -> ColumnTransformer:
    """Scale numerics + one-hot categoricals. Shared by every candidate model."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def make_xgb(scale_pos_weight: float, params: dict | None = None) -> XGBClassifier:
    """Construct an XGBoost classifier with imbalance reweighting baked in."""
    return XGBClassifier(
        **(params or DEFAULT_XGB_PARAMS),
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr", random_state=42, n_jobs=-1,
    )


def build_pipeline(scale_pos_weight: float, params: dict | None = None) -> Pipeline:
    """Preprocessing + isotonic-calibrated XGBoost as one fitted-as-a-unit pipeline.

    Kept as the canonical builder used by the notebook and tests. Calibration
    makes the probabilities the NBA layer consumes honest, not just rank-ordered.
    """
    xgb = make_xgb(scale_pos_weight, params)
    calibrated = CalibratedClassifierCV(xgb, method="isotonic", cv=3)
    return Pipeline([("prep", build_preprocessor()), ("clf", calibrated)])


def build_baseline_pipeline() -> Pipeline:
    """Logistic-regression baseline — the bar XGBoost must clear to be worth it."""
    logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
    return Pipeline([("prep", build_preprocessor()), ("clf", logreg)])


def search_xgb_params(X_train, y_train, scale_pos_weight: float, n_iter: int = 25) -> dict:
    """Light RandomizedSearchCV over XGBoost hyperparameters, scored on PR-AUC.

    Returns the best params dict. We search the uncalibrated tree (calibration is
    applied afterward) so the search stays fast and focused on ranking quality.
    """
    search_space = {
        "n_estimators": [200, 300, 400, 600, 800],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.02, 0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    }
    pipe = Pipeline([("prep", build_preprocessor()),
                     ("clf", make_xgb(scale_pos_weight))])
    search = RandomizedSearchCV(
        pipe, {f"clf__{k}": v for k, v in search_space.items()},
        n_iter=n_iter, scoring="average_precision", cv=3,
        random_state=42, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    print(f"Search best CV PR-AUC={search.best_score_:.4f}  params={best}")
    return best


def _evaluate(pipeline, X_test, y_test) -> dict:
    """Score a fitted pipeline on the held-out split (PR-AUC + ROC-AUC)."""
    proba = pipeline.predict_proba(X_test)[:, 1]
    return {"pr_auc": round(float(average_precision_score(y_test, proba)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, proba)), 4)}


def load_data(path: str = "data/customers.csv") -> pd.DataFrame:
    """Load the customer table, impute the only missing column, drop unlabeled rows."""
    df = pd.read_csv(path)
    df["nps_score"] = df["nps_score"].fillna(df["nps_score"].median())
    return df.dropna(subset=[TARGET])


def train(data_path: str = "data/customers.csv", fast: bool = False) -> dict:
    """Compare candidate models, calibrate + persist the winner, log the comparison."""
    df = load_data(data_path)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    spw = neg / pos  # scale_pos_weight = #negatives / #positives

    # --- Candidate 1: logistic-regression baseline -----------------------------
    baseline = build_baseline_pipeline().fit(X_train, y_train)
    comparison = {"logistic_regression": _evaluate(baseline, X_test, y_test)}

    # --- Candidate 2: XGBoost with known-good defaults --------------------------
    xgb_default = build_pipeline(spw, DEFAULT_XGB_PARAMS).fit(X_train, y_train)
    comparison["xgboost_default"] = _evaluate(xgb_default, X_test, y_test)

    # --- Candidate 3: XGBoost with searched hyperparameters ---------------------
    if fast:
        best_params = DEFAULT_XGB_PARAMS
    else:
        best_params = search_xgb_params(X_train, y_train, spw)
    tuned = build_pipeline(spw, best_params).fit(X_train, y_train)
    comparison["xgboost_tuned"] = _evaluate(tuned, X_test, y_test)

    # Deploy whichever candidate actually wins on PR-AUC (real model selection,
    # not "XGBoost because we said so"). Fitted candidates kept for selection.
    fitted = {"logistic_regression": baseline,
              "xgboost_default": xgb_default, "xgboost_tuned": tuned}
    winner_name = max(comparison, key=lambda k: comparison[k]["pr_auc"])
    winner = fitted[winner_name]

    metrics = {
        "winner": winner_name,
        "winner_metrics": comparison[winner_name],
        "model_comparison": comparison,
        "best_params": best_params,
        "base_churn_rate": round(float(y.mean()), 4),
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
    }

    os.makedirs("models", exist_ok=True)
    joblib.dump(winner, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model -> {MODEL_PATH}  (winner: {winner_name})")
    for name, m in comparison.items():
        flag = "  <-- deployed" if name == winner_name else ""
        print(f"  {name:22s} PR-AUC={m['pr_auc']:.4f}  ROC-AUC={m['roc_auc']:.4f}{flag}")
    print(f"Base churn rate: {metrics['base_churn_rate']:.1%}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the SMB churn model.")
    parser.add_argument("--fast", action="store_true",
                        help="skip the hyperparameter search (use known-good params)")
    args = parser.parse_args()
    train(fast=args.fast)
