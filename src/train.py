"""Train the SMB churn model and serialize a serving-ready pipeline.

The artifact saved to models/churn_model.joblib is a single sklearn object
bundling preprocessing and a calibrated classifier, so the serving layer can
score raw customer records with the same transformations used in training.

Model selection:
    1. Fit a logistic-regression baseline.
    2. Run a small RandomizedSearchCV over XGBoost hyperparameters (PR-AUC scored).
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
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# Column groups, shared with the API and batch scorer.
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
    """Scale numeric features and one-hot encode categoricals."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def make_xgb(scale_pos_weight: float, params: dict | None = None) -> XGBClassifier:
    """Construct an XGBoost classifier with class-imbalance reweighting."""
    return XGBClassifier(
        **(params or DEFAULT_XGB_PARAMS),
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr", random_state=42, n_jobs=-1,
    )


def build_pipeline(scale_pos_weight: float, params: dict | None = None) -> Pipeline:
    """Build the preprocessing + isotonic-calibrated XGBoost pipeline.

    Used by train(), the notebook, and the tests. Calibration matters because
    the economics layer multiplies these probabilities by dollar amounts.
    """
    xgb = make_xgb(scale_pos_weight, params)
    calibrated = CalibratedClassifierCV(xgb, method="isotonic", cv=3)
    return Pipeline([("prep", build_preprocessor()), ("clf", calibrated)])


def build_baseline_pipeline() -> Pipeline:
    """Logistic-regression baseline for the model comparison."""
    logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
    return Pipeline([("prep", build_preprocessor()), ("clf", logreg)])


def search_xgb_params(X_train, y_train, scale_pos_weight: float, n_iter: int = 25) -> dict:
    """RandomizedSearchCV over XGBoost hyperparameters, scored on PR-AUC.

    Searches the uncalibrated model (calibration is applied afterward) to keep
    the search fast. Returns the best params dict.
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


def impute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature prep shared by training and scoring (median-fill NPS).

    Does not touch the target column: scoring files have no churn label, so
    label handling belongs in load_data.
    """
    df = df.copy()
    df["nps_score"] = df["nps_score"].fillna(df["nps_score"].median())
    return df


def load_data(path: str = "data/customers.csv") -> pd.DataFrame:
    """Load the labeled training table: impute features, then drop unlabeled rows."""
    df = impute_features(pd.read_csv(path))
    return df.dropna(subset=[TARGET])


def load_scoring_data(path: str) -> pd.DataFrame:
    """Load a customer file for scoring: impute features, keep every row.

    Unlike load_data, this does not require a `churn` column, since scoring
    files are customers whose outcome is unknown.
    """
    return impute_features(pd.read_csv(path))


def train(data_path: str = "data/customers.csv", fast: bool = False,
          model_path: str = MODEL_PATH, metrics_path: str = METRICS_PATH) -> dict:
    """Compare candidate models, persist the winner, and log the comparison.

    model_path/metrics_path are parameterized so tests can train to a temp
    location without overwriting the committed artifact.
    """
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

    # Deploy whichever candidate wins on held-out PR-AUC.
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

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump(winner, model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model -> {model_path}  (winner: {winner_name})")
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
