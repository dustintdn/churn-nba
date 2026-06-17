"""Batch scoring: score a whole customer file and emit a ranked action worklist.

This is how the engine is actually used day-to-day: point it at the current
customer export, get back a CSV sorted by churn risk with the recommended
next-best-action per row — ready for the retention team to work top-down.

Usage (run as a module from the repo root so `src` imports resolve):
    python -m src.batch_score                                   # data/customers.csv -> data/scored_customers.csv
    python -m src.batch_score --input path.csv --output out.csv
    python -m src.batch_score --top 100                         # only the riskiest N rows
"""

import argparse

import joblib
import pandas as pd

from src.recommend import recommend_action
from src.train import CATEGORICAL_FEATURES, ID_COL, MODEL_PATH, NUMERIC_FEATURES, load_data


def score_dataframe(df: pd.DataFrame, model=None) -> pd.DataFrame:
    """Score a customer DataFrame in memory and return it ranked with actions.

    Shared by the batch CLI and the Streamlit dashboard so scoring logic lives in
    exactly one place. Expects the raw feature columns; tolerates an optional id.
    """
    model = model or joblib.load(MODEL_PATH)

    # One vectorized scoring pass for calibrated churn probabilities.
    features = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    df = df.copy()
    df["churn_probability"] = model.predict_proba(features)[:, 1].round(4)

    # Per-row recommendation from the raw record (no SHAP needed for batch use).
    recs = [recommend_action(p, row) for p, row in
            zip(df["churn_probability"], features.to_dict("records"), strict=True)]
    df["risk_tier"] = [r.risk_tier for r in recs]
    df["top_risk_driver"] = [r.top_driver for r in recs]
    df["recommended_action"] = [r.action for r in recs]

    out_cols = ([ID_COL] if ID_COL in df.columns else []) + [
        "churn_probability", "risk_tier", "top_risk_driver", "recommended_action"]
    return df.sort_values("churn_probability", ascending=False)[out_cols]


def score_file(input_path: str, output_path: str, top: int | None = None) -> pd.DataFrame:
    """Load customers, score every row, attach the next-best-action, write ranked CSV."""
    df = load_data(input_path)  # reuses the same null-handling as training
    ranked = score_dataframe(df)
    if top:
        ranked = ranked.head(top)

    ranked.to_csv(output_path, index=False)
    tiers = ranked["risk_tier"].value_counts().to_dict()
    print(f"Scored {len(df):,} customers -> wrote {len(ranked):,} rows to {output_path}")
    print(f"Risk tiers in output: {tiers}")
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-score customers for churn + actions.")
    parser.add_argument("--input", default="data/customers.csv")
    parser.add_argument("--output", default="data/scored_customers.csv")
    parser.add_argument("--top", type=int, default=None, help="keep only the riskiest N rows")
    args = parser.parse_args()
    score_file(args.input, args.output, args.top)


if __name__ == "__main__":
    main()
