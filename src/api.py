"""FastAPI serving layer for the SMB churn + next-best-action engine.

This is the model-to-production boundary: a raw customer record comes in over
HTTP, gets validated by a Pydantic schema, scored by the SAME pipeline that was
trained offline, and comes back with both a calibrated churn probability and a
concrete recommended action.

Run locally:
    uvicorn src.api:app --reload

Then POST a customer record to /predict (see README for an example).
"""

from functools import lru_cache

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.recommend import recommend_action
from src.train import CATEGORICAL_FEATURES, MODEL_PATH, NUMERIC_FEATURES

app = FastAPI(
    title="SMB Churn + Next-Best-Action Engine",
    description="Predicts customer churn risk and recommends a retention action.",
    version="1.0.0",
)


class CustomerRecord(BaseModel):
    """Validated inbound customer record. Field constraints reject bad data
    at the edge so the model never sees nonsense (negative tenure, etc.)."""
    customer_id: str = Field(..., examples=["SMB-00042"])
    tenure_months: int = Field(..., ge=0, le=600)
    contract_type: str = Field(..., examples=["Month-to-month"])
    plan_tier: str = Field(..., examples=["Growth"])
    payment_method: str = Field(..., examples=["Credit card"])
    monthly_spend: float = Field(..., ge=0)
    logins_per_week: float = Field(..., ge=0)
    last_login_days: int = Field(..., ge=0)
    active_campaigns: int = Field(..., ge=0)
    support_tickets_90d: int = Field(..., ge=0)
    discount_pct: float = Field(..., ge=0, le=100)
    price_increase_recent: int = Field(..., ge=0, le=1)
    has_account_manager: int = Field(..., ge=0, le=1)
    nps_score: float = Field(..., ge=-100, le=100)


class PredictionResponse(BaseModel):
    """Structured serving response: the score AND the decision it drives."""
    customer_id: str
    churn_probability: float
    risk_tier: str
    top_driver: str
    recommended_action: str
    rationale: str


@lru_cache(maxsize=1)
def get_model():
    """Load the serialized pipeline once and cache it for the process lifetime."""
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail=f"Model artifact not found at {MODEL_PATH}. Run `python src/train.py` first.",
        )


@app.get("/health")
def health() -> dict:
    """Liveness + readiness probe. Reports whether the model is loadable."""
    try:
        get_model()
        model_ready = True
    except HTTPException:
        model_ready = False
    return {"status": "ok", "model_loaded": model_ready}


@app.post("/predict", response_model=PredictionResponse)
def predict(record: CustomerRecord) -> PredictionResponse:
    """Score one customer and return churn probability + next-best-action."""
    model = get_model()

    # Build a single-row frame with exactly the columns the pipeline expects.
    raw = record.model_dump()
    features = {k: raw[k] for k in NUMERIC_FEATURES + CATEGORICAL_FEATURES}
    X = pd.DataFrame([features])

    # Calibrated probability of the positive (churn) class.
    churn_prob = float(model.predict_proba(X)[0, 1])

    # The NBA layer reads the raw record to pick the highest-leverage action.
    rec = recommend_action(churn_prob, features)

    return PredictionResponse(
        customer_id=record.customer_id,
        churn_probability=round(churn_prob, 4),
        risk_tier=rec.risk_tier,
        top_driver=rec.top_driver,
        recommended_action=rec.action,
        rationale=rec.rationale,
    )
