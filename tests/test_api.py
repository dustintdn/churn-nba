"""Integration tests for the FastAPI serving layer.

Exercises the real app via Starlette's TestClient: health probe, a well-formed
prediction (asserting response shape + types), and validation rejection of a bad
record. Requires a trained model artifact (run `python src/train.py --fast`).
"""

import pytest
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)

VALID_RECORD = {
    "customer_id": "SMB-TEST-1", "tenure_months": 3,
    "contract_type": "Month-to-month", "plan_tier": "Growth",
    "payment_method": "Manual invoice", "monthly_spend": 420.5,
    "logins_per_week": 0.5, "last_login_days": 34, "active_campaigns": 1,
    "support_tickets_90d": 4, "discount_pct": 2.0, "price_increase_recent": 1,
    "has_account_manager": 0, "nps_score": -20,
}


def test_health_reports_model_loaded():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_well_formed_response():
    resp = client.post("/predict", json=VALID_RECORD)
    assert resp.status_code == 200
    body = resp.json()
    # Shape: all contract fields present.
    for key in ("customer_id", "churn_probability", "risk_tier",
                "top_driver", "recommended_action", "rationale"):
        assert key in body
    # Types + ranges.
    assert isinstance(body["churn_probability"], float)
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_tier"] in {"Low", "Medium", "High"}
    assert body["customer_id"] == "SMB-TEST-1"


def test_predict_rejects_invalid_record():
    bad = {**VALID_RECORD, "tenure_months": -5}  # violates ge=0 constraint
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_rejects_missing_field():
    incomplete = {k: v for k, v in VALID_RECORD.items() if k != "monthly_spend"}
    resp = client.post("/predict", json=incomplete)
    assert resp.status_code == 422
