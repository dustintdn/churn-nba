# SMB Churn + Next-Best-Action Engine

> **Which customers are most likely to churn — and what specific action should we take to retain each one?**

A churn model is only half a solution. This project pairs a calibrated churn predictor with a
**next-best-action (NBA) layer** that turns every prediction into a concrete retention decision,
and serves the whole thing through a **production FastAPI endpoint**.

---

## Business Question

For a small/mid-size-business (SMB) subscription platform, identify the accounts most likely to
cancel **and** recommend the single highest-leverage retention action for each one.

## Approach

A calibrated **XGBoost** classifier scores each customer's churn risk from behavioral and account
features. **SHAP** explains every prediction down to its dominant driver, and a transparent rules
layer maps that driver to a costed retention action (discount, support call, re-onboarding, etc.).
The trained pipeline is serialized once and served behind a FastAPI `/predict` endpoint, so the
same logic that ran in analysis runs in production.

## Key Findings

- **Top churn drivers:** month-to-month contracts, support-ticket burden, recent price increases,
  and low product engagement push risk up; tenure, NPS, and a dedicated account manager pull it down.
- **Model performance:** **ROC-AUC 0.81**, **PR-AUC 0.42** — a **3.3×** lift over the ~13% base rate.
  Probabilities are **isotonic-calibrated**, so a "30% risk" score really means ~30% churn.
- **At the decision threshold:** focusing the retention team on the **top 15%** of accounts by risk
  catches **~45% of all churners** at **~38% precision** — versus 13% if they called at random.

## From Prediction to Action

The NBA layer reads each at-risk customer's strongest churn driver and recommends a matching action,
with a one-line rationale a rep can read aloud. For example, a high-risk account flagged for a
**recent price increase** with little existing discount is routed to *"Offer a loyalty discount /
lock-in pricing."* A disengaged account gets re-onboarding outreach; a high-support-volume account
gets a proactive senior-rep call. The team works a **prioritized, reasoned worklist** instead of a
flat risk score — and low-risk customers are explicitly left alone to protect retention budget.

## Production

The fitted pipeline (`models/churn_model.joblib`) is served by a FastAPI app
(`src/api.py`) exposing:

- `GET /health` — liveness + model-readiness probe
- `POST /predict` — accepts a Pydantic-validated customer record; returns the calibrated churn
  probability **and** the recommended next-best-action

**Example request:**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "SMB-00042",
    "tenure_months": 3,
    "contract_type": "Month-to-month",
    "plan_tier": "Growth",
    "payment_method": "Manual invoice",
    "monthly_spend": 420.50,
    "logins_per_week": 0.5,
    "last_login_days": 34,
    "active_campaigns": 1,
    "support_tickets_90d": 4,
    "discount_pct": 2.0,
    "price_increase_recent": 1,
    "has_account_manager": 0,
    "nps_score": -20
  }'
```

**Example response:**

```json
{
  "customer_id": "SMB-00042",
  "churn_probability": 0.8485,
  "risk_tier": "High",
  "top_driver": "Recent price increase",
  "recommended_action": "Offer a loyalty discount / lock-in pricing",
  "rationale": "Customer absorbed a recent price increase with little existing discount."
}
```

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) regenerate the synthetic dataset
python src/generate_data.py

# 3. Train the model -> writes models/churn_model.joblib
python src/train.py

# 4. Launch the serving API
uvicorn src.api:app --reload

# 5. Hit the endpoint (see the curl example above), or open the interactive
#    docs at http://127.0.0.1:8000/docs

# 6. Explore the full analysis
jupyter notebook notebooks/churn_analysis.ipynb
```

## Project Structure

```
smb-churn-nba-engine/
├── README.md
├── requirements.txt
├── data/
│   └── customers.csv          # synthetic SMB customer dataset
├── notebooks/
│   └── churn_analysis.ipynb   # 5-phase analysis (EDA -> model -> SHAP -> NBA)
├── src/
│   ├── generate_data.py       # reproducible synthetic data generator
│   ├── train.py               # preprocessing + calibrated XGBoost pipeline -> joblib
│   ├── recommend.py           # next-best-action rules layer
│   └── api.py                 # FastAPI serving app
└── models/
    └── churn_model.joblib     # serialized serving-ready pipeline
```

## Caveats

- **Synthetic data.** The dataset is *simulated* (`src/generate_data.py`) with a known churn
  relationship so the repo is fully reproducible without external credentials. On real data the
  feature set, drivers, and metric levels would differ; the *methodology* is what transfers.
- **NBA rules are hypotheses.** The action mappings are sensible, auditable starting points. In
  practice each rule's retention lift should be validated with a holdout A/B experiment before the
  business trusts it — the model surfaces *who* and *why*; experiments confirm *what works*.
- **Point-in-time scoring.** The model scores a snapshot of customer state. A production deployment
  would add monitoring for feature/label drift and periodic recalibration as behavior shifts.
