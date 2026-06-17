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

A tuned, calibrated **XGBoost** classifier — selected over a logistic-regression baseline via a
held-out PR-AUC bake-off — scores each customer's churn risk from behavioral and account features.
**SHAP** explains every prediction down to its dominant driver, and a transparent rules layer maps
that driver to a costed retention action (discount, support call, re-onboarding, etc.). The trained
pipeline is serialized once and served behind a FastAPI `/predict` endpoint (and a Streamlit
dashboard), so the same logic that ran in analysis runs in production.

## Key Findings

- **Top churn drivers:** month-to-month contracts, support-ticket burden, recent price increases,
  and low product engagement push risk up; tenure, NPS, and a dedicated account manager pull it down.
- **Model selection:** a tuned **XGBoost** (chosen via `RandomizedSearchCV`) is benchmarked against a
  logistic-regression baseline and deployed automatically as the winner — **ROC-AUC 0.84**, **PR-AUC
  0.57**, a **3.4×** lift over the ~17% base rate. Probabilities are **isotonic-calibrated**, so a
  "30% risk" score really means ~30% churn.
- **At the decision threshold:** focusing the retention team on the **top 15%** of accounts by risk
  catches **~52% of all churners** at **~58% precision** — versus 17% if they called at random.

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
    "tenure_months": 14,
    "contract_type": "Month-to-month",
    "plan_tier": "Growth",
    "payment_method": "Credit card",
    "monthly_spend": 520,
    "logins_per_week": 3.0,
    "last_login_days": 12,
    "active_campaigns": 3,
    "support_tickets_90d": 1,
    "discount_pct": 5.0,
    "price_increase_recent": 1,
    "has_account_manager": 0,
    "nps_score": 20
  }'
```

**Example response:**

```json
{
  "customer_id": "SMB-00042",
  "churn_probability": 0.8563,
  "risk_tier": "High",
  "top_driver": "Recent price increase",
  "recommended_action": "Offer a loyalty discount / lock-in pricing",
  "rationale": "Customer absorbed a recent price increase with little existing discount."
}
```

### Interactive dashboard

For non-engineers, a **Streamlit dashboard** wraps the same model: score one customer from a
form, or upload a CSV and download a ranked action worklist.

```bash
streamlit run app/dashboard.py     # opens at http://localhost:8501
```

### Batch scoring

The day-to-day usage pattern — score an entire customer file into a ranked worklist:

```bash
python -m src.batch_score --input data/customers.csv --output data/scored_customers.csv
```

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) regenerate the synthetic dataset
python src/generate_data.py

# 3. Train the model (model bake-off + tuning) -> writes models/churn_model.joblib
python src/train.py            # add --fast to skip the hyperparameter search

# 4. Run the tests
pytest

# 5. Launch the serving API ...
uvicorn src.api:app --reload   # docs at http://127.0.0.1:8000/docs
#    ... or the dashboard
streamlit run app/dashboard.py

# 6. Explore the full analysis
jupyter notebook notebooks/churn_analysis.ipynb
```

### Run with Docker

Bring up the API **and** the dashboard together:

```bash
docker compose up        # API on :8000, dashboard on :8501
```

## Project Structure

```
smb-churn-nba-engine/
├── README.md
├── requirements.txt
├── Dockerfile / docker-compose.yml   # API + dashboard, one command
├── data/
│   └── customers.csv          # synthetic SMB customer dataset
├── notebooks/
│   └── churn_analysis.ipynb   # 5-phase analysis (EDA -> model -> SHAP -> NBA)
├── app/
│   └── dashboard.py           # Streamlit consumer (single + batch scoring)
├── src/
│   ├── generate_data.py       # reproducible synthetic data generator
│   ├── train.py               # model selection + calibrated pipeline -> joblib
│   ├── recommend.py           # next-best-action rules layer
│   ├── batch_score.py         # score a whole file into a ranked worklist
│   └── api.py                 # FastAPI serving app
├── tests/                     # NBA rules, API integration, end-to-end model
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
