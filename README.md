# SMB Churn + Next-Best-Action Engine

[![CI](https://github.com/dustintdn/churn-nba-dev/actions/workflows/ci.yml/badge.svg)](https://github.com/dustintdn/churn-nba-dev/actions/workflows/ci.yml)

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
- **Targeting by dollars, not just risk:** ranking the worklist by *expected net value* instead of
  churn probability captures **~58% more retention value at the same team budget** (≈$142k vs $90k
  across the held-out book in the notebook) — because a moderate-risk high-spend account can be worth
  far more to save than a near-certain-churn low-spend one.

## From Prediction to Action

The NBA layer reads each at-risk customer's strongest churn driver and recommends a matching action,
with a one-line rationale a rep can read aloud. For example, a high-risk account flagged for a
**recent price increase** with little existing discount is routed to *"Offer a loyalty discount /
lock-in pricing."* A disengaged account gets re-onboarding outreach; a high-support-volume account
gets a proactive senior-rep call. An **expected-value layer** then prices each action — value at
risk, expected value saved, net value, and ROI — so the team works a **dollar-prioritized worklist**
and low-risk customers are explicitly left alone to protect retention budget.

### How the economics work

Each customer's dollar impact is derived from four values:

| Metric | Formula | Example (85% churn risk, $520/mo spend) |
|---|---|---|
| **Customer value** | monthly spend × 12-month horizon × 70% gross margin | $520 × 12 × 0.70 = **$4,368** |
| **Value at risk** | churn probability × customer value | 0.85 × $4,368 = **$3,713** |
| **Expected value saved** | churn probability × action lift × customer value | 0.85 × 0.30 × $4,368 = **$1,114** |
| **Net value** | expected value saved − action cost | $1,114 − $150 = **$964** |

The **action lift** (the share of otherwise-churning customers the action saves) and **action cost**
(one-time cost to execute) vary per action type — for example, a loyalty discount costs $150 with
a 30% lift, while assigning an account manager costs $500 with a 40% lift. The 70% gross margin
and 12-month horizon are configurable assumptions in `src/economics.py`.

**These are assumptions, not measured effects.** In production, each action's cost and lift would
be estimated from holdout A/B experiments. They live in one place (`ACTION_ECONOMICS` in
`src/economics.py`) so they're easy to tune as real data comes in.

## Production

The fitted pipeline (`models/churn_model.joblib`) is served by a FastAPI app
(`src/api.py`) exposing:

- `GET /health` — liveness + model-readiness probe
- `POST /predict` — accepts a Pydantic-validated customer record; returns the calibrated churn
  probability, the recommended next-best-action, **and** its expected-value economics
- `POST /predict/batch` — scores a list of records and returns them ranked by expected net value
  (worklist order)

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
  "rationale": "Customer absorbed a recent price increase with little existing discount.",
  "value_at_risk": 3740.38,
  "expected_value_saved": 1122.11,
  "net_value": 972.11,
  "roi": 6.48
}
```

### Interactive dashboard

For non-engineers, a **Streamlit dashboard** wraps the same model: score one customer from a form, or upload a CSV and download a ranked action worklist. Dashboard provides predicted churn probability and risk tier, top risk driver and recommended next-best-action, and the expected-value economics.

![Single-customer scoring in the Streamlit dashboard - churn probability and risk tier, top risk driver and recommended next-best-action, and the expected-value economics.](docs/images/dashboard_single_customer_pred.png)

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
│   ├── README.md              # data dictionary (column definitions)
│   └── customers.csv          # synthetic SMB customer dataset
├── notebooks/
│   └── churn_analysis.ipynb   # 5-phase analysis (EDA -> model -> SHAP -> NBA)
├── app/
│   └── dashboard.py           # Streamlit consumer (single + batch scoring)
├── src/
│   ├── generate_data.py       # reproducible synthetic data generator
│   ├── train.py               # model selection + calibrated pipeline -> joblib
│   ├── recommend.py           # next-best-action rules layer
│   ├── economics.py           # expected-value / ROI layer (risk -> dollars)
│   ├── batch_score.py         # score a whole file into a ranked worklist
│   └── api.py                 # FastAPI serving app
├── tests/                     # NBA rules, API integration, end-to-end model
└── models/
    └── churn_model.joblib     # serialized serving-ready pipeline
```

See [`MODEL_CARD.md`](MODEL_CARD.md) for the model's intended use, training data, evaluation, and limitations.

## Caveats

- **Synthetic data.** The dataset is *simulated* (`src/generate_data.py`) with a known churn
  relationship so the repo is fully reproducible without external credentials. On real data the
  feature set, drivers, and metric levels would differ; the *methodology* is what transfers.
- **NBA rules are hypotheses.** The action mappings are sensible, auditable starting points. In
  practice each rule's retention lift should be validated with a holdout A/B experiment before the
  business trusts it — the model surfaces *who* and *why*; experiments confirm *what works*.
- **Point-in-time scoring.** The model scores a snapshot of customer state. A production deployment
  would add monitoring for feature/label drift and periodic recalibration as behavior shifts.
