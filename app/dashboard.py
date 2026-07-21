"""Streamlit dashboard for the churn + NBA engine.

Two ways to use the model without touching the API:
  1. Score one customer from a form and see risk + recommended action.
  2. Upload a CSV of customers and get a ranked, downloadable action worklist.

Run from the repo root:
    streamlit run app/dashboard.py
"""

import sys
from pathlib import Path

# Make the repo root importable when Streamlit runs this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
import streamlit as st

from src.batch_score import score_dataframe
from src.economics import expected_value
from src.recommend import recommend_action
from src.train import CATEGORICAL_FEATURES, MODEL_PATH, NUMERIC_FEATURES, impute_features

# Risk colors, matching the notebook palette.
TIER_COLOR = {"High": "#D1495B", "Medium": "#E8A87C", "Low": "#2E86AB"}

st.set_page_config(page_title="SMB Churn + Next-Best-Action", page_icon="📉", layout="wide")


@st.cache_resource
def load_model():
    """Load the serialized pipeline once per session."""
    return joblib.load(MODEL_PATH)


def render_single_customer(model):
    """Form -> single prediction with risk tier and recommended action."""
    st.subheader("Score a single customer")
    c1, c2, c3 = st.columns(3)
    rec = {
        "tenure_months": c1.number_input("Tenure (months)", 0, 120, 3),
        "monthly_spend": c1.number_input("Monthly spend ($)", 0.0, 10000.0, 420.0),
        "logins_per_week": c1.number_input("Logins / week", 0.0, 50.0, 0.5),
        "last_login_days": c1.number_input("Days since last login", 0, 365, 34),
        "active_campaigns": c2.number_input("Active campaigns", 0, 50, 1),
        "support_tickets_90d": c2.number_input("Support tickets (90d)", 0, 50, 4),
        "discount_pct": c2.number_input("Discount (%)", 0.0, 100.0, 2.0),
        "nps_score": c2.number_input("NPS score", -100, 100, -20),
        "contract_type": c3.selectbox("Contract", ["Month-to-month", "Annual", "Two-year"]),
        "plan_tier": c3.selectbox("Plan tier", ["Starter", "Growth", "Pro", "Enterprise"], 1),
        "payment_method": c3.selectbox("Payment", ["Credit card", "Bank transfer", "PayPal", "Manual invoice"], 3),
        "price_increase_recent": int(c3.checkbox("Recent price increase", True)),
        "has_account_manager": int(c3.checkbox("Has account manager", False)),
    }

    if st.button("Predict churn risk", type="primary"):
        X = pd.DataFrame([rec])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        prob = float(model.predict_proba(X)[0, 1])
        action = recommend_action(prob, rec)
        ev = expected_value(prob, rec["monthly_spend"], action.action)
        m1, m2 = st.columns([1, 2])
        m1.metric("Churn probability", f"{prob:.0%}")
        m1.markdown(f"**Risk tier:** :{_tier_badge(action.risk_tier)}")
        m2.markdown(f"**Top risk driver:** {action.top_driver}")
        m2.markdown(f"**Recommended action:** {action.action}")
        m2.caption(action.rationale)
        st.divider()
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Value at risk", f"${ev.value_at_risk:,.0f}")
        e2.metric("Expected value saved", f"${ev.expected_value_saved:,.0f}")
        e3.metric("Action cost", f"${ev.action_cost:,.0f}")
        e4.metric("Net value", f"${ev.net_value:,.0f}")


def _tier_badge(tier: str) -> str:
    """Map a tier to a Streamlit colored-badge token."""
    return {"High": "red[High]", "Medium": "orange[Medium]", "Low": "blue[Low]"}[tier]


def render_batch(model):
    """CSV upload -> ranked, downloadable action worklist."""
    st.subheader("Score a customer file")
    st.caption("Upload a CSV with the standard customer columns "
               "(see data/customers.csv). Rows are ranked by expected net value.")
    upload = st.file_uploader("Customer CSV", type="csv")
    if upload is not None:
        try:
            df = pd.read_csv(upload)
        except Exception as exc:  # malformed / non-CSV upload
            st.error(f"Couldn't read that file as CSV: {exc}")
            return

        missing = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c not in df.columns]
        if missing:
            st.error(
                "Uploaded CSV is missing required column(s): "
                f"**{', '.join(missing)}**. Expected the standard customer columns "
                "(see data/customers.csv)."
            )
            return

        df = impute_features(df)  # median-fill scattered NPS nulls, same as scoring
        ranked = score_dataframe(df, model)
        st.dataframe(ranked, use_container_width=True, hide_index=True)
        st.download_button("Download ranked worklist (CSV)",
                           ranked.to_csv(index=False), "scored_customers.csv", "text/csv")
        counts = ranked["risk_tier"].value_counts()
        st.bar_chart(counts, color="#D1495B")


def main():
    st.title("📉 SMB Churn + Next-Best-Action Engine")
    st.markdown("Predict which customers are likely to churn — and the specific "
                "action to retain each one.")
    model = load_model()
    tab1, tab2 = st.tabs(["Single customer", "Batch upload"])
    with tab1:
        render_single_customer(model)
    with tab2:
        render_batch(model)


if __name__ == "__main__":
    main()
