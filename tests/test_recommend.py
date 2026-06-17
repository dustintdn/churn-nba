"""Unit tests for the next-best-action rules layer.

These pin down the business logic: each rule fires on the right signal, the
priority ordering resolves correctly when multiple risks are present, and
low-risk customers are left alone.
"""

from src.recommend import recommend_action, recommend_from_shap, risk_tier

# A baseline "healthy" record; individual tests perturb one signal at a time.
HEALTHY = dict(
    tenure_months=48, monthly_spend=500, logins_per_week=5, last_login_days=3,
    active_campaigns=4, support_tickets_90d=0, discount_pct=15,
    price_increase_recent=0, has_account_manager=1, nps_score=60,
    contract_type="Annual", plan_tier="Pro", payment_method="Credit card",
)


def test_risk_tiers_bucket_correctly():
    assert risk_tier(0.80) == "High"
    assert risk_tier(0.30) == "Medium"
    assert risk_tier(0.05) == "Low"


def test_low_risk_gets_monitor_no_intervention():
    rec = recommend_action(0.05, HEALTHY)
    assert rec.risk_tier == "Low"
    assert "Monitor" in rec.action


def test_price_increase_rule_fires():
    feats = {**HEALTHY, "price_increase_recent": 1, "discount_pct": 2}
    rec = recommend_action(0.7, feats)
    assert rec.top_driver == "Recent price increase"
    assert "discount" in rec.action.lower()


def test_support_burden_rule_fires():
    rec = recommend_action(0.7, {**HEALTHY, "support_tickets_90d": 5})
    assert rec.top_driver == "High support burden"
    assert "support call" in rec.action.lower()


def test_low_engagement_rule_fires():
    rec = recommend_action(0.7, {**HEALTHY, "last_login_days": 40})
    assert rec.top_driver == "Low engagement"


def test_priority_ordering_price_beats_support():
    # When BOTH price-shock and support burden are present, price wins (higher priority).
    feats = {**HEALTHY, "price_increase_recent": 1, "discount_pct": 1,
             "support_tickets_90d": 5}
    rec = recommend_action(0.7, feats)
    assert rec.top_driver == "Recent price increase"


def test_month_to_month_rule_fires_when_no_higher_priority():
    feats = {**HEALTHY, "contract_type": "Month-to-month"}
    rec = recommend_action(0.6, feats)
    assert rec.top_driver == "Month-to-month contract"


def test_fallback_for_atrisk_with_no_dominant_driver():
    # High risk but every specific signal is healthy -> generic check-in.
    rec = recommend_action(0.6, HEALTHY)
    assert rec.top_driver == "General churn risk"


def test_shap_driver_maps_to_action():
    shap_vals = {"num__support_tickets_90d": 1.2, "num__tenure_months": -0.5}
    rec = recommend_from_shap(0.7, HEALTHY, shap_vals)
    assert "support" in rec.action.lower()


def test_shap_low_risk_defers_to_monitor():
    rec = recommend_from_shap(0.05, HEALTHY, {"num__support_tickets_90d": 1.0})
    assert rec.risk_tier == "Low"
