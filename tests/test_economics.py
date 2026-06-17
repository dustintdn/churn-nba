"""Unit tests for the expected-value / ROI layer."""

from src.economics import ACTION_ECONOMICS, expected_value


def test_value_at_risk_scales_with_probability_and_spend():
    low = expected_value(0.2, 500, "Re-engagement / onboarding outreach")
    high = expected_value(0.8, 500, "Re-engagement / onboarding outreach")
    assert high.value_at_risk > low.value_at_risk
    bigger = expected_value(0.5, 2000, "Re-engagement / onboarding outreach")
    smaller = expected_value(0.5, 500, "Re-engagement / onboarding outreach")
    assert bigger.value_at_risk > smaller.value_at_risk


def test_net_value_can_favor_high_value_over_high_risk():
    # The core insight: rank by net value, not raw probability.
    high_risk_low_value = expected_value(0.85, 200, "Offer a loyalty discount / lock-in pricing")
    mod_risk_high_value = expected_value(0.45, 2400, "Assign a dedicated account manager")
    assert mod_risk_high_value.net_value > high_risk_low_value.net_value


def test_monitor_action_has_zero_cost_and_no_roi():
    ev = expected_value(0.05, 500, "Monitor — no intervention needed")
    assert ev.action_cost == 0
    assert ev.expected_value_saved == 0
    assert ev.roi is None


def test_net_value_is_saved_minus_cost():
    ev = expected_value(0.6, 1000, "Proactive support call from a senior rep")
    assert round(ev.net_value, 2) == round(ev.expected_value_saved - ev.action_cost, 2)


def test_unknown_action_uses_fallback_economics():
    ev = expected_value(0.5, 1000, "Some action we never defined")
    assert ev.action_cost == 50  # fallback cost
    assert ev.net_value is not None


def test_all_recommended_actions_have_economics():
    # Guards against an NBA action label with no cost/lift mapping.
    for econ in ACTION_ECONOMICS.values():
        assert "cost" in econ and "lift" in econ
        assert 0.0 <= econ["lift"] <= 1.0
