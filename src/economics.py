"""Expected-value layer: translate churn risk + an action into dollars.

A churn probability tells you *who* is at risk; the recommended action tells you
*what to do*. This module adds the third question a business actually asks:
**is the intervention worth it, and which customers return the most per dollar?**

For each customer we estimate:
    value at risk        = P(churn) x customer value over the horizon
    expected value saved = P(churn) x action's retention lift x customer value
    net value of action  = expected value saved - action cost
    ROI                  = net value / action cost

Ranking by *net value* (not raw probability) is the key business insight: a
moderate-risk, high-spend account can be worth far more to save than a
near-certain-churn, low-spend one.

IMPORTANT: the per-action cost and lift numbers below are **assumptions**, not
measured effects. In production each would be estimated from a holdout
experiment (see README caveats). They live in one place so they're easy to tune.
"""

from dataclasses import asdict, dataclass

# Assumed economics per recommended action: one-time cost ($) to execute, and the
# retention "lift" = the share of otherwise-churning customers the action saves.
ACTION_ECONOMICS = {
    "Offer a loyalty discount / lock-in pricing": {"cost": 150, "lift": 0.30},
    "Proactive support call from a senior rep":   {"cost": 80,  "lift": 0.25},
    "Re-engagement / onboarding outreach":        {"cost": 40,  "lift": 0.20},
    "Incentivize an annual contract upgrade":     {"cost": 120, "lift": 0.35},
    "Assign a dedicated account manager":         {"cost": 500, "lift": 0.40},
    "Executive check-in to rebuild trust":        {"cost": 200, "lift": 0.30},
    "Standard retention check-in":                {"cost": 30,  "lift": 0.10},
    "Monitor — no intervention needed":           {"cost": 0,   "lift": 0.0},
}
# Fallback for any unmapped action label.
_DEFAULT_ECON = {"cost": 50, "lift": 0.15}

# Assumptions for converting spend into the value we're protecting.
GROSS_MARGIN = 0.70        # share of revenue that is margin (what we actually lose)
HORIZON_MONTHS = 12        # value horizon for an avoided churn


@dataclass
class EconomicValue:
    """Per-customer economics of acting on a churn prediction."""
    customer_value: float        # margin value over the horizon
    value_at_risk: float         # expected loss if we do nothing
    expected_value_saved: float  # expected margin the action recovers
    action_cost: float           # cost to execute the action
    net_value: float             # expected_value_saved - action_cost
    roi: float | None            # net_value / action_cost (None if cost is 0)

    def to_dict(self) -> dict:
        return asdict(self)


def expected_value(churn_prob: float, monthly_spend: float, action: str,
                   margin: float = GROSS_MARGIN,
                   horizon_months: int = HORIZON_MONTHS) -> EconomicValue:
    """Compute the dollar economics of taking `action` on one customer."""
    econ = ACTION_ECONOMICS.get(action, _DEFAULT_ECON)

    # Margin value we stand to lose if this customer churns over the horizon.
    customer_value = monthly_spend * horizon_months * margin
    value_at_risk = churn_prob * customer_value

    # The action only recovers a fraction (its lift) of the at-risk value.
    expected_value_saved = churn_prob * econ["lift"] * customer_value
    net_value = expected_value_saved - econ["cost"]
    roi = (net_value / econ["cost"]) if econ["cost"] else None

    return EconomicValue(
        customer_value=round(customer_value, 2),
        value_at_risk=round(value_at_risk, 2),
        expected_value_saved=round(expected_value_saved, 2),
        action_cost=float(econ["cost"]),
        net_value=round(net_value, 2),
        roi=round(roi, 2) if roi is not None else None,
    )
