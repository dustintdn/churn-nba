"""Next-best-action (NBA) layer: turn a churn prediction into a decision.

A probability alone doesn't tell the retention team what to DO. This module maps
the customer's strongest churn drivers to a concrete, costed intervention via a
transparent rules layer. The rules are deliberately simple and auditable — in
production each rule's lift would be validated with a holdout experiment before
being trusted (see README caveats).

The driver signal can come from two places:
  - SHAP values per feature (preferred; used in the notebook), or
  - the raw feature values themselves (used by the API, which scores one record
    at a time and reads the customer's risk factors directly).

Both paths funnel into `recommend_action`, which returns a structured decision.
"""

from dataclasses import asdict, dataclass


@dataclass
class Recommendation:
    """A single, actionable retention recommendation for one customer."""
    risk_tier: str            # Low / Medium / High
    top_driver: str           # the dominant churn risk factor
    action: str               # the recommended next-best-action
    rationale: str            # one line the retention rep can read aloud

    def to_dict(self) -> dict:
        return asdict(self)


def risk_tier(churn_prob: float, high: float = 0.5, medium: float = 0.25) -> str:
    """Bucket a probability into an action-oriented tier."""
    if churn_prob >= high:
        return "High"
    if churn_prob >= medium:
        return "Medium"
    return "Low"


def _identify_driver(features: dict) -> tuple[str, str, str]:
    """Inspect a raw customer record and return (driver, action, rationale).

    Rules are ordered by retention priority. The FIRST matching condition wins,
    so the customer gets the single highest-leverage action rather than a list.
    """
    # 1) Price shock — the clearest, most addressable churn trigger.
    if features.get("price_increase_recent", 0) == 1 and features.get("discount_pct", 0) < 10:
        return ("Recent price increase",
                "Offer a loyalty discount / lock-in pricing",
                "Customer absorbed a recent price increase with little existing discount.")

    # 2) High support friction — fix the experience before they leave.
    if features.get("support_tickets_90d", 0) >= 3:
        return ("High support burden",
                "Proactive support call from a senior rep",
                "Elevated support volume signals unresolved friction.")

    # 3) Disengagement — they've stopped showing up.
    if features.get("last_login_days", 0) >= 21 or features.get("logins_per_week", 99) < 2:
        return ("Low engagement",
                "Re-engagement / onboarding outreach",
                "Login activity has dropped well below a healthy cadence.")

    # 4) No commitment — convert month-to-month into a committed contract.
    if features.get("contract_type") == "Month-to-month":
        return ("Month-to-month contract",
                "Incentivize an annual contract upgrade",
                "No contractual commitment makes this customer easy to lose.")

    # 5) High value with no human owner — assign coverage.
    if features.get("monthly_spend", 0) >= 1000 and features.get("has_account_manager", 0) == 0:
        return ("High value, unmanaged",
                "Assign a dedicated account manager",
                "High-spend account has no dedicated relationship owner.")

    # 6) Detractor — sentiment is the lever.
    if features.get("nps_score", 0) <= 0:
        return ("Low satisfaction (NPS)",
                "Executive check-in to rebuild trust",
                "Detractor-level satisfaction needs a senior relationship touch.")

    # Fallback for at-risk customers with no single dominant driver.
    return ("General churn risk",
            "Standard retention check-in",
            "Elevated risk without a single dominant driver — start with a check-in.")


def recommend_action(churn_prob: float, features: dict) -> Recommendation:
    """Top-level entry point used by the API: prob + raw record -> Recommendation.

    Low-risk customers get a 'monitor' action rather than a costly intervention,
    so the team spends retention budget where it moves the needle.
    """
    tier = risk_tier(churn_prob)
    if tier == "Low":
        return Recommendation(
            risk_tier=tier,
            top_driver="None (healthy)",
            action="Monitor — no intervention needed",
            rationale="Predicted churn risk is low; reserve outreach for higher-risk accounts.",
        )

    driver, action, rationale = _identify_driver(features)
    return Recommendation(tier, driver, action, rationale)


def recommend_from_shap(churn_prob: float, features: dict,
                        shap_values: dict, top_k: int = 1) -> Recommendation:
    """SHAP-driven variant used in the notebook.

    Picks the feature(s) pushing risk UP the most for THIS customer, then reuses
    the same rule mapping. This keeps the notebook's explanation and the API's
    decision logic consistent.
    """
    tier = risk_tier(churn_prob)
    if tier == "Low":
        return recommend_action(churn_prob, features)

    # Rank features by positive (risk-increasing) SHAP contribution.
    risk_drivers = sorted(
        ((f, v) for f, v in shap_values.items() if v > 0),
        key=lambda kv: kv[1], reverse=True,
    )[:top_k]

    # Map the dominant SHAP driver onto a known retention lever.
    if risk_drivers:
        top_feature = risk_drivers[0][0]
        action_map = {
            "price_increase_recent": ("Recent price increase",
                                      "Offer a loyalty discount / lock-in pricing"),
            "discount_pct": ("Price sensitivity",
                             "Offer a loyalty discount / lock-in pricing"),
            "support_tickets_90d": ("High support burden",
                                    "Proactive support call from a senior rep"),
            "last_login_days": ("Low engagement",
                                "Re-engagement / onboarding outreach"),
            "logins_per_week": ("Low engagement",
                                "Re-engagement / onboarding outreach"),
            "contract_type": ("Month-to-month contract",
                              "Incentivize an annual contract upgrade"),
            "monthly_spend": ("High value account",
                             "Assign a dedicated account manager"),
            "nps_score": ("Low satisfaction (NPS)",
                         "Executive check-in to rebuild trust"),
            "tenure_months": ("Early-tenure risk",
                             "Onboarding / success outreach"),
        }
        # Some one-hot features arrive prefixed (e.g. "cat__contract_type_..."):
        # match on the substring so the mapping is robust to encoding names.
        for key, (driver, action) in action_map.items():
            if key in top_feature:
                return Recommendation(
                    tier, driver, action,
                    f"SHAP flags '{top_feature}' as the dominant risk driver for this customer.",
                )

    # Fall back to the rule-based reading of the raw record.
    return recommend_action(churn_prob, features)
