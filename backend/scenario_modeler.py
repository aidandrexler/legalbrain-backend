import math
import random
from dataclasses import dataclass

BADGE_WEIGHTS = {
    "B1_insider_transfer": 2,
    "B2_retained_control": 2,
    "B3_concealment": 3,
    "B4_absconding": 3,
    "B5_asset_removal": 2,
    "B6_insolvency_at_transfer": 3,
    "B7_timing_before_debt": 2,
    "B8_substantially_all_assets": 3,
    "B9_flight_risk": 3,
    "B10_post_transfer_insolvency": 3,
    "B11_insider_before_suit": 2,
}
MAX_SCORE = 28


@dataclass
class BadgeResult:
    badges: list
    score: float
    max_score: int
    classification: str
    claude_input: str


def compute_fraud_badges(client_data: dict) -> BadgeResult:
    badges = []

    if client_data.get("transfer_to_insider"):
        badges.append({"id": "B1", "description": "Transfer to insider", "weight": BADGE_WEIGHTS["B1_insider_transfer"]})
    if client_data.get("debtor_retained_control"):
        badges.append({"id": "B2", "description": "Debtor retained control after transfer", "weight": BADGE_WEIGHTS["B2_retained_control"]})
    if client_data.get("transfer_concealed"):
        badges.append({"id": "B3", "description": "Transfer was concealed", "weight": BADGE_WEIGHTS["B3_concealment"]})
    if client_data.get("debtor_absconded"):
        badges.append({"id": "B4", "description": "Debtor absconded", "weight": BADGE_WEIGHTS["B4_absconding"]})
    if client_data.get("assets_removed_or_concealed"):
        badges.append({"id": "B5", "description": "Assets removed or concealed", "weight": BADGE_WEIGHTS["B5_asset_removal"]})
    if client_data.get("insolvent_at_transfer"):
        badges.append({"id": "B6", "description": "Insolvent at time of transfer", "weight": BADGE_WEIGHTS["B6_insolvency_at_transfer"]})
    if client_data.get("transfer_before_substantial_debt"):
        badges.append({"id": "B7", "description": "Transfer shortly before substantial debt incurred", "weight": BADGE_WEIGHTS["B7_timing_before_debt"]})
    if client_data.get("transferred_substantially_all_assets"):
        badges.append({"id": "B8", "description": "Transferred substantially all assets", "weight": BADGE_WEIGHTS["B8_substantially_all_assets"]})
    if client_data.get("sued_or_threatened_before_transfer"):
        badges.append({"id": "B9", "description": "Sued or threatened before transfer", "weight": BADGE_WEIGHTS["B9_flight_risk"]})
    if client_data.get("insolvent_after_transfer"):
        badges.append({"id": "B10", "description": "Became insolvent shortly after transfer", "weight": BADGE_WEIGHTS["B10_post_transfer_insolvency"]})
    if client_data.get("insider_transfer_before_suit"):
        badges.append({"id": "B11", "description": "Insider transfer before suit filed", "weight": BADGE_WEIGHTS["B11_insider_before_suit"]})

    score = sum(b["weight"] for b in badges)

    if score <= 5:
        classification = "Class 1 — Low Risk"
    elif score <= 11:
        classification = "Class 2 — Moderate Risk"
    elif score <= 17:
        classification = "Class 3 — Elevated Risk"
    elif score <= 23:
        classification = "Class 4 — High Risk"
    else:
        classification = "Class 5 — Critical Risk"

    badge_lines = "\n".join(
        f"- {b['id']}: {b['description']} (weight {b['weight']})" for b in badges
    )
    claude_input = (
        f"BADGE ANALYSIS RESULT — DO NOT RECALCULATE\n"
        f"Score: {score}/{MAX_SCORE}\n"
        f"Classification: {classification}\n"
        f"Triggered badges:\n{badge_lines or 'None'}"
    )

    return BadgeResult(
        badges=badges,
        score=score,
        max_score=MAX_SCORE,
        classification=classification,
        claude_input=claude_input,
    )


RED_FLAG_RULES: list = []  # populated from extraction_rules table at runtime


def run_red_flag_check(client_data: dict, tenant_rules: list) -> list:
    triggered = []
    all_rules = RED_FLAG_RULES + (tenant_rules or [])
    for rule in all_rules:
        field = rule.get("condition_field")
        operator = rule.get("condition_operator", "equals")
        value = rule.get("condition_value")
        client_value = client_data.get(field)
        triggered_flag = False
        if operator == "equals" and str(client_value) == str(value):
            triggered_flag = True
        elif operator == "greater_than" and client_value and float(client_value) > float(value):
            triggered_flag = True
        elif operator == "less_than" and client_value and float(client_value) < float(value):
            triggered_flag = True
        elif operator == "is_true" and client_value is True:
            triggered_flag = True
        elif operator == "is_false" and client_value is False:
            triggered_flag = True
        elif operator == "contains" and value and str(value).lower() in str(client_value or "").lower():
            triggered_flag = True
        if triggered_flag:
            triggered.append({
                "flag": rule.get("flag_level", "HIGH"),
                "message": rule.get("alert_message", ""),
                "citation": rule.get("citations", []),
                "condition_field": field,
            })
    return triggered


def run_grat_analysis(client_data: dict, rate_7520: float) -> dict:
    asset_value = float(client_data.get("estate_size_estimate") or 1000000)
    mean_return = 0.06
    volatility = 0.15
    simulations = 1000
    results = []
    for _ in range(simulations):
        value = asset_value
        annual_payment = asset_value * (rate_7520 / 100)
        for year in range(2):
            annual_return = random.gauss(mean_return, volatility)
            value = value * (1 + annual_return) - annual_payment
        results.append(max(value, 0))
    results.sort()
    p10 = results[int(simulations * 0.10)]
    median = results[int(simulations * 0.50)]
    p90 = results[int(simulations * 0.90)]
    zeroed_out_feasible = rate_7520 < mean_return * 100
    return {
        "p10": round(p10, 2),
        "median": round(median, 2),
        "p90": round(p90, 2),
        "zeroed_out_feasible": zeroed_out_feasible,
        "optimal_term_years": 2,
        "rate_7520_used": rate_7520,
        "asset_value": asset_value,
    }
