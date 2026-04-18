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

    if triggered:
        try:
            from main import get_supabase
            _severity = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
            highest = min(triggered, key=lambda f: _severity.get(f["flag"], 99))["flag"]
            get_supabase().table("telemetry_events").insert({
                "event_type": "red_flag_triggered",
                "flag_level": highest,
                "jurisdiction": client_data.get("jurisdiction", "FL"),
            }).execute()
        except Exception:
            pass

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


def run_all_scenarios(client_data: dict) -> dict:
    estate = float(client_data.get("estate_size_estimate") or 0)
    is_physician = client_data.get("is_physician", False)
    trust_funded = client_data.get("trust_funded", False)
    entity_type = client_data.get("practice_entity_type") or client_data.get("entity_type") or ""
    homestead = client_data.get("homestead_declared", False)
    payer_pct = float(client_data.get("top_3_payer_concentration_pct") or 0)

    def assess(asset_protected, threat):
        if asset_protected:
            return "T5 — Protected"
        elif estate < threat * 0.5:
            return "T2 — 30 days"
        elif estate < threat:
            return "T3 — 6 months"
        else:
            return "T4 — 1-2 years"

    scenario_a = {
        "name": "Nuclear Verdict ($3-5M judgment)",
        "what_breaks_first": "Unprotected liquid assets, unfunded trust assets" if not trust_funded else "Non-exempt assets outside trust",
        "what_survives": "Homestead (if declared), retirement accounts (ERISA/IRA), funded trust assets",
        "time_to_failure": assess(trust_funded and homestead, 3000000),
        "pa_gap_flag": "PA entity provides ZERO charging order protection" if "PA" in entity_type.upper() else None,
    }

    scenario_b = {
        "name": "Payer Audit + Recoupment Clawback",
        "what_breaks_first": "Practice operating accounts, accounts receivable" if is_physician else "N/A",
        "what_survives": "Personal assets if practice properly segregated",
        "time_to_failure": "T1 — Immediate" if (is_physician and payer_pct > 70) else "T3 — 6 months",
        "payer_concentration_flag": f"Top 3 payers = {payer_pct}% of revenue — CRITICAL" if payer_pct > 70 else None,
    }

    scenario_c = {
        "name": "Business Failure + Personal Guarantee Called",
        "what_breaks_first": "Personal assets pledged as guarantee collateral",
        "what_survives": "Homestead, retirement accounts, assets with no guarantee",
        "time_to_failure": assess(not client_data.get("has_personal_guarantee"), estate * 0.3),
    }

    scenario_d = {
        "name": "Divorce — Equitable Distribution",
        "what_breaks_first": "Marital assets, jointly titled property, commingled trust assets",
        "what_survives": "Separate property maintained with clean title, pre-marital assets with tracing",
        "time_to_failure": "T3 — 6 months" if client_data.get("divorce_risk_flag") else "T5 — Not applicable",
    }

    scenario_e = {
        "name": "IRS Audit + Back Taxes + Tax Liens",
        "what_breaks_first": "Non-exempt assets subject to federal tax lien (supersedes most state exemptions)",
        "what_survives": "Qualified retirement accounts (partial), homestead (partial — IRS can force sale)",
        "time_to_failure": "T3 — 6 months",
    }

    return {
        "scenario_a": scenario_a,
        "scenario_b": scenario_b,
        "scenario_c": scenario_c,
        "scenario_d": scenario_d,
        "scenario_e": scenario_e,
    }
