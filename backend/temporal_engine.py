import datetime
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

today = datetime.date.today()

_RATE_DEFAULTS = {
    "rate_7520": 4.6,
    "afr_short": 4.1,
    "afr_mid": 4.2,
    "afr_long": 4.5,
}


@dataclass
class PlanningWindow:
    window_type: str
    urgency: str  # CRITICAL, URGENT, HIGH, MEDIUM, LOW
    title: str
    reason: str
    strategies: list = field(default_factory=list)
    window_expires: Optional[datetime.date] = None
    action_required: str = ""


def get_current_rates(supabase_client, tenant_id: str) -> dict:
    """Fetch most recent rate from rate_table for this tenant. Fallback to defaults."""
    try:
        result = (
            supabase_client.table("rate_table")
            .select("rate_7520, afr_short, afr_mid, afr_long")
            .eq("tenant_id", tenant_id)
            .order("effective_month", desc=True)
            .limit(1)
            .execute()
        )
        if result.data:
            row = result.data[0]
            return {
                "rate_7520": float(row.get("rate_7520") or _RATE_DEFAULTS["rate_7520"]),
                "afr_short": float(row.get("afr_short") or _RATE_DEFAULTS["afr_short"]),
                "afr_mid": float(row.get("afr_mid") or _RATE_DEFAULTS["afr_mid"]),
                "afr_long": float(row.get("afr_long") or _RATE_DEFAULTS["afr_long"]),
            }
    except Exception as exc:
        logger.warning("rate_table fetch failed for tenant %s: %s", tenant_id, exc)
    return dict(_RATE_DEFAULTS)


def check_client_windows(client: dict, rates: dict) -> list[PlanningWindow]:
    """Check a single client for all planning windows. Returns list of PlanningWindow."""
    windows = []
    rate_7520 = float(rates.get("rate_7520", 4.6))
    afr_mid = float(rates.get("afr_mid", 4.2))

    # 1. GRAT window — rate based
    if rate_7520 < 4.0:
        windows.append(PlanningWindow(
            "GRAT_OPTIMAL", "HIGH",
            "GRAT Window Optimal — §7520 Below 4%",
            f"Current rate {rate_7520}% makes zeroed-out GRAT highly favorable",
            strategies=["grat"],
            action_required="Initiate GRAT discussion and asset selection immediately.",
        ))
    elif rate_7520 < 5.0:
        windows.append(PlanningWindow(
            "GRAT_FAVORABLE", "MEDIUM",
            "GRAT Window Favorable",
            f"§7520 at {rate_7520}% — GRAT viable",
            strategies=["grat"],
            action_required="Evaluate GRAT for high-growth assets; model zeroed-out structure.",
        ))

    # 2. IDGT installment sale window — AFR based
    if afr_mid < 4.5:
        windows.append(PlanningWindow(
            "IDGT_SALE_FAVORABLE", "MEDIUM",
            "IDGT Installment Sale Window Open",
            f"AFR mid-term at {afr_mid}% — installment sale to IDGT marginally favorable",
            strategies=["idgt_installment_sale"],
            action_required="Model installment sale with current AFR before rate environment shifts.",
        ))

    # 3. Pre-sale window
    sale_date = client.get("anticipated_sale_date")
    if sale_date:
        if isinstance(sale_date, str):
            sale_date = datetime.date.fromisoformat(sale_date[:10])
        days_until = (sale_date - today).days
        if 0 < days_until <= 90:
            windows.append(PlanningWindow(
                "PRE_SALE_90", "CRITICAL",
                "Practice Sale: 90-Day Planning Window",
                "Pre-transaction planning must begin immediately",
                strategies=["idgt_installment_sale", "grat", "ilit"],
                window_expires=sale_date,
                action_required="Engage transaction counsel. Freeze all new transfers. Begin entity restructure.",
            ))
        elif 0 < days_until <= 180:
            windows.append(PlanningWindow(
                "PRE_SALE_180", "URGENT",
                "Practice Sale: Begin Pre-Transaction Planning",
                f"Sale in {days_until} days — pre-transaction planning window open",
                strategies=["idgt_installment_sale", "grat"],
                window_expires=sale_date,
                action_required="Schedule planning meeting. Obtain business valuation. Assess entity structure.",
            ))

    # 4. Trust unfunded
    trust_signed = client.get("trust_signed_date")
    trust_funded = client.get("trust_funded", False)
    if trust_signed and not trust_funded:
        if isinstance(trust_signed, str):
            trust_signed = datetime.date.fromisoformat(trust_signed[:10])
        days_since = (today - trust_signed).days
        if days_since >= 30:
            windows.append(PlanningWindow(
                "TRUST_UNFUNDED_CRITICAL", "CRITICAL",
                f"Trust Unfunded — {days_since} Days Since Signing",
                "Signed trust with no retitled assets provides zero protection",
                action_required="Send funding checklist immediately. Schedule retitling call this week.",
            ))
        elif days_since >= 14:
            windows.append(PlanningWindow(
                "TRUST_UNFUNDED_HIGH", "HIGH",
                "Trust Implementation: Funding Not Confirmed",
                f"Trust signed {days_since} days ago — begin retitling",
                action_required="Confirm funding status with client. Send deed preparation instructions.",
            ))

    # 5. Annual gift exclusion Q4
    if today.month >= 10:
        dec_31 = datetime.date(today.year, 12, 31)
        windows.append(PlanningWindow(
            "ANNUAL_GIFT_Q4", "MEDIUM",
            "Annual Gift Exclusion — Act Before Dec 31",
            "Use $19,000 per donee exclusion before year end",
            strategies=["basic_gifting"],
            window_expires=dec_31,
            action_required="Confirm gift amounts and wire instructions before December 20.",
        ))

    # 6. Divorce risk + SLAT
    if client.get("divorce_risk_flag") and client.get("has_slat"):
        windows.append(PlanningWindow(
            "DIVORCE_SLAT_RISK", "URGENT",
            "Divorce Risk: SLAT Exposure — Review Immediately",
            "Divorce could collapse SLAT — immediate review required",
            action_required="Review SLAT trust document for divorce provisions. Model access loss scenario.",
        ))

    # 7. Payer concentration — physician
    payer_pct = float(client.get("top_3_payer_concentration_pct") or 0)
    if client.get("is_physician") and payer_pct > 70:
        windows.append(PlanningWindow(
            "PAYER_CONCENTRATION_CRITICAL", "CRITICAL",
            f"Payer Concentration {payer_pct:.0f}% — Gassman Founding Case Pattern",
            "Top 3 payers exceed 70% of revenue — catastrophic vulnerability",
            action_required="Run payer diversification analysis. Ensure practice entity provides maximum isolation.",
        ))

    # 8. PA structure — physician
    if client.get("is_physician") and (client.get("practice_entity_type") == "PA"):
        windows.append(PlanningWindow(
            "PA_CHARGING_ORDER_GAP", "CRITICAL",
            "PA Entity: Zero Charging Order Protection",
            "PA provides no charging order protection under F.S. §605.0503 — restructure required",
            action_required="Begin PA → PLLC conversion. Form PLLC first, operate 30 days concurrent, then dissolve PA.",
        ))

    # 9. Portability election — surviving spouse
    spouse_death_date = client.get("spouse_death_date")
    if spouse_death_date:
        if isinstance(spouse_death_date, str):
            spouse_death_date = datetime.date.fromisoformat(spouse_death_date[:10])
        days_since_death = (today - spouse_death_date).days
        if 0 < days_since_death < 270:
            deadline = spouse_death_date + datetime.timedelta(days=270)
            windows.append(PlanningWindow(
                "PORTABILITY_ELECTION", "CRITICAL",
                "Portability Election: Deadline Approaching",
                f"Estate tax return must be filed within 9 months of death to elect portability",
                window_expires=deadline,
                action_required=f"File Form 706 by {deadline.isoformat()} to preserve deceased spouse's unused exemption.",
            ))

    # 10. DAPT 2-year seasoning complete
    dapt_creation_date = client.get("dapt_creation_date")
    if dapt_creation_date:
        if isinstance(dapt_creation_date, str):
            dapt_creation_date = datetime.date.fromisoformat(dapt_creation_date[:10])
        days_since_creation = (today - dapt_creation_date).days
        if days_since_creation >= 730:
            windows.append(PlanningWindow(
                "DAPT_SEASON_2YR", "HIGH",
                "DAPT: 2-Year Seasoning Complete (Nevada/SD/WY)",
                f"DAPT created {days_since_creation} days ago — state lookback period satisfied",
                action_required="Confirm additional contributions now fully seasoned. Review for §548(e) 10-year federal clock.",
            ))

    # 11. Beneficiary designations — 3-year review
    last_beneficiary_review = client.get("last_beneficiary_review_date")
    if last_beneficiary_review:
        if isinstance(last_beneficiary_review, str):
            last_beneficiary_review = datetime.date.fromisoformat(last_beneficiary_review[:10])
        years_since = (today - last_beneficiary_review).days / 365
        if years_since >= 3:
            windows.append(PlanningWindow(
                "BENEFICIARY_REVIEW_NEEDED", "HIGH",
                "Beneficiary Designations: 3+ Years Since Last Review",
                f"Designations last reviewed {years_since:.1f} years ago — SECURE Act may affect stretch IRA provisions",
                action_required="Send beneficiary designation audit checklist. Review all IRAs, 401(k)s, life insurance, and annuities.",
            ))
    elif not last_beneficiary_review:
        windows.append(PlanningWindow(
            "BENEFICIARY_REVIEW_NEEDED", "HIGH",
            "Beneficiary Designations: No Review Date on Record",
            "No beneficiary review date recorded — designations may be outdated or missing",
            action_required="Request all beneficiary designation forms from client. Audit against current plan.",
        ))

    return windows


def run_temporal_engine(supabase_client, tenant_id: str, client_id: Optional[str] = None) -> dict:
    """
    Run temporal engine for one client or all active clients for a tenant.
    Inserts action_items into the database for each open window found.
    Returns summary dict.
    """
    rates = get_current_rates(supabase_client, tenant_id)

    # Fetch clients
    if client_id:
        result = (
            supabase_client.table("clients")
            .select("*")
            .eq("id", client_id)
            .eq("tenant_id", tenant_id)
            .execute()
        )
    else:
        result = (
            supabase_client.table("clients")
            .select("*")
            .eq("tenant_id", tenant_id)
            .in_("client_status", ["intake", "active", "documents_signed"])
            .execute()
        )

    clients = result.data or []
    summary = {"clients_checked": 0, "windows_found": 0, "critical": 0, "urgent": 0, "high": 0, "medium": 0}

    for client in clients:
        windows = check_client_windows(client, rates)
        summary["clients_checked"] += 1
        summary["windows_found"] += len(windows)

        for window in windows:
            summary[window.urgency.lower()] = summary.get(window.urgency.lower(), 0) + 1

            # Skip if an identical open action item already exists for this client + window_type
            existing = (
                supabase_client.table("action_items")
                .select("id")
                .eq("tenant_id", tenant_id)
                .eq("client_id", client["id"])
                .eq("window_type", window.window_type)
                .eq("status", "open")
                .execute()
            )
            if existing.data:
                continue

            due_date = window.window_expires.isoformat() if window.window_expires else None
            supabase_client.table("action_items").insert({
                "tenant_id": tenant_id,
                "client_id": client["id"],
                "title": window.title,
                "description": f"{window.reason}\n\nAction: {window.action_required}",
                "urgency": window.urgency,
                "source": "temporal_engine",
                "window_type": window.window_type,
                "status": "open",
                "due_date": due_date,
            }).execute()

    try:
        supabase_client.table("telemetry_events").insert({
            "event_type": "temporal_scan",
            "metadata": {
                "clients_checked": summary["clients_checked"],
                "windows_found": summary["windows_found"],
            },
        }).execute()
    except Exception as exc:
        logger.warning("Telemetry insert failed: %s", exc)

    return summary
