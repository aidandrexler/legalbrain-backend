import datetime


def get_supabase():
    from main import get_supabase as _get
    return _get()


async def provision_sandbox(tenant_id: str):
    supabase = get_supabase()

    # Guard: do not provision twice
    tenant = (
        supabase.table("tenants")
        .select("sandbox_provisioned")
        .eq("id", tenant_id)
        .single()
        .execute()
    )
    if tenant.data and tenant.data.get("sandbox_provisioned"):
        return {"success": True, "already_provisioned": True}

    try:
        trust_date = "2021-03-15"

        client_result = supabase.table("clients").insert({
            "tenant_id": tenant_id,
            "first_name": "Margaret",
            "last_name": "Roy",
            "estate_size_estimate": 4200000,
            "marital_status": "married",
            "spouse_first_name": "Thomas",
            "spouse_last_name": "Roy",
            "is_physician": False,
            "homestead_declared": True,
            "trust_funded": False,
            "trust_signed_date": trust_date,
            "entity_type": "LLC",
            "num_children": 2,
            "goal_weights": {
                "tax_efficiency": 0.35,
                "risk_reduction": 0.30,
                "liquidity": 0.20,
                "simplicity": 0.15,
            },
            "client_status": "active",
            "is_sandbox": True,
        }).execute()
        client_id = client_result.data[0]["id"]

        SANDBOX_OUTPUT = """## RISK ARCHITECTURE ANALYSIS — The Roy Family Estate

### RISK MATRIX

| Asset | Severity | Occurrence | Detection | RPN | Class |
|-------|----------|------------|-----------|-----|-------|
| Primary Residence ($1.8M) | 6 | 4 | 3 | 72 | YELLOW |
| LLC Membership Interest ($1.2M) | 5 | 4 | 4 | 80 | YELLOW |
| Revocable Trust Assets ($800K) | 9 | 9 | 9 | 729 | RED |
| Brokerage Account ($400K) | 7 | 6 | 5 | 210 | RED |

### CRITICAL FLAGS

FLAG 1 — TRUST SIGNED BUT NOT FUNDED (CRITICAL)
The Roy Family Trust was signed March 15, 2021 — over 3 years ago. No assets have been retitled into the trust. A signed trust with no funded assets provides ZERO probate avoidance and ZERO creditor protection.

FLAG 2 — BROKERAGE ACCOUNT EXPOSED (HIGH)
The $400K brokerage account is held individually with no beneficiary designation. It will pass through probate and is fully exposed to creditors.

### PRIORITY ACTIONS
1. Immediately retitle primary residence to trust via deed to trustee
2. Retitle LLC membership interest via assignment of membership interest
3. Add TOD designation to brokerage account pointing to trust
4. Review and update all beneficiary designations

### FL EXEMPTIONS APPLIED
- Homestead: Protected under FL Const. Art. X Section 4 (declared)
- LLC: Charging order protection under F.S. Section 605.0503
- Retirement accounts: Would be protected under ERISA/Section 222.21 (none present)"""

        supabase.table("diagnostics").insert({
            "tenant_id": tenant_id,
            "client_id": client_id,
            "diagnostic_type": "risk_architecture",
            "status": "complete",
            "output_text": SANDBOX_OUTPUT,
            "attorney_reviewed": False,
        }).execute()

        for item in [
            {
                "title": "Trust Funding Overdue — 3+ Years",
                "urgency": "CRITICAL",
                "description": "Roy Family Trust signed March 2021. No assets retitled. Immediate action required.",
                "source": "diagnostic",
            },
            {
                "title": "Brokerage Account: No Beneficiary Designation",
                "urgency": "HIGH",
                "description": "$400K brokerage will pass through probate without TOD designation.",
                "source": "diagnostic",
            },
            {
                "title": "Annual Gift Exclusion Not Being Utilized",
                "urgency": "MEDIUM",
                "description": "With $4.2M estate, systematic annual gifting ($19K per donee) would reduce estate tax exposure.",
                "source": "temporal_engine",
            },
        ]:
            supabase.table("action_items").insert({
                "tenant_id": tenant_id,
                "client_id": client_id,
                **item,
                "status": "open",
            }).execute()

        supabase.table("tenants").update({"sandbox_provisioned": True}).eq("id", tenant_id).execute()
        return {"success": True, "client_id": client_id}

    except Exception as e:
        return {"success": False, "error": str(e)}
