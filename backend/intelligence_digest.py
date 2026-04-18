import logging
from datetime import datetime, timedelta

log = logging.getLogger("legalbrain-digest")


def get_supabase():
    from main import get_supabase as _get_supabase
    return _get_supabase()


# ============================================================
# LEVEL 1: PATTERN RECOGNITION
# Detects when the same correction type fires repeatedly
# on the same diagnostic type for the same tenant
# ============================================================

def run_pattern_recognition(tenant_id: str) -> list:
    """
    Scan recent corrections (last 30 days) for patterns.
    If same correction_type fires 3+ times on same diagnostic_type: flag as pattern.
    Returns list of detected patterns as dicts.
    """
    supabase = get_supabase()
    cutoff = (datetime.utcnow() - timedelta(days=30)).isoformat()
    result = (
        supabase.table("diagnostics")
        .select("diagnostic_type,correction_type,correction_significance")
        .eq("tenant_id", tenant_id)
        .eq("attorney_reviewed", True)
        .neq("correction_type", "none")
        .neq("correction_type", None)
        .gte("reviewed_at", cutoff)
        .execute()
    )

    corrections = result.data or []
    pattern_counts = {}
    for c in corrections:
        key = (c["diagnostic_type"], c["correction_type"])
        pattern_counts[key] = pattern_counts.get(key, 0) + 1

    patterns = []
    for (diag_type, corr_type), count in pattern_counts.items():
        if count >= 3:
            patterns.append({
                "diagnostic_type": diag_type,
                "correction_type": corr_type,
                "count": count,
                "detected_at": datetime.utcnow().isoformat(),
            })
    return patterns


# ============================================================
# LEVEL 2: RETRIEVAL WEIGHT UPDATES
# Adjusts how heavily each namespace is weighted for each
# diagnostic type based on correction patterns
# ============================================================

def update_retrieval_weights(tenant_id: str, patterns: list):
    """
    If wrong_jurisdiction corrections cluster on a diagnostic:
    reduce weight of secondary_law namespace for that diagnostic.
    If factual_error corrections cluster: reduce practitioner_patterns weight.
    Weights stored in a retrieval_weights table (namespace, diagnostic_type, weight).
    Since retrieval_weights table may not exist yet, wrap in try/except.
    """
    supabase = get_supabase()
    for pattern in patterns:
        diag_type = pattern["diagnostic_type"]
        corr_type = pattern["correction_type"]
        try:
            if corr_type == "wrong_jurisdiction":
                supabase.table("retrieval_weights").upsert({
                    "tenant_id": tenant_id,
                    "namespace": "secondary_law",
                    "diagnostic_type": diag_type,
                    "weight": 0.6,
                    "updated_at": datetime.utcnow().isoformat(),
                }).execute()
            elif corr_type == "factual_error":
                supabase.table("retrieval_weights").upsert({
                    "tenant_id": tenant_id,
                    "namespace": "practitioner_patterns",
                    "diagnostic_type": diag_type,
                    "weight": 0.5,
                    "updated_at": datetime.utcnow().isoformat(),
                }).execute()
        except Exception as e:
            log.warning(f"Retrieval weight update failed: {e}")


# ============================================================
# LEVEL 3: SYSTEM PROMPT IMPROVEMENT PROPOSALS
# When a pattern is detected, proposes a new rule for the
# attorney to approve or reject in the UI
# ============================================================

def propose_system_improvements(tenant_id: str, patterns: list, anthropic_key: str):
    """
    For each detected pattern, call Claude to propose a new system prompt rule.
    Insert proposal into system_improvements table with status=pending.
    Attorney reviews and approves/rejects in the Brain page UI.
    """
    if not patterns or not anthropic_key:
        return

    import anthropic
    client = anthropic.Anthropic(api_key=anthropic_key)

    for pattern in patterns:
        try:
            prompt = (
                f"An attorney has corrected the \"{pattern['diagnostic_type']}\" diagnostic "
                f"{pattern['count']} times with correction type \"{pattern['correction_type']}\".\n\n"
                f"Propose a single, specific rule (1-2 sentences) that should be added to the "
                f"{pattern['diagnostic_type']} system prompt to prevent this type of correction. "
                f"Be concrete and actionable.\n"
                f"Output only the rule text, nothing else."
            )
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            proposed_rule = response.content[0].text.strip()

            get_supabase().table("system_improvements").insert({
                "tenant_id": tenant_id,
                "diagnostic_type": pattern["diagnostic_type"],
                "proposed_rule": proposed_rule,
                "correction_count": pattern["count"],
                "correction_pattern": pattern,
                "status": "pending",
                "is_active": False,
            }).execute()

        except Exception as e:
            log.warning(f"Improvement proposal failed: {e}")


# ============================================================
# LEVEL 4: POPULATION ANALYTICS
# Gated at 50 completed engagements per tenant
# Analyzes patterns across all clients
# ============================================================

def run_population_analytics(tenant_id: str) -> dict:
    """
    Only runs when tenant has 50+ completed diagnostics.
    Returns population-level insights: most common risk flags,
    most recommended strategies, most common temporal windows.
    """
    supabase = get_supabase()
    count_result = (
        supabase.table("diagnostics")
        .select("id", count="exact")
        .eq("tenant_id", tenant_id)
        .eq("status", "complete")
        .execute()
    )

    total = count_result.count or 0
    if total < 50:
        return {"status": "gated", "completed_diagnostics": total, "required": 50}

    action_result = (
        supabase.table("action_items")
        .select("urgency,source,window_type")
        .eq("tenant_id", tenant_id)
        .execute()
    )

    items = action_result.data or []
    urgency_counts = {}
    window_counts = {}
    for item in items:
        u = item.get("urgency", "MEDIUM")
        urgency_counts[u] = urgency_counts.get(u, 0) + 1
        wt = item.get("window_type")
        if wt:
            window_counts[wt] = window_counts.get(wt, 0) + 1

    return {
        "status": "active",
        "completed_diagnostics": total,
        "urgency_distribution": urgency_counts,
        "top_planning_windows": sorted(
            window_counts.items(), key=lambda x: x[1], reverse=True
        )[:5],
    }


# ============================================================
# MAIN DIGEST RUNNER
# Called by POST /api/v1/intelligence-digest endpoint
# ============================================================

async def run_intelligence_digest(tenant_id: str, anthropic_key: str) -> dict:
    """
    Runs all 4 levels in sequence.
    Returns digest summary for the attorney.
    """
    log.info(f"Running intelligence digest for tenant {tenant_id}")

    # Level 1: Pattern detection
    patterns = run_pattern_recognition(tenant_id)

    # Level 2: Retrieval weight updates
    if patterns:
        update_retrieval_weights(tenant_id, patterns)

    # Level 3: Propose improvements
    if patterns and anthropic_key:
        propose_system_improvements(tenant_id, patterns, anthropic_key)

    # Level 4: Population analytics (gated)
    population = run_population_analytics(tenant_id)

    return {
        "tenant_id": tenant_id,
        "run_at": datetime.utcnow().isoformat(),
        "patterns_detected": len(patterns),
        "patterns": patterns,
        "improvements_proposed": len(patterns),
        "population_analytics": population,
    }
