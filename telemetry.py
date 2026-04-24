def get_supabase():
    from main import get_supabase as _get
    return _get()


def log_event(
    event_type: str,
    diagnostic_type: str = None,
    strategy_key: str = None,
    flag_level: str = None,
    jurisdiction: str = None,
    metadata: dict = None,
):
    """
    Log anonymized telemetry event. No tenant_id, no client_id, no PII ever.
    Wrapped in try/except so telemetry failure never breaks the app.
    """
    try:
        get_supabase().table("telemetry_events").insert({
            "event_type": event_type,
            "diagnostic_type": diagnostic_type,
            "strategy_key": strategy_key,
            "flag_level": flag_level,
            "jurisdiction": jurisdiction or "FL",
            "metadata": metadata or {},
        }).execute()
    except Exception:
        pass
