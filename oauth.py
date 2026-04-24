import os
import datetime
import requests
from fastapi import HTTPException


def get_supabase():
    from main import get_supabase as _get
    return _get()


def encrypt_token(token: str) -> str:
    """
    Base64 encoding as placeholder — replace with Fernet symmetric encryption
    before production. Add: pip install cryptography, use ENCRYPTION_KEY env var.
    """
    import base64
    return base64.b64encode(token.encode()).decode()


def decrypt_token(token: str) -> str:
    import base64
    return base64.b64decode(token.encode()).decode()


def store_oauth_token(tenant_id: str, provider: str, token_response: dict):
    expires_in = token_response.get("expires_in", 3600)
    expires_at = (
        datetime.datetime.utcnow() + datetime.timedelta(seconds=expires_in)
    ).isoformat()
    get_supabase().table("oauth_tokens").upsert({
        "tenant_id": tenant_id,
        "provider": provider,
        "access_token": encrypt_token(token_response["access_token"]),
        "refresh_token": encrypt_token(token_response["refresh_token"])
            if token_response.get("refresh_token") else None,
        "expires_at": expires_at,
        "scope": token_response.get("scope"),
        "updated_at": datetime.datetime.utcnow().isoformat(),
    }).execute()


def get_valid_token(tenant_id: str, provider: str) -> str:
    result = (
        get_supabase().table("oauth_tokens")
        .select("*")
        .eq("tenant_id", tenant_id)
        .eq("provider", provider)
        .single()
        .execute()
    )
    if not result.data:
        raise HTTPException(
            status_code=401, detail=f"No {provider} connection. Connect in Settings."
        )
    token_row = result.data
    expires_at = datetime.datetime.fromisoformat(token_row["expires_at"].replace("Z", ""))
    if expires_at < datetime.datetime.utcnow() + datetime.timedelta(minutes=5):
        return refresh_token(tenant_id, provider, decrypt_token(token_row["refresh_token"]))
    return decrypt_token(token_row["access_token"])


def refresh_token(tenant_id: str, provider: str, refresh_tok: str) -> str:
    CLIO_CLIENT_ID = os.environ.get("CLIO_CLIENT_ID", "")
    CLIO_CLIENT_SECRET = os.environ.get("CLIO_CLIENT_SECRET", "")
    LEAP_CLIENT_ID = os.environ.get("LEAP_CLIENT_ID", "")
    LEAP_CLIENT_SECRET = os.environ.get("LEAP_CLIENT_SECRET", "")

    if provider == "clio":
        resp = requests.post(
            "https://app.clio.com/oauth/token",
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_tok,
                "client_id": CLIO_CLIENT_ID,
                "client_secret": CLIO_CLIENT_SECRET,
            },
            timeout=30,
        )
    elif provider == "leap":
        # LEAP token endpoint — verify against current LEAP API docs before going live
        resp = requests.post(
            "https://api.leapaws.com/oauth2/token",
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_tok,
                "client_id": LEAP_CLIENT_ID,
                "client_secret": LEAP_CLIENT_SECRET,
            },
            timeout=30,
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")

    if resp.status_code != 200:
        raise HTTPException(status_code=401, detail=f"Token refresh failed for {provider}")
    token_response = resp.json()
    store_oauth_token(tenant_id, provider, token_response)
    return token_response["access_token"]


def get_clio_authorize_url(tenant_id: str) -> str:
    CLIO_CLIENT_ID = os.environ.get("CLIO_CLIENT_ID", "")
    APP_URL = os.environ.get("APP_URL", "")
    return (
        f"https://app.clio.com/oauth/authorize"
        f"?response_type=code&client_id={CLIO_CLIENT_ID}"
        f"&redirect_uri={APP_URL}/api/v1/oauth/clio/callback"
        f"&scope=contacts:read+matters:read+matters:write+custom_fields:read"
        f"&state={tenant_id}"
    )


def get_leap_authorize_url(tenant_id: str) -> str:
    LEAP_CLIENT_ID = os.environ.get("LEAP_CLIENT_ID", "")
    APP_URL = os.environ.get("APP_URL", "")
    return (
        f"https://api.leapaws.com/oauth2/authorize"
        f"?response_type=code&client_id={LEAP_CLIENT_ID}"
        f"&redirect_uri={APP_URL}/api/v1/oauth/leap/callback"
        f"&state={tenant_id}"
    )
