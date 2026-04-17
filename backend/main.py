import os
import logging

import anthropic
import httpx
import openai
import uvicorn
from fastapi import FastAPI
from supabase import Client, create_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LegalBrain AI", version="2.0.0")

_supabase_client: Client | None = None


def get_supabase() -> Client:
    global _supabase_client
    if _supabase_client is None:
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
        _supabase_client = create_client(url, key)
    return _supabase_client


def get_tenant_keys(tenant_id: str) -> dict:
    sb = get_supabase()
    result = (
        sb.table("tenants")
        .select("anthropic_api_key_encrypted, openai_api_key_encrypted")
        .eq("id", tenant_id)
        .single()
        .execute()
    )
    data = result.data or {}
    return {
        "anthropic_api_key": data.get("anthropic_api_key_encrypted"),
        "openai_api_key": data.get("openai_api_key_encrypted"),
    }


@app.get("/api/v1/health")
async def health():
    return {"status": "healthy", "version": "2.0.0"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
