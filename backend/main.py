import os
import logging
import datetime
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("legalbrain")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

_supabase_client = None
_openai_client = None


def get_supabase():
    global _supabase_client
    if _supabase_client is None:
        from supabase import create_client
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return _supabase_client


def get_tenant(tenant_id: str) -> dict:
    """Fetch tenant row including encrypted API keys."""
    result = get_supabase().table("tenants").select("*").eq("id", tenant_id).single().execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Tenant not found")
    return result.data


def get_tenant_anthropic_key(tenant: dict) -> str:
    key = tenant.get("anthropic_api_key_encrypted") or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise HTTPException(status_code=400, detail="No Anthropic API key configured. Add your key in Settings.")
    return key


def get_tenant_openai_key(tenant: dict) -> str:
    key = tenant.get("openai_api_key_encrypted") or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise HTTPException(status_code=400, detail="No OpenAI API key configured. Add your key in Settings.")
    return key


# kept for backwards-compat with modules that import get_tenant_keys
def get_tenant_keys(tenant_id: str) -> dict:
    tenant = get_tenant(tenant_id)
    return {
        "anthropic_api_key": tenant.get("anthropic_api_key_encrypted"),
        "openai_api_key": tenant.get("openai_api_key_encrypted"),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    from pii_masker import get_presidio
    get_presidio()
    log.info("LegalBrain AI orchestrator starting...")
    yield
    log.info("Shutting down.")


app = FastAPI(title="LegalBrain AI", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── MODELS ──────────────────────────────────────────────────────────────────

class DiagnosticRequest(BaseModel):
    tenant_id: str
    diagnostic_type: str
    client_id: str
    include_badge_check: bool = False
    include_topsis: bool = False
    include_scenarios: bool = False


class ResearchRequest(BaseModel):
    tenant_id: str
    question: str
    client_id: Optional[str] = None
    namespaces: Optional[list] = None


class IngestRequest(BaseModel):
    tenant_id: str
    content: str
    source_id: str
    description: str
    namespace: str
    confidence_tier: int = 3
    source_type: str
    citation: str
    replaces_source: Optional[str] = None


class TemporalRequest(BaseModel):
    tenant_id: str
    client_id: Optional[str] = None


class MatterSyncRequest(BaseModel):
    tenant_id: str
    provider: str
    payload: dict


class ImproveRequest(BaseModel):
    tenant_id: str
    improvement_id: str
    decision: str
    final_rule: Optional[str] = None


class ExtractionRequest(BaseModel):
    tenant_id: str
    source_title: str
    source_type: str
    authority_level: str = "secondary_law"
    confidence_tier: int = 2
    jurisdiction: str = "FL"
    user_source_id: Optional[str] = None
    tier: int = 2
    text: Optional[str] = None
    storage_path: Optional[str] = None


# ── ENDPOINT 1: HEALTH ──────────────────────────────────────────────────────

@app.get("/api/v1/health")
async def health_check():
    checks = {}
    try:
        result = get_supabase().table("tenants").select("id", count="exact").limit(1).execute()
        chunks = get_supabase().table("legal_knowledge").select("id", count="exact").execute()
        checks["supabase"] = {"status": "ok", "tenants": result.count or 0}
        checks["pgvector"] = {"status": "ok", "chunks": chunks.count or 0}
    except Exception as e:
        checks["supabase"] = {"status": "error", "detail": str(e)}
    try:
        rate = (
            get_supabase()
            .table("rate_table")
            .select("rate_7520")
            .order("effective_month", desc=True)
            .limit(1)
            .execute()
        )
        rate_val = rate.data[0]["rate_7520"] if rate.data else None
        checks["rate_table"] = {"status": "ok", "rate_7520": rate_val}
    except Exception as e:
        checks["rate_table"] = {"status": "error", "detail": str(e)}
    overall = "healthy" if all(v.get("status") == "ok" for v in checks.values()) else "degraded"
    return {
        "status": overall,
        "version": "2.0.0",
        "checks": checks,
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }


# ── ENDPOINT 2: EXTRACT (queue job) ─────────────────────────────────────────

@app.post("/api/v1/extract")
async def start_extraction(req: ExtractionRequest):
    import asyncio
    from extraction import process_job

    job = get_supabase().table("extraction_jobs").insert({
        "tenant_id": req.tenant_id,
        "source_title": req.source_title,
        "source_type": req.source_type,
        "authority_level": req.authority_level,
        "confidence_tier": req.confidence_tier,
        "jurisdiction": req.jurisdiction,
        "user_source_id": req.user_source_id,
        "tier": req.tier,
        "raw_text": req.text,
        "storage_path": req.storage_path,
        "run_prompt_b": req.tier <= 2,
        "run_prompt_c": req.tier == 1,
        "run_prompt_d": req.tier == 1,
        "status": "pending",
    }).execute()
    job_id = job.data[0]["id"]
    asyncio.create_task(process_job(job_id))
    return {"success": True, "job_id": job_id, "status": "pending"}


# ── ENDPOINT 3: EXTRACTION STATUS ───────────────────────────────────────────

@app.get("/api/v1/extract/{job_id}/status")
async def extraction_status(job_id: str):
    result = (
        get_supabase()
        .table("extraction_jobs")
        .select("status,progress_pct,current_step,chunks_created,error_message")
        .eq("id", job_id)
        .single()
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="Job not found")
    return result.data
