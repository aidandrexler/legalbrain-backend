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


# ── ENDPOINT 4: RUN DIAGNOSTIC ──────────────────────────────────────────────

DIAGNOSTIC_SYSTEM_PROMPTS = {
"risk_architecture": """You are a legal Risk Architecture analyst for LegalBrain AI. Assess the client's complete creditor exposure and protection gaps.

MANDATORY CHECKS — run these regardless of client profile:
- FL homestead: declared? conversion history? 1,215-day bankruptcy rule applies to homestead acquired within that period
- Retirement beneficiaries: ERISA override? Ex-spouse still listed? SECURE Act 10-year rule applied?
- Life insurance: §2042 incidents of ownership? §2035 three-year rule exposure?
- Entity structure: PA = ZERO charging order protection in FL under F.S. §605.0503 — always flag
- Trust funding: signed trust with no retitled assets = critical failure — always flag

PHYSICIAN ADDITIONS (if is_physician=true):
- PA vs PLLC charging order gap — flag immediately if PA
- Payer concentration >70% from top 3 payers — flag as Gassman Founding Case Pattern
- Malpractice + umbrella adequacy
- Buy-sell existence and funding

OUTPUT FORMAT:
1. RISK MATRIX: Each asset — Severity(1-10) × Occurrence(1-10) × Detection(1-10) = RPN, Class RED/YELLOW/GREEN
2. CRITICAL FLAGS: Immediate action required
3. PRIORITY ACTIONS: Top 5 by RPN
4. FL EXEMPTIONS APPLIED: What protects each asset

FRAMING: Never use "asset protection" as stated purpose. Use "structural resilience" or "risk architecture".""",

"estate_tax_architecture": """You are an Estate and Tax Architecture analyst for LegalBrain AI.

OBBBA RULE: TCJA sunset permanently eliminated. Exemption permanent at $13.99M individual / $27.98M married. Never imply sunset urgency.
CONNELLY RULE: Life insurance proceeds inflate entity value in redemption buy-sells (2024 SCOTUS). Flag on every buy-sell review.
FL DAPT: FL has no DAPT statute. For DAPT → Nevada, South Dakota, Wyoming.
SECURE ACT: Most non-spouse beneficiaries → 10-year rule. Conduit vs accumulation trust analysis required when trust named as IRA beneficiary.

OUTPUT FORMAT:
1. ESTATE TAX EXPOSURE: Current vs exemption, projected growth, buffer
2. PROBATE VULNERABILITY: Trust funded? Beneficiary designations coordinated?
3. RANKED STRATEGY LIST: Explain why each strategy fits this client's specific goal weights
4. CONTRAINDICATIONS: Strategies excluded and why
5. IMMEDIATE ACTIONS: What to do first""",

"temporal_planning": """You are a Temporal Planning Intelligence analyst for LegalBrain AI.

You receive pre-computed temporal windows from the Python engine. Write narrative only — explain what each window means, why it matters, and what happens if the client misses it.

OUTPUT FORMAT:
1. OPEN WINDOWS: Each planning window currently open, urgency, plain-English explanation
2. CLOSING SOON: Windows expiring within 60 days with specific deadlines
3. RATE ENVIRONMENT: What current §7520 and AFR rates mean for strategy selection
4. CALENDAR: Specific dates the attorney needs to act by""",

"plan_integrity_audit": """You are a Plan Integrity Auditor for LegalBrain AI. Stress-test existing estate plans against current law.

DOCUMENT CHECKLIST:
- Trust signed but unfunded? → Critical failure
- POA current? (pre-2011 FL POA = outdated statute)
- Healthcare directive compliant with FL §765?
- Beneficiary designations reviewed post-SECURE Act?
- Buy-sell reviewed post-Connelly (2024 SCOTUS)?
- Any sunset language remaining? → Remove immediately (OBBBA permanent)
- Spendthrift provision present?
- Digital assets provision present (FL §740)?

ADVERSARIAL ANALYSIS:
- §2036 retained interest risks
- §2038 modification/revocation risks
- Fraudulent transfer vulnerabilities (timing, badges, insolvency at transfer)
- Connelly buy-sell exposure
- Reciprocal trust doctrine (SLAT pairs)

5-SCENARIO STRESS TEST:
A) Nuclear verdict ($3-5M)
B) Payer audit + recoupment (physician clients: always run)
C) Business failure + personal guarantee
D) Divorce — equitable distribution
E) IRS audit + tax liens + federal lien priority

OUTPUT FORMAT:
1. FINDINGS REPORT: Each finding with severity CRITICAL/HIGH/MEDIUM/LOW
2. CURRENT LAW COMPLIANCE: What is outdated and which law changed it
3. SCENARIO RESULTS: What survives each stress test
4. REMEDIATION PLAN: Specific fixes in priority order""",

"advisor_intelligence": """You are the Advisor Intelligence module for LegalBrain AI. Produce attorney-facing communication and research outputs.

For legal research: structure as:
ANSWER (1-2 sentence direct answer)
ANALYSIS (substantive explanation with citations)
PRACTICE NOTES (practical considerations and exceptions)
CITATIONS (all sources cited)

For meeting prep: CLIENT SNAPSHOT (3 sentences), OPEN ACTION ITEMS, PLANNING WINDOWS OPEN NOW, 5 TALKING POINTS (numbered), CLIENT NEXT STEPS

Be precise. Cite everything. Do not fabricate citations.""",
}


@app.post("/api/v1/run")
async def run_diagnostic(req: DiagnosticRequest):
    from pii_masker import mask_client_data, restore_pii
    from scenario_modeler import compute_fraud_badges, run_all_scenarios, run_red_flag_check
    from goal_weighting import compute_topsis
    import anthropic

    tenant = get_tenant(req.tenant_id)
    anthropic_key = get_tenant_anthropic_key(tenant)

    client_result = (
        get_supabase().table("clients").select("*").eq("id", req.client_id).single().execute()
    )
    if not client_result.data:
        raise HTTPException(status_code=404, detail="Client not found")
    client_data = client_result.data

    diag = get_supabase().table("diagnostics").insert({
        "tenant_id": req.tenant_id,
        "client_id": req.client_id,
        "diagnostic_type": req.diagnostic_type,
        "status": "running",
    }).execute()
    diag_id = diag.data[0]["id"]

    try:
        badge_result = None
        topsis_result = None
        scenario_result = None

        if req.include_badge_check or req.diagnostic_type == "risk_architecture":
            badge_result = compute_fraud_badges(client_data)

        if req.include_topsis or req.diagnostic_type == "estate_tax_architecture":
            goal_weights = client_data.get("goal_weights") or {}
            topsis_result = compute_topsis(client_data, goal_weights)

        if req.include_scenarios or req.diagnostic_type == "plan_integrity_audit":
            scenario_result = run_all_scenarios(client_data)

        tenant_rules = (
            get_supabase().table("extraction_rules")
            .select("*")
            .eq("tenant_id", req.tenant_id)
            .eq("rule_type", "red_flag")
            .eq("is_active", True)
            .execute()
            .data or []
        )
        red_flags = run_red_flag_check(client_data, tenant_rules)

        mask_result = mask_client_data(str(client_data))

        user_message = f"CLIENT PROFILE:\n{mask_result.masked_text}"
        if badge_result:
            user_message += f"\n\nBADGE ANALYSIS (FINAL — DO NOT RECALCULATE):\n{badge_result.claude_input}"
        if topsis_result:
            user_message += f"\n\nSTRATEGY RANKING (FINAL — DO NOT RE-RANK):\n{topsis_result}"
        if scenario_result:
            user_message += f"\n\nSCENARIO RESULTS (FINAL):\n{scenario_result}"
        if red_flags:
            user_message += f"\n\nRED FLAGS TRIGGERED:\n{red_flags}"

        approved_rules = (
            get_supabase().table("system_improvements")
            .select("final_rule")
            .eq("tenant_id", req.tenant_id)
            .eq("diagnostic_type", req.diagnostic_type)
            .eq("is_active", True)
            .execute()
            .data or []
        )
        system_prompt = DIAGNOSTIC_SYSTEM_PROMPTS.get(
            req.diagnostic_type, "You are a legal planning analyst."
        )
        if approved_rules:
            system_prompt += "\n\nATTORNEY-APPROVED RULES:\n" + "\n".join(
                r["final_rule"] for r in approved_rules if r.get("final_rule")
            )

        ai_client = anthropic.Anthropic(api_key=anthropic_key)
        response = ai_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        output = response.content[0].text
        output = restore_pii(output, mask_result.restore_map)

        get_supabase().table("diagnostics").update({
            "status": "complete",
            "output_text": output,
            "badge_result": badge_result.__dict__ if badge_result else None,
            "topsis_result": topsis_result.__dict__ if topsis_result else None,
            "scenario_result": scenario_result,
            "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
        }).eq("id", diag_id).execute()

        try:
            get_supabase().table("telemetry_events").insert({
                "event_type": "diagnostic_run",
                "diagnostic_type": req.diagnostic_type,
                "jurisdiction": client_data.get("primary_jurisdiction", "FL"),
            }).execute()
        except Exception:
            pass

        return {
            "success": True,
            "diagnostic_id": diag_id,
            "output": output,
            "diagnostic_type": req.diagnostic_type,
        }

    except Exception as e:
        get_supabase().table("diagnostics").update({"status": "failed"}).eq("id", diag_id).execute()
        raise HTTPException(status_code=500, detail=str(e))


# ── ENDPOINT 5: RESEARCH ─────────────────────────────────────────────────────

@app.post("/api/v1/research")
async def quick_research(req: ResearchRequest):
    from pii_masker import mask_client_data, restore_pii
    import anthropic
    import requests as req_lib

    tenant = get_tenant(req.tenant_id)
    anthropic_key = get_tenant_anthropic_key(tenant)
    openai_key = get_tenant_openai_key(tenant)

    embed_resp = req_lib.post(
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"},
        json={"model": "text-embedding-3-small", "input": req.question.replace("\n", " ")},
        timeout=30,
    )
    embedding = embed_resp.json()["data"][0]["embedding"]

    namespaces = req.namespaces or [
        "primary_law", "case_law", "secondary_law", "irs_guidance", "practitioner_patterns"
    ]
    results = get_supabase().rpc("hybrid_search_legal", {
        "p_tenant_id": req.tenant_id,
        "query_text": req.question,
        "query_embedding": embedding,
        "match_count": 12,
        "filter_namespaces": namespaces,
    }).execute()

    retrieved_chunks = results.data or []
    context = "\n\n".join(
        f"[{c['namespace'].upper()} — {c['citation']}]\n{c['content']}"
        for c in retrieved_chunks
    )

    mask_result = mask_client_data(req.question)
    user_message = (
        f"RESEARCH QUESTION: {mask_result.masked_text}\n\n"
        f"RELEVANT LAW FROM KNOWLEDGE BASE:\n{context}"
    )

    ai_client = anthropic.Anthropic(api_key=anthropic_key)
    response = ai_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=3000,
        system=DIAGNOSTIC_SYSTEM_PROMPTS["advisor_intelligence"],
        messages=[{"role": "user", "content": user_message}],
    )
    output = restore_pii(response.content[0].text, mask_result.restore_map)
    return {
        "success": True,
        "answer": output,
        "sources_retrieved": len(retrieved_chunks),
        "question": req.question,
    }


# ── ENDPOINT 6: INGEST ───────────────────────────────────────────────────────

@app.post("/api/v1/ingest")
async def ingest_knowledge(req: IngestRequest):
    import requests as req_lib

    tenant = get_tenant(req.tenant_id)
    openai_key = get_tenant_openai_key(tenant)

    chunk_size, overlap = 1200, 150
    chunks, start = [], 0
    while start < len(req.content):
        chunk = req.content[start: start + chunk_size]
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
        start += chunk_size - overlap

    if not chunks:
        raise HTTPException(status_code=400, detail="Content too short to ingest")

    embed_resp = req_lib.post(
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"},
        json={
            "model": "text-embedding-3-small",
            "input": [c.replace("\n", " ") for c in chunks],
        },
        timeout=60,
    )
    embeddings = [d["embedding"] for d in embed_resp.json()["data"]]

    if req.replaces_source:
        get_supabase().table("legal_knowledge").update({
            "is_superseded": True,
            "superseded_by": f"Replaced by {req.source_id}",
        }).eq("tenant_id", req.tenant_id).eq("source", req.replaces_source).execute()

    rows = [
        {
            "tenant_id": req.tenant_id,
            "content": chunk,
            "embedding": emb,
            "source": req.source_id,
            "source_title": req.description,
            "namespace": req.namespace,
            "confidence_tier": req.confidence_tier,
            "source_type": req.source_type,
            "citation": req.citation,
            "is_superseded": False,
            "is_global": False,
        }
        for chunk, emb in zip(chunks, embeddings)
    ]
    get_supabase().table("legal_knowledge").insert(rows).execute()
    return {"success": True, "chunks_created": len(rows), "source_id": req.source_id}


# ── ENDPOINT 7: TEMPORAL ─────────────────────────────────────────────────────

@app.post("/api/v1/temporal")
async def run_temporal(req: TemporalRequest):
    from temporal_engine import run_temporal_engine
    result = run_temporal_engine(get_supabase(), req.tenant_id, req.client_id)
    return {"success": True, **result}


# ── ENDPOINT 8: IMPROVE ──────────────────────────────────────────────────────

@app.post("/api/v1/improve")
async def handle_improvement(req: ImproveRequest):
    update: dict = {"status": req.decision, "reviewed_at": datetime.datetime.utcnow().isoformat()}
    if req.decision in ("approved", "edited_approved"):
        update["is_active"] = True
        update["final_rule"] = req.final_rule
    elif req.decision == "rejected":
        update["is_active"] = False
    get_supabase().table("system_improvements").update(update).eq(
        "id", req.improvement_id
    ).eq("tenant_id", req.tenant_id).execute()
    return {"success": True, "decision": req.decision}


# ── ENDPOINT 9: MATTER SYNC ──────────────────────────────────────────────────

@app.post("/api/v1/matter-sync")
async def matter_sync(req: MatterSyncRequest):
    provider = req.provider.lower()
    payload = req.payload
    client_data: dict = {"tenant_id": req.tenant_id, "client_status": "intake"}

    if provider == "clio":
        contact = payload.get("contact", {})
        matter = payload.get("matter", {})
        custom = payload.get("custom_fields", {})
        client_data.update({
            "clio_matter_id": str(matter["id"]) if matter.get("id") else None,
            "first_name": contact.get("first_name", ""),
            "last_name": contact.get("last_name", ""),
            "email": contact.get("primary_email_address"),
            "phone": contact.get("primary_phone_number"),
            "estate_size_estimate": custom.get("estate_value"),
            "is_physician": custom.get("is_physician", False),
        })
    elif provider == "leap":
        client_data.update({
            "leap_matter_id": str(payload["MatterId"]) if payload.get("MatterId") else None,
            "first_name": payload.get("ClientFirstName", ""),
            "last_name": payload.get("ClientLastName", ""),
            "email": payload.get("ClientEmail"),
        })

    existing = None
    if client_data.get("clio_matter_id"):
        existing = (
            get_supabase().table("clients").select("id")
            .eq("tenant_id", req.tenant_id)
            .eq("clio_matter_id", client_data["clio_matter_id"])
            .execute().data
        )
    elif client_data.get("leap_matter_id"):
        existing = (
            get_supabase().table("clients").select("id")
            .eq("tenant_id", req.tenant_id)
            .eq("leap_matter_id", client_data["leap_matter_id"])
            .execute().data
        )

    if existing:
        client_id = existing[0]["id"]
        get_supabase().table("clients").update(client_data).eq("id", client_id).execute()
        created = False
    else:
        result = get_supabase().table("clients").insert(client_data).execute()
        client_id = result.data[0]["id"]
        created = True

    return {"success": True, "client_id": client_id, "created": created, "provider": provider}


# ── ENDPOINT 10: INTELLIGENCE DIGEST ────────────────────────────────────────

@app.post("/api/v1/intelligence-digest")
async def intelligence_digest(tenant_id: str):
    from intelligence_digest import run_intelligence_digest
    tenant = get_tenant(tenant_id)
    anthropic_key = get_tenant_anthropic_key(tenant)
    result = await run_intelligence_digest(tenant_id, anthropic_key)
    return {"success": True, **result}


# ── OAUTH ENDPOINTS ──────────────────────────────────────────────────────────

from fastapi.responses import RedirectResponse


@app.get("/api/v1/oauth/clio/connect")
async def clio_connect(tenant_id: str):
    from oauth import get_clio_authorize_url
    url = get_clio_authorize_url(tenant_id)
    return RedirectResponse(url=url)


@app.get("/api/v1/oauth/clio/callback")
async def clio_callback(code: str, state: str):
    import requests as req_lib
    from oauth import store_oauth_token
    APP_URL = os.environ.get("APP_URL", "")
    resp = req_lib.post(
        "https://app.clio.com/oauth/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "client_id": os.environ.get("CLIO_CLIENT_ID", ""),
            "client_secret": os.environ.get("CLIO_CLIENT_SECRET", ""),
            "redirect_uri": f"{APP_URL}/api/v1/oauth/clio/callback",
        },
        timeout=30,
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=400, detail="Clio OAuth failed")
    store_oauth_token(state, "clio", resp.json())
    return RedirectResponse(url=f"{APP_URL}/settings?connected=clio")


@app.get("/api/v1/oauth/leap/connect")
async def leap_connect(tenant_id: str):
    from oauth import get_leap_authorize_url
    url = get_leap_authorize_url(tenant_id)
    return RedirectResponse(url=url)


@app.get("/api/v1/oauth/leap/callback")
async def leap_callback(code: str, state: str):
    import requests as req_lib
    from oauth import store_oauth_token
    APP_URL = os.environ.get("APP_URL", "")
    resp = req_lib.post(
        "https://api.leapaws.com/oauth2/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "client_id": os.environ.get("LEAP_CLIENT_ID", ""),
            "client_secret": os.environ.get("LEAP_CLIENT_SECRET", ""),
            "redirect_uri": f"{APP_URL}/api/v1/oauth/leap/callback",
        },
        timeout=30,
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=400, detail="LEAP OAuth failed")
    store_oauth_token(state, "leap", resp.json())
    return RedirectResponse(url=f"{APP_URL}/settings?connected=leap")


# ── SANDBOX PROVISION ENDPOINT ───────────────────────────────────────────────

@app.post("/api/v1/sandbox/provision")
async def provision_sandbox_endpoint(tenant_id: str):
    from sandbox import provision_sandbox
    result = await provision_sandbox(tenant_id)
    return result
