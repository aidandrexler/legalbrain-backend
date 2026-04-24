import json
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from typing import Optional

import anthropic
import openai

from main import get_supabase, get_tenant_keys

logger = logging.getLogger(__name__)

# ── Supabase helpers ────────────────────────────────────────────────────────

def fetch_job(job_id: str) -> dict:
    result = (
        get_supabase()
        .table("extraction_jobs")
        .select("*")
        .eq("id", job_id)
        .single()
        .execute()
    )
    return result.data


def _update_job(job_id: str, **fields) -> None:
    get_supabase().table("extraction_jobs").update(
        {**fields, "updated_at": datetime.now(timezone.utc).isoformat()}
    ).eq("id", job_id).execute()


# ── Stage 1: PDF text extraction ────────────────────────────────────────────

def extract_text_from_pdf(storage_path: str, tenant_id: str) -> str:
    sb = get_supabase()
    data = sb.storage.from_("sources").download(storage_path)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        from unstructured.partition.pdf import partition_pdf
        elements = partition_pdf(filename=tmp_path)
        return "\n\n".join(str(el) for el in elements if str(el).strip())
    finally:
        os.unlink(tmp_path)


# ── Stage 2: Prompt A — 14-section legal extraction ─────────────────────────

PROMPT_A_SYSTEM = """You are a senior Legal Knowledge Engineer building a deterministic
expert system for a legal practice. You extract structured legal rules, decision trees,
and operational logic from source material.

CRITICAL HALLUCINATION PREVENTION RULE — NON-NEGOTIABLE:
Every extraction operates ONLY on text the user has provided. Never infer, assume, or add
legal rules from training data. If the source text does not say it, do not include it.
Flag gaps: [RULE NOT IN SOURCE TEXT — verify before using]"""

PROMPT_A_USER = """Extract all legal logic from this source material.

Source: {source_title}
Jurisdiction: {jurisdiction}
Authority Level: {authority_level}

Produce the following 14 sections:

I. BLACK LETTER RULES — every primary rule as a precise declarative sentence with exact citation
II. STATUTORY/CASE TRIGGERS — every factual condition that must be true, with burden of proof
III. EXCEPTIONS AND FAILURE MODES — every exception, piercing vulnerability, disqualifying event
IV. JURISDICTIONAL NUANCE — state-specific variations, federal preemption, choice of law
V. DRAFTING DIRECTIVES — every clause the source states MUST or SHOULD be in documents
VI. CONTRAINDICATIONS — client profiles where this strategy is dangerous
VII. CLIENT BEHAVIORAL FAILURE PATTERNS — what clients do that destroys structures
VIII. OPPOSING COUNSEL ATTACKS — how a plaintiff\u2019s attorney attacks these structures
IX. TIMING DEPENDENCIES — optimal windows, too-late consequences, critical deadlines
X. ADVISOR COORDINATION — specific actions required of CPAs, trustees, advisors

XI. AUTHORITY MAP \u2014 format as JSON:
{{"controlling_law": [], "persuasive_authority": [], "commentary_only": [], "superseded_by": [], "conflict_notes": ""}}

XII. DECISION TREE NODES \u2014 for every if/then logic pattern, produce node objects:
{{"node_id": "D001", "question": "...", "yes_path": "...", "no_path": "...", "citations": [], "required_facts": []}}
Minimum 5 nodes per major topic area.

XIII. REQUIRED CLIENT FACTS \u2014 flat list of every client data field needed to trigger any rule:
["entity_type", "homestead_declared", "trust_funded", ...]

XIV. RED FLAG RULES \u2014 facts that immediately require attorney escalation:
{{"flag": "CRITICAL|HIGH|MEDIUM", "condition": "...", "message": "...", "citation": "..."}}

Source material:
{source_text}"""


def run_prompt_a(
    text: str,
    source_title: str,
    jurisdiction: str,
    authority_level: str,
    anthropic_key: str,
) -> str:
    client = anthropic.Anthropic(api_key=anthropic_key)
    user_msg = PROMPT_A_USER.format(
        source_title=source_title,
        jurisdiction=jurisdiction,
        authority_level=authority_level,
        source_text=text[:150_000],  # stay within context window
    )
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=16000,
        system=PROMPT_A_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    return message.content[0].text


# ── Stage 3: Prompt B — formal decision trees ───────────────────────────────

PROMPT_B_SYSTEM = """You are a senior Legal Knowledge Engineer building formal decision trees
from extracted legal rules. Convert unstructured legal logic into precise if/then branch structures."""

PROMPT_B_USER = """Using the Prompt A extraction below, build formal decision trees.

For each major planning question this source addresses, produce a complete decision tree
in this format:
DECISION TREE: [Topic]
ROOT QUESTION: [The threshold question]
\u251c\u2500\u2500 YES \u2192 [Next question or conclusion]
\u2502   \u251c\u2500\u2500 YES \u2192 [Conclusion with citation]
\u2502   \u2514\u2500\u2500 NO  \u2192 [Alternative path]
\u2514\u2500\u2500 NO  \u2192 [Alternative path or dead end]

Include at minimum:
- Eligibility decision trees (who qualifies for this protection)
- Implementation decision trees (how to implement correctly)
- Attack/defense decision trees (how creditors attack, how to defend)
- Timing decision trees (when to act, when too late)

Prompt A extraction:
{prompt_a_output}"""


def run_prompt_b(prompt_a_output: str, anthropic_key: str) -> str:
    client = anthropic.Anthropic(api_key=anthropic_key)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8000,
        system=PROMPT_B_SYSTEM,
        messages=[{"role": "user", "content": PROMPT_B_USER.format(prompt_a_output=prompt_a_output)}],
    )
    return message.content[0].text


# ── Prompt C — JSONLogic translation (Tier 1 only) ──────────────────────────

PROMPT_C_SYSTEM = """You are a Legal Logic Engineer. Convert legal decision rules into
machine-executable JSONLogic expressions."""

PROMPT_C_USER = """Convert the key decision nodes from the extraction below into JSONLogic format.

For each decision rule, produce:
{{
  "rule_id": "R001",
  "description": "Human-readable description",
  "logic": {{JSONLogic expression}},
  "conclusion_true": "What to do when condition is true",
  "conclusion_false": "What to do when condition is false",
  "citation": "F.S. \u00a7XXX"
}}

Extraction:
{prompt_a_output}"""


def run_prompt_c(prompt_a_output: str, anthropic_key: str) -> str:
    client = anthropic.Anthropic(api_key=anthropic_key)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8000,
        system=PROMPT_C_SYSTEM,
        messages=[{"role": "user", "content": PROMPT_C_USER.format(prompt_a_output=prompt_a_output)}],
    )
    return message.content[0].text


# ── Prompt D — Clause library (Tier 1 only) ──────────────────────────────────

PROMPT_D_SYSTEM = """You are a Legal Drafting Engineer. Extract every clause, provision,
and document language recommendation from the source material."""

PROMPT_D_USER = """Extract the clause library from this source.

For every clause, provision, or document language the source recommends, produce:
{{
  "clause_id": "CL001",
  "clause_type": "spendthrift|no_contest|governing_law|trustee_succession|...",
  "use_case": "When this clause is needed",
  "jurisdiction": "FL",
  "statutory_basis": "F.S. \u00a7XXX",
  "text": "The actual clause language if provided in the source",
  "contraindications": ["Self-settled trusts", "Medicaid planning context"],
  "approved": false
}}

Source extraction:
{prompt_a_output}"""


def run_prompt_d(prompt_a_output: str, anthropic_key: str) -> str:
    client = anthropic.Anthropic(api_key=anthropic_key)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8000,
        system=PROMPT_D_SYSTEM,
        messages=[{"role": "user", "content": PROMPT_D_USER.format(prompt_a_output=prompt_a_output)}],
    )
    return message.content[0].text


# ── Bundle assembly ──────────────────────────────────────────────────────────

def _extract_json_objects(text: str, section_marker: str) -> list[dict]:
    """Pull JSON objects from a numbered section of Claude output."""
    idx = text.find(section_marker)
    if idx == -1:
        return []
    section_text = text[idx:]
    # Trim at the next Roman-numeral section header
    nxt = re.search(r"\n[IVX]+\.", section_text[len(section_marker):])
    if nxt:
        section_text = section_text[: len(section_marker) + nxt.start()]

    results = []
    for m in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}", section_text, re.DOTALL):
        try:
            results.append(json.loads(m.group()))
        except json.JSONDecodeError:
            continue
    return results


def _extract_authority_map(text: str) -> dict:
    m = re.search(r"XI\.[^\{]*(\{[\s\S]*?\})", text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return {}


def _extract_required_facts(text: str) -> list[str]:
    idx = text.find("XIII.")
    if idx == -1:
        return []
    snippet = text[idx: idx + 2000]
    nxt = re.search(r"\n[IVX]+\.", snippet[5:])
    if nxt:
        snippet = snippet[: 5 + nxt.start()]
    return re.findall(r'"([a-z_][a-z_0-9]*)"', snippet)


def build_bundle(
    job: dict,
    prompt_a_output: str,
    prompt_b_output: Optional[str],
    prompt_c_output: Optional[str] = None,
    prompt_d_output: Optional[str] = None,
) -> dict:
    return {
        "metadata": {
            "job_id": job["id"],
            "tenant_id": job["tenant_id"],
            "source_title": job["source_title"],
            "source_type": job["source_type"],
            "authority_level": job["authority_level"],
            "confidence_tier": job["confidence_tier"],
            "jurisdiction": job.get("jurisdiction", "FL"),
            "user_source_id": job.get("user_source_id"),
        },
        "prompt_a": {
            "raw": prompt_a_output,
            "authority_map": _extract_authority_map(prompt_a_output),
            "decision_tree_nodes": _extract_json_objects(prompt_a_output, "XII."),
            "required_client_facts": _extract_required_facts(prompt_a_output),
            "red_flag_rules": _extract_json_objects(prompt_a_output, "XIV."),
        },
        "prompt_b": {"raw": prompt_b_output} if prompt_b_output else None,
        "prompt_c": {"raw": prompt_c_output} if prompt_c_output else None,
        "prompt_d": {"raw": prompt_d_output} if prompt_d_output else None,
    }


# ── Chunking + ingestion ─────────────────────────────────────────────────────

_NAMESPACE_MAP = {
    "primary_law": "primary_law",
    "major_treatise": "secondary_law",
    "cle": "secondary_law",
    "case_law": "case_law",
    "practitioner": "practitioner_patterns",
}


def _chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start: start + chunk_size])
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]


def chunk_and_ingest(bundle: dict, job_id: str, tenant_id: str, openai_key: str) -> int:
    sb = get_supabase()
    oa = openai.OpenAI(api_key=openai_key)
    meta = bundle["metadata"]
    namespace = _NAMESPACE_MAP.get(meta["authority_level"], "secondary_law")

    # Combine all prompt outputs into one corpus
    corpus_parts = [bundle["prompt_a"]["raw"]]
    for key in ("prompt_b", "prompt_c", "prompt_d"):
        if bundle.get(key) and bundle[key].get("raw"):
            corpus_parts.append(bundle[key]["raw"])
    full_text = "\n\n".join(corpus_parts)

    chunks = _chunk_text(full_text)
    total_inserted = 0
    BATCH = 20

    for i in range(0, len(chunks), BATCH):
        batch = chunks[i: i + BATCH]
        emb_resp = oa.embeddings.create(model="text-embedding-3-small", input=batch)
        rows = [
            {
                "tenant_id": tenant_id,
                "extraction_job_id": job_id,
                "content": batch[j],
                "embedding": obj.embedding,
                "source": meta.get("user_source_id") or meta["source_title"],
                "source_title": meta["source_title"],
                "namespace": namespace,
                "confidence_tier": meta["confidence_tier"],
                "source_type": meta["source_type"],
                "jurisdiction": meta["jurisdiction"],
                "topic_tags": [],
            }
            for j, obj in enumerate(emb_resp.data)
        ]
        sb.table("legal_knowledge").insert(rows).execute()
        total_inserted += len(batch)

    # Persist decision tree nodes into extraction_rules
    for node in bundle["prompt_a"].get("decision_tree_nodes", []):
        try:
            sb.table("extraction_rules").insert({
                "tenant_id": tenant_id,
                "extraction_job_id": job_id,
                "rule_id": node.get("node_id", f"DN_{total_inserted}"),
                "rule_type": "decision_node",
                "question": node.get("question"),
                "yes_path": node.get("yes_path"),
                "no_path": node.get("no_path"),
                "citations": node.get("citations", []),
                "required_facts": node.get("required_facts", []),
                "jurisdiction": meta["jurisdiction"],
                "source": meta["source_title"],
            }).execute()
        except Exception as exc:
            logger.warning("Decision node insert failed (%s): %s", node.get("node_id"), exc)

    # Persist red flag rules into extraction_rules
    for idx, flag in enumerate(bundle["prompt_a"].get("red_flag_rules", [])):
        try:
            sb.table("extraction_rules").insert({
                "tenant_id": tenant_id,
                "extraction_job_id": job_id,
                "rule_id": f"RF_{idx:04d}",
                "rule_type": "red_flag",
                "flag_level": flag.get("flag", "HIGH"),
                "condition_field": flag.get("condition"),
                "alert_message": flag.get("message"),
                "citations": [flag["citation"]] if flag.get("citation") else [],
                "jurisdiction": meta["jurisdiction"],
                "source": meta["source_title"],
            }).execute()
        except Exception as exc:
            logger.warning("Red flag insert failed: %s", exc)

    return total_inserted


# ── Conflict detection ───────────────────────────────────────────────────────

def run_conflict_detection(
    tenant_id: str,
    new_chunk_ids: list[str],
    confidence_tier: int,
    source_title: str,
) -> int:
    """
    For each new chunk, find similar existing chunks (cosine > 0.85) and ask Claude
    whether the new text contradicts or supersedes the old text. Marks superseded
    chunks with is_superseded=True.
    Returns count of chunks marked superseded.
    """
    sb = get_supabase()
    keys = get_tenant_keys(tenant_id)
    anthropic_key = keys.get("anthropic_api_key")
    if not anthropic_key or not new_chunk_ids:
        return 0

    client = anthropic.Anthropic(api_key=anthropic_key)
    superseded_count = 0

    for chunk_id in new_chunk_ids:
        try:
            new_row = (
                sb.table("legal_knowledge")
                .select("content, embedding")
                .eq("id", chunk_id)
                .single()
                .execute()
            )
            if not new_row.data:
                continue
            new_content = new_row.data["content"]
            new_embedding = new_row.data["embedding"]

            similar = sb.rpc("hybrid_search_legal", {
                "p_tenant_id": tenant_id,
                "query_text": new_content[:500],
                "query_embedding": new_embedding,
                "match_count": 5,
                "alpha": 1.0,
                "beta": 0.0,
            }).execute()

            for existing in (similar.data or []):
                if existing["id"] == chunk_id:
                    continue
                if (existing.get("hybrid_score") or 0) < 0.85:
                    continue

                response = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=10,
                    messages=[{"role": "user", "content": (
                        f"Old text:\n{existing['content']}\n\n"
                        f"New text:\n{new_content}\n\n"
                        f"Does the new text directly contradict or update the old text? "
                        f"Answer YES or NO only."
                    )}],
                )
                if response.content[0].text.strip().upper().startswith("YES"):
                    sb.table("legal_knowledge").update({
                        "is_superseded": True,
                        "superseded_by": f"{chunk_id} — Superseded by: {source_title}",
                    }).eq("id", existing["id"]).execute()
                    superseded_count += 1

        except Exception as exc:
            logger.warning("Conflict check failed for chunk %s: %s", chunk_id, exc)

    return superseded_count


# ── Main orchestrator ────────────────────────────────────────────────────────

async def process_job(job_id: str) -> None:
    sb = get_supabase()

    def update(status: str, pct: int, step: str = "", **extra) -> None:
        sb.table("extraction_jobs").update(
            {"status": status, "progress_pct": pct, "current_step": step,
             "updated_at": datetime.now(timezone.utc).isoformat(), **extra}
        ).eq("id", job_id).execute()

    try:
        job = fetch_job(job_id)
        tenant_id = job["tenant_id"]
        keys = get_tenant_keys(tenant_id)
        anthropic_key = keys["anthropic_api_key"]
        openai_key = keys["openai_api_key"]

        if not anthropic_key:
            raise ValueError("No Anthropic API key configured for this tenant.")
        if not openai_key:
            raise ValueError("No OpenAI API key configured for this tenant.")

        # Stage 1 — extract text from PDF if needed
        raw_text: str = job.get("raw_text") or ""
        if not raw_text and job.get("storage_path"):
            update("extracting_text", 5, "Downloading and extracting PDF text...")
            raw_text = extract_text_from_pdf(job["storage_path"], tenant_id)
            sb.table("extraction_jobs").update(
                {"raw_text": raw_text, "updated_at": datetime.now(timezone.utc).isoformat()}
            ).eq("id", job_id).execute()
            # Delete source file immediately after text is extracted
            sb.storage.from_("sources").remove([job["storage_path"]])
            sb.table("extraction_jobs").update(
                {"storage_path": None, "updated_at": datetime.now(timezone.utc).isoformat()}
            ).eq("id", job_id).execute()

        if not raw_text:
            raise ValueError("No text to process — provide raw_text or a valid storage_path.")

        # Stage 2 — Prompt A (always)
        update("running_prompt_a", 20, "Extracting rules and decision trees (Prompt A)...")
        prompt_a = run_prompt_a(
            text=raw_text,
            source_title=job["source_title"],
            jurisdiction=job.get("jurisdiction", "FL"),
            authority_level=job.get("authority_level", "secondary_law"),
            anthropic_key=anthropic_key,
        )

        # Stage 3 — Prompt B (tier 1 or 2)
        prompt_b: Optional[str] = None
        if job.get("run_prompt_b", True):
            update("running_prompt_b", 45, "Building formal decision trees (Prompt B)...")
            prompt_b = run_prompt_b(prompt_a, anthropic_key)

        # Stage 4 — Prompt C (tier 1 only)
        prompt_c: Optional[str] = None
        if job.get("run_prompt_c", False):
            update("running_prompt_c", 60, "Generating JSONLogic rules (Prompt C)...")
            prompt_c = run_prompt_c(prompt_a, anthropic_key)

        # Stage 5 — Prompt D (tier 1 only, clause content)
        prompt_d: Optional[str] = None
        if job.get("run_prompt_d", False):
            update("running_prompt_d", 72, "Extracting clause library (Prompt D)...")
            prompt_d = run_prompt_d(prompt_a, anthropic_key)

        # Stage 6 — assemble bundle
        update("building_bundle", 80, "Assembling extraction bundle...")
        bundle = build_bundle(job, prompt_a, prompt_b, prompt_c, prompt_d)
        sb.table("extraction_jobs").update(
            {"bundle_json": bundle, "updated_at": datetime.now(timezone.utc).isoformat()}
        ).eq("id", job_id).execute()

        # Stage 7 — chunk and ingest into pgvector
        update("ingesting", 85, "Generating embeddings and ingesting into knowledge base...")
        chunks_created = chunk_and_ingest(bundle, job_id, tenant_id, openai_key)

        # Conflict detection against existing knowledge base
        try:
            new_chunks_result = (
                sb.table("legal_knowledge")
                .select("id")
                .eq("extraction_job_id", job_id)
                .execute()
            )
            new_chunk_ids = [row["id"] for row in (new_chunks_result.data or [])]
            run_conflict_detection(
                tenant_id, new_chunk_ids, job.get("confidence_tier", 2), job["source_title"]
            )
        except Exception as exc:
            logger.warning("Conflict detection failed: %s", exc)

        # Telemetry
        try:
            sb.table("telemetry_events").insert({
                "event_type": "extraction_complete",
                "metadata": {
                    "source_type": job["source_type"],
                    "confidence_tier": job.get("confidence_tier", 2),
                    "chunks_created": chunks_created,
                    "jurisdiction": job.get("jurisdiction", "FL"),
                },
            }).execute()
        except Exception as exc:
            logger.warning("Telemetry insert failed: %s", exc)

        update("complete", 100, "Extraction complete.", chunks_created=chunks_created)
        logger.info("Job %s complete — %d chunks ingested.", job_id, chunks_created)

    except Exception as exc:
        logger.exception("Extraction job %s failed.", job_id)
        sb.table("extraction_jobs").update({
            "status": "failed",
            "error_message": str(exc),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", job_id).execute()
