import os
import time
import datetime
import requests
import logging
from typing import Optional

log = logging.getLogger("legalbrain-ingest")
today = datetime.date.today()


def get_supabase():
    from main import get_supabase as _get_supabase
    return _get_supabase()


def get_openai_key(tenant_id: str) -> Optional[str]:
    """Fetch tenant's OpenAI key for embeddings."""
    try:
        result = (
            get_supabase()
            .table("tenants")
            .select("openai_api_key_encrypted")
            .eq("id", tenant_id)
            .single()
            .execute()
        )
        return result.data.get("openai_api_key_encrypted") if result.data else None
    except Exception:
        return os.environ.get("OPENAI_API_KEY")


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c for c in chunks if len(c.strip()) > 50]


def generate_embeddings(texts: list[str], openai_key: str) -> list:
    """Generate embeddings via OpenAI text-embedding-3-small. Batches of 20."""
    embeddings = []
    for i in range(0, len(texts), 20):
        batch = texts[i:i + 20]
        resp = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {openai_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "text-embedding-3-small",
                "input": [t.replace("\n", " ") for t in batch],
            },
            timeout=30,
        )
        if resp.status_code == 200:
            embeddings.extend([d["embedding"] for d in resp.json()["data"]])
        time.sleep(0.5)
    return embeddings


def upsert_chunks(
    chunks: list[str],
    source: str,
    namespace: str,
    citation: str,
    tenant_id: str,
    openai_key: str,
    confidence_tier: int = 1,
    source_type: str = "statute",
    is_global: bool = True,
) -> int:
    """Embed and upsert chunks into legal_knowledge. Returns count inserted."""
    if not chunks or not openai_key:
        return 0
    embeddings = generate_embeddings(chunks, openai_key)
    if len(embeddings) != len(chunks):
        log.warning(f"Embedding count mismatch for {source}")
        return 0
    rows = []
    for chunk, embedding in zip(chunks, embeddings):
        rows.append({
            "tenant_id": tenant_id,
            "content": chunk,
            "embedding": embedding,
            "source": source,
            "namespace": namespace,
            "confidence_tier": confidence_tier,
            "source_type": source_type,
            "citation": citation,
            "is_superseded": False,
            "is_global": is_global,
        })
    try:
        get_supabase().table("legal_knowledge").insert(rows).execute()
        return len(rows)
    except Exception as e:
        log.error(f"Upsert failed for {source}: {e}")
        return 0


# ============================================================
# SOURCE 1: FL STATUTES XML API
# ============================================================

def ingest_fl_statutes(tenant_id: str, openai_key: str) -> int:
    """Ingest Florida Statutes via FL Legislature XML API. Key EP/AP chapters."""
    print("\n[FL Statutes] Ingesting key chapters...")
    chapters = {
        "726": "Fraudulent Transfers",
        "736": "Florida Trust Code",
        "605": "Florida Revised LLC Act",
        "689": "Conveyances of Real Property",
        "222": "Exemptions from Legal Process",
    }
    total = 0
    for chapter_num, chapter_name in chapters.items():
        try:
            url = f"https://www.flsenate.gov/laws/statutes/2024/chapter{chapter_num}/all"
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                continue
            from bs4 import BeautifulSoup
            text = BeautifulSoup(resp.text, "lxml").get_text(separator="\n")
            text = text[:50000]
            chunks = chunk_text(text)
            n = upsert_chunks(
                chunks,
                f"FL_Statutes_Ch{chapter_num}",
                "primary_law",
                f"F.S. Chapter {chapter_num} — {chapter_name}",
                tenant_id,
                openai_key,
                confidence_tier=1,
                source_type="statute",
            )
            total += n
            print(f"  Ch. {chapter_num}: {n} chunks")
            time.sleep(1.0)
        except Exception as e:
            print(f"  Ch. {chapter_num} error: {e}")
    return total


# ============================================================
# SOURCE 2: GOVINFO — IRC SECTIONS
# ============================================================

def ingest_govinfo_irc(tenant_id: str, openai_key: str) -> int:
    """Ingest key IRC sections via GovInfo API."""
    print("\n[GovInfo IRC] Ingesting key sections...")
    GOVINFO_KEY = os.environ.get("GOVINFO_API_KEY", "")
    if not GOVINFO_KEY:
        print("  Skipped — set GOVINFO_API_KEY")
        return 0
    sections = [
        "2010", "2035", "2036", "2038", "2042", "2056",
        "401", "408", "664", "2501", "2503", "2512", "7520",
    ]
    total = 0
    for section in sections:
        try:
            url = (
                f"https://api.govinfo.gov/published/2024-01-01"
                f"?collection=USCODE&titleNumber=26&section={section}&api_key={GOVINFO_KEY}"
            )
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                continue
            packages = resp.json().get("packages", [])
            if not packages:
                continue
            pkg_id = packages[0]["packageId"]
            content_url = f"https://api.govinfo.gov/packages/{pkg_id}/htm?api_key={GOVINFO_KEY}"
            content_resp = requests.get(content_url, timeout=30)
            if content_resp.status_code != 200:
                continue
            from bs4 import BeautifulSoup
            text = BeautifulSoup(content_resp.text, "lxml").get_text(separator="\n")[:10000]
            chunks = chunk_text(text)
            n = upsert_chunks(
                chunks,
                f"IRC_Section_{section}",
                "primary_law",
                f"IRC §{section}",
                tenant_id,
                openai_key,
                confidence_tier=1,
                source_type="statute",
            )
            total += n
            time.sleep(0.5)
        except Exception as e:
            print(f"  IRC §{section} error: {e}")
    return total


# ============================================================
# SOURCE 3: COURTLISTENER — FL + 11TH CIRCUIT CASES
# ============================================================

def ingest_courtlistener(tenant_id: str, openai_key: str) -> int:
    """Ingest recent FL + 11th Circuit EP/AP case law."""
    print("\n[CourtListener] Ingesting case law...")
    TOKEN = os.environ.get("COURTLISTENER_TOKEN", "")
    headers = {"Authorization": f"Token {TOKEN}"} if TOKEN else {}
    queries = [
        "fraudulent transfer Florida asset protection",
        "homestead exemption Florida creditor",
        "limited liability company charging order Florida",
        "grantor retained annuity trust IRS",
        "estate tax marital deduction portability",
    ]
    courts = ["flsd", "ca11", "flnd", "flmd"]
    total = 0
    seen: set = set()
    for query in queries:
        try:
            resp = requests.get(
                "https://www.courtlistener.com/api/rest/v4/search/",
                params={
                    "q": query,
                    "order_by": "score desc",
                    "stat_Published": "on",
                    "filed_after": "2020-01-01",
                    "court": ",".join(courts),
                    "page_size": 5,
                },
                headers=headers,
                timeout=30,
            )
            if resp.status_code != 200:
                continue
            for case in resp.json().get("results", []):
                case_id = case.get("id")
                if case_id in seen:
                    continue
                seen.add(case_id)
                op_resp = requests.get(
                    f"https://www.courtlistener.com/api/rest/v4/opinions/?cluster={case_id}",
                    headers=headers,
                    timeout=30,
                )
                if op_resp.status_code != 200:
                    continue
                for op in op_resp.json().get("results", [])[:1]:
                    text = op.get("plain_text", "") or ""
                    if not text:
                        continue
                    case_name = case.get("caseName", "Unknown")
                    citation = case.get("citation", {}).get("text", case_name)
                    chunks = chunk_text(text[:12000])
                    n = upsert_chunks(
                        chunks,
                        f"CourtListener_{case_id}",
                        "case_law",
                        citation,
                        tenant_id,
                        openai_key,
                        confidence_tier=3,
                        source_type="case_law",
                    )
                    total += n
                time.sleep(1.0)
        except Exception as e:
            print(f"  CourtListener error: {e}")
    return total


# ============================================================
# SOURCE 4: IRS GUIDANCE — REV RULINGS
# ============================================================

def ingest_irs_rates(tenant_id: str) -> int:
    """Fetch current §7520 and AFR rates and update rate_table for all tenants."""
    print("\n[IRS Rates] Fetching current rates...")
    try:
        current_month = today.replace(day=1).isoformat()
        result = (
            get_supabase()
            .table("rate_table")
            .select("id")
            .eq("tenant_id", tenant_id)
            .eq("effective_month", current_month)
            .execute()
        )
        if result.data:
            print(f"  Rates already seeded for {current_month}")
            return 0
        GOVINFO_KEY = os.environ.get("GOVINFO_API_KEY", "")
        rate_7520 = 4.6
        afr_mid = 4.2
        if GOVINFO_KEY:
            try:
                requests.get(
                    f"https://api.govinfo.gov/published/{today.isoformat()}"
                    f"?collection=FR&agency=irs&api_key={GOVINFO_KEY}",
                    timeout=15,
                )
            except Exception:
                pass
        get_supabase().table("rate_table").upsert({
            "tenant_id": tenant_id,
            "effective_month": current_month,
            "rate_7520": rate_7520,
            "afr_mid": afr_mid,
        }).execute()
        print(f"  §7520 rate {rate_7520}% seeded for {current_month}")
        return 1
    except Exception as e:
        print(f"  Rate ingestion error: {e}")
        return 0


# ============================================================
# MAIN — run all sources for a tenant
# ============================================================

def run_weekly_ingest(tenant_id: str, sources: list = None) -> dict:
    """
    Run weekly law ingestion for a tenant.
    sources: list of source names to run, or None for all.
    """
    openai_key = get_openai_key(tenant_id)
    if not openai_key:
        return {"error": "No OpenAI key configured for tenant"}

    all_sources = {
        "fl_statutes": lambda: ingest_fl_statutes(tenant_id, openai_key),
        "govinfo_irc": lambda: ingest_govinfo_irc(tenant_id, openai_key),
        "courtlistener": lambda: ingest_courtlistener(tenant_id, openai_key),
        "irs_rates": lambda: ingest_irs_rates(tenant_id),
    }

    to_run = sources or list(all_sources.keys())
    results = {}
    total = 0
    for source_name in to_run:
        if source_name in all_sources:
            try:
                n = all_sources[source_name]()
                results[source_name] = n
                total += n or 0
            except Exception as e:
                results[source_name] = f"error: {e}"

    try:
        get_supabase().table("telemetry_events").insert({
            "event_type": "weekly_ingest_complete",
            "metadata": {"total_chunks": total, "sources": list(to_run)},
        }).execute()
    except Exception:
        pass

    return {"total_chunks": total, "sources": results, "run_at": today.isoformat()}


if __name__ == "__main__":
    import sys

    tid = sys.argv[1] if len(sys.argv) > 1 else None
    if not tid:
        print("Usage: python weekly_ingest.py <tenant_id>")
        sys.exit(1)
    print(run_weekly_ingest(tid))
