const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

const ANTHROPIC_API_KEY = Deno.env.get("ANTHROPIC_API_KEY")!;
const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;

const SYSTEM_PROMPTS: Record<string, string> = {
  risk_architecture: `You are a legal Risk Architecture analyst for LegalBrain AI. Assess the client's complete creditor exposure and protection gaps.

MANDATORY CHECKS:
- FL homestead: declared? conversion history? 1,215-day bankruptcy rule?
- Retirement beneficiaries: ERISA override? Ex-spouse listed? SECURE Act 10-year rule?
- Life insurance: section 2042 incidents of ownership? Section 2035 three-year rule?
- Entity structure: PA = ZERO charging order protection in FL under F.S. section 605.0503 — always flag
- Trust funding: signed trust with no retitled assets = critical failure — always flag

PHYSICIAN ADDITIONS (if is_physician):
- PA vs PLLC charging order gap — flag immediately
- Payer concentration greater than 70% from top 3 payers — flag as critical
- Malpractice and umbrella adequacy
- Buy-sell existence and funding

OUTPUT FORMAT:
1. RISK MATRIX: Each asset — Severity x Occurrence x Detection = RPN, Class RED/YELLOW/GREEN
2. CRITICAL FLAGS: Immediate action required
3. PRIORITY ACTIONS: Top 5 by RPN
4. FL EXEMPTIONS APPLIED

FRAMING: Never use "asset protection" as stated purpose. Use "structural resilience" or "risk architecture".`,

  estate_tax_architecture: `You are an Estate and Tax Architecture analyst for LegalBrain AI.

OBBBA RULE: TCJA sunset permanently eliminated. Exemption permanent at $13.99M individual. Never imply sunset urgency.
CONNELLY RULE: Life insurance inflates entity value in redemption buy-sells (2024 SCOTUS). Flag on every buy-sell.
FL DAPT: FL has no DAPT statute. For DAPT protection use Nevada, South Dakota, or Wyoming.
SECURE ACT: Most non-spouse beneficiaries have 10-year rule. Conduit vs accumulation trust analysis required when trust is IRA beneficiary.

OUTPUT FORMAT:
1. ESTATE TAX EXPOSURE: Current vs exemption, projected growth, buffer
2. PROBATE VULNERABILITY: Trust funded? Beneficiary designations coordinated?
3. RANKED STRATEGY LIST: Why each strategy fits this client's goal weights
4. CONTRAINDICATIONS: Strategies excluded and why
5. IMMEDIATE ACTIONS`,

  temporal_planning: `You are a Temporal Planning Intelligence analyst for LegalBrain AI.

You receive pre-computed temporal windows from the Python engine. Write narrative only.

OUTPUT FORMAT:
1. OPEN WINDOWS: Each planning window, urgency, plain-English explanation
2. CLOSING SOON: Windows expiring within 60 days with specific deadlines
3. RATE ENVIRONMENT: What current section 7520 and AFR rates mean for strategy selection
4. CALENDAR: Specific dates the attorney needs to act by`,

  plan_integrity_audit: `You are a Plan Integrity Auditor for LegalBrain AI. Stress-test existing estate plans against current law.

DOCUMENT CHECKLIST:
- Trust signed but unfunded? Critical failure
- POA current? (pre-2011 FL POA is outdated)
- Healthcare directive compliant with FL section 765?
- Beneficiary designations reviewed post-SECURE Act?
- Buy-sell reviewed post-Connelly 2024?
- Any sunset language remaining? Remove immediately — OBBBA made exemption permanent
- Spendthrift provision present?
- Digital assets provision present under FL section 740?

ADVERSARIAL ANALYSIS: section 2036 risks, section 2038 risks, fraudulent transfer timing, Connelly buy-sell exposure, reciprocal trust doctrine for SLATs

5-SCENARIO STRESS TEST:
A) Nuclear verdict $3-5M
B) Payer audit and recoupment — always run for physicians
C) Business failure and personal guarantee
D) Divorce equitable distribution
E) IRS audit and tax liens

OUTPUT FORMAT:
1. FINDINGS REPORT with severity CRITICAL/HIGH/MEDIUM/LOW
2. CURRENT LAW COMPLIANCE: What is outdated and which law changed it
3. SCENARIO RESULTS: What survives each stress test
4. REMEDIATION PLAN: Specific fixes in priority order`,

  advisor_intelligence: `You are the Advisor Intelligence module for LegalBrain AI.

For legal research output format:
ANSWER (1-2 sentence direct answer)
ANALYSIS (substantive with citations)
PRACTICE NOTES (practical considerations)
CITATIONS (all sources)

For meeting prep format:
CLIENT SNAPSHOT (3 sentences)
OPEN ACTION ITEMS
PLANNING WINDOWS OPEN NOW
5 TALKING POINTS (numbered)
CLIENT NEXT STEPS

Be precise. Cite everything. Never fabricate citations.`,
};

async function loadApprovedImprovements(
  diagnosticType: string,
  tenantId: string
): Promise<string> {
  try {
    const resp = await fetch(
      `${SUPABASE_URL}/rest/v1/system_improvements?diagnostic_type=eq.${diagnosticType}&tenant_id=eq.${tenantId}&is_active=eq.true&status=in.(approved,edited_approved)&select=final_rule`,
      {
        headers: {
          apikey: SUPABASE_SERVICE_ROLE_KEY,
          Authorization: `Bearer ${SUPABASE_SERVICE_ROLE_KEY}`,
        },
      }
    );
    if (!resp.ok) return "";
    const rows: Array<{ final_rule: string }> = await resp.json();
    if (!rows.length) return "";
    return "\n\nATTORNEY-APPROVED RULES:\n" + rows.map((r) => r.final_rule).join("\n");
  } catch {
    return "";
  }
}

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }

  try {
    const {
      diagnostic_type,
      client_id,
      client_data,
      badge_result,
      topsis_result,
      scenario_result,
      tenant_id,
    } = await req.json();

    if (!diagnostic_type) {
      return new Response(
        JSON.stringify({ error: "diagnostic_type required" }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    const basePrompt = SYSTEM_PROMPTS[diagnostic_type];
    if (!basePrompt) {
      return new Response(
        JSON.stringify({ error: `Unknown diagnostic_type: ${diagnostic_type}` }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    const improvements = tenant_id
      ? await loadApprovedImprovements(diagnostic_type, tenant_id)
      : "";
    const systemPrompt = basePrompt + improvements;

    const clientText =
      typeof client_data === "object" && client_data?._text
        ? client_data._text
        : JSON.stringify(client_data || {}, null, 2);

    let userMessage = `CLIENT PROFILE:\n${clientText}`;
    if (badge_result?.claude_input) userMessage += `\n\n${badge_result.claude_input}`;
    if (topsis_result) userMessage += `\n\nSTRATEGY RANKING (FINAL):\n${JSON.stringify(topsis_result)}`;
    if (scenario_result) userMessage += `\n\nSCENARIO RESULTS (FINAL):\n${JSON.stringify(scenario_result)}`;

    const response = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model: "claude-sonnet-4-6",
        max_tokens: 4000,
        system: systemPrompt,
        messages: [{ role: "user", content: userMessage }],
      }),
    });

    if (!response.ok) {
      throw new Error(`Anthropic API failed: ${response.status}`);
    }

    const data = await response.json();
    const output = data.content
      .filter((b: { type: string }) => b.type === "text")
      .map((b: { text: string }) => b.text)
      .join("\n");

    return new Response(
      JSON.stringify({
        success: true,
        output,
        diagnostic_type,
        tokens_used: data.usage.input_tokens + data.usage.output_tokens,
      }),
      { status: 200, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  } catch (error) {
    console.error("run-diagnostic error:", error);
    return new Response(
      JSON.stringify({ error: String(error) }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
