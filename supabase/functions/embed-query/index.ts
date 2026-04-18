import Anthropic from "https://esm.sh/@anthropic-ai/sdk@0.27.0";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const OPENAI_API_KEY = Deno.env.get("OPENAI_API_KEY")!;

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }

  try {
    const { query, match_count = 10, namespaces, tenant_id } = await req.json();

    if (!query) {
      return new Response(JSON.stringify({ error: "query required" }), {
        status: 400,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    // Generate embedding via OpenAI
    const embedResp = await fetch("https://api.openai.com/v1/embeddings", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${OPENAI_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "text-embedding-3-small",
        input: query.replace(/\n/g, " "),
      }),
    });

    if (!embedResp.ok) {
      throw new Error(`OpenAI embedding failed: ${embedResp.status}`);
    }

    const embedData = await embedResp.json();
    const embedding = embedData.data[0].embedding;

    // Call hybrid_search_legal RPC
    const searchResp = await fetch(
      `${SUPABASE_URL}/rest/v1/rpc/hybrid_search_legal`,
      {
        method: "POST",
        headers: {
          "apikey": SUPABASE_SERVICE_ROLE_KEY,
          "Authorization": `Bearer ${SUPABASE_SERVICE_ROLE_KEY}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          p_tenant_id: tenant_id,
          query_text: query,
          query_embedding: embedding,
          match_count: match_count,
          filter_namespaces: namespaces || null,
        }),
      }
    );

    if (!searchResp.ok) {
      throw new Error(`Hybrid search failed: ${searchResp.status}`);
    }

    const results = await searchResp.json();

    return new Response(
      JSON.stringify({
        success: true,
        query,
        results: results || [],
        count: results?.length || 0,
      }),
      {
        status: 200,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      }
    );
  } catch (error) {
    console.error("embed-query error:", error);
    return new Response(
      JSON.stringify({ error: String(error) }),
      {
        status: 500,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      }
    );
  }
});
