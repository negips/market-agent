"""
extract_llm_features.jl

For each confidence-scored NSE company, fetch the most recent conference call
or earnings release from Tijori and run Claude's structured-output extraction
to produce the 14-scalar LLMFeatures vector.

Results are saved to website/data/llm_features/{SYMBOL}.json.
The script is resumable — already-extracted symbols are skipped unless --refresh.

Requires ANTHROPIC_API_KEY in .env or environment.

Prerequisites:
  - Tijori sidecar running (node sidecar/server_http.js)
  - website/data/nse_companies_latest.json present
  - ANTHROPIC_API_KEY in environment

Usage:
  julia --project=packages/StockSwingPredictor scripts/extract_llm_features.jl
  julia --project=packages/StockSwingPredictor scripts/extract_llm_features.jl 50
  julia --project=packages/StockSwingPredictor scripts/extract_llm_features.jl --refresh
"""

using StockSwingPredictor, TijoriData, JSON3, Dates, Printf

const REPO_ROOT       = joinpath(@__DIR__, "..")
const COMPANIES_FILE  = joinpath(REPO_ROOT, "website", "data", "nse_companies_latest.json")
const LLM_FEATURES_DIR = joinpath(REPO_ROOT, "website", "data", "llm_features")
const SIDECAR_PORT    = 3001

function main()
    if "--help" in ARGS || "-h" in ARGS
        println("""
Usage:
  julia --project=packages/StockSwingPredictor scripts/extract_llm_features.jl [N] [--refresh]

Arguments:
  N          Top N companies by market cap (default: all scored).
  --refresh  Re-extract all, ignoring cached files.

Output:
  website/data/llm_features/{SYMBOL}.json
""")
        return
    end

    api_key = get(ENV, "ANTHROPIC_API_KEY", "")
    if isempty(api_key)
        # Try loading from .env
        env_path = joinpath(REPO_ROOT, ".env")
        if isfile(env_path)
            for line in readlines(env_path)
                m = match(r"^ANTHROPIC_API_KEY\s*=\s*(.+)$", strip(line))
                isnothing(m) || (api_key = strip(string(m[1])))
            end
        end
    end
    isempty(api_key) && error("ANTHROPIC_API_KEY not set. Add it to .env or export it.")

    refresh = "--refresh" in ARGS
    args    = filter(a -> a != "--refresh", ARGS)
    top_n   = length(args) >= 1 ? parse(Int, args[1]) : typemax(Int)

    # ── Connect to sidecar ─────────────────────────────────────────────────────

    TijoriData.configure!(port=SIDECAR_PORT)
    TijoriData.is_running() || error("Sidecar not reachable on port $SIDECAR_PORT. " *
                                      "Run: node sidecar/server_http.js")
    @info "Sidecar OK"

    # ── Load companies ─────────────────────────────────────────────────────────

    isfile(COMPANIES_FILE) || error("Not found: $COMPANIES_FILE")
    raw   = JSON3.read(read(COMPANIES_FILE, String))
    all_c = collect(raw.companies)

    eligible = filter(all_c) do c
        !isnothing(get(c, :confidence, nothing)) &&
        !isempty(string(get(c, :slug, "")))
    end
    sort!(eligible, by = c -> Float64(get(c, :market_cap_cr, 0.0)), rev=true)
    todo = first(eligible, top_n)

    mkpath(LLM_FEATURES_DIR)
    @info "Processing $(length(todo)) companies → $LLM_FEATURES_DIR"

    ok = skipped = failed = 0

    for (i, c) in enumerate(todo)
        sym  = string(get(c, :symbol, ""))
        slug = string(get(c, :slug,   ""))
        name = string(get(c, :name,   sym))

        out_path = joinpath(LLM_FEATURES_DIR, "$(sym).json")
        if !refresh && isfile(out_path)
            skipped += 1
            continue
        end

        try
            kb = get_knowledge_base(slug)
            llm = extract_features_from_kb(kb, today(); api_key=api_key)

            open(out_path, "w") do io
                JSON3.pretty(io, Dict(
                    "symbol"       => sym,
                    "slug"         => slug,
                    "extracted_at" => string(now(UTC)),
                    "features"     => Dict(
                        "management_tone"           => llm.management_tone,
                        "guidance_direction"        => llm.guidance_direction,
                        "guidance_specificity"      => llm.guidance_specificity,
                        "demand_outlook"            => llm.demand_outlook,
                        "margin_commentary"         => llm.margin_commentary,
                        "competitive_pressure"      => llm.competitive_pressure,
                        "new_wins_announced"        => llm.new_wins_announced,
                        "capex_expansion"           => llm.capex_expansion,
                        "buyback_or_dividend"       => llm.buyback_or_dividend,
                        "mgmt_language_hedging"     => llm.mgmt_language_hedging,
                        "auditor_concerns"          => llm.auditor_concerns,
                        "related_party_flags"       => llm.related_party_flags,
                        "contingent_liability_flag" => llm.contingent_liability_flag,
                        "extraction_confidence"     => llm.extraction_confidence,
                        "doc_age_days"              => llm.doc_age_days,
                    ),
                ))
            end

            ok += 1
            @info @sprintf("[%d/%d] %-30s  conf=%.2f",
                           i, length(todo), name[1:min(30,length(name))],
                           llm.extraction_confidence)
        catch e
            failed += 1
            @warn @sprintf("[%d/%d] %-30s  ERROR: %s",
                           i, length(todo), name[1:min(30,length(name))],
                           sprint(showerror, e)[1:80])
        end

        # LLM calls are slower — no extra sleep needed, Claude adds natural latency
    end

    println()
    @printf("LLM extraction complete:\n")
    @printf("  Extracted: %d\n  Skipped:   %d\n  Failed:    %d\n", ok, skipped, failed)
end

main()
