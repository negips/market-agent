"""
build_universe_snapshot.jl

Builds the TradingGame candidate universe: `CompanyConfidence`-filtered
(score >= MIN_CONFIDENCE_SCORE, reusing the confidence data
`run_confidence_checks.jl` already computed into `nse_companies_latest.json`
— no live Tijori sidecar calls here), ranked by market cap, capped at
`N_CANDIDATE_STOCKS`, and restricted to symbols with cached price history.

Usage:
  julia --project=packages/TradingGame scripts/build_universe_snapshot.jl
  julia --project=packages/TradingGame scripts/build_universe_snapshot.jl --n 100

Prerequisites:
  website/data/nse_companies_latest.json  (generate_nse_list.jl, then
                                            run_confidence_checks.jl)
  website/data/inference_cache.bson       (build_cache.jl)

Output: website/data/trading_game/universe_latest.json
"""

using TradingGame, StockSwingPredictor, Printf

const REPO_ROOT          = joinpath(@__DIR__, "..")
const CACHE_FILE         = joinpath(REPO_ROOT, "website", "data", "inference_cache.bson")
const NSE_COMPANIES_FILE = joinpath(REPO_ROOT, "website", "data", "nse_companies_latest.json")
const OUTPUT_FILE        = joinpath(REPO_ROOT, "website", "data", "trading_game", "universe_latest.json")

function parse_args()
    n = N_CANDIDATE_STOCKS
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a in ("--help", "-h")
            println("""
Usage:
  julia --project=packages/TradingGame scripts/build_universe_snapshot.jl [options]

Options:
  --n N   Candidate universe cap (default: $N_CANDIDATE_STOCKS)

Output: $OUTPUT_FILE
""")
            exit(0)
        elseif a == "--n"; n = parse(Int, ARGS[i+1]); i += 2
        else; i += 1
        end
    end
    return n
end

function main()
    n = parse_args()

    isfile(NSE_COMPANIES_FILE) || error(
        "Not found: $NSE_COMPANIES_FILE\nRun: julia scripts/generate_nse_list.jl && " *
        "julia --project=packages/CompanyConfidence scripts/run_confidence_checks.jl")
    isfile(CACHE_FILE) || error(
        "Not found: $CACHE_FILE\nRun: julia --project=packages/StockSwingPredictor scripts/build_cache.jl")

    @info "Loading inference cache…"
    cache = load_inference_cache(CACHE_FILE)
    @info "  $(length(cache.companies)) companies with cached price history"

    @info "Building candidate universe (confidence >= $MIN_CONFIDENCE_SCORE, top $n by market cap)…"
    entries = build_candidate_universe(cache, NSE_COMPANIES_FILE; n_candidates=n)

    isempty(entries) && error(
        "No candidates passed the confidence filter and had cached price history — " *
        "check $NSE_COMPANIES_FILE and $CACHE_FILE cover the same universe")

    save_universe_snapshot(entries, OUTPUT_FILE)
    @info "Saved $(length(entries)) candidates → $OUTPUT_FILE"

    println()
    for e in entries[1:min(10, length(entries))]
        @printf("  %-12s %-32s mcap ₹%.0f Cr   conf %.0f\n", e.symbol, e.name, e.market_cap_cr, e.confidence_score)
    end
    length(entries) > 10 && println("  … and $(length(entries) - 10) more")
end

main()
