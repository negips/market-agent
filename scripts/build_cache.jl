"""
build_cache.jl

Build the inference cache from all cached OHLCV CSVs.

Reads every {SYMBOL}_daily.csv and {SYMBOL}_hourly.csv in website/data/ohlcv/,
aligns them onto shared time axes, forward-fills missing bars, and writes a
single binary file (~42 MB). This file is loaded once at the start of each
live session and used for both inference and training batch assembly.

Run this after initial OHLCV collection, and again each morning after
update_ohlcv.jl to incorporate the latest bars.

Prerequisites:
  - website/data/ohlcv/ populated       (run collect_ohlcv.jl first)
  - website/data/nse_companies_latest.json with confidence scores

Usage:
  julia --project=packages/StockSwingPredictor scripts/build_cache.jl
  julia --project=packages/StockSwingPredictor scripts/build_cache.jl 200  # top-N only
"""

using StockSwingPredictor, JSON3, Dates

const REPO_ROOT      = joinpath(@__DIR__, "..")
const COMPANIES_FILE = joinpath(REPO_ROOT, "website", "data", "nse_companies_latest.json")
const OHLCV_DIR      = joinpath(REPO_ROOT, "website", "data", "ohlcv")
const CACHE_FILE     = joinpath(REPO_ROOT, "website", "data", "inference_cache.bson")

const CONFIDENCE_THRESHOLD = 40

function main()
    if "--help" in ARGS || "-h" in ARGS
        println("""
Usage:
  julia --project=packages/StockSwingPredictor scripts/build_cache.jl [N]

Arguments:
  N   Top N companies by market cap to include (default: all with confidence > $CONFIDENCE_THRESHOLD).

Output:
  website/data/inference_cache.bson  — ~42 MB, loaded at session start
""")
        return
    end

    top_n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : typemax(Int)

    isfile(COMPANIES_FILE) ||
        error("Not found: $COMPANIES_FILE\nRun: julia scripts/generate_nse_list.jl")

    raw      = JSON3.read(read(COMPANIES_FILE, String))
    all_c    = collect(raw.companies)
    eligible = filter(all_c) do c
        conf = get(c, :confidence, nothing)
        !isnothing(conf) &&
        !isempty(string(get(c, :symbol, ""))) &&
        get(conf, :score, 0) > CONFIDENCE_THRESHOLD
    end
    sort!(eligible, by = c -> Float64(get(c, :market_cap_cr, 0.0)), rev=true)

    universe  = first(eligible, top_n)
    companies = [string(c.symbol) for c in universe]

    @info "Universe: $(length(companies)) companies (confidence > $CONFIDENCE_THRESHOLD)"
    @info "Output: $CACHE_FILE"

    build_inference_cache(OHLCV_DIR, companies, CACHE_FILE)

    # Write the market universe JSON so the model's column ordering is auditable.
    universe_file = joinpath(REPO_ROOT, "website", "data", "market_universe.json")
    universe_json = [(rank=i, symbol=string(c.symbol),
                      name=string(get(c, :name, "")),
                      market_cap_cr=Float64(get(c, :market_cap_cr, 0.0)),
                      confidence=Int(get(get(c, :confidence, Dict()), :score, 0)))
                     for (i, c) in enumerate(universe)]
    open(universe_file, "w") do io
        JSON3.pretty(io, Dict("generated_at" => string(Dates.now()),
                              "n_market_companies" => length(universe),
                              "confidence_threshold" => CONFIDENCE_THRESHOLD,
                              "companies" => universe_json))
    end
    @info "Market universe → $universe_file ($(length(universe)) companies)"
end

main()
