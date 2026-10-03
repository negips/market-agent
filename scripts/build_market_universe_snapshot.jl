"""
build_market_universe_snapshot.jl

Builds the TradingGame candidate universe: filters `nse_companies_latest.json`
to the confidence-passing (score >= MIN_CONFIDENCE_SCORE, reusing the
confidence data `run_confidence_checks.jl` already computed — no live Tijori
sidecar calls here), cache-available pool, ranked by market cap, then splits
it into a train list and a held-out validation list per the chosen
`--strategy`.

Strategies (--strategy NAME):
  shared-topcap     (default) top --n by market cap, identical list for train and val
  disjoint-topcap    top (--n-train + --n-val) by market cap, split disjoint and
                     stratified by market-cap decile
  random              uniformly random --n from the WHOLE confidence-passing pool
                     (not restricted to top-market-cap); add --disjoint for
                     independent random train/val draws
  random-bucketed      random selection from named market-cap bands, one or more
                     --band LO:HI:QUOTA (HI may be 'inf'); add --disjoint for
                     independent random train/val draws per band

See TradingGame.UniverseStrategy's docstring (universe.jl) for the full
rationale of each strategy, and why train/val are free to use different
company counts and/or different companies entirely (no symbol-identity
embedding or positional encoding anywhere downstream).

Usage:
  julia --project=packages/TradingGame scripts/build_market_universe_snapshot.jl
  julia --project=packages/TradingGame scripts/build_market_universe_snapshot.jl --n 100
  julia --project=packages/TradingGame scripts/build_market_universe_snapshot.jl \\
      --strategy disjoint-topcap --n-train 60 --n-val 20 --seed 42
  julia --project=packages/TradingGame scripts/build_market_universe_snapshot.jl \\
      --strategy random --n 60 --disjoint --seed 42
  julia --project=packages/TradingGame scripts/build_market_universe_snapshot.jl \\
      --strategy random-bucketed --band 0:5000:15 --band 5000:inf:15

Prerequisites:
  website/data/nse_companies_latest.json  (generate_nse_list.jl, then
                                            run_confidence_checks.jl)
  website/data/inference_cache.bson       (build_cache.jl)

Output: website/data/trading_game/universe_latest.json
  {"strategy": ..., "strategy_params": ..., "train_candidates": [...], "val_candidates": [...]}
"""

using TradingGame, StockSwingPredictor, Printf

const REPO_ROOT          = joinpath(@__DIR__, "..")
const CACHE_FILE         = joinpath(REPO_ROOT, "website", "data", "inference_cache.bson")
const NSE_COMPANIES_FILE = joinpath(REPO_ROOT, "website", "data", "nse_companies_latest.json")
const OUTPUT_FILE        = joinpath(REPO_ROOT, "website", "data", "trading_game", "universe_latest.json")

"""Parse `LO:HI:QUOTA` (HI may be `inf`/`infinity`, case-insensitive) into a
`(lo, hi) => quota` pair for `BucketedRandom.quotas`."""
function _parse_band(spec::String)
    parts = split(spec, ":")
    length(parts) == 3 || error("--band expects LO:HI:QUOTA, got '$spec'")
    lo_s, hi_s, q_s = parts
    hi = lowercase(hi_s) in ("inf", "infinity") ? Inf : parse(Float64, hi_s)
    return (parse(Float64, lo_s), hi) => parse(Int, q_s)
end

function parse_args()
    strategy_name = "shared-topcap"
    n        = N_CANDIDATE_STOCKS
    n_train  = N_CANDIDATE_STOCKS
    n_val    = N_CANDIDATE_STOCKS
    disjoint = false
    seed     = nothing
    bands    = Pair{Tuple{Float64, Float64}, Int}[]

    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a in ("--help", "-h")
            println("""
Usage:
  julia --project=packages/TradingGame scripts/build_market_universe_snapshot.jl [options]

Strategies (--strategy NAME):
  shared-topcap      (default) top --n by market cap, same list for train and val
  disjoint-topcap     top (--n-train + --n-val) by market cap, split disjoint and
                      stratified by market-cap decile
  random               uniformly random --n from the whole confidence-passing pool;
                      add --disjoint for independent random train/val draws
  random-bucketed       random selection from named market-cap bands (--band
                      LO:HI:QUOTA, repeatable, HI may be 'inf'); add --disjoint
                      for independent random train/val draws per band

Options:
  --n N                Candidate cap for shared-topcap/random (default: $N_CANDIDATE_STOCKS)
  --n-train N           Train-side cap for disjoint-topcap (default: $N_CANDIDATE_STOCKS)
  --n-val N             Val-side cap for disjoint-topcap (default: $N_CANDIDATE_STOCKS)
  --band LO:HI:QUOTA    One market-cap band for random-bucketed (repeatable)
  --disjoint            Independent random train/val draws (random, random-bucketed only)
  --seed N              Reproducible split/draw (disjoint-topcap, random, random-bucketed)

Output: $OUTPUT_FILE
""")
            exit(0)
        elseif a == "--strategy"; strategy_name = ARGS[i+1]; i += 2
        elseif a == "--n";        n = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--n-train";  n_train = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--n-val";    n_val   = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--disjoint"; disjoint = true; i += 1
        elseif a == "--seed";     seed = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--band";     push!(bands, _parse_band(ARGS[i+1])); i += 2
        else; i += 1
        end
    end

    strategy = if strategy_name == "shared-topcap"
        SharedTopMarketCap(n=n)
    elseif strategy_name == "disjoint-topcap"
        DisjointTopMarketCap(n_train=n_train, n_val=n_val, seed=seed)
    elseif strategy_name == "random"
        RandomUniverse(n=n, disjoint=disjoint, seed=seed)
    elseif strategy_name == "random-bucketed"
        isempty(bands) && error("--strategy random-bucketed requires at least one --band LO:HI:QUOTA")
        BucketedRandom(quotas=bands, disjoint=disjoint, seed=seed)
    else
        error("Unknown --strategy '$strategy_name'. Expected: shared-topcap, disjoint-topcap, random, random-bucketed")
    end
    return strategy
end

function _print_entries(label::String, entries)
    println("\n$label candidates:")
    for e in entries[1:min(10, length(entries))]
        @printf("  %-12s %-32s mcap ₹%.0f Cr   conf %.0f\n", e.symbol, e.name, e.market_cap_cr, e.confidence_score)
    end
    length(entries) > 10 && println("  … and $(length(entries) - 10) more")
end

function main()
    strategy = parse_args()

    isfile(NSE_COMPANIES_FILE) || error(
        "Not found: $NSE_COMPANIES_FILE\nRun: julia scripts/generate_nse_list.jl && " *
        "julia --project=packages/CompanyConfidence scripts/run_confidence_checks.jl")
    isfile(CACHE_FILE) || error(
        "Not found: $CACHE_FILE\nRun: julia --project=packages/StockSwingPredictor scripts/build_cache.jl")

    @info "Loading inference cache…"
    cache = load_inference_cache(CACHE_FILE)
    @info "  $(length(cache.companies)) companies with cached price history"

    @info "Building eligible pool (confidence >= $MIN_CONFIDENCE_SCORE, ranked by market cap)…"
    pool = eligible_candidates(cache, NSE_COMPANIES_FILE)
    isempty(pool) && error(
        "No candidates passed the confidence filter and had cached price history — " *
        "check $NSE_COMPANIES_FILE and $CACHE_FILE cover the same universe")
    @info "  $(length(pool)) companies eligible"

    @info "Applying strategy: $(typeof(strategy))"
    (; train, val) = build_universes(strategy, pool)

    isempty(train) && error("Strategy produced an empty train universe — check --n/--n-train/--band " *
                             "quotas against the eligible pool size ($(length(pool)))")
    isempty(val) && error("Strategy produced an empty val universe — check --n/--n-val/--band " *
                           "quotas against the eligible pool size ($(length(pool)))")

    save_universe_snapshot(strategy, train, val, OUTPUT_FILE)
    @info "Saved $(length(train)) train / $(length(val)) val candidates → $OUTPUT_FILE"

    _print_entries("Train", train)
    _print_entries("Val", val)
end

main()
