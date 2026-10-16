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
  julia --project=packages/TradingGame scripts/game/build_market_universe_snapshot.jl --config configs/bse_v3.toml
  julia --project=packages/TradingGame scripts/game/build_market_universe_snapshot.jl
  julia --project=packages/TradingGame scripts/game/build_market_universe_snapshot.jl --n 100
  julia --project=packages/TradingGame scripts/game/build_market_universe_snapshot.jl \\
      --strategy disjoint-topcap --n-train 60 --n-val 20 --seed 42
  julia --project=packages/TradingGame scripts/game/build_market_universe_snapshot.jl \\
      --strategy random --n 60 --disjoint --seed 42
  julia --project=packages/TradingGame scripts/game/build_market_universe_snapshot.jl \\
      --strategy random-bucketed --band 0:5000:15 --band 5000:inf:15

Prerequisites:
  website/data/{nse,bse}_companies_latest.json  — the file for the cache's exchange
                                            (generate_nse_list.jl + run_confidence_checks.jl,
                                            or generate_bse_list.jl + run_bse_confidence_checks.jl)
  website/data/inference_cache.bson       (build_cache.jl; records its exchange and bar length)

Output: website/data/trading_game/universe_latest.json
  {"strategy": ..., "strategy_params": ..., "train_candidates": [...], "val_candidates": [...]}
"""

using TradingGame, StockSwingPredictor, Printf

include(joinpath(@__DIR__, "..", "lib", "run_config.jl"))
using .RunConfigs

const REPO_ROOT          = joinpath(@__DIR__, "..", "..")
const CACHE_FILE         = joinpath(REPO_ROOT, "website", "data", "inference_cache.bson")
const DATA_DIR           = joinpath(REPO_ROOT, "website", "data")

"""`--min-mcap`, set by `parse_args`."""
const MIN_MARKET_CAP_CR = Ref(0.0)
"""`--keep-val`, set by `parse_args`."""
const KEEP_VAL = Ref(false)
const OUTPUT_FILE       = joinpath(REPO_ROOT, "website", "data", "trading_game", "universe_latest.json")

function parse_args()
    config_path = RunConfigs.take_config_arg!(ARGS)
    print_config = RunConfigs.take_flag!(ARGS, "--print-config")
    cfg = config_path === nothing ? nothing : RunConfigs.load_or_exit(config_path)
    u = Dict{String, Any}("strategy" => "shared-topcap", "n" => N_CANDIDATE_STOCKS,
                          "n_train" => N_CANDIDATE_STOCKS, "n_val" => N_CANDIDATE_STOCKS,
                          "disjoint" => false, "seed" => nothing, "bands" => String[], "min_mcap" => 0.0)
    merge!(u, RunConfigs.section_opts(cfg, "universe"))
    strategy_name = u["strategy"]
    n        = u["n"]
    n_train  = u["n_train"]
    n_val    = u["n_val"]
    disjoint = u["disjoint"]
    seed     = u["seed"]
    bands    = String[u["bands"]...]
    MIN_MARKET_CAP_CR[] = u["min_mcap"]

    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a in ("--help", "-h")
            println("""
Usage:
  julia --project=packages/TradingGame scripts/game/build_market_universe_snapshot.jl [options]

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
  --min-mcap CR         Drop companies below this market cap (₹ Cr) before the strategy runs.
                        Advised for BSE, where thin small caps have mostly forward-filled bars.
  --seed N              Reproducible split/draw (disjoint-topcap, random, random-bucketed)
  --keep-val            Keep the val companies of the existing universe_latest.json unchanged and draw only a new
                        train set from the pool without them (rotating the train companies while the validation
                        set stays comparable). Errors if a kept val company is no longer eligible.
  --config PATH        TOML run file; this script reads its [universe] section (configs/*.toml).
                        Command-line flags override it.
  --print-config        Show the resolved settings and exit

Output: $OUTPUT_FILE
""")
            exit(0)
        elseif a == "--strategy"; strategy_name = ARGS[i+1]; i += 2
        elseif a == "--n";        n = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--n-train";  n_train = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--n-val";    n_val   = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--disjoint"; disjoint = true; i += 1
        elseif a == "--min-mcap"; MIN_MARKET_CAP_CR[] = parse(Float64, ARGS[i+1]); i += 2
        elseif a == "--keep-val"; KEEP_VAL[] = true; i += 1
        elseif a == "--seed";     seed = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--band";     push!(bands, ARGS[i+1]); i += 2
        else; error("Unknown argument: $a (see --help)")
        end
    end

    u = Dict{String, Any}("strategy" => strategy_name, "n" => n, "n_train" => n_train, "n_val" => n_val,
                          "disjoint" => disjoint, "seed" => seed, "bands" => bands, "min_mcap" => MIN_MARKET_CAP_CR[])
    if print_config
        RunConfigs.print_resolved(stdout, "build_market_universe_snapshot.jl", config_path, u)
        exit(0)
    end
    strategy = RunConfigs.universe_strategy(u)
    return strategy
end

function _print_entries(label::String, entries)
    println("\n$label candidates:")
    for e in entries[1:min(10, length(entries))]
        @printf("  %-12s %-32s mcap ₹%.0f Cr   conf %.0f\n", e.symbol, e.name, e.market_cap_cr, e.confidence_score)
    end
    length(entries) > 10 && println("  … and $(length(entries) - 10) more")
end

"""
    kept_val_entries(pool) -> Vector{UniverseEntry}

The val companies of the existing `universe_latest.json`, taken from `pool` in their saved order. Errors if the file is
missing or a saved val company is no longer in the pool (a changed `--min-mcap` or confidence list), since a silently
different validation set would make losses from before and after incomparable.
"""
function kept_val_entries(pool)
    isfile(OUTPUT_FILE) || error("--keep-val needs an existing $OUTPUT_FILE; build one first without it")
    saved = load_universe_snapshot(OUTPUT_FILE).val
    by_symbol = Dict(e.symbol => e for e in pool)
    gone = [s for s in saved if !haskey(by_symbol, s)]
    isempty(gone) || error("--keep-val: $(length(gone)) saved val companies are no longer eligible: " *
                            join(first(gone, 10), ", ") * (length(gone) > 10 ? ", …" : ""))
    return [by_symbol[s] for s in saved]
end

function main()
    strategy = parse_args()

    isfile(CACHE_FILE) || error(
        "Not found: $CACHE_FILE\nRun: julia --project=packages/StockSwingPredictor scripts/ohlcv/build_cache.jl")

    @info "Loading inference cache…"
    cache = load_inference_cache(CACHE_FILE)
    @info "  $(length(cache.companies)) $(uppercase(cache.exchange)) companies with cached price history " *
          "($(cache.bar_minutes)-minute bars)"

    companies_file = joinpath(DATA_DIR, "$(cache.exchange)_companies_latest.json")
    isfile(companies_file) || error("Not found: $companies_file\nRun the $(cache.exchange) list + confidence scripts first")

    @info "Building eligible pool (confidence >= $MIN_CONFIDENCE_SCORE, ranked by market cap)…"
    pool = eligible_candidates(cache, companies_file; min_market_cap_cr=MIN_MARKET_CAP_CR[])
    isempty(pool) && error(
        "No candidates passed the confidence filter and had cached price history — " *
        "check $companies_file and $CACHE_FILE cover the same universe")
    @info "  $(length(pool)) companies eligible"

    kept_val = KEEP_VAL[] ? kept_val_entries(pool) : nothing
    if kept_val !== nothing
        kept_symbols = Set(e.symbol for e in kept_val)
        pool = filter(e -> !(e.symbol in kept_symbols), pool)
        @info "  --keep-val: $(length(kept_val)) val companies kept, $(length(pool)) left for the train draw"
    end

    @info "Applying strategy: $(typeof(strategy))"
    (; train, val) = build_universes(strategy, pool)
    kept_val === nothing || (val = kept_val)

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
