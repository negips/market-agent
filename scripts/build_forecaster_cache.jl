"""
build_forecaster_cache.jl

Build the return cache for the v4 forecaster at any bar size (1min, 5min, 15min, hourly or daily):
every symbol of the train and val candidate universes on one axis of trading days × bars per day
(375 / 75 / 25 / 7 / 1 bars; bars are stamped by their start, 09:15 first), stored as log-returns
plus a traded flag (see `TradingGame.MinuteCache`, `build_forecaster_cache`).

The columns are `universe_latest.json`'s train candidates followed by its val candidates.
The date range is `[forecaster_cache] from/to` if given, else the span of `[dates]`
(with ten days of margin before it for the context features), else all data on disk.

Usage:
  julia -t auto --project=packages/TradingGame scripts/build_forecaster_cache.jl --config configs/bse_v3.toml
  julia -t auto --project=packages/TradingGame scripts/build_forecaster_cache.jl --granularity 15min --exchange bse
  julia -t auto --project=packages/TradingGame scripts/build_forecaster_cache.jl --granularity daily --from 2015-01-01

Options:
  --config PATH       TOML run file; reads [forecaster_cache] and [dates] (configs/*.toml)
  --print-config      Show the resolved settings and exit
  --granularity G     1min (default), 5min, 15min, hourly or daily; reads ohlcv/{exchange}/{dir} for that bar size
  --exchange nse|bse  Which ohlcv/{exchange} tree to read (default: bse)
  --from DATE         First day to keep (yyyy-mm-dd)
  --to DATE           Last day to keep
  --out PATH          Output file (default: website/data/forecaster_cache_{granularity}.bson)
  --min-day-fraction X  A day joins the axis when this fraction of symbols were active (default: 0.2)

Prerequisites: website/data/trading_game/universe_latest.json (build_market_universe_snapshot.jl)
and the CSVs of that bar size for every symbol in it (collect_bse_ohlcv.jl --1min-only, --5min-only, ...).
A symbol with no CSV of that bar size is an error naming it (BSE hourly files cover only part of the universe).
Run Julia with `-t auto`: the CSVs are read in parallel.
"""

using TradingGame, Dates

include(joinpath(@__DIR__, "run_config.jl"))
using .RunConfigs

const REPO_ROOT     = joinpath(@__DIR__, "..")
const DATA_DIR      = joinpath(REPO_ROOT, "website", "data")
const UNIVERSE_FILE = joinpath(DATA_DIR, "trading_game", "universe_latest.json")
const CONTEXT_MARGIN_DAYS = 10

function parse_args()
    config_path = RunConfigs.take_config_arg!(ARGS)
    print_config = RunConfigs.take_flag!(ARGS, "--print-config")
    cfg = config_path === nothing ? nothing : RunConfigs.load_or_exit(config_path)
    opts = Dict{String, Any}("exchange" => "bse", "granularity" => "1min", "out" => nothing,
                             "from" => nothing, "to" => nothing, "min_day_fraction" => TradingGame.MIN_DAY_SYMBOL_FRACTION)
    dates = RunConfigs.section_opts(cfg, "dates")
    explicit_dates = [get(dates, k, nothing) for k in ("train_start", "train_end", "val_start", "val_end")]
    given = filter(!isnothing, explicit_dates)
    if !isempty(given)
        opts["from"] = minimum(given) - Day(CONTEXT_MARGIN_DAYS)
        opts["to"] = maximum(given)
    end
    merge!(opts, RunConfigs.section_opts(cfg, "forecaster_cache"))
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a in ("-h", "--help")
            println(match(r"\"\"\"\n(.*?)\"\"\""s, read(@__FILE__, String)).captures[1])
            exit(0)
        elseif a == "--granularity";      opts["granularity"] = lowercase(ARGS[i+1]); i += 2
        elseif a == "--exchange";         opts["exchange"] = lowercase(ARGS[i+1]); i += 2
        elseif a == "--from";             opts["from"] = Date(ARGS[i+1]); i += 2
        elseif a == "--to";               opts["to"] = Date(ARGS[i+1]); i += 2
        elseif a == "--out";              opts["out"] = abspath(ARGS[i+1]); i += 2
        elseif a == "--min-day-fraction"; opts["min_day_fraction"] = parse(Float64, ARGS[i+1]); i += 2
        else; error("Unknown argument: $a")
        end
    end
    opts["exchange"] in ("nse", "bse") || error("--exchange must be nse or bse")
    haskey(GRANULARITIES, opts["granularity"]) || error("--granularity must be one of $(join(sort(collect(keys(GRANULARITIES))), ", "))")
    opts["out"] = opts["out"] === nothing ? normpath(joinpath(DATA_DIR, "forecaster_cache_$(opts["granularity"]).bson")) : abspath(joinpath(REPO_ROOT, opts["out"]))
    if print_config
        RunConfigs.print_resolved(stdout, "build_forecaster_cache.jl", config_path, opts)
        exit(0)
    end
    return opts
end

function main()
    opts = parse_args()
    isfile(UNIVERSE_FILE) || error("Not found: $UNIVERSE_FILE\nRun: julia --project=packages/TradingGame scripts/build_market_universe_snapshot.jl")
    universe = load_universe_snapshot(UNIVERSE_FILE)
    symbols = unique(vcat(universe.train, universe.val))
    overlap = length(universe.train) + length(universe.val) - length(symbols)
    ohlcv_dir = joinpath(DATA_DIR, "ohlcv", opts["exchange"], GRANULARITIES[opts["granularity"]].dir)
    from = something(opts["from"], Date(2010, 1, 1))
    to = something(opts["to"], today())
    @info "Forecaster cache ($(opts["granularity"])): $(length(symbols)) symbols ($(length(universe.train)) train, $(length(universe.val)) val, $overlap shared), " *
          "$from .. $to, $(Threads.nthreads()) thread(s)" * (Threads.nthreads() == 1 ? " — read CSVs in parallel with: julia -t auto --project=packages/TradingGame scripts/build_forecaster_cache.jl ..." : "")
    t0 = time()
    cache = build_forecaster_cache(ohlcv_dir, symbols; granularity=opts["granularity"], from=from, to=to, min_day_fraction=opts["min_day_fraction"], verbose=false)
    save_minute_cache(cache, opts["out"])
    traded = sum(cache.traded) / length(cache.traded)
    @info "Done in $(round(time() - t0, digits=1)) s → $(opts["out"])"
    @info "  $(length(cache.dates)) trading days ($(first(cache.dates)) .. $(last(cache.dates))), $(TradingGame.n_bars(cache)) bars, " *
          "$(round(100 * traded, digits=1))% of symbol-minutes traded, return scale $(round(cache.scale, digits=4))%"
end

main()
