"""
ar_baseline.jl — linear autoregressive baseline for the v4 minute forecaster.

Fits a ridge regression of the next return on the last `--lags` returns of the same stock (shared across stocks) plus a
scale regression on the time of day and recent |returns|, on the training companies and dates, and scores it on exactly
the evaluation sets of `train_minute_forecaster.jl` (same windows, targets and metrics). Writes `ar_baseline.json` next to
the run's `val_report.json` and prints the two side by side, so the transformer's skill and likelihood gain can be read
against what a linear model on lagged returns already gets.

Usage:
  julia --project=packages/TradingGame scripts/forecaster/ar_baseline.jl --config configs/bse_v3.toml
  julia --project=packages/TradingGame scripts/forecaster/ar_baseline.jl --config configs/bse_v3.toml --lags 30

Options (the [forecaster] and [dates] sections of --config supply window, val_windows, loss, nu and the date windows):
  --config PATH        TOML run file (configs/*.toml)
  --granularity G      forecaster cache to score on (1min default; 5min, 15min, hourly, daily)
  --lags K             latest returns used by both regressions (default 10)
  --fit-windows N      random training windows the regressions are fitted on (default 3000)
  --ridge X            ridge penalty relative to the mean diagonal of XᵀX (default 1e-4)
  --out-dir DIR        run directory holding val_report.json; ar_baseline.json is written here (default website/data/forecaster)
  --seed N             seed of the training-window draw (default 42)
  -h, --help           this text
"""

using TradingGame, Dates, JSON3, Random, Printf

include(joinpath(@__DIR__, "..", "lib", "run_config.jl"))
using .RunConfigs

const REPO_ROOT     = joinpath(@__DIR__, "..", "..")
const DATA_DIR      = joinpath(REPO_ROOT, "website", "data")
const UNIVERSE_FILE = joinpath(DATA_DIR, "trading_game", "universe_latest.json")

function parse_args()
    config_path = RunConfigs.take_config_arg!(ARGS)
    cfg = config_path === nothing ? nothing : RunConfigs.load_or_exit(config_path)
    opts = Dict{String, Any}("window" => 96, "val_windows" => 256, "loss" => "student_t", "nu" => 4.0,
        "val_window" => "trailing", "val_days" => 60,
        "train_start" => nothing, "train_end" => nothing, "val_start" => nothing, "val_end" => nothing,
        "granularity" => "1min", "out" => nothing, "lags" => TradingGame.AR_DEFAULT_LAGS, "fit_windows" => 3000, "ridge" => 1e-4,
        "out_dir" => nothing, "seed" => 42)
    for sec in ("dates", "forecaster_cache", "forecaster")
        merge!(opts, Dict(k => v for (k, v) in RunConfigs.section_opts(cfg, sec) if k in ("out", "granularity", keys(opts)...)))
    end
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a in ("-h", "--help")
            println(match(r"\"\"\"\n(.*?)\"\"\""s, read(@__FILE__, String)).captures[1])
            exit(0)
        elseif a == "--granularity"; opts["granularity"] = lowercase(ARGS[i+1]); i += 2
        elseif a == "--out-dir"; opts["out_dir"] = abspath(ARGS[i+1]); i += 2
        elseif a in ("--lags", "--fit-windows", "--seed")
            i < length(ARGS) || error("$a needs a value")
            opts[replace(a[3:end], "-" => "_")] = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--ridge"
            i < length(ARGS) || error("$a needs a value")
            opts["ridge"] = parse(Float64, ARGS[i+1]); i += 2
        else
            error("Unknown argument: $a (see --help)")
        end
    end
    forecast_loss_kind(opts["loss"]) === :huber && error("the AR baseline needs a likelihood: loss must be student_t or gaussian")
    return opts
end

function columns_of(cache::MinuteCache, symbols::Vector{String}, role::String)::Vector{Int}
    idx = Dict(s => i for (i, s) in enumerate(cache.symbols))
    gone = [s for s in symbols if !haskey(idx, s)]
    isempty(gone) || error("$(length(gone)) $role companies are not in the minute cache (e.g. $(first(gone))); rebuild it with build_forecaster_cache.jl")
    return [idx[s] for s in symbols]
end

_nn(x) = x isa Real && !isfinite(x) ? nothing : x

function main()
    opts = parse_args()
    cache_path = opts["out"] === nothing ? normpath(joinpath(DATA_DIR, "forecaster_cache_$(opts["granularity"]).bson")) : abspath(joinpath(REPO_ROOT, opts["out"]))
    cache = load_minute_cache(cache_path)
    opts["out_dir"] === nothing && (opts["out_dir"] = joinpath(DATA_DIR, cache.granularity == "1min" ? "forecaster" : "forecaster_" * cache.granularity))
    universe = load_universe_snapshot(UNIVERSE_FILE)
    train_cols = columns_of(cache, universe.train, "train")
    val_cols = columns_of(cache, universe.val, "val")
    train_start, train_end, val_start, val_end = resolve_date_windows(
        first(cache.dates), last(cache.dates), opts["val_window"], opts["val_days"],
        opts["train_start"], opts["train_end"], opts["val_start"], opts["val_end"])
    M = opts["window"]
    train_range = window_starts(cache, train_start, train_end, M)
    val_range = window_starts(cache, val_start, val_end, M)
    loss = forecast_loss_kind(opts["loss"])
    sets = [EvalSet("val", val_cols, spread_starts(val_range, opts["val_windows"], M)),
            EvalSet("val_train_companies", train_cols, spread_starts(val_range, opts["val_windows"], M)),
            EvalSet("train", train_cols, spread_starts(train_range, opts["val_windows"], M))]
    @info "Fitting AR($(opts["lags"])) on $(opts["fit_windows"]) windows of $(length(train_cols)) train companies ($train_start .. $train_end)"
    model = fit_ar_baseline(cache, train_cols, train_range, M; lags=opts["lags"], loss=loss, nu=opts["nu"],
                            windows=opts["fit_windows"], ridge=opts["ridge"], rng=MersenneTwister(opts["seed"]))
    @printf("location coefficients (lag 0 = the latest return): %s\n", join((@sprintf("%.4f", a) for a in model.mean_coef), "  "))

    report = Dict{String, Any}("lags" => opts["lags"], "fit_windows" => opts["fit_windows"], "distribution" => opts["loss"], "nu" => opts["nu"],
                               "mean_coef" => model.mean_coef, "sets" => Dict{String, Any}())
    results = Dict(es.name => evaluate_ar_baseline(model, cache, es, M) for es in sets)
    for (n, r) in results
        report["sets"][n] = Dict("n_targets" => r.n_targets, "nll" => _nn(r.nll), "nll_const_sigma" => _nn(r.nll_const), "nll_ewma_sigma" => _nn(r.nll_ewma),
            "nll_gain_vs_const_sigma" => _nn(r.nll_const - r.nll), "nll_gain_vs_ewma_sigma" => _nn(r.nll_ewma - r.nll),
            "skill_vs_zero" => _nn(r.skill), "sign_hit_rate" => _nn(r.hit_rate), "coverage_68" => _nn(r.cov68), "coverage_95" => _nn(r.cov95),
            "per_slot_skill" => [_nn(s) for s in r.per_slot_skill])
    end
    mkpath(opts["out_dir"])
    open(joinpath(opts["out_dir"], "ar_baseline.json"), "w") do io
        JSON3.pretty(io, report)
    end

    ref_path = joinpath(opts["out_dir"], "val_report.json")
    ref = isfile(ref_path) ? JSON3.read(read(ref_path, String)) : nothing
    f(x) = x === nothing ? "      —" : @sprintf("%7.4f", x)
    p(x) = x === nothing ? "     —" : @sprintf("%5.2f%%", 100 * x)
    println()
    @printf("%-22s %-12s %8s %9s %9s %7s %7s %7s\n", "set", "model", "NLL", "gain/cσ", "gain/ewma", "skill", "hit", "cov95")
    for es in sets
        a = report["sets"][es.name]
        row(name, d) = @printf("%-22s %-12s %s %s %s %s %s %s\n", startswith(name, "AR") ? es.name : "", name, f(d["nll"]), f(d["nll_gain_vs_const_sigma"]),
                               f(d["nll_gain_vs_ewma_sigma"]), p(d["skill_vs_zero"]), p(d["sign_hit_rate"]), p(d["coverage_95"]))
        row("AR($(opts["lags"]))", a)
        if ref !== nothing && haskey(ref, Symbol(es.name))
            b = ref[Symbol(es.name)]
            row("transformer", Dict("nll" => b.nll, "nll_gain_vs_const_sigma" => b.nll_gain_vs_const_sigma, "nll_gain_vs_ewma_sigma" => b.nll_gain_vs_ewma_sigma,
                                    "skill_vs_zero" => b.skill_vs_zero, "sign_hit_rate" => b.sign_hit_rate, "coverage_95" => b.coverage_95))
        end
    end
    println("\nWritten: ", joinpath(opts["out_dir"], "ar_baseline.json"))
end

main()
