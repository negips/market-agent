"""
train_minute_forecaster.jl

Train the v4 minute forecaster (`TradingGame.Forecaster`): from the last `window` one-minute states
of N stocks, predict each stock's next-minute return and the next timestamp. Supervised, no
simulator, no portfolio, no news — see website/architecture.html (Forecaster_v4 tab) and
website/mathematics.html for the layers.

Train and validation use the universe's train and val companies on the train and val dates
(`[dates]`), so the report separates "new dates" from "new companies": the evaluation sets are
  val                  val companies, val dates        (decides the checkpoint)
  val_train_companies  train companies, val dates
  train                train companies, train dates    (a sample of the training windows)

Usage:
  julia --project=packages/TradingGame scripts/train_minute_forecaster.jl --config configs/bse_v3.toml
  julia --project=packages/TradingGame scripts/train_minute_forecaster.jl --config configs/bse_v3.toml --device gpu
  julia --project=packages/TradingGame scripts/train_minute_forecaster.jl --config configs/bse_v3.toml --epochs 1 --steps-per-epoch 20
  julia --project=packages/TradingGame scripts/train_minute_forecaster.jl --config configs/bse_v3.toml --resume --epochs 10

Options (each also a key of the TOML's [forecaster]; a flag overrides the file):
  --config PATH         TOML run file; reads [forecaster], [dates] and [forecaster_cache] (granularity picks the cache)
  --print-config        Show the resolved settings and exit
  --window M            Minutes of history per window (default 96)
  --d D                 Model width; a multiple of --heads (default 64)
  --heads H             Attention heads (default 4)
  --blocks L            Causal time blocks (default 4)
  --dropout P           Dropout (default 0.1)
  --epochs N            Epochs (default 20)
  --steps-per-epoch N   Random batches per epoch (default 500)
  --batch B             Windows per batch (default 32)
  --lr X                Peak learning rate; warm-up then cosine to a tenth (default 3e-4)
  --weight-decay X      AdamW decay (default 0.01)
  --warmup-steps N      Warm-up steps (default 200)
  --val-windows N       Windows per evaluation set (default 256)
  --device cpu|gpu      Falls back to cpu when CUDA is not functional (default cpu)
  --loss L              student_t (default), gaussian or huber: the return objective. The first two predict a
                        location and a scale per stock and minute and minimise the negative log-likelihood; huber
                        fits the location alone.
  --nu X                Student-t degrees of freedom, above 2 (default 4)
  --nll-beta X          β-NLL exponent: each term is weighted by σ^(2β) held constant (default 0.5; 0 = plain NLL)
  --dtype T             float32 (default), float16 or float64: the number type of the forward and backward
                        pass. float16 keeps Float32 master weights and scales the loss dynamically; the
                        checkpoint is always saved as Float32.
  --seed N              Reproducible weights and batches
  --granularity G       Which forecaster cache to read (1min, 5min, 15min, hourly, daily; the file
                        website/data/forecaster_cache_{G}.bson). The model takes its bar size from the cache.
  --val-window MODE, --val-days N, --train-start/--train-end/--val-start/--val-end DATE
                        The date windows, as in train_trading_policy.jl
  --out-dir PATH        Where checkpoint, logs and the report go (default website/data/forecaster); a sweep
                        gives each run its own directory
  --eval-only           Do not train: score the saved forecaster.bson on the evaluation sets and rewrite
                        val_report.json (use it to re-score a finished run, e.g. after a metric change)
  --resume              Load forecaster.bson and continue the epoch numbering
                        (--epochs then means "how many more"; the learning-rate schedule restarts).
                        Also the [forecaster] key `resume`; --no-resume overrides a file's resume = true.
  --init-from PATH      A fresh run (epoch 1, cleared logs) whose weights start from the checkpoint at PATH
                        instead of random. The architecture (window, d, heads, blocks) must equal the
                        config's; dropout, lr and the rest may differ. Also the [forecaster] key `init_from`
                        (a relative path there is taken from the repo root). Exclusive with --resume and --eval-only.

Prerequisites: website/data/forecaster_cache_{granularity}.bson (build_forecaster_cache.jl) and universe_latest.json.

Outputs under website/data/forecaster/ (website/data/forecaster_{G}/ for any granularity but 1min): forecaster.bson (best checkpoint), epoch_log.jsonl,
timing.json, run_config.json, val_report.json (per-stock and per-slot skill, baselines), and
live_status.json (rewritten every 10 steps: state, epoch, step, running loss, ETA). Watch the run at
website/tradinggamelive.html (the "Forecaster v4" tab).
Stop cleanly with `touch website/data/forecaster/STOP` (after the current epoch) or
`STOP_NOW` (immediately; the best checkpoint is already on disk).
"""

using TradingGame, Dates, JSON3, Random, CUDA, Printf

include(joinpath(@__DIR__, "run_config.jl"))
using .RunConfigs

const REPO_ROOT     = joinpath(@__DIR__, "..")
const DATA_DIR      = joinpath(REPO_ROOT, "website", "data")
const DEFAULT_OUT_DIR = joinpath(DATA_DIR, "forecaster")
const UNIVERSE_FILE = joinpath(DATA_DIR, "trading_game", "universe_latest.json")

const FLAGS = Dict(
    "--window" => ("window", Int), "--d" => ("d", Int), "--heads" => ("heads", Int), "--blocks" => ("blocks", Int),
    "--dropout" => ("dropout", Float64), "--epochs" => ("epochs", Int), "--steps-per-epoch" => ("steps_per_epoch", Int),
    "--batch" => ("batch", Int), "--lr" => ("lr", Float64), "--weight-decay" => ("weight_decay", Float64),
    "--warmup-steps" => ("warmup_steps", Int), "--val-windows" => ("val_windows", Int), "--device" => ("device", String), "--dtype" => ("dtype", String), "--loss" => ("loss", String),
    "--nu" => ("nu", Float64), "--nll-beta" => ("nll_beta", Float64),
    "--seed" => ("seed", Int), "--val-window" => ("val_window", String), "--val-days" => ("val_days", Int),
    "--train-start" => ("train_start", Date), "--train-end" => ("train_end", Date),
    "--val-start" => ("val_start", Date), "--val-end" => ("val_end", Date))

function parse_args()
    config_path = RunConfigs.take_config_arg!(ARGS)
    print_config = RunConfigs.take_flag!(ARGS, "--print-config")
    cfg = config_path === nothing ? nothing : RunConfigs.load_or_exit(config_path)
    opts = Dict{String, Any}(
        "window" => 96, "d" => 64, "heads" => 4, "blocks" => 4, "dropout" => 0.1,
        "epochs" => 20, "steps_per_epoch" => 500, "batch" => 32, "lr" => 3e-4, "weight_decay" => 0.01,
        "warmup_steps" => 200, "val_windows" => 256, "device" => "cpu", "dtype" => "float32", "loss" => "student_t", "nu" => 4.0, "nll_beta" => 0.5, "seed" => nothing,
        "val_window" => "trailing", "val_days" => 60,
        "train_start" => nothing, "train_end" => nothing, "val_start" => nothing, "val_end" => nothing,
        "granularity" => "1min", "out" => nothing, "resume" => false, "init_from" => "", "eval_only" => false, "out_dir" => nothing)
    for sec in ("dates", "forecaster_cache", "forecaster")
        merge!(opts, Dict(k => v for (k, v) in RunConfigs.section_opts(cfg, sec) if k in ("out", "granularity", keys(opts)...)))
    end
    isempty(opts["init_from"]) || (opts["init_from"] = abspath(joinpath(REPO_ROOT, opts["init_from"])))
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a in ("-h", "--help")
            println(match(r"\"\"\"\n(.*?)\"\"\""s, read(@__FILE__, String)).captures[1])
            exit(0)
        elseif a == "--resume"; opts["resume"] = true; i += 1
        elseif a == "--no-resume"; opts["resume"] = false; i += 1
        elseif a == "--init-from"
            i < length(ARGS) || error("--init-from needs a path")
            opts["init_from"] = abspath(ARGS[i+1]); i += 2
        elseif a == "--granularity"; opts["granularity"] = lowercase(ARGS[i+1]); i += 2
        elseif a == "--eval-only"; opts["eval_only"] = true; i += 1
        elseif a == "--out-dir"; opts["out_dir"] = abspath(ARGS[i+1]); i += 2
        elseif haskey(FLAGS, a)
            i < length(ARGS) || error("$a needs a value")
            key, T = FLAGS[a]
            opts[key] = T === String ? ARGS[i+1] : T === Date ? Date(ARGS[i+1]) : parse(T, ARGS[i+1])
            i += 2
        else
            error("Unknown argument: $a (see --help)")
        end
    end
    opts["device"] in ("cpu", "gpu") || error("--device must be cpu or gpu")
    opts["resume"] && !isempty(opts["init_from"]) && error("resume and init_from are mutually exclusive: resume continues the run in --out-dir, init_from starts a fresh one")
    opts["eval_only"] && !isempty(opts["init_from"]) && error("--eval-only scores the checkpoint in --out-dir; it cannot be combined with init_from")
    forecast_dtype(opts["dtype"])
    forecast_loss_kind(opts["loss"])
    (forecast_loss_kind(opts["loss"]) === :huber || opts["nu"] > 2) || error("--nu must exceed 2 for a finite variance")
    opts["val_window"] in ("trailing", "same") || error("--val-window must be trailing or same")
    if print_config
        RunConfigs.print_resolved(stdout, "train_minute_forecaster.jl", config_path, opts)
        exit(0)
    end
    return opts, config_path
end

function resolve_device(requested::String)::Symbol
    requested == "cpu" && return :cpu
    if !CUDA.functional()
        @warn "--device gpu requested but CUDA.functional() is false — falling back to cpu"
        return :cpu
    end
    @info "Training on GPU: $(CUDA.name(CUDA.device()))"
    return :gpu
end

"""Columns of `cache` holding `symbols`, in order; an error naming any that are missing."""
function columns_of(cache::MinuteCache, symbols::Vector{String}, role::String)::Vector{Int}
    idx = Dict(s => i for (i, s) in enumerate(cache.symbols))
    gone = [s for s in symbols if !haskey(idx, s)]
    isempty(gone) || error("$(length(gone)) $role companies are not in the minute cache (e.g. $(first(gone))); " *
                           "rebuild it: julia -t auto --project=packages/TradingGame scripts/build_forecaster_cache.jl --granularity $(cache.granularity)")
    return [idx[s] for s in symbols]
end

_nan_to_nothing(x) = x isa Real && !isfinite(x) ? nothing : x

"""`val_report.json`: the full result of every evaluation set, with per-stock skill keyed by symbol."""
function write_report(path::String, model, cache, sets::Vector{EvalSet}, cfg, device, dtype, loss, nu)
    report = Dict{String, Any}()
    for es in sets
        r = evaluate_forecaster(model, cache, es, cfg.window; device=device, dtype=dtype, loss=loss, nu=nu)
        report[es.name] = Dict(
            "n_windows" => r.n_windows, "n_targets" => r.n_targets, "loss" => r.loss, "mse" => r.mse,
            "mse_zero" => r.mse_zero, "mse_repeat_last" => r.mse_last, "skill_vs_zero" => r.skill,
            "skill_repeat_last_vs_zero" => r.skill_last, "sign_hit_rate" => r.hit_rate,
            "timestamp_minute_acc" => r.minute_acc, "timestamp_gap_acc" => r.gap_acc, "timestamp_weekday_acc" => r.weekday_acc,
            "distribution" => string(loss), "nu" => nu,
            "nll" => _nan_to_nothing(r.nll), "nll_const_sigma" => _nan_to_nothing(r.nll_const), "nll_ewma_sigma" => _nan_to_nothing(r.nll_ewma),
            "nll_gain_vs_const_sigma" => _nan_to_nothing(r.nll_const - r.nll), "nll_gain_vs_ewma_sigma" => _nan_to_nothing(r.nll_ewma - r.nll),
            "coverage_68" => _nan_to_nothing(r.cov68), "coverage_95" => _nan_to_nothing(r.cov95),
            "std_by_slot" => [Dict("from_minute" => (i - 1) * cache.bucket_bars * cache.bar_minutes, "predicted_std" => _nan_to_nothing(a), "realised_rms" => _nan_to_nothing(b))
                              for (i, (a, b)) in enumerate(zip(r.std_pred_slot, r.rms_slot))],
            "per_stock_skill" => Dict(cache.symbols[c] => _nan_to_nothing(s) for (c, s) in zip(es.cols, r.per_stock_skill)),
            "per_slot_skill" => [Dict("from_minute" => (i - 1) * cache.bucket_bars * cache.bar_minutes, "skill" => _nan_to_nothing(s))
                                 for (i, s) in enumerate(r.per_slot_skill)])
    end
    open(path, "w") do io
        JSON3.pretty(io, report)
    end
end

function main()
    opts, config_path = parse_args()
    haskey(GRANULARITIES, opts["granularity"]) || error("--granularity must be one of $(join(sort(collect(keys(GRANULARITIES))), ", "))")
    cache_path = opts["out"] === nothing ? normpath(joinpath(DATA_DIR, "forecaster_cache_$(opts["granularity"]).bson")) : abspath(joinpath(REPO_ROOT, opts["out"]))
    isfile(cache_path) || error("Not found: $cache_path\nBuild it: julia -t auto --project=packages/TradingGame scripts/build_forecaster_cache.jl --granularity $(opts["granularity"])")
    cache = load_minute_cache(cache_path)
    OUT_DIR = opts["out_dir"] === nothing ? (cache.granularity == "1min" ? DEFAULT_OUT_DIR : DEFAULT_OUT_DIR * "_" * cache.granularity) : opts["out_dir"]
    mkpath(OUT_DIR)
    config_path === nothing || cp(config_path, joinpath(OUT_DIR, "config.toml"); force=true)
    checkpoint = joinpath(OUT_DIR, "forecaster.bson")
    isfile(UNIVERSE_FILE) || error("Not found: $UNIVERSE_FILE\nRun: julia --project=packages/TradingGame scripts/build_market_universe_snapshot.jl")
    universe = load_universe_snapshot(UNIVERSE_FILE)
    train_cols = columns_of(cache, universe.train, "train")
    val_cols   = columns_of(cache, universe.val, "val")
    @info "Forecaster cache ($(cache.granularity), $(cache.bars_per_day) bars/day): $(length(cache.dates)) days ($(first(cache.dates)) .. $(last(cache.dates))), " *
          "$(length(train_cols)) train / $(length(val_cols)) val companies"

    train_start, train_end, val_start, val_end = resolve_date_windows(
        first(cache.dates), last(cache.dates), opts["val_window"], opts["val_days"],
        opts["train_start"], opts["train_end"], opts["val_start"], opts["val_end"])
    @info "Train window: $train_start .. $train_end   Val window: $val_start .. $val_end"

    if opts["resume"] || opts["eval_only"]
        isfile(checkpoint) || error("--resume/--eval-only needs a checkpoint, but there is none at $checkpoint")
        model, mcfg, _ = load_forecaster(checkpoint)
        mcfg.bars_per_day == cache.bars_per_day || error("the checkpoint was trained on $(mcfg.bars_per_day) bars/day, but the $(cache.granularity) cache has $(cache.bars_per_day)")
        for (k, have) in (("window", mcfg.window), ("d", mcfg.d), ("heads", mcfg.heads), ("blocks", mcfg.blocks))
            opts[k] != have && @warn "--resume keeps the checkpoint's $k = $have (the config says $(opts[k]))"
        end
    else
        mcfg = ForecasterConfig(window=opts["window"], d=opts["d"], heads=opts["heads"], blocks=opts["blocks"],
                                dropout=Float32(opts["dropout"]), bars_per_day=cache.bars_per_day)
        if isempty(opts["init_from"])
            model = Forecaster(mcfg; seed=opts["seed"])
        else
            model = warm_start_forecaster(mcfg, opts["init_from"]; seed=opts["seed"])
            @info "Fresh run, weights warm-started from $(opts["init_from"])"
        end
    end
    M = mcfg.window

    train_range = window_starts(cache, train_start, train_end, M)
    val_range   = window_starts(cache, val_start, val_end, M)
    isempty(train_range) && error("No training window fits in $train_start .. $train_end with window $M")
    isempty(val_range) && error("No validation window fits in $val_start .. $val_end with window $M")
    sets = [EvalSet("val", val_cols, spread_starts(val_range, opts["val_windows"], M)),
            EvalSet("val_train_companies", train_cols, spread_starts(val_range, opts["val_windows"], M)),
            EvalSet("train", train_cols, spread_starts(train_range, opts["val_windows"], M))]
    @info "$(length(train_range)) possible training windows; evaluation sets of $(length(sets[1].starts)) windows"

    device = resolve_device(opts["device"])
    if opts["eval_only"]
        write_report(joinpath(OUT_DIR, "val_report.json"), model, cache, sets, mcfg, device, forecast_dtype(opts["dtype"]),
                     forecast_loss_kind(opts["loss"]), opts["nu"])
        println("Report: ", joinpath(OUT_DIR, "val_report.json"))
        return
    end
    epoch_log = joinpath(OUT_DIR, "epoch_log.jsonl")
    epoch_offset = 0
    if opts["resume"]
        isfile(epoch_log) && (epoch_offset = maximum(Int(JSON3.read(l).epoch) for l in eachline(epoch_log) if !isempty(strip(l)); init=0))
        @info "Resumed — last completed epoch: $epoch_offset"
    else
        for f in ("epoch_log.jsonl", "timing.json", "val_report.json", "live_status.json")
            isfile(joinpath(OUT_DIR, f)) && rm(joinpath(OUT_DIR, f))
        end
    end

    open(joinpath(OUT_DIR, "run_config.json"), "w") do io
        JSON3.pretty(io, Dict(
            "model" => Dict("window" => M, "d" => mcfg.d, "heads" => mcfg.heads, "blocks" => mcfg.blocks, "dropout" => mcfg.dropout, "bars_per_day" => mcfg.bars_per_day, "granularity" => cache.granularity,
                            "parameters" => n_parameters(model)),
            "training" => Dict(k => opts[k] for k in ("epochs", "steps_per_epoch", "batch", "lr", "weight_decay", "warmup_steps", "val_windows", "seed")),
            "resolved_device" => string(device), "dtype" => opts["dtype"], "loss" => opts["loss"], "nu" => opts["nu"], "nll_beta" => opts["nll_beta"],
            "dates" => Dict("train_start" => string(train_start), "train_end" => string(train_end),
                            "val_start" => string(val_start), "val_end" => string(val_end)),
            "train_companies" => universe.train, "val_companies" => universe.val,
            "return_scale_percent" => cache.scale, "resumed" => opts["resume"], "init_from" => isempty(opts["init_from"]) ? nothing : opts["init_from"],
            "started" => string(now())))
    end
    @info "Forecaster: $(n_parameters(model)) parameters (window $M, d $(mcfg.d), $(mcfg.heads) heads, $(mcfg.blocks) blocks)"

    rng = opts["seed"] === nothing ? Random.default_rng() : MersenneTwister(opts["seed"])
    log = train_forecaster!(model, cache, train_cols, train_range, sets;
                            epochs=opts["epochs"], steps_per_epoch=opts["steps_per_epoch"], batch=opts["batch"],
                            lr=opts["lr"], weight_decay=opts["weight_decay"], warmup_steps=opts["warmup_steps"],
                            device=device, dtype=forecast_dtype(opts["dtype"]), loss=forecast_loss_kind(opts["loss"]), nu=opts["nu"], beta=opts["nll_beta"],
                            rng=rng, out_dir=OUT_DIR, epoch_offset=epoch_offset,
                            meta=Dict{String, Any}("loss" => opts["loss"], "nu" => opts["nu"], "nll_beta" => opts["nll_beta"], "dtype" => opts["dtype"]))
    write_report(joinpath(OUT_DIR, "val_report.json"), model, cache, sets, mcfg, device, forecast_dtype(opts["dtype"]),
                 forecast_loss_kind(opts["loss"]), opts["nu"])

    best = log["history"]
    @printf("\nDone. %d epochs run, best epoch %d (val loss %.4f)%s\n", log["epochs_run"], log["best_epoch"], log["best_loss"],
            isempty(log["stopped"]) ? "" : ", stopped by " * log["stopped"])
    println("Report: ", joinpath(OUT_DIR, "val_report.json"))
end

main()
