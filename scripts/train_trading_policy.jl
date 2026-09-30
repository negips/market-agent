"""
train_trading_policy.jl

Drives `TradingGame.train_policy!`: loads the inference cache + candidate
universe, splits the cache's date range into a training window and a
held-out validation tail, builds a fresh `ActorCriticPolicy`, and trains.

Usage:
  julia --project=packages/TradingGame scripts/train_trading_policy.jl
  julia --project=packages/TradingGame scripts/train_trading_policy.jl --iterations 500
  julia --project=packages/TradingGame scripts/train_trading_policy.jl --val-days 40 --live

Prerequisites:
  website/data/inference_cache.bson              (build_cache.jl)
  website/data/trading_game/universe_latest.json (build_universe_snapshot.jl)

Outputs (under website/data/trading_game/):
  policy.bson          — best checkpoint (by held-out return)
  episode_log.jsonl     — per-iteration train/val return, streamed
  live_status.json      — current episode's live trajectory, with --live (default on)

To stop cleanly:   touch website/data/trading_game/STOP        (checkpoint saved)
To stop hard:      touch website/data/trading_game/STOP_NOW    (no save)
"""

using TradingGame, StockSwingPredictor, Dates, Printf

const REPO_ROOT   = joinpath(@__DIR__, "..")
const DATA_DIR    = joinpath(REPO_ROOT, "website", "data", "trading_game")
const CACHE_FILE  = joinpath(REPO_ROOT, "website", "data", "inference_cache.bson")
const UNIVERSE_FILE = joinpath(DATA_DIR, "universe_latest.json")

function parse_args()
    opts = Dict{String, Any}(
        "iterations"  => 300,
        "initial_cash" => 1_000_000.0,
        "val_days"    => 60,
        "eval_every"  => 10,
        "lr"          => 3f-4,
        "live"        => true,
    )
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a in ("--help", "-h")
            println("""
Usage:
  julia --project=packages/TradingGame scripts/train_trading_policy.jl [options]

Options:
  --iterations N      Training iterations (default: $(opts["iterations"]))
  --initial-cash N    Starting cash per episode (default: $(opts["initial_cash"]))
  --val-days N        Held-out tail length in calendar days (default: $(opts["val_days"]))
  --eval-every N       Run a held-out episode every N iterations (default: $(opts["eval_every"]))
  --lr N               Adam learning rate (default: $(opts["lr"]))
  --no-live            Disable live_status.json streaming
""")
            exit(0)
        elseif a == "--iterations";   opts["iterations"]   = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--initial-cash"; opts["initial_cash"] = parse(Float64, ARGS[i+1]); i += 2
        elseif a == "--val-days";     opts["val_days"]     = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--eval-every";   opts["eval_every"]   = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--lr";           opts["lr"]           = parse(Float32, ARGS[i+1]); i += 2
        elseif a == "--no-live";      opts["live"]         = false; i += 1
        else; i += 1
        end
    end
    return opts
end

function main()
    opts = parse_args()
    mkpath(DATA_DIR)

    isfile(CACHE_FILE) || error(
        "Not found: $CACHE_FILE\nRun: julia --project=packages/StockSwingPredictor scripts/build_cache.jl")
    isfile(UNIVERSE_FILE) || error(
        "Not found: $UNIVERSE_FILE\nRun: julia --project=packages/TradingGame scripts/build_universe_snapshot.jl")

    @info "Loading inference cache…"
    cache = load_inference_cache(CACHE_FILE)
    universe = load_universe_snapshot(UNIVERSE_FILE)
    @info "  $(length(cache.companies)) companies cached, $(length(universe)) candidates in universe"

    cache_start, cache_end = first(cache.dates), last(cache.dates)
    val_start   = cache_end - Day(opts["val_days"])
    train_start = cache_start
    train_end   = val_start - Day(1)

    train_start >= train_end && error(
        "Cache date range ($cache_start .. $cache_end) is too short for a " *
        "$(opts["val_days"])-day held-out tail — shrink --val-days or extend the cache")

    @info "Train window: $train_start .. $train_end"
    @info "Val window:   $val_start .. $cache_end"

    train_config = EpisodeConfig(initial_cash=opts["initial_cash"], start_date=train_start,
                                  end_date=train_end, candidate_universe=universe)
    val_config   = EpisodeConfig(initial_cash=opts["initial_cash"], start_date=val_start,
                                  end_date=cache_end, candidate_universe=universe)

    env    = TradingGameEnv(cache)
    policy = ActorCriticPolicy()

    policy, log = train_policy!(policy, env, train_config; val_config=val_config,
        iterations=opts["iterations"], eval_every=opts["eval_every"], lr=opts["lr"],
        checkpoint_path=joinpath(DATA_DIR, "policy.bson"),
        episode_log_path=joinpath(DATA_DIR, "episode_log.jsonl"),
        stop_file=joinpath(DATA_DIR, "STOP"),
        live_path=opts["live"] ? joinpath(DATA_DIR, "live_status.json") : "")

    save_training_log(log, joinpath(DATA_DIR, "training_log.json"))

    @printf("\nDone. %d iterations, best return %.4f (iter %d)\n",
            log["iterations_run"], log["best_return"], log["best_iteration"])
end

main()
