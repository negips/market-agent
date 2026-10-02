"""
train_trading_policy.jl

Drives `TradingGame.train_policy!`: loads the inference cache + candidate
universe, splits the cache's date range into a training window and a
held-out validation tail, builds a fresh `ActorCriticPolicy`, and trains.

Usage:
  julia --project=packages/TradingGame scripts/train_trading_policy.jl
  julia --project=packages/TradingGame scripts/train_trading_policy.jl --iterations 500
  julia --project=packages/TradingGame scripts/train_trading_policy.jl --val-days 40 --live
  julia --project=packages/TradingGame scripts/train_trading_policy.jl --resume --iterations 100
  julia --project=packages/TradingGame scripts/train_trading_policy.jl --seed 42
  julia --project=packages/TradingGame scripts/train_trading_policy.jl --init-from other_run/policy.bson
  julia --project=packages/TradingGame scripts/train_trading_policy.jl --device gpu

Prerequisites:
  website/data/inference_cache.bson              (build_cache.jl)
  website/data/trading_game/universe_latest.json (build_universe_snapshot.jl)

Outputs (under website/data/trading_game/):
  policy.bson          — best checkpoint (by held-out return)
  episode_log.jsonl     — per-iteration train/val return, streamed
  live_status.json      — current episode's live trajectory, with --live (default on)
  val_runs.jsonl         — every held-out episode's full value curve + trades,
                           one appended line per eval — with --live (default on)
  run_config.json        — effective training-config flags, for --resume (see below)

To stop cleanly:   touch website/data/trading_game/STOP        (checkpoint saved)
To stop hard:      touch website/data/trading_game/STOP_NOW    (no save)

To restart after a STOP: re-run with --resume. This loads policy.bson (the
last checkpoint) instead of a fresh policy, and continues episode_log.jsonl's
iteration numbering from where it left off (`--iterations` means "how many
MORE iterations to run", not a new total) — same convention as
StockSwingPredictor's scripts/train_model.jl --resume. A STOP_NOW abort has
no checkpoint newer than the last improvement before it, so --resume after
one just continues from that same last-good checkpoint (no data lost, some
unsaved training since then is simply redone).

--init-from PATH is different from --resume: it starts a genuinely FRESH run
(iteration numbering restarts at 1, episode_log.jsonl/val_runs.jsonl are
cleared, best_return tracking restarts at -Inf — so the first checkpoint
write is unconditional, same as any fresh run's first improvement) but
initialises the policy's
weights from an existing checkpoint at PATH instead of random init. Useful
for warm-starting a new run (different window/hyperparameters/universe) from
weights already trained elsewhere, without inheriting that run's log or
iteration count. --resume and --init-from are mutually exclusive.

--seed N makes a FRESH policy's initial weights reproducible (via
Random.seed!, see ActorCriticPolicy's docstring) and also seeds the PPO
rollout's stochastic action sampling, so two runs with the same --seed (and
otherwise identical arguments) produce identical training trajectories.
Ignored by --resume and --init-from's weight loading (the weights come from
a checkpoint, not fresh init) but still seeds the rollout sampling in both
cases.

--device gpu moves the policy to GPU once, up front — requires
CUDA.functional() (falls back to cpu with a warning if not, same as
train_model.jl). --minibatch defaults to 256 on gpu / 32 on cpu when not
given explicitly (larger batches better amortise transfer/kernel-launch
overhead) — same convention as train_model.jl's batch-size default.
collect_rollout always runs its own forward pass on CPU regardless of
--device — its one-bar-at-a-time calls measured roughly 10x SLOWER on GPU
than CPU for this network (host round-trip + 120 individual GRU-step kernel
launches per call dominate over the tiny per-call compute), so --device gpu
only accelerates ppo_update!'s minibatched passes, where batching actually
helps — see ppo.jl's docstrings.

Every run writes its effective --initial-cash/--val-days/--eval-every/--lr/
--entropy/--seed/--device/--minibatch into run_config.json (alongside
policy.bson).
--resume reads it back and uses those values for any of those flags NOT
also given explicitly on the resume command line — an explicit flag on the
command line always wins over the saved value. This is what makes a bare
`--resume` reproduce the original run's config instead of silently reverting
to script defaults (e.g. --val-days back to 60, corrupting the train/val
split against what the checkpoint was actually trained on). Delete
run_config.json (or pass the flags explicitly) to intentionally change
config on resume.
"""

using TradingGame, StockSwingPredictor, Dates, Printf, JSON3, Random, CUDA

const REPO_ROOT   = joinpath(@__DIR__, "..")
const DATA_DIR    = joinpath(REPO_ROOT, "website", "data", "trading_game")
const CACHE_FILE  = joinpath(REPO_ROOT, "website", "data", "inference_cache.bson")
const UNIVERSE_FILE = joinpath(DATA_DIR, "universe_latest.json")

"""Opts persisted to/restored from `run_config.json` on `--resume` — training
config that must stay consistent across a resumed run, not runtime-only
flags like `--live`/`--resume`/`--init-from`/`--iterations` (the latter means
"how many more" each time by design, so it's never something to restore)."""
const RESUMABLE_KEYS = ("initial_cash", "val_days", "eval_every", "lr", "entropy_coef", "seed", "device", "minibatch")

function parse_args()
    opts = Dict{String, Any}(
        "iterations"  => 300,
        "initial_cash" => 1_000_000.0,
        "val_days"    => 60,
        "eval_every"  => 10,
        "lr"          => 3f-4,
        "entropy_coef" => ENTROPY_COEF,
        "live"        => true,
        "resume"      => false,
        "seed"        => nothing,
        "init_from"   => "",
        "device"      => "cpu",
        "minibatch"   => nothing,
    )
    explicit = Set{String}()
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
  --entropy N           PPO entropy bonus weight — higher keeps the policy's
                        action distribution spread out for longer, at the
                        cost of noisier rollouts (default: $(opts["entropy_coef"]))
  --no-live            Disable live_status.json streaming
  --resume             Load policy.bson and continue iteration numbering from
                        episode_log.jsonl instead of starting a fresh policy
  --init-from PATH     Fresh run (iteration 1, cleared log) but weights loaded
                        from PATH instead of random init. Mutually exclusive
                        with --resume.
  --seed N             Reproducible initial weights (fresh/--init-from) and
                        PPO rollout sampling
  --device NAME        cpu or gpu (default: cpu). gpu requires
                        CUDA.functional() — falls back to cpu with a warning
  --minibatch N         PPO minibatch size (default: 256 on gpu, 32 on cpu)
""")
            exit(0)
        elseif a == "--iterations";   opts["iterations"]   = parse(Int, ARGS[i+1]); push!(explicit, "iterations"); i += 2
        elseif a == "--initial-cash"; opts["initial_cash"] = parse(Float64, ARGS[i+1]); push!(explicit, "initial_cash"); i += 2
        elseif a == "--val-days";     opts["val_days"]     = parse(Int, ARGS[i+1]); push!(explicit, "val_days"); i += 2
        elseif a == "--eval-every";   opts["eval_every"]   = parse(Int, ARGS[i+1]); push!(explicit, "eval_every"); i += 2
        elseif a == "--lr";           opts["lr"]           = parse(Float32, ARGS[i+1]); push!(explicit, "lr"); i += 2
        elseif a == "--entropy";      opts["entropy_coef"] = parse(Float64, ARGS[i+1]); push!(explicit, "entropy_coef"); i += 2
        elseif a == "--no-live";      opts["live"]         = false; i += 1
        elseif a == "--resume";       opts["resume"]       = true; i += 1
        elseif a == "--init-from";    opts["init_from"]    = ARGS[i+1]; i += 2
        elseif a == "--seed";         opts["seed"]         = parse(Int, ARGS[i+1]); push!(explicit, "seed"); i += 2
        elseif a == "--device";       opts["device"]       = ARGS[i+1]; push!(explicit, "device"); i += 2
        elseif a == "--minibatch";    opts["minibatch"]    = parse(Int, ARGS[i+1]); push!(explicit, "minibatch"); i += 2
        else; i += 1
        end
    end
    opts["resume"] && !isempty(opts["init_from"]) &&
        error("--resume and --init-from are mutually exclusive")
    opts["device"] in ("cpu", "gpu") ||
        error("Unknown --device '$(opts["device"])'. Expected: cpu, gpu")
    return opts, explicit
end

"""Resolve `--device` to a `:cpu`/`:gpu` symbol, falling back to `:cpu` if gpu
was requested but no functional CUDA backend is available — same convention
as `train_model.jl`'s `_resolve_device`."""
function _resolve_device(requested::String)::Symbol
    requested == "cpu" && return :cpu
    if !CUDA.functional()
        @warn "--device gpu requested but CUDA.functional() is false — falling back to cpu"
        return :cpu
    end
    @info "Training on GPU: $(CUDA.name(CUDA.device()))"
    return :gpu
end

"""Resolve `--minibatch` to a concrete size, defaulting to 256 on gpu / 32 on
cpu when not given explicitly — same convention as `train_model.jl`'s
`_resolve_batch`."""
function _resolve_minibatch(requested::Union{Int, Nothing}, device::Symbol)::Int
    isnothing(requested) || return requested
    return device === :gpu ? 256 : 32
end

_fmt_money(v) = "₹" * replace(@sprintf("%.0f", v), r"(\d)(?=(\d{3})+(?!\d))" => s"\1,")

"""Print this run's effective hyperparameters and the rule-derived constants
from `constants.jl`. The latter are NOT saved to `run_config.json` (only
CLI-exposed flags are — see `RESUMABLE_KEYS`) and have no flag of their own,
so if they're hand-edited directly in `constants.jl` between runs (rather
than through a CLI flag), this printout is the only record of what was
actually in effect for a given run — check it before relying on a comparison
across runs. Called once, after every opt has been fully resolved
(`--resume`-restored values applied, `device`/`minibatch` auto-resolved)."""
function _print_training_params(opts::Dict, device::Symbol, minibatch::Int, n_candidates::Int,
                                 train_start::Date, train_end::Date, val_start::Date, val_end::Date)
    n_max = n_max_holdings(n_candidates)
    println("═"^64)
    println("Training parameters")
    println("═"^64)
    println("Run config:")
    @printf("  %-22s %d\n",  "iterations:",   opts["iterations"])
    @printf("  %-22s %s\n",  "device:",       device)
    @printf("  %-22s %d\n",  "minibatch:",    minibatch)
    @printf("  %-22s %s\n",  "lr:",           opts["lr"])
    @printf("  %-22s %s\n",  "entropy_coef:", opts["entropy_coef"])
    @printf("  %-22s %d\n",  "eval_every:",   opts["eval_every"])
    @printf("  %-22s %d\n",  "val_days:",     opts["val_days"])
    @printf("  %-22s %s\n",  "initial_cash:", _fmt_money(opts["initial_cash"]))
    @printf("  %-22s %s\n",  "seed:",         something(opts["seed"], "none"))
    @printf("  %-22s %s\n",  "resume:",       opts["resume"])
    @printf("  %-22s %s\n",  "init_from:",    isempty(opts["init_from"]) ? "none" : opts["init_from"])
    @printf("  %-22s %d\n",  "n_candidates:", n_candidates)
    println("  train window:          $train_start .. $train_end")
    println("  val window:            $val_start .. $val_end")
    println()
    println("Rule-derived constants (constants.jl):")
    @printf("  %-26s %s\n",     "FEE_RATE:",                  FEE_RATE)
    @printf("  %-26s %s\n",     "SETTLEMENT_DAYS:",           SETTLEMENT_DAYS)
    @printf("  %-26s %s\n",     "MIN_HOLD_DAYS:",             MIN_HOLD_DAYS)
    @printf("  %-26s %s\n",     "MAX_HOLD_DAYS:",             MAX_HOLD_DAYS)
    @printf("  %-26s %s\n",     "MAX_POSITION_FRACTION:",     MAX_POSITION_FRACTION)
    @printf("  %-26s %s (→ N_MAX = %d for this universe)\n",
                                 "N_MAX_HOLDINGS_FRACTION:",  N_MAX_HOLDINGS_FRACTION, n_max)
    @printf("  %-26s %s\n",     "MAX_CASH_FRACTION:",         MAX_CASH_FRACTION)
    @printf("  %-26s %s\n",     "CASH_CEILING_PENALTY_COEF:", CASH_CEILING_PENALTY_COEF)
    @printf("  %-26s %s\n",     "REBUY_COOLDOWN_DAYS:",       REBUY_COOLDOWN_DAYS)
    @printf("  %-26s %s / %s\n","GAMMA / GAE_LAMBDA:",        GAMMA, GAE_LAMBDA)
    @printf("  %-26s %s\n",     "CLIP_EPS:",                  CLIP_EPS)
    @printf("  %-26s %s\n",     "VALUE_LOSS_COEF:",           VALUE_LOSS_COEF)
    println("═"^64)
end

"""Highest `iteration` field logged in `log_path`, or 0 if it doesn't exist
yet — same convention as `train_model.jl`'s `_last_completed_epoch`, used so
`--resume` continues iteration numbering instead of restarting at 1."""
function _last_completed_iteration(log_path::String)::Int
    isfile(log_path) || return 0
    max_iter = 0
    for line in eachline(log_path)
        isempty(strip(line)) && continue
        try
            max_iter = max(max_iter, Int(JSON3.read(line)[:iteration]))
        catch
        end
    end
    return max_iter
end

"""Write the `RESUMABLE_KEYS` subset of `opts` to `path` as JSON — the
*requested* values (e.g. `device="gpu"` even if it later falls back to cpu,
`minibatch=nothing` if left on auto), not resolved/derived ones, so a later
`--resume` re-derives exactly the same way the original run did.

`n_candidates` is informational only (not a `RESUMABLE_KEYS` entry — nothing
reads it back on `--resume`, since the candidate universe always comes from
`universe_latest.json` at load time, not from this file): how many companies
`UNIVERSE_FILE` held for *this* run, so `run_config.json` is a quick way to
see what universe size a checkpoint was actually trained against without
cross-referencing `universe_latest.json`'s own `generated_at`/history."""
function save_run_config(opts::Dict, path::String, n_candidates::Int)
    open(path, "w") do io
        JSON3.pretty(io, merge(Dict(k => opts[k] for k in RESUMABLE_KEYS),
                                Dict("n_candidates" => n_candidates)))
    end
end

"""Fill in any `RESUMABLE_KEYS` entry of `opts` NOT in `explicit` from the
saved `run_config.json` at `path`, so `--resume` reproduces the original
run's config instead of silently reverting to script defaults. A flag given
explicitly on this invocation always overrides the saved value. No-op
(with a warning) if `path` doesn't exist — an old checkpoint predating this
feature, or one someone moved by hand."""
function load_run_config!(opts::Dict, explicit::Set{String}, path::String)
    if !isfile(path)
        @warn "--resume: no saved run config at $path — using CLI/defaults for any flag not explicitly given"
        return nothing
    end
    saved = JSON3.read(read(path, String))
    for k in RESUMABLE_KEYS
        k in explicit && continue
        haskey(saved, Symbol(k)) || continue
        v = saved[Symbol(k)]
        if v === nothing
            opts[k] = k in ("seed", "minibatch") ? nothing : opts[k]
        elseif k == "lr"
            opts[k] = Float32(v)
        elseif k == "entropy_coef"
            opts[k] = Float64(v)
        elseif k in ("val_days", "eval_every", "seed", "minibatch")
            opts[k] = Int(v)
        else
            opts[k] = v
        end
    end
    @info "Loaded saved run config from $path for flags not given explicitly"
    return nothing
end

function main()
    opts, explicit = parse_args()
    mkpath(DATA_DIR)
    config_path = joinpath(DATA_DIR, "run_config.json")
    opts["resume"] && load_run_config!(opts, explicit, config_path)

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

    env = TradingGameEnv(cache)

    checkpoint_path  = joinpath(DATA_DIR, "policy.bson")
    episode_log_path = joinpath(DATA_DIR, "episode_log.jsonl")
    val_curve_path   = joinpath(DATA_DIR, "val_runs.jsonl")

    iteration_offset = 0
    if opts["resume"]
        isfile(checkpoint_path) ||
            error("--resume requested but no checkpoint found at $checkpoint_path")
        policy, hp, _ = load_policy(checkpoint_path)
        iteration_offset = _last_completed_iteration(episode_log_path)
        @info "Resumed policy — last completed iteration: $iteration_offset"
    elseif !isempty(opts["init_from"])
        isfile(opts["init_from"]) ||
            error("--init-from: not found: $(opts["init_from"])")
        policy, hp, _ = load_policy(opts["init_from"])
        @info "Fresh run, weights warm-started from $(opts["init_from"])"
        # Fresh iteration numbering and log, unlike --resume — see the module
        # docstring's --init-from vs --resume note. val_runs.jsonl follows the
        # same lifecycle as episode_log.jsonl — a fresh iteration-1 run means a
        # fresh held-out-curve history too.
        isfile(episode_log_path) && rm(episode_log_path)
        isfile(val_curve_path) && rm(val_curve_path)
    else
        policy = ActorCriticPolicy(seed=opts["seed"])
        hp = (embed_dim=64, macro_embed_dim=16, attn_heads=4, critic_hidden=[64, 32])
        @info "Fresh policy" * (opts["seed"] === nothing ? "" : " (seed=$(opts["seed"]))")
        # episode_log.jsonl/val_runs.jsonl are both opened in append mode
        # inside train_policy! (so --resume can keep history) — a fresh run
        # must clear any stale ones.
        isfile(episode_log_path) && rm(episode_log_path)
        isfile(val_curve_path) && rm(val_curve_path)
    end

    rng       = opts["seed"] === nothing ? Random.default_rng() : MersenneTwister(opts["seed"])
    device    = _resolve_device(opts["device"])
    minibatch = _resolve_minibatch(opts["minibatch"], device)

    save_run_config(opts, config_path, length(universe))
    _print_training_params(opts, device, minibatch, length(universe), train_start, train_end, val_start, cache_end)

    policy, log = train_policy!(policy, env, train_config; val_config=val_config,
        iterations=opts["iterations"], eval_every=opts["eval_every"], lr=opts["lr"],
        entropy_coef=opts["entropy_coef"], rng=rng,
        minibatch_size=minibatch, device=device,
        checkpoint_path=checkpoint_path,
        episode_log_path=episode_log_path,
        stop_file=joinpath(DATA_DIR, "STOP"),
        live_path=opts["live"] ? joinpath(DATA_DIR, "live_status.json") : "",
        val_curve_path=opts["live"] ? val_curve_path : "",
        iteration_offset=iteration_offset,
        embed_dim=hp.embed_dim, macro_embed_dim=hp.macro_embed_dim,
        attn_heads=hp.attn_heads, critic_hidden=hp.critic_hidden)

    save_policy_training_log(log, joinpath(DATA_DIR, "training_log.json"))

    @printf("\nDone. %d iterations, best return %.4f (iter %d)\n",
            log["iterations_run"], log["best_return"], log["best_iteration"])
end

main()
