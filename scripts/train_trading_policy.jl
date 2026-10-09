"""
train_trading_policy.jl

Drives `TradingGame.train_policy!`: loads the inference cache + the train/val
candidate universes (see `build_market_universe_snapshot.jl` — a
`UniverseStrategy` may give train and val different companies, not just
different dates), resolves the train/val date windows, builds a fresh
`ActorCriticPolicy`, and trains.

Usage:
  julia --project=packages/TradingGame scripts/train_trading_policy.jl
  julia --project=packages/TradingGame scripts/train_trading_policy.jl --iterations 500
  julia --project=packages/TradingGame scripts/train_trading_policy.jl --val-days 40 --live
  julia --project=packages/TradingGame scripts/train_trading_policy.jl --val-window same
  julia --project=packages/TradingGame scripts/train_trading_policy.jl --val-start 2024-06-01 --val-end 2024-12-31
  julia --project=packages/TradingGame scripts/train_trading_policy.jl --resume --iterations 100
  julia --project=packages/TradingGame scripts/train_trading_policy.jl --seed 42
  julia --project=packages/TradingGame scripts/train_trading_policy.jl --init-from other_run/policy.bson
  julia --project=packages/TradingGame scripts/train_trading_policy.jl --device gpu
  julia --project=packages/TradingGame scripts/train_trading_policy.jl --no-macro --no-news

--val-window MODE (default: trailing):
  trailing   val = the last --val-days of the cache; train = everything before
             that. Sound even when train/val use disjoint companies (see
             build_market_universe_snapshot.jl) — there's no leakage risk
             left to guard against, but it keeps results comparable to runs
             that always used this split.
  same       train and val both span the FULL cache date range — only valid
             generalization-wise once train/val use different companies;
             makes full use of the cache's data on both sides instead of
             carving out a held-out date tail.
--train-start/--train-end/--val-start/--val-end (ISO yyyy-mm-dd) override
whichever bound --val-window would otherwise have picked — e.g. `--val-window
same --val-start 2024-06-01` uses the full cache range for train but starts
val partway through it, for deliberately validating against a specific
regime.

For any of those four NOT given explicitly here, the fallback (before the
raw cache-bounds default) is website/data/trading_game/date_window.json,
written by prepare_training_data.jl right after it resolves its own
window — so running that script once with your intended date flags, then
this one with none at all, reproduces the exact same window with nothing
to repeat. A --resume'd run's saved run_config.json still takes priority
over that file, same as an explicit flag would.

Prerequisites:
  website/data/inference_cache.bson              (build_cache.jl)
  website/data/trading_game/universe_latest.json (build_market_universe_snapshot.jl)

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
const MACRO_DIR   = normpath(joinpath(REPO_ROOT, "website", "data", "ohlcv", "macro"))
const OHLCV_1MIN_DIR = normpath(joinpath(REPO_ROOT, "website", "data", "ohlcv", "nse", "1min"))   # news snapshots are NSE-only
const NEWS_DB_FILE = normpath(joinpath(REPO_ROOT, "website", "data", "news_signals.db"))

"""Opts persisted to/restored from `run_config.json` on `--resume` — training
config that must stay consistent across a resumed run, not runtime-only
flags like `--live`/`--resume`/`--init-from`/`--iterations` (the latter means
"how many more" each time by design, so it's never something to restore).
`val_window`/`train_start`/`train_end`/`val_start`/`val_end` are included for
the same reason `val_days` always was — a `--resume` that silently picked up
a different date-window mode or explicit override than the checkpoint was
actually trained on would corrupt the train/val split just as surely as
`val_days` reverting to its default would."""
const RESUMABLE_KEYS = ("initial_cash", "val_days", "eval_every", "lr", "entropy_coef", "seed", "device", "minibatch",
                         "val_window", "train_start", "train_end", "val_start", "val_end",
                         "use_macro", "use_news", "news_db", "game_version", "cash_penalty", "hold_penalty", "illegal_penalty", "history_encoder", "reward_mode", "reward_window_days")

"""`--game-version` accepts `1`, `2`, `3`, `v1`, `v2`, `v3`."""
function _parse_game_version(raw::AbstractString)::Int
    v = lowercase(strip(raw))
    v in ("1", "v1") && return 1
    v in ("2", "v2") && return 2
    v in ("3", "v3") && return 3
    error("Unknown --game-version '$raw'. Expected: 1, 2, 3, v1, v2, v3")
end

"""The `GameRules` selected by `opts` — `rules_v1`/`rules_v2` with the penalty
flags applied. `--hold-penalty` is meaningless under v1 (it has a forced exit
instead), so a non-zero value there is a warning, not silently ignored."""
function build_rules(opts::Dict)::GameRules
    if opts["game_version"] == 3
        return rules_v3(cash_penalty=something(opts["cash_penalty"], 0.0), hold_penalty=opts["hold_penalty"],
                        illegal_penalty=something(opts["illegal_penalty"], ILLEGAL_PENALTY_COEF_V3),
                        history_encoder=Symbol(something(opts["history_encoder"], :direct)),
                        terminal_reward=(something(opts["reward_mode"], "terminal") == "terminal"),
                        reward_window_days=opts["reward_window_days"])
    elseif opts["game_version"] == 2
        return rules_v2(cash_penalty=something(opts["cash_penalty"], 0.0), hold_penalty=opts["hold_penalty"],
                        illegal_penalty=something(opts["illegal_penalty"], 0.0))
    end
    opts["hold_penalty"] != 0.0 &&
        @warn "--hold-penalty only applies to --game-version 2 (v1 force-exits at MAX_HOLD_DAYS instead); ignoring"
    return rules_v1(cash_penalty=opts["cash_penalty"])
end

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
        "val_window"  => "trailing",
        "train_start" => nothing,
        "train_end"   => nothing,
        "val_start"   => nothing,
        "val_end"     => nothing,
        "use_macro"   => true,
        "use_news"    => true,
        "news_db"     => NEWS_DB_FILE,
        "game_version" => 1,
        "cash_penalty"      => nothing,
        "hold_penalty" => 0.0,
        "illegal_penalty" => nothing,
        "val_steps_every" => 1,
        "history_encoder" => nothing,
        "reward_mode" => nothing,
        "reward_window_days" => 0,
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
  --val-window MODE     trailing (default) or same — see the module docstring
  --train-start DATE    Explicit yyyy-mm-dd override, beats --val-window for
  --train-end DATE       this specific bound
  --val-start DATE
  --val-end DATE
  --no-macro            Don't build macro context (SP500/VIX/etc.) — zero
                        placeholder instead, same as before this flag existed
  --no-news             Don't load news features — zero placeholder instead
  --news-db PATH        Classified-signals DB (default: $(opts["news_db"]))
  --game-version V      1 (default, the original game), 2, or 3. 3 = the v2 rules
                        on a BSE 15-minute cache only (any other cache is
                        refused); checkpoints are interchangeable with v2's.
                        v2: cash is a token
                        in the policy, decisions fill at the SAME bar's close,
                        and a stock held MAX_HOLD_DAYS_V2+ days is penalised
                        instead of force-sold. Accepts 1/2/v1/v2. A v1
                        checkpoint can't be resumed/warm-started as v2.
  --cash-penalty X           Cash-ceiling penalty coefficient: per bar, X * max(0,
                        cash/value - MAX_CASH_FRACTION). Default 0.0 under v2
                        (off); under v1 default is CASH_CEILING_PENALTY_COEF
                        ($(CASH_CEILING_PENALTY_COEF)) unless given.
  --val-steps-every N   Write the full per-decision-step log (probabilities, action,
                        value, book state, price for every 15-min bar) of every Nth
                        validation run to website/data/trading_game/val_steps/
                        iter_NNNNN.bson. Default 1 (every run, ~4 MB each for a
                        year of 15-min bars); 0 disables. Read with
                        TradingGame.load_val_steps / scripts/val_steps_summary.jl.
  --reward-mode M       v3 only: terminal (default) = 0 on every bar and, at the end
                        of each reward window, (V_end - V_start)/V_start minus the
                        penalties accumulated in that window in rupees (each illegal
                        move costs its --illegal-penalty share of the portfolio value
                        when it was attempted), over V_start, the portfolio value at
                        the start of the window. stepwise = the weekly log-return
                        plus per-bar penalties.
  --reward-window-days N  terminal reward: trading days per reward window (a shorter
                        last window is flushed at the episode's end). 0 (default) =
                        one window, the whole train/val period. Each window is
                        normalised by the value at its own start. With 0, PPO uses
                        discount 1; with N > 0 the bar-scaled gamma and lambda = 1.
  --history-encoder E   v3 only: how the 70-return price history enters the policy.
                        direct (default): straight into the fusion layer, no
                        recurrence. gru: through the GRU encoder first.
  --illegal-penalty X   v2/v3: charge for each illegal move, as a fraction of portfolio
                        value (0.01 = 1%): under v3's terminal reward it is added to the
                        window's penalty ledger, otherwise taken from that bar's reward. Illegal = a move
                        the rules refuse: buy with no / too little cash, in rebuy
                        cooldown, beyond the holdings cap, or over the position
                        cap (the in-cap part still fills). The trade is rejected
                        either way. Default 0.0 under v2, $(ILLEGAL_PENALTY_COEF_V3) under v3;
                        v1 ignores it. v3 also masks sells of stocks with
                        nothing sellable out of the policy's distribution before
                        it samples.
  --hold-penalty X      v2 only: per bar, X * (share of portfolio value in
                        stocks held MAX_HOLD_DAYS_V2+ days). Default 0.0 (off).
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
        elseif a == "--val-window";   opts["val_window"]   = ARGS[i+1]; push!(explicit, "val_window"); i += 2
        elseif a == "--train-start";  opts["train_start"]  = Date(ARGS[i+1]); push!(explicit, "train_start"); i += 2
        elseif a == "--train-end";    opts["train_end"]    = Date(ARGS[i+1]); push!(explicit, "train_end"); i += 2
        elseif a == "--val-start";    opts["val_start"]    = Date(ARGS[i+1]); push!(explicit, "val_start"); i += 2
        elseif a == "--val-end";      opts["val_end"]      = Date(ARGS[i+1]); push!(explicit, "val_end"); i += 2
        elseif a == "--no-macro";     opts["use_macro"]    = false; push!(explicit, "use_macro"); i += 1
        elseif a == "--no-news";      opts["use_news"]     = false; push!(explicit, "use_news"); i += 1
        elseif a == "--news-db";      opts["news_db"]      = ARGS[i+1]; push!(explicit, "news_db"); i += 2
        elseif a == "--game-version"; opts["game_version"] = _parse_game_version(ARGS[i+1]); push!(explicit, "game_version"); i += 2
        elseif a == "--cash-penalty";      opts["cash_penalty"]      = parse(Float64, ARGS[i+1]); push!(explicit, "cash_penalty"); i += 2
        elseif a == "--val-steps-every"; opts["val_steps_every"] = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--reward-window-days"; opts["reward_window_days"] = parse(Int, ARGS[i+1]); push!(explicit, "reward_window_days"); i += 2
        elseif a == "--reward-mode"; (ARGS[i+1] in ("terminal", "stepwise") || error("--reward-mode must be terminal or stepwise")); opts["reward_mode"] = ARGS[i+1]; push!(explicit, "reward_mode"); i += 2
        elseif a == "--history-encoder"; opts["history_encoder"] = Symbol(lowercase(ARGS[i+1])); push!(explicit, "history_encoder"); i += 2
        elseif a == "--illegal-penalty"; opts["illegal_penalty"] = parse(Float64, ARGS[i+1]); push!(explicit, "illegal_penalty"); i += 2
        elseif a == "--hold-penalty"; opts["hold_penalty"] = parse(Float64, ARGS[i+1]); push!(explicit, "hold_penalty"); i += 2
        else; i += 1
        end
    end
    opts["resume"] && !isempty(opts["init_from"]) &&
        error("--resume and --init-from are mutually exclusive")
    opts["device"] in ("cpu", "gpu") ||
        error("Unknown --device '$(opts["device"])'. Expected: cpu, gpu")
    opts["val_window"] in ("trailing", "same") ||
        error("Unknown --val-window '$(opts["val_window"])'. Expected: trailing, same")
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
function _print_training_params(opts::Dict, device::Symbol, minibatch::Int,
                                 n_candidates_train::Int, n_candidates_val::Int,
                                 train_start::Date, train_end::Date, val_start::Date, val_end::Date)
    n_max_train = n_max_holdings(n_candidates_train)
    n_max_val   = n_max_holdings(n_candidates_val)
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
    @printf("  %-22s %s\n",  "val_window:",   opts["val_window"])
    @printf("  %-22s %s\n",  "initial_cash:", _fmt_money(opts["initial_cash"]))
    @printf("  %-22s %s\n",  "seed:",         something(opts["seed"], "none"))
    @printf("  %-22s %s\n",  "resume:",       opts["resume"])
    @printf("  %-22s %s\n",  "init_from:",    isempty(opts["init_from"]) ? "none" : opts["init_from"])
    @printf("  %-22s %s\n",  "use_macro:",    opts["use_macro"])
    @printf("  %-22s %s\n",  "use_news:",     opts["use_news"])
    @printf("  %-22s %s\n",  "game_version:", opts["game_version"])
    @printf("  %-22s %s\n",  "cash_penalty:", something(opts["cash_penalty"], opts["game_version"] >= 2 ? 0.0 : CASH_CEILING_PENALTY_COEF))
    @printf("  %-22s %s\n",  "illegal_penalty:", opts["game_version"] >= 2 ? something(opts["illegal_penalty"], opts["game_version"] == 3 ? ILLEGAL_PENALTY_COEF_V3 : 0.0) : "n/a (v1)")
    @printf("  %-22s %s\n",  "hold_penalty:", opts["game_version"] >= 2 ? opts["hold_penalty"] : "n/a (v1)")
    @printf("  %-22s %d\n",  "n_candidates (train):", n_candidates_train)
    @printf("  %-22s %d\n",  "n_candidates (val):",   n_candidates_val)
    println("  train window:          $train_start .. $train_end")
    println("  val window:            $val_start .. $val_end")
    println()
    println("Rule-derived constants (constants.jl):")
    @printf("  %-26s %s\n",     "FEE_RATE:",                  FEE_RATE)
    @printf("  %-26s %s\n",     "SETTLEMENT_DAYS:",           SETTLEMENT_DAYS)
    @printf("  %-26s %s\n",     "MIN_HOLD_DAYS:",             MIN_HOLD_DAYS)
    @printf("  %-26s %s\n",     "MAX_HOLD_DAYS (v1):",        MAX_HOLD_DAYS)
    @printf("  %-26s %s\n",     "MAX_HOLD_DAYS_V2:",          MAX_HOLD_DAYS_V2)
    @printf("  %-26s %s\n",     "MAX_POSITION_FRACTION:",     MAX_POSITION_FRACTION)
    @printf("  %-26s %s (→ N_MAX = %d train / %d val)\n",
                                 "N_MAX_HOLDINGS_FRACTION:",  N_MAX_HOLDINGS_FRACTION, n_max_train, n_max_val)
    @printf("  %-26s %s\n",     "MAX_CASH_FRACTION:",         MAX_CASH_FRACTION)
    @printf("  %-26s %s\n",     "CASH_CEILING_PENALTY_COEF:", CASH_CEILING_PENALTY_COEF)
    @printf("  %-26s %s\n",     "REBUY_COOLDOWN_DAYS:",       REBUY_COOLDOWN_DAYS)
    @printf("  %-26s %s / %s\n","GAMMA / GAE_LAMBDA:",        GAMMA, GAE_LAMBDA)
    @printf("  %-26s %s\n",     "CLIP_EPS:",                  CLIP_EPS)
    @printf("  %-26s %s\n",     "VALUE_LOSS_COEF:",           VALUE_LOSS_COEF)
    println("═"^64)
end

"""Window length fed straight to the policy's fusion layer under `rules`, or 0
when a GRU encodes the history instead (every non-v3 game, and `--history-encoder gru`)."""
_history_bars(rules::GameRules, cache)::Int =
    rules.history_encoder == :direct && rules.use_history ? obs_window_bars(rules, cache) : 0

"""A checkpoint is only usable under the game version it was built for: v2's
policy has a cash token and a wider portfolio input that a v1 checkpoint lacks,
and v3's has no macro branch or news inputs, so loading across versions can't work."""
function _check_policy_matches_rules(policy::ActorCriticPolicy, rules::GameRules, path::String, cache)
    has_token = policy.cash_encoder !== nothing
    has_macro = policy.macro_encoder !== nothing
    has_news, has_sf = policy_stock_inputs(policy)
    has_ch    = policy.hourly_encoder === nothing ? 1 : size(policy.hourly_encoder.cell.Wi, 2)
    has_merged = policy.global_in_fusion > 0
    has_critic = size(policy.critic_head.layers[1].weight, 2) > size(policy.actor_head.weight, 2)
    (has_token, has_macro, has_news, has_ch, has_sf, policy.history_bars, has_merged, has_critic) ==
        (rules.cash_token, rules.use_macro, rules.use_news, n_price_channels(rules), n_stock_features(rules),
         _history_bars(rules, cache), rules.portfolio_in_fusion, rules.portfolio_to_critic) && return nothing
    error("$path has cash token=$has_token, macro branch=$has_macro, news inputs=$has_news, $has_ch price channel(s), " *
          "$has_sf per-stock features, but game v$(rules.version) needs $(rules.cash_token), $(rules.use_macro), " *
          "$(rules.use_news), $(n_price_channels(rules)), $(n_stock_features(rules)), history window $(_history_bars(rules, cache)) " *
          "(this one: $(policy.history_bars); 0 = GRU), portfolio scalars in fusion $(rules.portfolio_in_fusion) " *
          "(this one: $has_merged), portfolio to critic $(rules.portfolio_to_critic) (this one: $has_critic) — v1, v2 and v3 policies are not " *
          "interchangeable; start a fresh policy (drop --resume/--init-from) or pass the matching --game-version")
end

"""Remove the per-step validation logs of an earlier run (a fresh run starts the
history over, like `episode_log.jsonl`/`val_runs.jsonl`)."""
function _clear_val_steps(dir::String)
    isdir(dir) || return nothing
    for f in readdir(dir; join=true)
        (endswith(f, ".bson") || endswith(f, ".json")) && rm(f)
    end
    return nothing
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

"""`RESUMABLE_KEYS` whose value is a `Union{Date,Nothing}`, not JSON-native —
`save_run_config`/`load_run_config!` stringify/parse these explicitly instead
of handing a `Date` straight to `JSON3` (which has no default encoding for
one)."""
const DATE_OPT_KEYS = ("train_start", "train_end", "val_start", "val_end")

"""Write the `RESUMABLE_KEYS` subset of `opts` to `path` as JSON — the
*requested* values (e.g. `device="gpu"` even if it later falls back to cpu,
`minibatch=nothing`/`train_start=nothing` if left on auto), not
resolved/derived ones, so a later `--resume` re-derives exactly the same way
the original run did.

Everything else written here — `n_candidates_train`/`n_candidates_val`,
`resolved_device`/`resolved_minibatch`/`resolved_train_start`/etc. (the
actually-resolved dates, under a `resolved_` prefix so they don't collide
with the raw, possibly-`nothing` `RESUMABLE_KEYS` entries of the same root
name), `n_max_holdings_train`/`n_max_holdings_val`, and `rule_constants` — is
informational only: none of it is a `RESUMABLE_KEYS` entry, so none of it is
read back by `load_run_config!` on `--resume` (the candidate universe always
comes from `universe_latest.json` at load time, not from this file; the rule
constants come from `constants.jl`, not a flag). It exists purely so
`tradinggamelive.html` can display this run's actual training parameters and
rule-derived constants — the same printout `_print_training_params` puts on
stdout — without needing console access."""
function save_run_config(opts::Dict, path::String, n_candidates_train::Int, n_candidates_val::Int,
                          resolved_device::Symbol, resolved_minibatch::Int,
                          train_start::Date, train_end::Date, val_start::Date, val_end::Date, cache)
    resumable = Dict{String, Any}()
    for k in RESUMABLE_KEYS
        v = opts[k]
        resumable[k] = (k in DATE_OPT_KEYS && v !== nothing) ? string(v) : v
    end
    open(path, "w") do io
        JSON3.pretty(io, merge(resumable, Dict(
            "n_candidates_train"   => n_candidates_train,
            "n_candidates_val"     => n_candidates_val,
            "exchange"             => cache.exchange,
            "has_history"          => has_history(cache),
            "bar_minutes"          => cache.bar_minutes,
            "resolved_device"      => string(resolved_device),
            "resolved_minibatch"   => resolved_minibatch,
            "resolved_train_start" => string(train_start),
            "resolved_train_end"   => string(train_end),
            "resolved_val_start"   => string(val_start),
            "resolved_val_end"     => string(val_end),
            "n_max_holdings_train" => n_max_holdings(n_candidates_train),
            "n_max_holdings_val"   => n_max_holdings(n_candidates_val),
            "rule_constants"       => Dict(
                "fee_rate"                  => FEE_RATE,
                "settlement_days"           => SETTLEMENT_DAYS,
                "min_hold_days"             => MIN_HOLD_DAYS,
                "max_hold_days"             => (opts["game_version"] >= 2 ? MAX_HOLD_DAYS_V2 : MAX_HOLD_DAYS),
                "max_position_fraction"     => MAX_POSITION_FRACTION,
                "n_max_holdings_fraction"   => N_MAX_HOLDINGS_FRACTION,
                "max_cash_fraction"         => MAX_CASH_FRACTION,
                "cash_ceiling_penalty_coef" => (opts["game_version"] >= 2 ? something(opts["cash_penalty"], 0.0) :
                                                  something(opts["cash_penalty"], CASH_CEILING_PENALTY_COEF)),
                "hold_penalty_coef"         => (opts["game_version"] >= 2 ? opts["hold_penalty"] : 0.0),
                "illegal_penalty_coef"      => build_rules(opts).illegal_penalty_coef,
                "terminal_reward"           => build_rules(opts).terminal_reward,
                "reward_window_days"        => build_rules(opts).reward_window_days,
                "game_version"              => opts["game_version"],
                "rebuy_cooldown_days"       => REBUY_COOLDOWN_DAYS,
                "gamma"                     => GAMMA,
                "gae_lambda"                => GAE_LAMBDA,
                "clip_eps"                  => CLIP_EPS,
                "value_loss_coef"           => VALUE_LOSS_COEF,
                "news_decision_severity_threshold" => NEWS_DECISION_SEVERITY_THRESHOLD,
            ),
        )))
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
    nullable_keys = ("seed", "minibatch", DATE_OPT_KEYS...)
    for k in RESUMABLE_KEYS
        k in explicit && continue
        haskey(saved, Symbol(k)) || continue
        v = saved[Symbol(k)]
        if v === nothing
            opts[k] = k in nullable_keys ? nothing : opts[k]
        elseif k in DATE_OPT_KEYS
            opts[k] = Date(String(v))
        elseif k == "lr"
            opts[k] = Float32(v)
        elseif k == "entropy_coef"
            opts[k] = Float64(v)
        elseif k in ("val_days", "eval_every", "seed", "minibatch")
            opts[k] = Int(v)
        else
            opts[k] = v
        end
        # Mark as settled, same as a CLI flag — this is what stops the
        # weaker prepare_training_data.jl-persisted date-window layer
        # (applied right before resolve_date_windows, see main()) from
        # overwriting a value --resume just restored.
        push!(explicit, k)
    end
    @info "Loaded saved run config from $path for flags not given explicitly"
    return nothing
end

"""
Fill in any of `train_start`/`train_end`/`val_start`/`val_end` NOT already
settled (a CLI flag, or a `--resume`d run's saved config — both already in
`explicit` by the time this runs) from `prepare_training_data.jl`'s
persisted `date_window.json`, if one exists. This is the weakest of the
three layers — CLI flag, then `--resume`'s saved config, then this — and is
what lets `train_trading_policy.jl` reproduce the exact window
`prepare_training_data.jl` was run with, with no date flags of its own.
No-op (no warning — this file is optional, unlike `run_config.json` on an
explicit `--resume`) if `path` doesn't exist."""
function load_date_window!(opts::Dict, explicit::Set{String}, path::String)
    isfile(path) || return nothing
    train_start, train_end, val_start, val_end = load_date_window(path)
    for (k, v) in zip(DATE_OPT_KEYS, (train_start, train_end, val_start, val_end))
        k in explicit || (opts[k] = v)
    end
    @info "Loaded resolved date window from $path for date flags not given explicitly/by --resume"
    return nothing
end

function main()
    opts, explicit = parse_args()
    mkpath(DATA_DIR)
    config_path = joinpath(DATA_DIR, "run_config.json")
    opts["resume"] && load_run_config!(opts, explicit, config_path)
    date_window_path = joinpath(DATA_DIR, "date_window.json")
    load_date_window!(opts, explicit, date_window_path)

    isfile(CACHE_FILE) || error(
        "Not found: $CACHE_FILE\nRun: julia --project=packages/StockSwingPredictor scripts/build_cache.jl")
    isfile(UNIVERSE_FILE) || error(
        "Not found: $UNIVERSE_FILE\nRun: julia --project=packages/TradingGame scripts/build_market_universe_snapshot.jl")

    @info "Loading inference cache…"
    cache = load_inference_cache(CACHE_FILE)
    universe = load_universe_snapshot(UNIVERSE_FILE)
    @info "  $(length(cache.companies)) $(uppercase(cache.exchange)) companies cached ($(cache.bar_minutes)-minute bars" *
          (has_history(cache) ? ", hourly history axis of $(length(cache.history_datetimes)) bars" : "") * ", " *
          "decisions every bar = $(decision_granularity(cache))), " *
          "$(length(universe.train)) train / $(length(universe.val)) val candidates in universe"
    if opts["use_news"] && cache.exchange != "nse"
        @warn "News features read NSE announcements keyed by NSE symbols; on a $(uppercase(cache.exchange)) cache most " *
              "symbols will have no signals and 1-minute snapshots are unavailable. Pass --no-news."
    end

    # Intraday axis, not daily — episodes step through intraday bars, whose
    # real Kite floor is shallower than daily's. See prepare_training_data.jl's
    # matching comment for the concrete numbers that motivated this.
    cache_start = Date(first(cache.hourly_datetimes))
    cache_end   = Date(last(cache.hourly_datetimes))
    train_start, train_end, val_start, val_end = resolve_date_windows(
        cache_start, cache_end, opts["val_window"], opts["val_days"],
        opts["train_start"], opts["train_end"], opts["val_start"], opts["val_end"])

    @info "Train window: $train_start .. $train_end"
    @info "Val window:   $val_start .. $val_end"

    rules = build_rules(opts)
    if !rules.use_macro && opts["use_macro"]
        @info "Game v$(rules.version) runs without the macro context (not loaded, no macro branch in the policy)"
        opts["use_macro"] = false
    end
    if !rules.use_news && opts["use_news"]
        @info "Game v$(rules.version) runs without news features (not loaded, no news inputs in the policy)"
        opts["use_news"] = false
    end
    @info "Game v$(rules.version): same-bar execution=$(rules.same_bar_execution), forced exit=$(rules.forced_exit), " *
          "max hold $(rules.max_hold_days)d, cash penalty=$(rules.cash_penalty_coef), hold penalty=$(rules.hold_penalty_coef), illegal penalty=$(rules.illegal_penalty_coef), reward=$(rules.terminal_reward ? (rules.reward_window_days == 0 ? "terminal (whole period)" : "terminal, every $(rules.reward_window_days) trading days") : "stepwise")"
    train_config = EpisodeConfig(initial_cash=opts["initial_cash"], start_date=train_start,
                                  end_date=train_end, candidate_universe=universe.train, rules=rules)
    val_config   = EpisodeConfig(initial_cash=opts["initial_cash"], start_date=val_start,
                                  end_date=val_end, candidate_universe=universe.val, rules=rules)

    macro_cache = nothing
    if opts["use_macro"]
        if isdir(MACRO_DIR)
            @info "Loading macro context from $MACRO_DIR…"
            macro_cache = build_macro_cache(MACRO_DIR)
        else
            @warn "Macro OHLCV dir not found, training without macro context: $MACRO_DIR " *
                  "(run collect_macro_ohlcv.jl, or pass --no-macro to silence this)"
        end
    end

    news_fn = (_env, _s, _h) -> zeros(Float32, N_NEWS_FEATURES)   # same neutral default as TradingGame's internal _zero_news
    news_hour_indices = Set{Int}()
    price_overrides   = Dict{Int, Dict{Int, Float32}}()
    if opts["use_news"]
        if isfile(opts["news_db"])
            @info "Loading news features from $(opts["news_db"])…"
            snapshot_symbols = unique(vcat(universe.train, universe.val))
            ohlcv_1min_dir   = isdir(OHLCV_1MIN_DIR) ? OHLCV_1MIN_DIR : nothing
            isnothing(ohlcv_1min_dir) && @warn "No 1-minute OHLCV dir at $OHLCV_1MIN_DIR — " *
                  "news-instant snapshots will be skipped (every news bar falls back to the hourly close). " *
                  "Collect it with collect_nse_ohlcv.jl --1min-only / backfill_ohlcv.jl for the candidate universe."
            news_cache = build_news_feature_cache(cache, opts["news_db"];
                snapshot_symbols=ohlcv_1min_dir === nothing ? String[] : snapshot_symbols,
                ohlcv_1min_dir=ohlcv_1min_dir)
            news_fn = news_feature_fn(news_cache)
            news_hour_indices = news_cache.decision_hours
            price_overrides   = news_cache.snapshots
            n_snapshot_bars = length(price_overrides)
            @info "  $(length(news_cache.market_events)) classified signals, " *
                  "$(length(news_hour_indices)) hourly bars at/above severity threshold, " *
                  "$n_snapshot_bars with a 1-minute instant-price snapshot"
        else
            @warn "News signals DB not found, training without news features: $(opts["news_db"]) " *
                  "(run scripts/backfill_news_signals.jl, or pass --no-news to silence this)"
        end
    end

    env = TradingGameEnv(cache; news_hour_indices=news_hour_indices, price_overrides=price_overrides)

    checkpoint_path  = joinpath(DATA_DIR, "policy.bson")
    episode_log_path = joinpath(DATA_DIR, "episode_log.jsonl")
    val_curve_path   = joinpath(DATA_DIR, "val_runs.jsonl")
    val_steps_dir    = joinpath(DATA_DIR, "val_steps")

    iteration_offset = 0
    if opts["resume"]
        isfile(checkpoint_path) ||
            error("--resume requested but no checkpoint found at $checkpoint_path")
        policy, hp, _ = load_policy(checkpoint_path)
        _check_policy_matches_rules(policy, rules, checkpoint_path, cache)
        iteration_offset = _last_completed_iteration(episode_log_path)
        @info "Resumed policy — last completed iteration: $iteration_offset"
    elseif !isempty(opts["init_from"])
        isfile(opts["init_from"]) ||
            error("--init-from: not found: $(opts["init_from"])")
        policy, hp, _ = load_policy(opts["init_from"])
        _check_policy_matches_rules(policy, rules, opts["init_from"], cache)
        @info "Fresh run, weights warm-started from $(opts["init_from"])"
        # Fresh iteration numbering and log, unlike --resume — see the module
        # docstring's --init-from vs --resume note. val_runs.jsonl follows the
        # same lifecycle as episode_log.jsonl — a fresh iteration-1 run means a
        # fresh held-out-curve history too.
        isfile(episode_log_path) && rm(episode_log_path)
        isfile(val_curve_path) && rm(val_curve_path)
        _clear_val_steps(val_steps_dir)
    else
        policy = ActorCriticPolicy(seed=opts["seed"], cash_token=rules.cash_token,
                                    portfolio_in_fusion=rules.portfolio_in_fusion,
                                    portfolio_to_critic=rules.portfolio_to_critic,
                                    portfolio_scalars=n_portfolio_scalars(rules),
                                    use_macro=rules.use_macro, use_news=rules.use_news,
                                    price_channels=n_price_channels(rules),
                                    stock_features=n_stock_features(rules),
                                    history_bars=_history_bars(rules, cache))
        hp = (embed_dim=64, macro_embed_dim=16, attn_heads=4, critic_hidden=[64, 32])
        @info "Fresh policy" * (opts["seed"] === nothing ? "" : " (seed=$(opts["seed"]))")
        # episode_log.jsonl/val_runs.jsonl are both opened in append mode
        # inside train_policy! (so --resume can keep history) — a fresh run
        # must clear any stale ones.
        isfile(episode_log_path) && rm(episode_log_path)
        isfile(val_curve_path) && rm(val_curve_path)
        _clear_val_steps(val_steps_dir)
    end

    rng       = opts["seed"] === nothing ? Random.default_rng() : MersenneTwister(opts["seed"])
    device    = _resolve_device(opts["device"])
    minibatch = _resolve_minibatch(opts["minibatch"], device)

    save_run_config(opts, config_path, length(universe.train), length(universe.val), device, minibatch,
                     train_start, train_end, val_start, val_end, cache)
    _print_training_params(opts, device, minibatch, length(universe.train), length(universe.val),
                            train_start, train_end, val_start, val_end)

    policy, log = train_policy!(policy, env, train_config; val_config=val_config,
        iterations=opts["iterations"], eval_every=opts["eval_every"], lr=opts["lr"],
        entropy_coef=opts["entropy_coef"], rng=rng,
        minibatch_size=minibatch, device=device,
        checkpoint_path=checkpoint_path,
        episode_log_path=episode_log_path,
        stop_file=joinpath(DATA_DIR, "STOP"),
        live_path=opts["live"] ? joinpath(DATA_DIR, "live_status.json") : "",
        val_curve_path=opts["live"] ? val_curve_path : "",
        val_steps_dir=val_steps_dir, val_steps_every=opts["val_steps_every"],
        iteration_offset=iteration_offset,
        macro_cache=macro_cache, news_fn=news_fn,
        embed_dim=hp.embed_dim, macro_embed_dim=hp.macro_embed_dim,
        attn_heads=hp.attn_heads, critic_hidden=hp.critic_hidden)

    save_policy_training_log(log, joinpath(DATA_DIR, "training_log.json"))

    @printf("\nDone. %d iterations, best return %.4f (iter %d)\n",
            log["iterations_run"], log["best_return"], log["best_iteration"])
end

main()
