"""
Streams a running episode's state to a JSON file the website polls
(`website/tradinggamelive.html`) — the only consumer of `live.jl`; `train.jl`
is oblivious to what "live" means, it just calls `live_cb(env, result)` (see
`ppo.jl`'s `collect_rollout`) if a `LiveTracker` is wired in.

Deliberately file-polling, not a socket/server: the website is a static site
with no backend (`serve.sh` is a plain file server), so "live" here means
"overwrite a JSON file every few bars, atomically, and let the page re-fetch
it on a timer" — the same pattern every other page on this site already uses
for `nse_companies_latest.json` etc.
"""

using Dates, JSON3

"""Preallocated size of `LiveTracker.curve_t`/`curve_v`, not a display window
— a full 5-year training episode at hourly cadence is ~8,600 bars, so this
only exists to bound memory/JSON size against a pathological config (e.g. a
decades-long window), never expected to bind in practice. It must stay well
above any real episode length: capping it near or below one means the chart's
oldest points (including the episode's actual start) stop being recorded
mid-run, which is exactly the "starting date keeps vanishing" bug this
constant used to cause.

Unlike the old growing-`Vector{Dict}` implementation (which evicted the
*oldest* point once this cap was hit, via `popfirst!`), the preallocated
array version simply stops recording new points once full — trading perfect
FIFO-eviction semantics in that pathological, never-expected-to-bind case for
a much simpler, allocation-free hot path (see `make_live_callback`'s
docstring for why the per-bar allocation this replaces mattered a lot more
than this edge-case trade-off does)."""
const LIVE_VALUE_CURVE_CAP = 20_000

"""Same safety-ceiling reasoning as `LIVE_VALUE_CURVE_CAP` — not a display
window. Measured directly on a real (untrained, exploratory) policy: several
trades per bar is common, not rare — tens of thousands over a full ~8,600-bar
episode, not "a few hundred" as earlier revisions of this comment assumed.
Both the "Recent decisions" feed and the buy/sell markers on the portfolio-
value chart (`website/tradinggamelive.html`) read the same `tracker.trades`
list, so a cap that actually binds silently drops the earliest trades from
*both*."""
const LIVE_TRADES_CAP = 50_000

"""Mutable, episode-spanning state for one live-viewer feed. Reused across
iterations (`start_episode!` resets the per-episode fields); `path=""`
disables writing entirely (`make_live_callback` returns `nothing` for it).

`curve_t`/`curve_v` are preallocated to `LIVE_VALUE_CURVE_CAP` once, up
front, and `curve_len` tracks how many leading entries are live for the
current episode (`curve_t[1:curve_len]`/`curve_v[1:curve_len]`) — see
`make_live_callback`'s docstring for why this had to be allocation-free per
entry. `trades` stays a plain growing `Vector{TradeEvent}` (push!/popfirst!,
not preallocated) — trade events turned out NOT to be rare (several per bar
is common for an exploratory policy, measured directly), but `TradeEvent`
is a concretely-typed `NamedTuple` (see `types.jl`), not a `Dict{String,Any}`,
which was the actual fix that mattered here: one compact allocation per
trade instead of the ~15+ a `Dict{String,Any}` costs per instance (one
object per key, one boxed object per `Any`-typed value). The outer Vector's
own amortized growth was never the issue — removing the error from `Dict`
boxing was."""
Base.@kwdef mutable struct LiveTracker
    path             :: String
    every_bars       :: Int = 5
    iteration        :: Int = 0
    phase            :: String = "train"
    bar_count        :: Int = 0
    curve_t          :: Vector{String}  = fill("", LIVE_VALUE_CURVE_CAP)
    curve_v          :: Vector{Float64} = zeros(Float64, LIVE_VALUE_CURVE_CAP)
    curve_len        :: Int = 0
    trades           :: Vector{TradeEvent} = TradeEvent[]
    started_at       :: String = ""
    update_progress  :: Union{Nothing, Dict{String, Any}} = nothing
end

"""Call at the start of each `collect_rollout` (before the rollout, not
inside it) to clear the previous episode's trajectory and tag the new one.
Also clears `update_progress` — a fresh rollout means the previous
iteration's PPO update (if any) is done."""
function start_episode!(tracker::LiveTracker; iteration::Int, phase::String)
    tracker.iteration  = iteration
    tracker.phase      = phase
    tracker.bar_count  = 0
    tracker.curve_len  = 0
    empty!(tracker.trades)
    tracker.started_at    = string(now(UTC))
    tracker.update_progress = nothing
    return nothing
end

"""Call at the start of `ppo_update!` (after the rollout it's training on) to
tag the tracker as being in the gradient-update phase and reset progress."""
function start_update!(tracker::LiveTracker; iteration::Int, k_epochs::Int, total_minibatches::Int)
    tracker.iteration = iteration
    tracker.phase      = "update"
    tracker.update_progress = Dict{String, Any}(
        "epoch" => 0, "k_epochs" => k_epochs,
        "minibatch" => 0, "total_minibatches" => total_minibatches,
        "loss" => 0.0,
    )
    return nothing
end

"""Write `status` to `tracker.path` atomically (`tmp` + `mv`) so the website
never reads a half-written file — it's a static file server with no locking."""
function _write_atomic(path::String, status)
    tmp = path * ".tmp"
    open(tmp, "w") do io
        JSON3.write(io, status)
    end
    mv(tmp, path; force=true)
    return nothing
end

"""Build the `live_cb` closure `collect_rollout`/`ppo.jl` calls after every
bar. Returns `nothing` if `tracker.path` is empty (live tracking disabled) —
callers can pass that straight through as `live_cb`.

The `curve_t`/`curve_v` write here is a plain indexed assignment into
preallocated arrays — no allocation. This matters a lot: `collect_rollout`
calls this on *every* bar of a ~8,600-bar episode, so the old
`push!(tracker.value_curve, Dict{String,Any}(...))` here allocated and
RETAINED (for the whole episode — nothing frees it until `start_episode!`
resets the tracker) one boxed, `Any`-valued `Dict` per bar. Measured directly:
that made the per-bar rate compound from ~31ms/bar to ~70ms/bar and climbing
over just the first 2,000 bars — the exact same ever-growing-retained-heap
GC-pressure pattern already fixed for `collect_rollout`'s own observation
buffer (`ppo.jl`), just discovered here separately after `collect_rollout`'s
fix alone didn't reproduce the expected speedup on a real full-length run."""
function make_live_callback(tracker::LiveTracker)
    isempty(tracker.path) && return nothing

    return function (env::TradingGameEnv, result::StepResult)
        tracker.bar_count += 1
        if tracker.curve_len < LIVE_VALUE_CURVE_CAP
            tracker.curve_len += 1
            tracker.curve_t[tracker.curve_len] = string(env.cache.hourly_datetimes[env.current_hour_idx])
            tracker.curve_v[tracker.curve_len] = result.info["portfolio_value"]
        end

        for ev in result.info["trades"]
            push!(tracker.trades, ev)
            length(tracker.trades) > LIVE_TRADES_CAP && popfirst!(tracker.trades)
        end

        # NOT `|| !isempty(result.info["trades"])` — that seemed reasonable
        # when trades were assumed rare (see `LIVE_TRADES_CAP`'s docstring for
        # why that assumption was wrong), but `_write_live_status` does
        # O(curve_len + trades_len) work (rebuilds `value_curve`'s JSON array,
        # serializes the full, uncapped-until-50k `trades` list). Forcing that
        # on every trade, for a policy trading several times a bar, meant
        # firing on ~96% of bars measured directly — an ever-larger array
        # rewritten on nearly every step, the dominant cost of a real rollout
        # by a wide margin. `every_bars` cadence alone still surfaces new
        # trades within a few bars — fast enough for a polling live viewer.
        due = tracker.bar_count % tracker.every_bars == 0 || result.done
        due && _write_live_status(tracker, env)
        return nothing
    end
end

"""Build the `progress_cb` closure `ppo_update!` (`ppo.jl`) calls after every
minibatch. `env` is the rollout's terminal state (unchanged during the update
phase — nothing about the portfolio moves while the network is training on
already-collected data), reused here only so `_write_live_status` can still
report it alongside the update progress. Returns `nothing` if `tracker.path`
is empty, same convention as `make_live_callback`."""
function make_update_callback(tracker::LiveTracker, env::TradingGameEnv)
    isempty(tracker.path) && return nothing

    return function (epoch::Int, minibatch::Int, total_minibatches::Int, loss::Real)
        tracker.update_progress["epoch"]     = epoch
        tracker.update_progress["minibatch"] = minibatch
        tracker.update_progress["loss"]      = loss
        _write_live_status(tracker, env)
        return nothing
    end
end

"""Snapshot `env.portfolio.holdings` with current price and unrealised P&L —
computed here (not stored on `Holding`) since it depends on the current bar."""
function _holdings_snapshot(env::TradingGameEnv)
    date_idx = env.cache.date_index[env.current_date]
    return [begin
        price = env.cache.hourly_closes[env.current_hour_idx, h.sym_idx]
        cost  = h.quantity * h.entry_price + h.entry_fee
        Dict{String, Any}(
            "symbol" => h.symbol, "quantity" => h.quantity,
            "entry_price" => h.entry_price, "current_price" => price,
            "unrealized_pnl" => h.quantity * price - cost,
            "unrealized_pnl_pct" => cost > 0 ? (h.quantity * price - cost) / cost * 100 : 0.0,
            "days_held" => date_idx - h.entry_date_idx,
        )
    end for h in env.portfolio.holdings]
end

"""Build the `[{"t":..., "value":...}, ...]`-shaped array `_write_live_status`
serializes, from `tracker`'s preallocated `curve_t`/`curve_v`. Allocates a
fresh (small, short-lived) array on each call — unlike the per-bar write this
reads from, this only runs when `_write_live_status` itself does (every
`every_bars` bars, not every bar), so it's cheap and never retained."""
function _value_curve_json(tracker::LiveTracker)
    return [(t=tracker.curve_t[i], value=tracker.curve_v[i]) for i in 1:tracker.curve_len]
end

function _write_live_status(tracker::LiveTracker, env::TradingGameEnv)
    status = Dict{String, Any}(
        "iteration"       => tracker.iteration,
        "phase"           => tracker.phase,
        "episode_started_at" => tracker.started_at,
        "updated_at"      => string(now(UTC)),
        "current_date"    => string(env.current_date),
        "n_candidates"    => length(env.candidate_order),
        "done"            => env.current_hour_idx >= env.end_hour_idx,
        "portfolio_value" => portfolio_value(env),
        "cash"            => env.portfolio.cash,
        "reserved_cash"   => isempty(env.portfolio.reserved) ? 0.0 : sum(l.amount for l in env.portfolio.reserved),
        "initial_cash"    => env.config === nothing ? 0.0 : env.config.initial_cash,
        "holdings"        => _holdings_snapshot(env),
        "value_curve"     => _value_curve_json(tracker),
        "recent_trades"   => tracker.trades,
        "update_progress" => tracker.update_progress,
    )
    _write_atomic(tracker.path, status)
    return nothing
end
