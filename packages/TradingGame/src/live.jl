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

"""Safety ceiling on `LiveTracker.value_curve`, not a display window — a full
5-year training episode at hourly cadence is ~8,600 bars, so this only exists
to bound memory/JSON size against a pathological config (e.g. a decades-long
window), never expected to bind in practice. It must stay well above any real
episode length: capping it near or below one means the chart's oldest points
(including the episode's actual start) get evicted mid-run, which is exactly
the "starting date keeps vanishing" bug this constant used to cause."""
const LIVE_VALUE_CURVE_CAP = 20_000

"""Same safety-ceiling reasoning as `LIVE_VALUE_CURVE_CAP` — not a display
window. A 60-candidate joint policy trading over an ~8,600-bar episode can
easily produce more than a few hundred buy/sell events, and both the "Recent
decisions" feed and the buy/sell markers on the portfolio-value chart
(`website/tradinggamelive.html`) read the same `tracker.trades` list, so a
cap that actually binds silently drops the earliest trades from *both*."""
const LIVE_TRADES_CAP = 50_000

"""Mutable, episode-spanning state for one live-viewer feed. Reused across
iterations (`start_episode!` resets the per-episode fields); `path=""`
disables writing entirely (`make_live_callback` returns `nothing` for it)."""
Base.@kwdef mutable struct LiveTracker
    path             :: String
    every_bars       :: Int = 5
    iteration        :: Int = 0
    phase            :: String = "train"
    bar_count        :: Int = 0
    value_curve      :: Vector{Dict{String, Any}} = Dict{String, Any}[]
    trades           :: Vector{Dict{String, Any}} = Dict{String, Any}[]
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
    empty!(tracker.value_curve)
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
callers can pass that straight through as `live_cb`."""
function make_live_callback(tracker::LiveTracker)
    isempty(tracker.path) && return nothing

    return function (env::TradingGameEnv, result::StepResult)
        tracker.bar_count += 1
        push!(tracker.value_curve, Dict{String, Any}(
            "t" => string(env.cache.hourly_datetimes[env.current_hour_idx]),
            "value" => result.info["portfolio_value"]))
        length(tracker.value_curve) > LIVE_VALUE_CURVE_CAP && popfirst!(tracker.value_curve)

        for ev in result.info["trades"]
            push!(tracker.trades, ev)
            length(tracker.trades) > LIVE_TRADES_CAP && popfirst!(tracker.trades)
        end

        due = tracker.bar_count % tracker.every_bars == 0 || result.done || !isempty(result.info["trades"])
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

function _write_live_status(tracker::LiveTracker, env::TradingGameEnv)
    status = Dict{String, Any}(
        "iteration"       => tracker.iteration,
        "phase"           => tracker.phase,
        "episode_started_at" => tracker.started_at,
        "updated_at"      => string(now(UTC)),
        "current_date"    => string(env.current_date),
        "done"            => env.current_hour_idx >= env.end_hour_idx,
        "portfolio_value" => portfolio_value(env),
        "cash"            => env.portfolio.cash,
        "reserved_cash"   => isempty(env.portfolio.reserved) ? 0.0 : sum(l.amount for l in env.portfolio.reserved),
        "initial_cash"    => env.config === nothing ? 0.0 : env.config.initial_cash,
        "holdings"        => _holdings_snapshot(env),
        "value_curve"     => tracker.value_curve,
        "recent_trades"   => tracker.trades,
        "update_progress" => tracker.update_progress,
    )
    _write_atomic(tracker.path, status)
    return nothing
end
