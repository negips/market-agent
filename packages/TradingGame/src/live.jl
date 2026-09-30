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

const LIVE_VALUE_CURVE_CAP = 2000   # ~1 full episode at hourly cadence; oldest points drop first
const LIVE_TRADES_CAP      = 100

"""Mutable, episode-spanning state for one live-viewer feed. Reused across
iterations (`start_episode!` resets the per-episode fields); `path=""`
disables writing entirely (`make_live_callback` returns `nothing` for it)."""
Base.@kwdef mutable struct LiveTracker
    path         :: String
    every_bars   :: Int = 5
    iteration    :: Int = 0
    phase        :: String = "train"
    bar_count    :: Int = 0
    value_curve  :: Vector{Dict{String, Any}} = Dict{String, Any}[]
    trades       :: Vector{Dict{String, Any}} = Dict{String, Any}[]
    started_at   :: String = ""
end

"""Call at the start of each `collect_rollout` (before the rollout, not
inside it) to clear the previous episode's trajectory and tag the new one."""
function start_episode!(tracker::LiveTracker; iteration::Int, phase::String)
    tracker.iteration  = iteration
    tracker.phase      = phase
    tracker.bar_count  = 0
    empty!(tracker.value_curve)
    empty!(tracker.trades)
    tracker.started_at = string(now(UTC))
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
    )
    _write_atomic(tracker.path, status)
    return nothing
end
