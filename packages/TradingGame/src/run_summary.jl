"""
Per-episode result statistics for a held-out validation run — the numbers the
live page's "Validation run summary" block shows next to the timings: final
portfolio value, drawdown, illegal-move penalties, trade counts, fees, realised
P&L and so on. `collect_rollout(…; summary=Ref{Any}(nothing))` accumulates them
bar by bar (a handful of additions per bar) and leaves a `NamedTuple` in the
`Ref`; `train_policy!` adds the iteration, the return and the timings and
writes the row to `val_summary.json`.
"""

"""Running totals behind a validation run's summary (see `_finish_summary`)."""
mutable struct _SummaryAccumulator
    initial_value :: Float64
    start_dt      :: Any
    prev_value    :: Float64
    peak          :: Float64
    min_value     :: Float64
    max_drawdown  :: Float64
    penalty_frac  :: Float64
    penalty_rupees :: Float64
    n_penalty_bars :: Int
    n_buys        :: Int
    n_sells       :: Int
    n_forced      :: Int
    fees          :: Float64
    realised_pnl  :: Float64
    n_wins        :: Int
    days_held_sum :: Int
    cash_frac_sum :: Float64
    lots_sum      :: Int
    n_hold        :: Int
    n_sell_act    :: Int
    n_buy_act     :: Int
    end_dt        :: Any
end

_SummaryAccumulator(value::Real, start_dt) =
    _SummaryAccumulator(value, start_dt, value, value, value, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0.0, 0.0, 0, 0, 0.0, 0, 0, 0, 0, start_dt)

"""Fold one bar's `StepResult` into `acc`. `action_idx` is the chosen HOLD/SELL/BUY index per candidate."""
function _accumulate!(acc::_SummaryAccumulator, env::TradingGameEnv, result::StepResult, action_idx::AbstractVector{Int})
    info  = result.info
    value = Float64(info["portfolio_value"])
    pen   = Float64(info["illegal_penalty"])
    if pen > 0
        acc.n_penalty_bars += 1
        acc.penalty_frac   += pen
        acc.penalty_rupees += pen * acc.prev_value
    end
    acc.peak      = max(acc.peak, value)
    acc.min_value = min(acc.min_value, value)
    acc.peak > 0 && (acc.max_drawdown = max(acc.max_drawdown, 1 - value / acc.peak))
    acc.prev_value = value
    acc.cash_frac_sum += Float64(info["cash_fraction"])
    acc.lots_sum      += length(env.portfolio.holdings)
    for ev in info["trades"]::Vector{TradeEvent}
        acc.fees += ev.fee
        if ev.kind == "buy"
            acc.n_buys += 1
        else
            ev.kind == "forced_exit" ? (acc.n_forced += 1) : (acc.n_sells += 1)
            acc.realised_pnl  += ev.pnl
            acc.days_held_sum += ev.days_held
            ev.pnl > 0 && (acc.n_wins += 1)
        end
    end
    for a in action_idx
        a == 1 ? (acc.n_hold += 1) : a == 2 ? (acc.n_sell_act += 1) : (acc.n_buy_act += 1)
    end
    acc.end_dt = env.cache.hourly_datetimes[env.current_hour_idx]
    return nothing
end

"""Close out `acc` into the summary `NamedTuple` (all numbers plain `Float64`/`Int`, dates as strings)."""
function _finish_summary(acc::_SummaryAccumulator, env::TradingGameEnv, n_bars::Int)
    value    = acc.prev_value
    reserved = isempty(env.portfolio.reserved) ? 0.0 : sum(l.amount for l in env.portfolio.reserved)
    cash     = env.portfolio.cash
    n_exits  = acc.n_sells + acc.n_forced
    r(x; d=2) = round(x; digits=d)
    return (initial_value=r(acc.initial_value), final_value=r(value),
            total_return=r(value / acc.initial_value - 1; d=5),
            peak_value=r(acc.peak), min_value=r(acc.min_value), max_drawdown=r(acc.max_drawdown; d=5),
            penalty_frac=r(acc.penalty_frac; d=5), penalty_rupees=r(acc.penalty_rupees), n_penalty_bars=acc.n_penalty_bars,
            n_buys=acc.n_buys, n_sells=acc.n_sells, n_forced_exits=acc.n_forced,
            fees_paid=r(acc.fees), realised_pnl=r(acc.realised_pnl),
            win_rate=n_exits > 0 ? r(acc.n_wins / n_exits; d=4) : nothing,
            avg_days_held=n_exits > 0 ? r(acc.days_held_sum / n_exits) : nothing,
            avg_cash_fraction=n_bars > 0 ? r(acc.cash_frac_sum / n_bars; d=4) : 0.0,
            avg_lots_held=n_bars > 0 ? r(acc.lots_sum / n_bars) : 0.0,
            n_bars=n_bars, start=string(acc.start_dt), finish=string(acc.end_dt),
            final_cash=r(cash), final_reserved=r(reserved), final_stocks=r(value - cash - reserved),
            actions=(hold=acc.n_hold, sell=acc.n_sell_act, buy=acc.n_buy_act))
end
