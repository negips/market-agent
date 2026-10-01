"""
The rule-exact market simulator: `reset!`/`step!` over an `InferenceCache`.

Trained at `TRAINING_DECISION_GRANULARITY = HOURLY` (see `constants.jl` for why)
— every hourly bar is a decision bar, so rule 8's "or immediately after a news
item" is currently a no-op (news can't fire *more* often than every bar). The
`news_hour_indices` mechanism is kept as a forward-compatible hook for a future
`MINUTE_15` cache.
"""

using Dates
using StockSwingPredictor: find_hourly_end, find_date

# ── Lifecycle ─────────────────────────────────────────────────────────────────────

"""
Start a new episode: resets cash/holdings/reserved cash, jumps the clock to the
first hourly bar at or before `config.start_date`'s market open, and restricts
the tradeable universe to `config.candidate_universe`.
"""
function reset!(env::TradingGameEnv, config::EpisodeConfig)
    env.config = config
    env.portfolio.cash = config.initial_cash
    empty!(env.portfolio.reserved)
    empty!(env.portfolio.holdings)

    start_hour = find_hourly_end(env.cache, DateTime(config.start_date, Time(9, 15)))
    start_hour == 0 && error(
        "TradingGameEnv.reset!: no hourly bars at or before episode start $(config.start_date)")
    env.current_hour_idx = start_hour
    env.current_date     = Date(env.cache.hourly_datetimes[start_hour])

    end_hour = find_hourly_end(env.cache, DateTime(config.end_date, Time(15, 30)))
    end_hour == 0 && error(
        "TradingGameEnv.reset!: no hourly bars at or before episode end $(config.end_date)")
    env.end_hour_idx = end_hour

    empty!(env.candidate_sym_idx)
    empty!(env.candidate_order)
    for sym in config.candidate_universe
        idx = get(env.cache.sym_index, sym, 0)
        idx == 0 && error("TradingGameEnv.reset!: candidate symbol not in cache universe: $sym")
        push!(env.candidate_sym_idx, idx)
        push!(env.candidate_order, idx)
    end

    return nothing
end

"""Total portfolio value: stock value (mark-to-market at the current hourly
close) + spendable cash + reserved cash (rules 2 and 7, the same formula)."""
function portfolio_value(env::TradingGameEnv)::Float64
    stocks = 0.0
    for h in env.portfolio.holdings
        stocks += h.quantity * env.cache.hourly_closes[env.current_hour_idx, h.sym_idx]
    end
    reserved = isempty(env.portfolio.reserved) ? 0.0 : sum(l.amount for l in env.portfolio.reserved)
    return stocks + env.portfolio.cash + reserved
end

# ── Step ──────────────────────────────────────────────────────────────────────────

"""
Advance one hourly bar. `raw_actions` is only applied on a decision bar — see
`is_decision_bar`; on any other bar it is ignored entirely (rule 8 enforced
structurally, not via a learned no-op).

Order of operations within a step (see module docs for the full rationale):
settle matured reserved cash → force-exit stale holdings (rule 9, runs on
every bar) → apply the voluntary action, if this is a decision bar (rule 8,
masked per rules 5 and 10 in `resolve_actions`) → mark portfolio value → reward.
"""
function step!(env::TradingGameEnv, raw_actions::JointAction=RawAction[])::StepResult
    env.config === nothing && error("TradingGameEnv.step!: call reset! before step!")
    prev_value = portfolio_value(env)

    _advance_clock!(env)
    date_idx = env.cache.date_index[env.current_date]

    n_settled = _settle_reserved_cash!(env.portfolio, date_idx)
    forced, forced_events = _force_exit_stale_holdings!(env, date_idx)

    n_executed    = 0
    voluntary_events = TradeEvent[]
    if is_decision_bar(env)
        resolved = resolve_actions(env, raw_actions, date_idx)
        voluntary_events = _apply_actions!(env, resolved, date_idx)
        n_executed = length(resolved)
    end

    value  = portfolio_value(env)
    reward = log(value / prev_value)
    done   = env.current_hour_idx >= env.end_hour_idx

    info = Dict{String, Any}(
        "portfolio_value" => value,
        "forced_exits"    => length(forced),
        "reserved_settled" => n_settled,
        "actions_executed" => n_executed,
        "trades"          => vcat(forced_events, voluntary_events),
    )
    return StepResult(reward, done, info)
end

"""Whether the current bar accepts a voluntary action. Always `true` under the
`HOURLY` training proxy (see `constants.jl`); `MINUTE_15` is not yet
implemented since it needs a rolling 15-minute `InferenceCache`."""
function is_decision_bar(env::TradingGameEnv)::Bool
    TRAINING_DECISION_GRANULARITY == HOURLY && return true
    error("is_decision_bar: MINUTE_15 decision granularity requires a 15-min InferenceCache, not yet implemented")
end

# ── Internals ─────────────────────────────────────────────────────────────────────

function _advance_clock!(env::TradingGameEnv)
    env.current_hour_idx += 1
    env.current_hour_idx > length(env.cache.hourly_datetimes) &&
        error("TradingGameEnv.step!: exhausted hourly series — `done` should have ended the episode before this call")
    env.current_date = Date(env.cache.hourly_datetimes[env.current_hour_idx])
    return nothing
end

"""Rule 4: move every `ReservedCashLot` whose settlement date has arrived into
spendable cash. Runs before forced exits and voluntary actions so newly-settled
cash is usable the same step it becomes available."""
function _settle_reserved_cash!(portfolio::Portfolio, date_idx::Int)::Int
    matured = filter(l -> l.available_date_idx <= date_idx, portfolio.reserved)
    isempty(matured) && return 0
    portfolio.cash += sum(l.amount for l in matured)
    filter!(l -> l.available_date_idx > date_idx, portfolio.reserved)
    return length(matured)
end

"""Rule 8: any holding at or past `MAX_HOLD_DAYS` is sold unconditionally at
the current bar's price, before the voluntary action is applied. Exempt from
rule 10's lock-up — a forced exit is not a voluntary sale."""
function _force_exit_stale_holdings!(env::TradingGameEnv, date_idx::Int)
    stale = filter(h -> date_idx - h.entry_date_idx >= MAX_HOLD_DAYS, env.portfolio.holdings)
    events = [_execute_sell!(env, h, date_idx; reason="forced_exit") for h in stale]
    filter!(h -> !(date_idx - h.entry_date_idx >= MAX_HOLD_DAYS), env.portfolio.holdings)
    return stale, events
end

"""Sell one lot at the current hourly close, crediting proceeds-minus-fee into
a new `ReservedCashLot` maturing `SETTLEMENT_DAYS` trading days from now (rules
4, 10, 11). Shared by both forced exits and voluntary sells — rule 11 does not
distinguish between them. Returns a `TradeEvent` for `StepResult.info`
(display/logging only — never consulted for rule decisions)."""
function _execute_sell!(env::TradingGameEnv, h::Holding, date_idx::Int; reason::String="sell")::TradeEvent
    price    = env.cache.hourly_closes[env.current_hour_idx, h.sym_idx]
    proceeds = h.quantity * price
    fee      = FEE_RATE * proceeds
    push!(env.portfolio.reserved,
          ReservedCashLot(proceeds - fee, date_idx + SETTLEMENT_DAYS, h.symbol))
    return (kind=reason, symbol=h.symbol, price=price, quantity=h.quantity, notional=proceeds, fee=fee,
            date=string(env.current_date), t=string(env.cache.hourly_datetimes[env.current_hour_idx]))
end

"""Execute an already-masked set of trades (see `resolve_actions`). Buys are
checked against available cash defensively — a violation here means the
masking layer has a bug, not that the policy chose an invalid action. Returns
the step's trade events (for `StepResult.info["trades"]`, display only)."""
function _apply_actions!(env::TradingGameEnv, resolved::Vector{ResolvedTrade}, date_idx::Int)
    events = TradeEvent[]
    for t in resolved
        if t.kind == SELL
            lots = filter(h -> h.sym_idx == t.sym_idx && date_idx - h.entry_date_idx >= MIN_HOLD_DAYS,
                          env.portfolio.holdings)
            for h in lots
                push!(events, _execute_sell!(env, h, date_idx; reason="sell"))
            end
            filter!(h -> !(h.sym_idx == t.sym_idx && date_idx - h.entry_date_idx >= MIN_HOLD_DAYS),
                    env.portfolio.holdings)

        elseif t.kind == BUY
            price = env.cache.hourly_closes[env.current_hour_idx, t.sym_idx]
            (isnan(price) || price <= 0) && continue

            # Shares trade in whole units — `t.notional` is a cash budget, not a
            # literal spend. Floor to the affordable whole-share count and price
            # the trade off *that*, not the requested notional; any remainder
            # (less than one share's worth) simply stays in cash unspent.
            qty = floor(t.notional / price)
            qty < 1 && continue

            notional = qty * price
            fee      = FEE_RATE * notional
            debit    = notional + fee
            debit > env.portfolio.cash + 1e-6 &&
                throw(CashConstraintViolation(debit, env.portfolio.cash))

            env.portfolio.cash -= debit
            symbol = env.cache.companies[t.sym_idx]
            push!(env.portfolio.holdings, Holding(
                symbol         = symbol,
                sym_idx        = t.sym_idx,
                entry_date_idx = date_idx,
                entry_hour_idx = env.current_hour_idx,
                quantity       = qty,
                entry_price    = price,
                entry_fee      = fee,
            ))
            push!(events, (kind="buy", symbol=symbol, price=price, quantity=qty, notional=notional, fee=fee,
                           date=string(env.current_date), t=string(env.cache.hourly_datetimes[env.current_hour_idx])))
        end
    end
    return events
end
