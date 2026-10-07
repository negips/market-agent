"""
The rule-exact market simulator: `reset!`/`step!` over an `InferenceCache`.

Trained at `TRAINING_DECISION_GRANULARITY = HOURLY` (see `constants.jl` for why)
— every hourly bar is a decision bar, so rule 8's "or immediately after a news
item" is currently a no-op (news can't fire *more* often than every bar). The
`news_hour_indices` mechanism is kept as a forward-compatible hook for a future
`MINUTE_15` cache.
"""

using Dates, Random
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
    empty!(env.portfolio.rebuy_cooldown)

    start_hour = find_hourly_end(env.cache, DateTime(config.start_date, Time(9, 15)))
    start_hour == 0 && error(
        "TradingGameEnv.reset!: no hourly bars at or before episode start $(config.start_date)")
    env.current_hour_idx = start_hour
    env.current_date     = Date(env.cache.hourly_datetimes[start_hour])

    env.reward_window_start_date_idx = env.cache.date_index[env.current_date]
    env.reward_window_start_value    = config.initial_cash
    env.cash_over_since_date_idx     = env.reward_window_start_date_idx   # episodes start at 100% cash, above any cap < 1

    env.daily_value_base_date_idx = env.reward_window_start_date_idx
    empty!(env.daily_values)
    push!(env.daily_values, config.initial_cash)

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

"""The mark/execution price for `sym_idx` at the current bar — the hourly
close, UNLESS `env.price_overrides` has a snapshot for `(env.current_hour_idx,
sym_idx)`. That override is how a qualifying news event's exact 1-minute
price (`news_features.jl`, built from real 1-minute OHLCV around the event's
timestamp, not the enclosing hourly bar) reaches every "price right now"
read in the simulator — `portfolio_breakdown`, `_execute_sell!`,
`_apply_actions!`'s BUY branch, `resolve_actions`'s rule-12 room check,
`observation.jl`'s current-bar price/holding features, and the two baseline
policies all go through this single function rather than reading
`cache.hourly_closes` directly, so the override logic exists exactly once.
`env.price_overrides` is empty by default, which makes this identical to a
plain `cache.hourly_closes[env.current_hour_idx, sym_idx]` read — see
`TradingGameEnv`'s docstring."""
function current_price(env::TradingGameEnv, sym_idx::Int)::Float32
    ov = _price_override(env, sym_idx)
    ov === nothing && return env.cache.hourly_closes[env.current_hour_idx, sym_idx]
    return ov
end

"""Raw news-instant override lookup for `sym_idx` at the current bar, or
`nothing` if none exists — `current_price` builds on this directly;
`observation.jl`'s price-window assembly also calls it directly (not
`current_price`) since it needs the un-substituted value to anchor-normalise
consistently with the rest of the window, not a final price."""
function _price_override(env::TradingGameEnv, sym_idx::Int)::Union{Nothing, Float32}
    overrides = get(env.price_overrides, env.current_hour_idx, nothing)
    overrides === nothing && return nothing
    return get(overrides, sym_idx, nothing)
end

"""Decomposes total portfolio value into its two components — stock value
(mark-to-market at the current price, see `current_price`) and cash value
(spendable cash plus reserved/settling cash, rules 4 and 7) — so callers that
need the breakdown (e.g. `step!`'s info dict, consumed by `live.jl` for the
website's stock-value/cash-value chart lines) don't duplicate the holdings
loop `portfolio_value` already does."""
function portfolio_breakdown(env::TradingGameEnv)::NamedTuple{(:value, :stocks_value, :cash_value), Tuple{Float64, Float64, Float64}}
    stocks = 0.0
    for h in env.portfolio.holdings
        stocks += h.quantity * current_price(env, h.sym_idx)
    end
    reserved = isempty(env.portfolio.reserved) ? 0.0 : sum(l.amount for l in env.portfolio.reserved)
    cash = env.portfolio.cash + reserved
    return (value=stocks + cash, stocks_value=stocks, cash_value=cash)
end

"""Total portfolio value: stock value (mark-to-market at the current hourly
close) + spendable cash + reserved cash (rules 2 and 7, the same formula)."""
portfolio_value(env::TradingGameEnv)::Float64 = portfolio_breakdown(env).value

# ── Step ──────────────────────────────────────────────────────────────────────────

"""
Advance one hourly bar. `raw_actions` is only applied on a decision bar — see
`is_decision_bar`; on any other bar it is ignored entirely (rule 8 enforced
structurally, not via a learned no-op).

Order of operations within a step (see module docs for the full rationale):
settle matured reserved cash → force-exit stale holdings (rule 9, runs on
every bar) → apply the voluntary action, if this is a decision bar (rule 8,
masked per rules 5 and 10 in `resolve_actions`) → mark portfolio value →
reward. The reward's log-return term is computed by `_log_return_reward!`
according to `TRAINING_REWARD_MODE` (`SPARSE_WINDOW` or `ROLLING_WINDOW` —
see `constants.jl`), with rule 14's cash-ceiling soft penalty subtracted on
top every single bar regardless of reward mode (see
`MAX_CASH_FRACTION`/`CASH_CEILING_PENALTY_COEF` — an independent, differently-
cadenced reward shaping term, not a structural mask like rules 12/13).

`rng` is forwarded unchanged to `resolve_actions`, which draws from it only
when rule 13's holdings cap binds on this bar (see that function's
docstring) — irrelevant to every other bar, so the default
`Random.default_rng()` is fine outside of `collect_rollout`, which forwards
its own `rng` here instead so a given seed covers the whole rollout.
"""
function step!(env::TradingGameEnv, raw_actions::JointAction=RawAction[];
                rng::AbstractRNG=Random.default_rng())::StepResult
    env.config === nothing && error("TradingGameEnv.step!: call reset! before step!")
    rules = env.config.rules
    rules.same_bar_execution && return _step_same_bar!(env, raw_actions, rules; rng=rng)

    _advance_clock!(env)
    date_idx = env.cache.date_index[env.current_date]

    n_settled = _settle_reserved_cash!(env.portfolio, date_idx)
    forced, forced_events = _force_exit_stale_holdings!(env, date_idx; max_hold_days=rules.max_hold_days)

    n_executed    = 0
    voluntary_events = TradeEvent[]
    if is_decision_bar(env)
        resolved = resolve_actions(env, raw_actions, date_idx; rng=rng)
        voluntary_events = _apply_actions!(env, resolved, date_idx)
        n_executed = length(resolved)
    end

    breakdown = portfolio_breakdown(env)
    value = breakdown.value
    done  = env.current_hour_idx >= env.end_hour_idx

    reward = _log_return_reward!(env, date_idx, value, done)

    # Rule 14 (soft): spendable cash above MAX_CASH_FRACTION of portfolio value
    # costs a per-bar reward penalty rather than being structurally blocked —
    # see `MAX_CASH_FRACTION`'s docstring in `constants.jl` for why a hard
    # ceiling isn't well-defined here (episode starts at 100% cash; matured
    # reserved cash lands back in cash passively, not via a masked action).
    # Independent of the log-return term's cadence above — always every bar.
    cash_fraction = value > 0 ? env.portfolio.cash / value : 0.0
    cash_excess   = max(0.0, cash_fraction - MAX_CASH_FRACTION)
    reward -= rules.cash_penalty_coef * cash_excess

    info = Dict{String, Any}(
        "portfolio_value"       => value,
        "stocks_value"          => breakdown.stocks_value,
        "cash_value"            => breakdown.cash_value,
        "cash_fraction"         => cash_fraction,
        "cash_ceiling_violated" => cash_excess > 0,
        "forced_exits"          => length(forced),
        "reserved_settled"      => n_settled,
        "actions_executed"      => n_executed,
        "trades"                => vcat(forced_events, voluntary_events),
    )
    return StepResult(reward, done, info)
end

"""Compute this bar's reward log-return term and advance whatever state
`mode` needs for next time — see `RewardMode`'s docstring in `constants.jl`
for the formula/tradeoffs of each mode. Called once per `step!`, after
`value`/`done` are known. Does NOT include rule 14's cash-ceiling penalty —
that's added by the caller, unconditionally, on top of whatever this
returns. `mode` defaults to the module-wide `TRAINING_REWARD_MODE`; exposed
as an explicit argument (rather than reading the constant directly) purely
so tests can exercise both branches without redefining a `const` at
runtime, which Julia doesn't support safely — `step!` itself never passes it
explicitly, so changing modes for a real run is still just the one-line
`TRAINING_REWARD_MODE` edit in `constants.jl`."""
function _log_return_reward!(env::TradingGameEnv, date_idx::Int, value::Float64, done::Bool;
                              mode::RewardMode=TRAINING_REWARD_MODE)::Float64
    if mode == SPARSE_WINDOW
        # 0 until REWARD_INTERVAL_DAYS trading days have passed since the last
        # checkpoint, then the FULL window's return in one lump sum. The final
        # bar force-flushes a shorter trailing window so the episode-total
        # reward still telescopes exactly to log(V_final/V_initial) — nothing
        # is silently dropped, it's just reported in ~weekly chunks.
        if done || date_idx - env.reward_window_start_date_idx >= REWARD_INTERVAL_DAYS
            reward = log(value / env.reward_window_start_value)
            env.reward_window_start_date_idx = date_idx
            env.reward_window_start_value    = value
            return reward
        end
        return 0.0
    else   # ROLLING_WINDOW
        # Record today's value once, the first time this trading day is seen
        # — daily_values ends up with exactly one entry per trading day, in
        # strictly consecutive date_idx order (the simulator never skips a
        # trading day), which is what makes the O(1) index lookup below valid.
        today_offset = date_idx - env.daily_value_base_date_idx + 1
        today_offset > length(env.daily_values) && push!(env.daily_values, value)

        lookback_offset = today_offset - REWARD_INTERVAL_DAYS
        lookback_offset < 1 && return 0.0   # not enough history yet (episode's first REWARD_INTERVAL_DAYS trading days)
        return log(value / env.daily_values[lookback_offset])
    end
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
function _force_exit_stale_holdings!(env::TradingGameEnv, date_idx::Int; max_hold_days::Int=MAX_HOLD_DAYS)
    stale = filter(h -> date_idx - h.entry_date_idx >= max_hold_days, env.portfolio.holdings)
    events = [_execute_sell!(env, h, date_idx; reason="forced_exit") for h in stale]
    filter!(h -> !(date_idx - h.entry_date_idx >= max_hold_days), env.portfolio.holdings)
    return stale, events
end

"""Game v2's `step!` (`rules.same_bar_execution`). The decision made on the
current bar is filled at THIS bar's close — the stand-in for a live system's
instantaneous price — and only then does the clock advance to mark the
result, so the reward for bar t's decision is the price move from bar t to
t+1. Order: settle reserved cash (rule 4) → apply the masked action → advance
the clock → settle again for the new date (so the next observation's cash
matches what `resolve_actions` will see) → mark value → reward minus the soft
penalties. There is no forced exit: lots held `rules.max_hold_days`+ days only
cost `rules.hold_penalty_coef` times their share of portfolio value per bar,
the stock analogue of the cash penalty."""
function _step_same_bar!(env::TradingGameEnv, raw_actions::JointAction, rules::GameRules;
                          rng::AbstractRNG=Random.default_rng())::StepResult
    date_idx = env.cache.date_index[env.current_date]
    n_settled = _settle_reserved_cash!(env.portfolio, date_idx)

    n_executed = 0
    events = TradeEvent[]
    if is_decision_bar(env)
        resolved = resolve_actions(env, raw_actions, date_idx; rng=rng)
        events = _apply_actions!(env, resolved, date_idx)
        n_executed = length(resolved)
    end

    _advance_clock!(env)
    new_date_idx = env.cache.date_index[env.current_date]
    n_settled += _settle_reserved_cash!(env.portfolio, new_date_idx)

    breakdown = portfolio_breakdown(env)
    value = breakdown.value
    done  = env.current_hour_idx >= env.end_hour_idx

    reward = _log_return_reward!(env, new_date_idx, value, done)

    cash_fraction = value > 0 ? env.portfolio.cash / value : 0.0
    cash_excess   = max(0.0, cash_fraction - MAX_CASH_FRACTION)
    reward -= rules.cash_penalty_coef * cash_excess
    if cash_excess > 0
        env.cash_over_since_date_idx == 0 && (env.cash_over_since_date_idx = new_date_idx)
    else
        env.cash_over_since_date_idx = 0
    end

    overdue_fraction = overdue_stock_fraction(env, new_date_idx, value, rules.max_hold_days)
    reward -= rules.hold_penalty_coef * overdue_fraction

    info = Dict{String, Any}(
        "portfolio_value"       => value,
        "stocks_value"          => breakdown.stocks_value,
        "cash_value"            => breakdown.cash_value,
        "cash_fraction"         => cash_fraction,
        "cash_ceiling_violated" => cash_excess > 0,
        "overdue_fraction"      => overdue_fraction,
        "forced_exits"          => 0,
        "reserved_settled"      => n_settled,
        "actions_executed"      => n_executed,
        "trades"                => events,
    )
    return StepResult(reward, done, info)
end

"""Share of portfolio value held in lots at least `max_hold_days` trading days
old, at the current bar's prices — the stock-side input to game v2's soft hold
penalty (`GameRules.hold_penalty_coef`), the analogue of `cash_fraction` for
the cash penalty."""
function overdue_stock_fraction(env::TradingGameEnv, date_idx::Int, value::Float64, max_hold_days::Int)::Float64
    value > 0 || return 0.0
    overdue = 0.0
    for h in env.portfolio.holdings
        date_idx - h.entry_date_idx >= max_hold_days || continue
        price = current_price(env, h.sym_idx)
        isnan(price) || (overdue += h.quantity * price)
    end
    return overdue / value
end

"""Sell one lot at the current hourly close, crediting proceeds-minus-fee into
a new `ReservedCashLot` maturing `SETTLEMENT_DAYS` trading days from now (rules
4, 10, 11), and starting that symbol's rule-15 rebuy cooldown
(`REBUY_COOLDOWN_DAYS` trading days). Shared by both forced exits and
voluntary sells — rule 11 does not distinguish between them, and neither does
rule 15's cooldown. Returns a `TradeEvent` for `StepResult.info`
(display/logging only — never consulted for rule decisions)."""
function _execute_sell!(env::TradingGameEnv, h::Holding, date_idx::Int; reason::String="sell")::TradeEvent
    price    = current_price(env, h.sym_idx)
    proceeds = h.quantity * price
    fee      = FEE_RATE * proceeds
    push!(env.portfolio.reserved,
          ReservedCashLot(proceeds - fee, date_idx + SETTLEMENT_DAYS, h.symbol))
    env.portfolio.rebuy_cooldown[h.sym_idx] = date_idx + REBUY_COOLDOWN_DAYS
    cost_basis = h.quantity * h.entry_price + h.entry_fee
    pnl        = proceeds - fee - cost_basis
    return (kind=reason, symbol=h.symbol, price=price, quantity=h.quantity, notional=proceeds, fee=fee,
            date=string(env.current_date), t=string(env.cache.hourly_datetimes[env.current_hour_idx]),
            entry_price=h.entry_price, pnl=pnl, ret=cost_basis > 0 ? pnl / cost_basis : 0.0,
            days_held=date_idx - h.entry_date_idx,
            p_hold=UNRECORDED_PROB, p_sell=UNRECORDED_PROB, p_buy=UNRECORDED_PROB)
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
            price = current_price(env, t.sym_idx)
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
                           date=string(env.current_date), t=string(env.cache.hourly_datetimes[env.current_hour_idx]),
                           entry_price=price, pnl=0.0, ret=0.0, days_held=0,
                           p_hold=UNRECORDED_PROB, p_sell=UNRECORDED_PROB, p_buy=UNRECORDED_PROB))
        end
    end
    return events
end
