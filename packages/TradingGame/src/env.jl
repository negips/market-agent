"""
The rule-exact market simulator: `reset!`/`step!` over an `InferenceCache`.

Every bar of the `InferenceCache` is a decision bar — hourly bars for an hourly
cache, 15-minute bars (rule 8's minimum interval) for a 15-minute one — so
rule 8's "or immediately after a news item" is a no-op (news can't fire *more*
often than every bar). The `news_hour_indices` mechanism is kept as a hook.
The cache's `bar_minutes` sets the cadence; see `decision_granularity`.
"""

using Dates, Random
using StockSwingPredictor: find_hourly_end, find_date, has_history

# ── Lifecycle ─────────────────────────────────────────────────────────────────────

"""
Start a new episode: resets cash/holdings/reserved cash, jumps the clock to the
first hourly bar at or before `config.start_date`'s market open, and restricts
the tradeable universe to `config.candidate_universe`.
"""
function reset!(env::TradingGameEnv, config::EpisodeConfig)
    r = config.rules
    wrong_exchange = !isempty(r.required_exchange) && env.cache.exchange != r.required_exchange
    wrong_bars     = r.required_bar_minutes != 0 && env.cache.bar_minutes != r.required_bar_minutes
    (wrong_exchange || wrong_bars) && error(
        "TradingGameEnv.reset!: game v$(r.version) needs a $(uppercase(r.required_exchange)) " *
        "$(r.required_bar_minutes)-minute cache, got $(uppercase(env.cache.exchange)) " *
        "$(env.cache.bar_minutes)-minute (build_cache.jl --exchange bse --granularity 15min)")
    r.use_history && !has_history(env.cache) && error(
        "TradingGameEnv.reset!: game v$(r.version) reads its price window from an hourly history axis, but this cache " *
        "has none (build_cache.jl --exchange bse --granularity 15min --history resample)")
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
    env.penalty_accum                = 0.0

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
    illegal = Ref(0.0)
    if is_decision_bar(env)
        resolved = resolve_actions(env, raw_actions, date_idx; rng=rng,
                                    penalty=illegal, illegal_cost=rules.illegal_penalty_coef)
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
    reward -= illegal[]

    info = Dict{String, Any}(
        "portfolio_value"       => value,
        "stocks_value"          => breakdown.stocks_value,
        "cash_value"            => breakdown.cash_value,
        "cash_fraction"         => cash_fraction,
        "cash_ceiling_violated" => cash_excess > 0,
        "illegal_penalty"       => illegal[],
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

"""The decision cadence implied by `cache.bar_minutes`: `MINUTE_15` for a
15-minute cache, `HOURLY` for an hourly one. Errors for any other bar length."""
function decision_granularity(cache)::DecisionGranularity
    cache.bar_minutes == 15 && return MINUTE_15
    cache.bar_minutes == 60 && return HOURLY
    error("decision_granularity: unsupported bar length $(cache.bar_minutes) min (expected 15 or 60)")
end

"""Whether the current bar accepts a voluntary action. Always `true`: every bar
of a 15-minute or hourly cache is a decision bar, and a 15-minute bar already
meets rule 8's minimum interval (`DECISION_INTERVAL_MIN`). Errors if the cache's
bars are shorter than that interval."""
function is_decision_bar(env::TradingGameEnv)::Bool
    decision_granularity(env.cache)
    env.cache.bar_minutes >= DECISION_INTERVAL_MIN && return true
    error("is_decision_bar: cache bars ($(env.cache.bar_minutes) min) are shorter than rule 8's $(DECISION_INTERVAL_MIN)-minute interval")
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
    illegal = Ref(0.0)
    value_at_decision = rules.terminal_reward ? portfolio_value(env) : 0.0   # what "1% of portfolio value" is taken of
    if is_decision_bar(env)
        resolved = resolve_actions(env, raw_actions, date_idx; rng=rng,
                                    penalty=illegal, illegal_cost=rules.illegal_penalty_coef)
        events = _apply_actions!(env, resolved, date_idx)
        n_executed = length(resolved)
    end

    _advance_clock!(env)
    new_date_idx = env.cache.date_index[env.current_date]
    n_settled += _settle_reserved_cash!(env.portfolio, new_date_idx)

    breakdown = portfolio_breakdown(env)
    value = breakdown.value
    done  = env.current_hour_idx >= env.end_hour_idx

    reward = rules.terminal_reward ? 0.0 : _log_return_reward!(env, new_date_idx, value, done)

    cash_fraction = value > 0 ? env.portfolio.cash / value : 0.0
    cash_excess   = max(0.0, cash_fraction - MAX_CASH_FRACTION)
    if cash_excess > 0
        env.cash_over_since_date_idx == 0 && (env.cash_over_since_date_idx = new_date_idx)
    else
        env.cash_over_since_date_idx = 0
    end
    overdue_fraction = overdue_stock_fraction(env, new_date_idx, value, rules.max_hold_days)

    if rules.terminal_reward
        # Penalties pile up in rupees (an illegal move: its share of the portfolio value when it was
        # attempted; the cash/hold penalties: their share of the value on that bar) and are paid,
        # together with the window's gain, when the reward window ends — every `reward_window_days`
        # trading days, and at the episode's last bar. Both are measured against the portfolio value
        # at the START of the window, which then becomes the base of the next one.
        env.penalty_accum += value_at_decision * illegal[] +
                             value * (rules.cash_penalty_coef * cash_excess + rules.hold_penalty_coef * overdue_fraction)
        due = done || (rules.reward_window_days > 0 &&
                       new_date_idx - env.reward_window_start_date_idx >= rules.reward_window_days)
        if due
            v0 = env.reward_window_start_value
            reward = (value - v0) / v0 - env.penalty_accum / v0
            env.reward_window_start_value    = value
            env.reward_window_start_date_idx = new_date_idx
            env.penalty_accum                = 0.0
        end
    else
        reward -= rules.cash_penalty_coef * cash_excess
        reward -= rules.hold_penalty_coef * overdue_fraction
        reward -= illegal[]
    end

    info = Dict{String, Any}(
        "portfolio_value"       => value,
        "stocks_value"          => breakdown.stocks_value,
        "cash_value"            => breakdown.cash_value,
        "cash_fraction"         => cash_fraction,
        "cash_ceiling_violated" => cash_excess > 0,
        "overdue_fraction"      => overdue_fraction,
        "illegal_penalty"       => illegal[],
        "penalty_accum"         => env.penalty_accum,
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
