"""
Action masking and cash-constraint normalisation.

`resolve_actions` is the single place rules 5 (purchases capped by available
cash), 10 (1-day lock-up before a voluntary sale), 12 (≤15% of portfolio value
per symbol) and 13 (≤`N_MAX_HOLDINGS` distinct symbols held at once) are
enforced. The rest of `step!` trusts its output unconditionally (see
`CashConstraintViolation`) — a policy's raw output should never need to
*learn* these constraints, only choose among the options this masking leaves
available.
"""

"""
Mask and normalise `raw` against the current portfolio and candidate universe.

- `SELL` on a symbol with no sellable lot (not held, or every lot still inside
  its `MIN_HOLD_DAYS` lock-up) becomes `HOLD`.
- `BUY` on a symbol outside `env.candidate_sym_idx` becomes `HOLD`.
- A `BUY` that would open a *new* (not already-held) position while the
  portfolio already holds `N_MAX_HOLDINGS` distinct symbols becomes `HOLD`
  (rule 13) — adding to an already-held symbol is unaffected. Based on
  holdings as they stood at the start of this step (a same-step sell doesn't
  free a slot for a same-step buy of a different new symbol), same convention
  as the cash check below excluding same-step sale proceeds.
- Surviving `BUY` weights are renormalised so total notional-plus-fee across
  all of them cannot exceed `env.portfolio.cash` as it stood at the start of
  this decision step (rule 5) — reserved cash and same-step sale proceeds are
  excluded, matching rule 4's settlement delay.
- Each `BUY`'s notional is then capped so that symbol's post-trade holding
  value can't exceed `MAX_POSITION_FRACTION` of total portfolio value (rule
  12, enforced at purchase time only — see the module docstring in
  `TradingGame.jl` for why organic price-drift above the cap is not force-
  trimmed). Cash freed up by this cap is left unspent this step, not
  redistributed to other buys — same choice already made when a buy is too
  small to afford one share.

# Returns
`Vector{ResolvedTrade}` — ready to pass to `apply_actions!` as-is.
"""
function resolve_actions(env::TradingGameEnv, raw::JointAction, date_idx::Int)::Vector{ResolvedTrade}
    sellable = Set{Int}()
    for h in env.portfolio.holdings
        (date_idx - h.entry_date_idx >= MIN_HOLD_DAYS) && push!(sellable, h.sym_idx)
    end

    held_symbols = Set(h.sym_idx for h in env.portfolio.holdings)
    n_held       = length(held_symbols)

    resolved = ResolvedTrade[]
    buys     = RawAction[]

    for a in raw
        if a.kind == SELL
            a.sym_idx in sellable && push!(resolved, ResolvedTrade(a.sym_idx, SELL, 0.0))
        elseif a.kind == BUY
            a.sym_idx in env.candidate_sym_idx || continue
            if a.sym_idx in held_symbols
                push!(buys, a)
            elseif n_held < N_MAX_HOLDINGS
                push!(buys, a)
                push!(held_symbols, a.sym_idx)   # reserves the slot against further buys this same step
                n_held += 1
            end
            # else: would open the (N_MAX_HOLDINGS+1)th distinct position — masked to HOLD (rule 13)
        end
        # HOLD (or a masked SELL/BUY) contributes nothing — absence from
        # `resolved` *is* the hold, apply_actions! only iterates trades.
    end

    if !isempty(buys)
        total_weight = sum(max(0.0, b.weight) for b in buys)
        if total_weight > 0
            cash        = env.portfolio.cash
            total_value = portfolio_value(env)
            for b in buys
                share    = max(0.0, b.weight) / total_weight
                notional = cash * share / (1 + FEE_RATE)

                existing_value = sum(h.quantity * env.cache.hourly_closes[env.current_hour_idx, h.sym_idx]
                                      for h in env.portfolio.holdings if h.sym_idx == b.sym_idx; init=0.0)
                room     = MAX_POSITION_FRACTION * total_value - existing_value
                notional = min(notional, max(0.0, room))

                notional > 0 && push!(resolved, ResolvedTrade(b.sym_idx, BUY, notional))
            end
        end
    end

    return resolved
end
