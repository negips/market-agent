"""
Action masking and cash-constraint normalisation.

`resolve_actions` is the single place rules 5 (purchases capped by available
cash) and 10 (1-day lock-up before a voluntary sale) are enforced. The rest of
`step!` trusts its output unconditionally (see `CashConstraintViolation`) — a
policy's raw output should never need to *learn* these constraints, only
choose among the options this masking leaves available.
"""

"""
Mask and normalise `raw` against the current portfolio and candidate universe.

- `SELL` on a symbol with no sellable lot (not held, or every lot still inside
  its `MIN_HOLD_DAYS` lock-up) becomes `HOLD`.
- `BUY` on a symbol outside `env.candidate_sym_idx` becomes `HOLD`.
- Surviving `BUY` weights are renormalised so total notional-plus-fee across
  all of them cannot exceed `env.portfolio.cash` as it stood at the start of
  this decision step (rule 5) — reserved cash and same-step sale proceeds are
  excluded, matching rule 4's settlement delay.

# Returns
`Vector{ResolvedTrade}` — ready to pass to `apply_actions!` as-is.
"""
function resolve_actions(env::TradingGameEnv, raw::JointAction, date_idx::Int)::Vector{ResolvedTrade}
    sellable = Set{Int}()
    for h in env.portfolio.holdings
        (date_idx - h.entry_date_idx >= MIN_HOLD_DAYS) && push!(sellable, h.sym_idx)
    end

    resolved = ResolvedTrade[]
    buys     = RawAction[]

    for a in raw
        if a.kind == SELL
            a.sym_idx in sellable && push!(resolved, ResolvedTrade(a.sym_idx, SELL, 0.0))
        elseif a.kind == BUY
            a.sym_idx in env.candidate_sym_idx && push!(buys, a)
        end
        # HOLD (or a masked SELL/BUY) contributes nothing — absence from
        # `resolved` *is* the hold, apply_actions! only iterates trades.
    end

    if !isempty(buys)
        total_weight = sum(max(0.0, b.weight) for b in buys)
        if total_weight > 0
            cash = env.portfolio.cash
            for b in buys
                share    = max(0.0, b.weight) / total_weight
                notional = cash * share / (1 + FEE_RATE)
                notional > 0 && push!(resolved, ResolvedTrade(b.sym_idx, BUY, notional))
            end
        end
    end

    return resolved
end
