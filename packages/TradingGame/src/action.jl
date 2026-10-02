"""
Action masking and cash-constraint normalisation.

`resolve_actions` is the single place rules 5 (purchases capped by available
cash), 10 (1-day lock-up before a voluntary sale), 12 (≤15% of portfolio value
per symbol), 13 (≤`n_max_holdings(N)` distinct symbols held at once) and 15
(a symbol can't be newly bought again for `REBUY_COOLDOWN_DAYS` trading days
after it was last sold) are enforced. The rest of `step!` trusts its output
unconditionally (see `CashConstraintViolation`) — a policy's raw output should
never need to *learn* these constraints, only choose among the options this
masking leaves available.

Rule 14 (cash can't exceed `MAX_CASH_FRACTION` of portfolio value) is
deliberately NOT masked here — it's enforced as a reward penalty in `env.jl`'s
`step!` instead. See `MAX_CASH_FRACTION`'s docstring in `constants.jl` for why.
"""

using Random

"""
Mask and normalise `raw` against the current portfolio and candidate universe.

- `SELL` on a symbol with no sellable lot (not held, or every lot still inside
  its `MIN_HOLD_DAYS` lock-up) becomes `HOLD`.
- `BUY` on a symbol outside `env.candidate_sym_idx` becomes `HOLD`.
- A `BUY` that would open a *new* (not already-held) position while the
  portfolio already holds `n_max_holdings(N)` distinct symbols (`N` = the
  candidate universe size for this episode) becomes `HOLD` (rule 13) — adding
  to an already-held symbol is unaffected. Based on holdings as they stood at
  the start of this step (a same-step sell doesn't free a slot for a
  same-step buy of a different new symbol), same convention as the cash check
  below excluding same-step sale proceeds. When more candidates want to open
  a new position than there are remaining slots, which ones get in is chosen
  uniformly at random via `rng` — not by each candidate's position in `raw`
  (i.e. in the candidate universe list), which would otherwise give
  earlier-listed candidates a standing edge every time the cap binds.
- A `BUY` that would open a *new* position in a symbol still inside its
  `REBUY_COOLDOWN_DAYS` window since it was last sold (voluntarily or via a
  rule-9 forced exit) becomes `HOLD` (rule 15) — checked against
  `env.portfolio.rebuy_cooldown`, set by `_execute_sell!`. Only applies to
  opening a *new* position: adding to a symbol that still has a separate,
  not-yet-sold lot open is unaffected (that symbol isn't "being rebought",
  it's just being added to), same `a.sym_idx in held_symbols` carve-out as
  rule 13 above.
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
function resolve_actions(env::TradingGameEnv, raw::JointAction, date_idx::Int;
                          rng::AbstractRNG=Random.default_rng())::Vector{ResolvedTrade}
    sellable = Set{Int}()
    for h in env.portfolio.holdings
        (date_idx - h.entry_date_idx >= MIN_HOLD_DAYS) && push!(sellable, h.sym_idx)
    end

    held_symbols = Set(h.sym_idx for h in env.portfolio.holdings)
    n_held       = length(held_symbols)
    n_max        = n_max_holdings(length(env.candidate_sym_idx))

    resolved      = ResolvedTrade[]
    buys          = RawAction[]
    new_positions = RawAction[]
    seen_new      = Set{Int}()   # defends against a malformed `raw` with duplicate sym_idx entries

    for a in raw
        if a.kind == SELL
            a.sym_idx in sellable && push!(resolved, ResolvedTrade(a.sym_idx, SELL, 0.0))
        elseif a.kind == BUY
            a.sym_idx in env.candidate_sym_idx || continue
            if a.sym_idx in held_symbols
                push!(buys, a)
            elseif date_idx >= get(env.portfolio.rebuy_cooldown, a.sym_idx, typemin(Int)) && !(a.sym_idx in seen_new)
                push!(new_positions, a)
                push!(seen_new, a.sym_idx)
            end
            # else: still inside its rule-15 rebuy cooldown — masked to HOLD
        end
        # HOLD (or a masked SELL/BUY) contributes nothing — absence from
        # `resolved` *is* the hold, apply_actions! only iterates trades.
    end

    # Rule 13: at most n_max distinct symbols held at once. `new_positions`
    # collects every candidate that wants to open a brand-new position;
    # shuffle before truncating to `remaining_slots` so which ones get in is
    # picked uniformly at random rather than by list order (see docstring).
    remaining_slots = max(0, n_max - n_held)
    length(new_positions) > remaining_slots && shuffle!(rng, new_positions)
    append!(buys, view(new_positions, 1:min(length(new_positions), remaining_slots)))

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
