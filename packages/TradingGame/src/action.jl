"""
Action masking and cash-constraint normalisation.

`resolve_actions` is the single place rules 5 (purchases capped by available
cash), 10 (1-day lock-up before a voluntary sale), 12 (≤15% of portfolio value
per symbol), 13 (≤`n_max_holdings(N)` distinct symbols held at once) and 15
(a symbol can't be newly bought again for `REBUY_COOLDOWN_DAYS` trading days
after it was last sold) are enforced. The rest of `step!` trusts its output
unconditionally (see `CashConstraintViolation`).

Rule enforcement is split in two:

1. **Impossible moves are masked before the policy samples** (`sellable_mask`,
   `mask_action_logits`): a SELL on a stock with nothing sellable is removed from
   that stock's distribution, so the HOLD/SELL/BUY probabilities are conditional
   on what can actually be done and the PPO log-probabilities are taken under the
   same masked distribution.
2. **Everything else is applied afterwards, here**, and any move the rules refuse
   is *penalised at once*: `resolve_actions` adds `illegal_cost` (a fraction of
   portfolio value, `GameRules.illegal_penalty_coef`) to a running `penalty`
   every time it rejects or trims one — a buy with no or too little cash, in
   rebuy cooldown, beyond rule 13's holdings cap, or over rule 12's position cap.
   Under v1/v2 the cost is 0 and the refused moves just vanish, as before.

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

# Illegal moves
Pass `penalty = Ref(0.0)` and `illegal_cost` to have `illegal_cost` added to it
for every move the rules refuse: a `SELL` with nothing sellable (a policy that
masks via `sellable_mask` never produces one; baselines can), a `BUY` of a symbol
in its rebuy cooldown, a `BUY` that finds no slot under rule 13's holdings cap, a
`BUY` with no cash to spend, a `BUY` the remaining cash cannot fund even one
share of, and a `BUY` that would exceed rule 12's position cap (the part within
the cap still executes). The caller (`step!`) subtracts the total from that
bar's reward straight away.

# Returns
`Vector{ResolvedTrade}` — ready to pass to `apply_actions!` as-is.
"""
function resolve_actions(env::TradingGameEnv, raw::JointAction, date_idx::Int;
                          rng::AbstractRNG=Random.default_rng(),
                          penalty::Union{Nothing, Base.RefValue{Float64}}=nothing,
                          illegal_cost::Float64=0.0)::Vector{ResolvedTrade}
    bump!() = penalty === nothing || (penalty[] += illegal_cost)
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
            a.sym_idx in sellable ? push!(resolved, ResolvedTrade(a.sym_idx, SELL, 0.0)) : bump!()
        elseif a.kind == BUY
            a.sym_idx in env.candidate_sym_idx || continue
            if a.sym_idx in held_symbols
                push!(buys, a)
            elseif date_idx >= get(env.portfolio.rebuy_cooldown, a.sym_idx, typemin(Int)) && !(a.sym_idx in seen_new)
                push!(new_positions, a)
                push!(seen_new, a.sym_idx)
            else
                bump!()
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
    for _ in 1:max(0, length(new_positions) - remaining_slots)
        bump!()
    end
    append!(buys, view(new_positions, 1:min(length(new_positions), remaining_slots)))

    if !isempty(buys)
        total_weight = sum(max(0.0, b.weight) for b in buys)
        if env.portfolio.cash <= 0
            for _ in buys; bump!(); end   # nothing to spend: every buy is illegal, then rejected
        elseif total_weight > 0
            cash        = env.portfolio.cash
            total_value = portfolio_value(env)
            for b in buys
                share    = max(0.0, b.weight) / total_weight
                notional = cash * share / (1 + FEE_RATE)

                existing_value = sum(h.quantity * current_price(env, h.sym_idx)
                                      for h in env.portfolio.holdings if h.sym_idx == b.sym_idx; init=0.0)
                room     = MAX_POSITION_FRACTION * total_value - existing_value
                flagged = notional > max(0.0, room) + 1e-9
                flagged && bump!()   # rule 12: the buy would take this symbol over its position cap
                notional = min(notional, max(0.0, room))

                price = current_price(env, b.sym_idx)
                if notional > 0 && !flagged && (isnan(price) || price <= 0 || floor(notional / price) < 1)
                    bump!()   # not enough cash for even one share
                    continue
                end

                notional > 0 && push!(resolved, ResolvedTrade(b.sym_idx, BUY, notional))
            end
        end
    end

    return resolved
end


# ── Pre-sampling mask (impossible moves) ────────────────────────────────────────

"""`(N,)` Bool: `true` where candidate `c` (in `env.candidate_order` order) has at
least one lot that may be sold now (held and past the `MIN_HOLD_DAYS` lock-up).
A SELL on any other stock is impossible, so it is removed from the policy's
distribution before sampling — see `mask_action_logits`."""
function sellable_mask(env::TradingGameEnv, date_idx::Int)::Vector{Bool}
    sellable = Set{Int}()
    for h in env.portfolio.holdings
        (date_idx - h.entry_date_idx >= MIN_HOLD_DAYS) && push!(sellable, h.sym_idx)
    end
    return Bool[sym_idx in sellable for sym_idx in env.candidate_order]
end

"""Logit value for a masked action. Large and negative rather than `-Inf`, so
`logsoftmax`/`softmax` and their gradients stay finite (`exp(-1e4)` underflows to
exactly `0` in `Float32`)."""
const MASKED_LOGIT = -1f4

"""`(3, N, B)` action logits with each non-sellable stock's SELL logit (row 2,
`ActionType` order HOLD/SELL/BUY) replaced by `MASKED_LOGIT`. `sell_ok` is
`(N, B)`. The softmax of the result is the policy's distribution *conditional on
the move being possible*."""
function mask_action_logits(logits::AbstractArray{<:Real}, sell_ok::AbstractArray{Bool})
    size(logits, 1) == 3 || error("mask_action_logits: expected 3 action rows, got $(size(logits, 1))")
    ok    = reshape(sell_ok, 1, size(sell_ok, 1), :)                  # (1, N, B)
    extra = MASKED_LOGIT .* (1f0 .- Float32.(ok))                      # 0 where SELL is possible
    z     = zero(extra)
    return logits .+ vcat(z, extra, z)                                  # row 2 = SELL
end
