"""
Random and heuristic policies with no neural network — used to validate the
simulator's rule compliance (Stage 1's "done" gate) before any RL code exists,
and later as training/evaluation baselines.
"""

using Random, Statistics

"""
Independently sample HOLD/SELL/BUY for every candidate, uniform buy weights.
Exists purely to exercise every code path in `resolve_actions`/`step!` under
adversarial-ish conditions (attempted sells on locked/unheld lots, attempted
buys beyond available cash, …) — good rule-compliance fuzzing, not a policy
anyone expects to make money.
"""
function random_policy(env::TradingGameEnv; rng::AbstractRNG=Random.default_rng())::JointAction
    actions = RawAction[]
    for sym_idx in env.candidate_sym_idx
        kind = rand(rng, (HOLD, SELL, BUY))
        weight = kind == BUY ? rand(rng) : 0.0
        push!(actions, RawAction(sym_idx, kind, weight))
    end
    return actions
end

"""
Simple momentum heuristic: buy candidates whose price rose over the last
`lookback_bars` hourly bars and aren't already held; sell held lots whose price
has fallen over the same window (masking handles the 1-day lock-up — an
attempted early sell just becomes a no-op HOLD). Buy weights are equal-split
across all momentum-positive candidates.
"""
function heuristic_policy(env::TradingGameEnv; lookback_bars::Int=7)::JointAction
    actions = RawAction[]
    t = env.current_hour_idx
    t <= lookback_bars && return actions   # not enough history yet — hold everything

    held_syms = Set(h.sym_idx for h in env.portfolio.holdings)

    for sym_idx in env.candidate_sym_idx
        p_now  = current_price(env, sym_idx)
        p_then = env.cache.hourly_closes[t - lookback_bars, sym_idx]
        (isnan(p_now) || isnan(p_then) || p_then <= 0) && continue
        momentum = (p_now - p_then) / p_then

        if sym_idx in held_syms
            momentum < 0 && push!(actions, RawAction(sym_idx, SELL, 0.0))
        elseif momentum > 0
            push!(actions, RawAction(sym_idx, BUY, 1.0))   # renormalised across all buys in resolve_actions
        end
    end

    return actions
end
