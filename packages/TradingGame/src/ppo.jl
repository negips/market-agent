"""
Hand-rolled PPO with GAE (no RL library exists in this Julia/Flux stack — see
the TradingGame module docstring). Covers the stochastic part of the action
space only: each candidate's HOLD/SELL/BUY choice is sampled from the actor's
categorical distribution and its log-probability drives the clipped surrogate
objective. The continuous buy-weight (`sigmoid(buy_weight_logit)`) is used
deterministically — `resolve_actions` renormalises it across whichever
candidates were sampled `BUY` — so PPO only needs to learn *which* stocks to
touch and *whether* to buy/sell/hold; sizing is a direct policy output rather
than a second sampled distribution. This keeps the credit-assignment problem
tractable for a hand-rolled implementation while still exercising the full
joint, cash-constrained action space each step.
"""

using Flux, Random, Statistics

"""Resolve `:cpu`/`:gpu` to Flux's `cpu`/`gpu` transfer functions — same
convention as `StockSwingPredictor.train!`'s `_to_device`."""
function _to_device(device::Symbol)
    device === :cpu && return cpu
    device === :gpu && return gpu
    error("Unknown device :$device — expected :cpu or :gpu")
end

# ── Rollout ───────────────────────────────────────────────────────────────────────

"""One decision step's rollout record — everything `ppo_update!` needs to
recompute the policy's log-probability under updated weights later.
Parametric for the same reason `Observation` is (see its docstring):
`collect_rollout` stores views into a preallocated per-episode tensor here,
not freshly heap-allocated arrays per step."""
struct RolloutStep{O<:Observation, AI<:AbstractVector{Int}, BW<:AbstractVector{Float32}}
    obs        :: O
    action_idx :: AI   # 1=HOLD, 2=SELL, 3=BUY per candidate (ActionType order)
    buy_weight :: BW   # sigmoid(buy_weight_logit) per candidate
    logprob    :: Float32          # Σ categorical log-prob across candidates, under the OLD policy
    value      :: Float32          # V(s) under the OLD policy
    reward     :: Float32
    done       :: Bool
end

"""Sample a 1-based index from a categorical distribution given as raw
probabilities (assumed to sum to ~1)."""
function _sample_categorical(probs::AbstractVector{<:Real}, rng::AbstractRNG)::Int
    r = rand(rng)
    c = 0.0
    for (i, p) in enumerate(probs)
        c += p
        r <= c && return i
    end
    return length(probs)
end

const _ACTION_TYPES = (HOLD, SELL, BUY)   # index i ↔ ActionType, matches the actor head's 3 logits

"""Stamp this bar's trade events with the actor's HOLD/SELL/BUY probabilities
for each event's own candidate (`probs` is `(3, N)`, `cand_pos` maps symbol →
candidate column). Forced exits get them too — the probability the policy
itself put on selling a position the rules were about to close anyway is the
clearest read on whether it ever chooses to sell. Replaces
`result.info["trades"]` in place; no-op on a bar with no trades."""
function _annotate_trade_probs!(result::StepResult, probs::AbstractMatrix{<:Real}, cand_pos::Dict{String, Int})
    trades = result.info["trades"]::Vector{TradeEvent}
    isempty(trades) && return nothing
    result.info["trades"] = TradeEvent[
        merge(ev, (p_hold=Float64(probs[1, cand_pos[ev.symbol]]),
                   p_sell=Float64(probs[2, cand_pos[ev.symbol]]),
                   p_buy =Float64(probs[3, cand_pos[ev.symbol]])))
        for ev in trades]
    return nothing
end

"""
Run one full episode (`reset!` then `step!` until `done`), sampling actions
from `policy` at every decision step and recording a `RolloutStep` per step.

`macro_cache`/`news_fn` are forwarded to `assemble_observation` unchanged —
omit them to use the documented neutral defaults (see `observation.jl`).
`greedy=true` picks each candidate's `argmax` action instead of sampling —
used for held-out evaluation rollouts, never for PPO training data (PPO needs
the exploration and the matching stochastic log-probability).

`live_cb`, when given, is called as `live_cb(env, result)` after every `step!`
— a hook for streaming this episode's progress (portfolio value, holdings,
trade events) to a live viewer (see `live.jl`); it never affects rollout
mechanics or training and defaults to a no-op.

Always runs its forward pass on CPU, regardless of `train_policy!`'s
`--device` — measured directly (see `packages/TradingGame/docs/` benchmark
notes): a single-bar (batch=1) forward pass through `hourly_encoder`'s
120-step GRU costs ~9.5ms on CPU, but the same call on GPU pays a host
round-trip plus 120 individual kernel launches for a batch this tiny, which
in practice runs roughly an order of magnitude SLOWER than CPU, not faster —
confirming what the old docstring here only warned about qualitatively. A
rollout is ~9,000+ such calls (one per hourly bar), so this is the single
biggest lever on rollout wall-clock time. If `policy` lives on GPU (trained
with `--device gpu`), a CPU copy is taken once up front (`cpu(policy)`,
*outside* the per-bar loop) and reused for the whole rollout — cheap relative
to ~9,000 avoided host round-trips, and correct: `ActorCriticPolicy`'s `GRU`
layers are stateless across calls when fed a full sequence each time (no
`Flux.reset!` needed — verified: calling a `Flux.GRU` twice with the same 3D
input gives identical output, confirming no hidden state persists between
calls here), so a fresh CPU snapshot of the current weights behaves
identically to calling the live GPU-resident `policy` directly. GPU stays
reserved for `ppo_update!`'s large minibatched passes, where it's the
workload that actually benefits (see `ppo_update!`'s docstring).

Preallocates the whole episode's observation tensors up front (`T` decision
bars is known exactly right after `reset!`: `env.end_hour_idx -
env.current_hour_idx`) instead of letting `assemble_observation` heap-allocate
a fresh `hourly`/`news`/`holding`/`macro_ctx`/`portfolio` array every single
bar. This matters because `buffer` must retain every bar's observation for
the whole episode (`ppo_update!` replays it across `k_epochs`), so those
per-bar allocations don't die young — they accumulate as live heap for the
entire rollout. Measured directly on a real run: with per-bar allocation, the
per-bar rate roughly DOUBLED from the start of a rollout to ~44% through it
(66ms/bar → ~150ms/bar), consistent with GC-scan cost growing with the ever-
growing live heap. Preallocating collapses that into a handful of large
up-front allocations (same total bytes, ~750MB for a full 5-year hourly
episode, but O(1) *objects* instead of O(T)), so each bar's `Observation`
is just a cheap `SubArray` view into a slice — no new backing array, no
growing-GC-pressure effect."""
function collect_rollout(env::TradingGameEnv, policy::ActorCriticPolicy, config::EpisodeConfig;
                          macro_cache::Union{Nothing, MacroCache}=nothing,
                          news_fn::Function=_zero_news,
                          greedy::Bool=false,
                          live_cb::Union{Nothing, Function}=nothing,
                          rng::AbstractRNG=Random.default_rng())
    policy_cpu = cpu(policy)
    reset!(env, config)

    T = env.end_hour_idx - env.current_hour_idx
    T <= 0 && error("collect_rollout: episode has no decision bars — check start_date/end_date")
    N = length(env.candidate_order)
    candidates = copy(env.candidate_order)   # invariant for the whole episode — one shared copy
    cand_pos   = Dict(env.cache.companies[s] => i for (i, s) in enumerate(candidates))

    hourly_buf    = Array{Float32}(undef, N_HOURLY_BARS_SHORT, N_PRICE_CHANNELS, N, T)
    macro_buf     = Array{Float32}(undef, N_MACRO_DAYS, N_MACRO_SERIES, T)
    news_buf      = Array{Float32}(undef, N_NEWS_FEATURES, N, T)
    holding_buf   = Array{Float32}(undef, N_HOLDING_FEATURES, N, T)
    portfolio_buf = Array{Float32}(undef, n_portfolio_scalars(config.rules), T)
    action_idx_buf = Array{Int}(undef, N, T)
    buy_weight_buf = Array{Float32}(undef, N, T)

    # `RolloutStep`/`Observation` are parametric (see their docstrings) — a
    # bare `Vector{RolloutStep}(undef, T)` would give `buffer` an ABSTRACT
    # element type (the unparameterized `RolloutStep`), making every later
    # field access type-unstable and defeating the point of this whole
    # preallocation. Compute the concrete, fully-parameterized type once from
    # the buffers' own view types (identical for every `t` — the indexing
    # *pattern* determines a view's type, not the runtime index value).
    ObsType  = Observation{typeof(@view hourly_buf[:, :, :, 1]), typeof(@view macro_buf[:, :, 1]),
                            typeof(@view news_buf[:, :, 1]), typeof(@view holding_buf[:, :, 1]),
                            typeof(@view portfolio_buf[:, 1])}
    StepType = RolloutStep{ObsType, typeof(@view action_idx_buf[:, 1]), typeof(@view buy_weight_buf[:, 1])}
    buffer = Vector{StepType}(undef, T)

    t = 0
    done = false
    while !done
        t += 1
        hourly_view    = @view hourly_buf[:, :, :, t]
        macro_view     = @view macro_buf[:, :, t]
        news_view      = @view news_buf[:, :, t]
        holding_view   = @view holding_buf[:, :, t]
        portfolio_view = @view portfolio_buf[:, t]
        assemble_observation!(hourly_view, macro_view, news_view, holding_view, portfolio_view,
                               env; macro_cache=macro_cache, news_fn=news_fn)
        obs = Observation(hourly_view, macro_view, news_view, holding_view, portfolio_view, candidates)

        batch = stack_observations([obs])
        action_logits, buy_weight_logit, value =
            policy_cpu(batch.hourly, batch.news, batch.holding, batch.macro_ctx, batch.portfolio)

        probs = Flux.softmax(action_logits[:, :, 1]; dims=1)   # (3, N)

        action_idx_view = @view action_idx_buf[:, t]
        logprob = 0f0
        for i in 1:N
            action_idx_view[i] = greedy ? argmax(view(probs, :, i)) : _sample_categorical(view(probs, :, i), rng)
            logprob += log(max(probs[action_idx_view[i], i], 1f-8))
        end
        buy_weight_view = @view buy_weight_buf[:, t]
        buy_weight_view .= Flux.sigmoid.(buy_weight_logit[:, 1])

        raw = RawAction[RawAction(sym_idx, _ACTION_TYPES[action_idx_view[i]], buy_weight_view[i])
                         for (i, sym_idx) in enumerate(obs.candidates)]
        result = step!(env, raw; rng=rng)
        _annotate_trade_probs!(result, probs, cand_pos)

        buffer[t] = RolloutStep(obs, action_idx_view, buy_weight_view, logprob, Float32(value[1]),
                                 Float32(result.reward), result.done)
        done = result.done
        live_cb !== nothing && live_cb(env, result)
    end
    t == T || error("collect_rollout: episode ran $t bars, expected exactly $T — " *
                     "preallocated buffer size assumption violated")
    return buffer
end

# ── GAE ───────────────────────────────────────────────────────────────────────────

"""
Generalised Advantage Estimation over one episode's `rewards`/`values`/`dones`.
`dones[t] == true` zeroes the bootstrap term at `t`, matching the standard
`δ_t = r_t + γ·V(s_{t+1})·(1-done_t) - V(s_t)` recursion. `advantages`,
`returns = advantages .+ values`.
"""
function compute_gae(rewards::Vector{Float32}, values::Vector{Float32}, dones::Vector{Bool};
                      gamma::Float64=GAMMA, gae_lambda::Float64=GAE_LAMBDA)
    T = length(rewards)
    advantages = zeros(Float32, T)
    gae = 0f0
    γ, λ = Float32(gamma), Float32(gae_lambda)
    for t in T:-1:1
        next_value       = t == T ? 0f0 : values[t + 1]
        next_nonterminal = dones[t] ? 0f0 : 1f0
        delta = rewards[t] + γ * next_value * next_nonterminal - values[t]
        gae   = delta + γ * λ * next_nonterminal * gae
        advantages[t] = gae
    end
    return advantages, advantages .+ values
end

# ── PPO update ────────────────────────────────────────────────────────────────────

"""
Run `k_epochs` shuffled minibatch passes of the clipped-surrogate PPO update
over `buffer`. Mutates `policy` and `opt_state` (from `Flux.setup`) in place.

Advantages are normalised (zero mean, unit std) once per call, before the
minibatch loop — standard PPO practice for stable gradient scale across
volatility regimes (see the TradingGame module docstring / plan's reward
section; this supersedes normalising the raw reward, which advantage
normalisation already subsumes for training-stability purposes).

`device` moves each minibatch (observations, the action mask, and the
advantage/return/old-logprob slices) to `:gpu` before the forward/backward
pass — `policy`/`opt_state` are expected to already live on `device` (see
`train_policy!`). This is the batched pass GPU support is actually for —
unlike `collect_rollout`'s one-bar-at-a-time calls, minibatches here are as
large as `minibatch_size`, which is where a GPU's throughput advantage shows
up (see `packages/TradingGame/docs/tradinggame_scaling.tex` for measured
memory/throughput at various batch sizes).

This is normally the *silent* half of an iteration from the caller's point of
view — `collect_rollout` steps the env bar by bar and can report per-bar
progress, but here there's nothing to report except the minibatch loop
itself, which for a full-scale run (N candidates × a multi-year window, one
minibatch per ~32-256 decision steps) can run for minutes with zero output
otherwise. `verbose=true` prints a `\r`-updating progress line (same
`\r`-line convention as `StockSwingPredictor.train!`'s batch progress);
`progress_cb`, if given, is called as `progress_cb(epoch, minibatch,
total_minibatches, loss)` after every minibatch update — used by
`train_policy!` to stream progress to `live_status.json` (see `live.jl`).
Both default to off/nothing so direct/test callers (small buffers, no
reason to want either) see no behaviour change.

# Returns
`Dict{String,Float32}` with `"loss"` (mean combined loss across all
minibatch updates this call) for episode-log diagnostics.
"""
function ppo_update!(policy::ActorCriticPolicy, opt_state, buffer::Vector{<:RolloutStep};
                      k_epochs::Int=4, minibatch_size::Int=32,
                      clip_eps::Float64=CLIP_EPS, value_loss_coef::Float64=VALUE_LOSS_COEF,
                      entropy_coef::Float64=ENTROPY_COEF,
                      gamma::Float64=GAMMA, gae_lambda::Float64=GAE_LAMBDA,
                      device::Symbol=:cpu,
                      verbose::Bool=false,
                      progress_cb::Union{Nothing, Function}=nothing,
                      rng::AbstractRNG=Random.default_rng())::Dict{String, Float32}
    to_dev = _to_device(device)
    T = length(buffer)
    N = length(buffer[1].obs.candidates)
    n_batches_per_epoch = cld(T, minibatch_size)
    total_minibatches    = k_epochs * n_batches_per_epoch
    progress_every        = max(1, total_minibatches ÷ 50)   # ~50 prints/writes over the whole call, any size

    rewards = Float32[s.reward  for s in buffer]
    values  = Float32[s.value   for s in buffer]
    dones   = Bool[s.done       for s in buffer]
    old_logprob = Float32[s.logprob for s in buffer]
    action_idx  = [s.action_idx for s in buffer]
    obs_all     = [s.obs        for s in buffer]

    advantages, returns = compute_gae(rewards, values, dones; gamma=gamma, gae_lambda=gae_lambda)
    advantages = (advantages .- mean(advantages)) ./ (std(advantages) + 1f-8)

    total_loss = 0f0
    n_updates  = 0

    for epoch in 1:k_epochs
        order = shuffle(rng, 1:T)
        for start in 1:minibatch_size:T
            idxs = order[start:min(start + minibatch_size - 1, T)]
            B = length(idxs)
            batch = stack_observations(obs_all[idxs])
            hourly, news, holding, macro_ctx, portfolio =
                to_dev.((batch.hourly, batch.news, batch.holding, batch.macro_ctx, batch.portfolio))

            action_mask = zeros(Float32, 3, N, B)
            for (b, i) in enumerate(idxs), c in 1:N
                action_mask[action_idx[i][c], c, b] = 1f0
            end
            action_mask = to_dev(action_mask)
            adv_b   = to_dev(advantages[idxs])
            ret_b   = to_dev(returns[idxs])
            oldlp_b = to_dev(old_logprob[idxs])

            loss, grads = Flux.withgradient(policy) do m
                action_logits, _, value = m(hourly, news, holding, macro_ctx, portfolio)

                logp_all = Flux.logsoftmax(action_logits; dims=1)
                probs    = Flux.softmax(action_logits; dims=1)
                new_logprob = vec(sum(logp_all .* action_mask; dims=(1, 2)))
                entropy     = vec(sum(-probs .* logp_all; dims=(1, 2)))

                ratio = exp.(new_logprob .- oldlp_b)
                surr1 = ratio .* adv_b
                surr2 = clamp.(ratio, Float32(1 - clip_eps), Float32(1 + clip_eps)) .* adv_b
                policy_loss = -mean(min.(surr1, surr2))
                value_loss  = mean(abs2.(value .- ret_b))

                policy_loss + Float32(value_loss_coef) * value_loss - Float32(entropy_coef) * mean(entropy)
            end

            Flux.update!(opt_state, policy, grads[1])
            total_loss += loss
            n_updates  += 1

            due = n_updates % progress_every == 0 || n_updates == total_minibatches
            if due && verbose
                print("\r  PPO update | epoch $epoch/$k_epochs | minibatch $n_updates/$total_minibatches | " *
                      "loss $(round(loss, sigdigits=4))    ")
                flush(stdout)
            end
            due && progress_cb !== nothing && progress_cb(epoch, n_updates, total_minibatches, Float64(loss))
        end
    end
    verbose && print("\r" * " "^88 * "\r")   # clear the progress line

    return Dict("loss" => total_loss / n_updates)
end
