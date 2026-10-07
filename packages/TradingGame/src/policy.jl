"""
Recurrent/transformer actor-critic policy — no CNN reuse from
`StockSwingPredictor` (per the confirmed architecture decision; see the
`TradingGame` module docstring).

Pipeline: per-stock GRU temporal encoder (weight-shared across candidates) →
concat news + holding-state features → fuse back to `embed_dim` → macro GRU +
portfolio `Dense` form a "portfolio token" → `MultiHeadAttention` over all `N`
stock embeddings plus the portfolio token (the "joint" part of the joint
policy — each stock's representation can attend to every other candidate and
to the shared cash state) → per-stock actor head + pooled critic head.

`policy.jl` itself only calls `Flux.gpu`/`Flux.cpu` — GPU dispatch activates
because `StockSwingPredictor` (a dependency) eagerly does `using CUDA, cuDNN`.
`TradingGame`'s own `CUDA` dependency (see `Project.toml`) exists only so
`test/test_policy.jl` can call `CUDA.functional()` directly.
"""

using Flux, BSON, Dates, Random

struct ActorCriticPolicy
    hourly_encoder    :: Flux.GRU
    fusion            :: Dense
    macro_encoder     :: Flux.GRU
    portfolio_encoder :: Dense
    attn              :: Flux.MultiHeadAttention
    actor_head        :: Dense
    critic_head       :: Chain
    cash_encoder      :: Union{Nothing, Dense}   # game v2 only — embeds the cash token (see `GameRules.cash_token`)
end

Flux.@layer ActorCriticPolicy

"""
Build a fresh `ActorCriticPolicy`. `embed_dim` is the per-stock embedding
width used throughout (hourly encoder output, fusion output, attention
embedding); `macro_embed_dim` is the macro-GRU's output width before it's
concatenated with the portfolio scalars.

`cash_token=true` (game v2) adds `cash_encoder` and widens the portfolio input
to `N_PORTFOLIO_SCALARS_V2`: cash becomes one more token in the attention set,
built from `[cash/value, reserved/value, cap utilisation, days-over-cap]`
(portfolio vector entries 1, 2, 5, 6), so each stock can attend to how much
cash there is and how long it has sat idle exactly as it attends to other
stocks. Its own output is not used — it only conditions the stocks and the
critic. Checkpoints record this (`save_policy`), so `load_policy` rebuilds
the matching architecture.

Every weight is Glorot-uniform, every bias zero — Flux's defaults for
`Dense`/`GRU`/`MultiHeadAttention`, since no `init=` is passed anywhere here.
Those defaults draw from Julia's *global* RNG, so two calls give different
initial weights unless you pass `seed`: when given, this calls
`Random.seed!(seed)` immediately before constructing the layers, making the
initial weights reproducible. This is a real (if standard) side effect on
global RNG state, same as calling `Random.seed!` anywhere else — it also
resets whatever random stream the rest of the program was drawing from.
Omit `seed` for the previous non-deterministic behaviour.
"""
function ActorCriticPolicy(; embed_dim::Int=64, macro_embed_dim::Int=16,
                            attn_heads::Int=4, critic_hidden::Vector{Int}=[64, 32],
                            seed::Union{Nothing, Int}=nothing,
                            cash_token::Bool=false)
    seed !== nothing && Random.seed!(seed)
    n_portfolio = cash_token ? N_PORTFOLIO_SCALARS_V2 : N_PORTFOLIO_SCALARS

    hourly_encoder    = GRU(N_PRICE_CHANNELS => embed_dim)
    fusion            = Dense(embed_dim + N_NEWS_FEATURES + N_HOLDING_FEATURES => embed_dim, relu)
    macro_encoder     = GRU(N_MACRO_SERIES => macro_embed_dim)
    portfolio_encoder = Dense(macro_embed_dim + n_portfolio => embed_dim, relu)
    attn              = MultiHeadAttention(embed_dim; nheads=attn_heads)
    actor_head        = Dense(embed_dim => 4)   # 3 action-type logits + 1 buy-weight logit

    critic_layers = Any[]
    in_dim = embed_dim
    for h in critic_hidden
        push!(critic_layers, Dense(in_dim => h, relu))
        in_dim = h
    end
    push!(critic_layers, Dense(in_dim => 1))
    critic_head = Chain(critic_layers...)

    cash_encoder = cash_token ? Dense(N_CASH_TOKEN_FEATURES => embed_dim, relu) : nothing

    return ActorCriticPolicy(hourly_encoder, fusion, macro_encoder, portfolio_encoder,
                              attn, actor_head, critic_head, cash_encoder)
end

"""
Forward pass over a batch produced by `stack_observations`.

- `hourly`    :: `(N_HOURLY_BARS_SHORT, N_PRICE_CHANNELS, N, B)`
- `news`      :: `(N_NEWS_FEATURES, N, B)`
- `holding`   :: `(N_HOLDING_FEATURES, N, B)`
- `macro_ctx` :: `(N_MACRO_DAYS, N_MACRO_SERIES, B)`
- `portfolio` :: `(N_PORTFOLIO_SCALARS, B)`

# Returns
`(action_logits, buy_weight_logit, value)`:
`action_logits :: (3, N, B)` (HOLD/SELL/BUY logits per candidate),
`buy_weight_logit :: (N, B)`, `value :: (B,)`.
"""
function (m::ActorCriticPolicy)(hourly::AbstractArray{<:Real, 4}, news::AbstractArray{<:Real, 3},
                                 holding::AbstractArray{<:Real, 3}, macro_ctx::AbstractArray{<:Real, 3},
                                 portfolio::AbstractMatrix{<:Real})
    bars, ch, N, B = size(hourly)
    embed_dim = size(m.actor_head.weight, 2)

    # ── Per-stock temporal encoder (weight-shared via the N*B batch fold) ────
    x = permutedims(hourly, (2, 1, 3, 4))                 # (ch, bars, N, B)
    x = reshape(x, ch, bars, N * B)
    h = m.hourly_encoder(x)                                # (embed, bars, N*B)
    stock_emb = reshape(h[:, end, :], embed_dim, N, B)      # last timestep only

    # ── News + holding-state fusion ──────────────────────────────────────────
    fused = vcat(stock_emb, reshape(news, :, N, B), reshape(holding, :, N, B))
    fused = m.fusion(reshape(fused, size(fused, 1), N * B))
    stock_emb = reshape(fused, embed_dim, N, B)

    # ── Macro + portfolio conditioning → one extra "portfolio token" ────────
    macro_x   = permutedims(macro_ctx, (2, 1, 3))           # (series, days, B)
    macro_h   = m.macro_encoder(macro_x)                    # (macro_embed, days, B)
    macro_emb = macro_h[:, end, :]                           # (macro_embed, B)
    port_tok  = m.portfolio_encoder(vcat(macro_emb, portfolio))   # (embed, B)
    port_tok  = reshape(port_tok, embed_dim, 1, B)

    # ── Cross-candidate attention (the "joint" decision) ─────────────────────
    if m.cash_encoder === nothing
        seq = cat(stock_emb, port_tok; dims=2)                  # (embed, N+1, B)
    else
        cash_in  = portfolio[[1, 2, 5, 6], :]                    # (N_CASH_TOKEN_FEATURES, B)
        cash_tok = reshape(m.cash_encoder(cash_in), embed_dim, 1, B)
        seq = cat(stock_emb, cash_tok, port_tok; dims=2)         # (embed, N+2, B)
    end
    attended, _ = m.attn(seq)
    stock_out = attended[:, 1:N, :]
    port_out  = attended[:, end, :]

    # ── Heads ─────────────────────────────────────────────────────────────────
    actor_out = m.actor_head(reshape(stock_out, embed_dim, N * B))   # (4, N*B)
    actor_out = reshape(actor_out, 4, N, B)
    action_logits    = actor_out[1:3, :, :]
    buy_weight_logit = actor_out[4, :, :]

    value = vec(m.critic_head(port_out))

    return action_logits, buy_weight_logit, value
end

# ── Persistence ────────────────────────────────────────────────────────────────────
#
# Mirrors StockSwingPredictor's save_model/load_model: CPU-resident weights so
# a GPU-trained checkpoint stays portable, plus the constructor hyperparameters
# needed to rebuild an identically-shaped policy before loading state into it.

"""Save `policy`'s weights (always CPU-resident, for portability) and the
hyperparameters needed to reconstruct it, to BSON at `path`."""
function save_policy(policy::ActorCriticPolicy, path::String;
                      embed_dim::Int, macro_embed_dim::Int, attn_heads::Int,
                      critic_hidden::Vector{Int}, meta::Dict=Dict())
    state = Flux.state(cpu(policy))
    cash_token = policy.cash_encoder !== nothing
    BSON.@save path state embed_dim macro_embed_dim attn_heads critic_hidden meta cash_token
    @info "Policy saved → $path"
end

"""Load an `ActorCriticPolicy` from BSON. Returns `(policy, hyperparams, meta)`
— `hyperparams` is the exact `(embed_dim, macro_embed_dim, attn_heads,
critic_hidden)` NamedTuple the policy was built with, so a resumed
`train_policy!` run can pass it straight back in and keep re-saving a
checkpoint with the same architecture (see that function's docstring for why
it needs these at all — `ActorCriticPolicy` doesn't carry them as a field)."""
function load_policy(path::String)
    BSON.@load path state embed_dim macro_embed_dim attn_heads critic_hidden meta
    d = BSON.load(path)
    cash_token = get(d, :cash_token, false)   # absent in checkpoints written before game v2
    policy = ActorCriticPolicy(embed_dim=embed_dim, macro_embed_dim=macro_embed_dim,
                                attn_heads=attn_heads, critic_hidden=critic_hidden,
                                cash_token=cash_token)
    Flux.loadmodel!(policy, state)
    hyperparams = (embed_dim=embed_dim, macro_embed_dim=macro_embed_dim,
                   attn_heads=attn_heads, critic_hidden=critic_hidden, cash_token=cash_token)
    return policy, hyperparams, meta
end
