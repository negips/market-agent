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
    hourly_encoder    :: Union{Nothing, Flux.GRU}   # nothing when `history_bars > 0` (game v3 direct): the window goes straight into `fusion`
    fusion            :: Dense
    macro_encoder     :: Union{Nothing, Flux.GRU}   # nothing when `use_macro=false` (game v3)
    portfolio_encoder :: Union{Nothing, Dense}   # nothing when the portfolio scalars go into `fusion` instead (`global_in_fusion > 0`)
    attn              :: Flux.MultiHeadAttention
    actor_head        :: Dense
    critic_head       :: Chain
    cash_encoder      :: Union{Nothing, Dense}   # game v2 only — embeds the cash token (see `GameRules.cash_token`)
    global_in_fusion  :: Int                     # > 0: width of the portfolio scalars concatenated onto every stock's fusion input (game v3); 0: separate portfolio token
    history_bars      :: Int                     # > 0: length of the price window fed directly to `fusion` (no GRU); 0: GRU encoder
end

Flux.@layer ActorCriticPolicy

"""
Build a fresh `ActorCriticPolicy`. `embed_dim` is the per-stock embedding
width used throughout (hourly encoder output, fusion output, attention
embedding); `macro_embed_dim` is the macro-GRU's output width before it's
concatenated with the portfolio scalars.

`portfolio_in_fusion=true` (game v3) removes the separate portfolio token and the
cash token: the `portfolio_scalars` portfolio numbers are repeated for every
stock and concatenated onto its fusion input, so the only tokens attention sees
are the N stocks, and the critic reads the mean of the attended stock tokens.
`portfolio_to_critic=true` (game v3) also appends the portfolio scalars to the
critic's input (`embed_dim + portfolio width` instead of `embed_dim`), on top of
whatever route they take to the actor; the critic then sees the book state
directly rather than only through attention.
`history_bars=70` (game v3, `history_encoder = :direct`) drops the GRU: the 70
returns of the price window are concatenated with the other per-stock features
and go straight into `fusion` (a single `Dense(70 + 4 → 64)`); it needs
`price_channels=1`. `stock_features=4` (game v3) adds the stock's instantaneous 15-minute price to
the three holding features that join each stock's embedding in `fusion`.
`price_channels=1` (game v3) feeds the GRU the normalised close only, with no
previous-day range channel; the window length needs no setting here (the GRU
takes any number of bars). `use_macro=false` / `use_news=false` (game v3) build the policy without the macro
GRU (the portfolio token is then encoded from the portfolio scalars alone) and
without the news inputs to `fusion`; the corresponding observation tensors are
simply ignored. Both are recorded in checkpoints.

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
                            cash_token::Bool=false, use_macro::Bool=true, use_news::Bool=true,
                            price_channels::Int=N_PRICE_CHANNELS,
                            stock_features::Int=N_HOLDING_FEATURES,
                            history_bars::Int=0,
                            portfolio_in_fusion::Bool=false, portfolio_scalars::Int=N_PORTFOLIO_SCALARS_V2,
                            portfolio_to_critic::Bool=false)
    seed !== nothing && Random.seed!(seed)
    n_portfolio = portfolio_in_fusion ? portfolio_scalars : (cash_token ? N_PORTFOLIO_SCALARS_V2 : N_PORTFOLIO_SCALARS)
    portfolio_in_fusion && (cash_token || use_macro) &&
        error("portfolio_in_fusion has no cash token and no macro branch (the portfolio scalars carry the cash state)")

    hourly_encoder    = history_bars > 0 ? nothing : GRU(price_channels => embed_dim)
    history_width     = history_bars > 0 ? history_bars : embed_dim
    global_width      = portfolio_in_fusion ? n_portfolio : 0
    fusion            = Dense(history_width + (use_news ? N_NEWS_FEATURES : 0) + stock_features + global_width => embed_dim, relu)
    macro_encoder     = use_macro ? GRU(N_MACRO_SERIES => macro_embed_dim) : nothing
    portfolio_encoder = portfolio_in_fusion ? nothing :
                        Dense((use_macro ? macro_embed_dim : 0) + n_portfolio => embed_dim, relu)
    attn              = MultiHeadAttention(embed_dim; nheads=attn_heads)
    actor_head        = Dense(embed_dim => 4)   # 3 action-type logits + 1 buy-weight logit

    critic_layers = Any[]
    in_dim = embed_dim + (portfolio_to_critic ? n_portfolio : 0)
    for h in critic_hidden
        push!(critic_layers, Dense(in_dim => h, relu))
        in_dim = h
    end
    push!(critic_layers, Dense(in_dim => 1))
    critic_head = Chain(critic_layers...)

    cash_encoder = cash_token ? Dense(N_CASH_TOKEN_FEATURES => embed_dim, relu) : nothing

    return ActorCriticPolicy(hourly_encoder, fusion, macro_encoder, portfolio_encoder,
                              attn, actor_head, critic_head, cash_encoder, global_width, history_bars)
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
    if m.hourly_encoder === nothing
        ch == 1 || error("direct history encoder needs a single price channel, got $ch")
        bars == m.history_bars || error("direct history encoder expects $(m.history_bars) bars, got $bars")
        stock_emb = reshape(hourly, bars, N, B)             # the window itself, no recurrence
    else
        x = permutedims(hourly, (2, 1, 3, 4))                 # (ch, bars, N, B)
        x = reshape(x, ch, bars, N * B)
        h = m.hourly_encoder(x)                                # (embed, bars, N*B)
        stock_emb = reshape(h[:, end, :], embed_dim, N, B)      # last timestep only
    end

    # ── News + holding-state fusion ──────────────────────────────────────────
    merged = m.global_in_fusion > 0
    use_news = size(m.fusion.weight, 2) > size(stock_emb, 1) + size(holding, 1) + m.global_in_fusion
    news_part   = use_news ? reshape(news, :, N, B) : similar(stock_emb, 0, N, B)
    if merged
        size(portfolio, 1) == m.global_in_fusion ||
            error("policy expects $(m.global_in_fusion) portfolio scalars, got $(size(portfolio, 1))")
    end
    book_part   = merged ? repeat(reshape(portfolio, :, 1, B), 1, N, 1) :   # the same book state beside every stock
                           similar(stock_emb, 0, N, B)
    fused = vcat(stock_emb, news_part, reshape(holding, :, N, B), book_part)
    fused = m.fusion(reshape(fused, size(fused, 1), N * B))
    stock_emb = reshape(fused, embed_dim, N, B)

    if merged
        # ── No extra tokens: attention is over the N stocks alone ───────────
        attended, _ = m.attn(stock_emb)                          # (embed, N, B)
        stock_out = attended
        port_out  = dropdims(sum(attended; dims=2); dims=2) ./ N  # critic reads the mean stock token
    else
        # ── Macro + portfolio conditioning → one extra "portfolio token" ────
        if m.macro_encoder === nothing
            port_tok = m.portfolio_encoder(portfolio)                 # (embed, B)
        else
            macro_x   = permutedims(macro_ctx, (2, 1, 3))           # (series, days, B)
            macro_h   = m.macro_encoder(macro_x)                    # (macro_embed, days, B)
            macro_emb = macro_h[:, end, :]                           # (macro_embed, B)
            port_tok  = m.portfolio_encoder(vcat(macro_emb, portfolio))   # (embed, B)
        end
        port_tok  = reshape(port_tok, embed_dim, 1, B)

        # ── Cross-candidate attention (the "joint" decision) ─────────────────
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
    end

    # ── Heads ─────────────────────────────────────────────────────────────────
    actor_out = m.actor_head(reshape(stock_out, embed_dim, N * B))   # (4, N*B)
    actor_out = reshape(actor_out, 4, N, B)
    action_logits    = actor_out[1:3, :, :]
    buy_weight_logit = actor_out[4, :, :]

    critic_in = size(m.critic_head.layers[1].weight, 2) > embed_dim ? vcat(port_out, portfolio) : port_out
    value = vec(m.critic_head(critic_in))

    return action_logits, buy_weight_logit, value
end

"""`(use_news, stock_features)` of `policy`, read off `fusion`'s input width
(embedding + optional news + 3 or 4 per-stock features — the four sums are
distinct, so the width identifies the combination)."""
function policy_stock_inputs(policy::ActorCriticPolicy)
    extra = size(policy.fusion.weight, 2) - policy.global_in_fusion -
            (policy.history_bars > 0 ? policy.history_bars : size(policy.actor_head.weight, 2))
    for sf in (N_HOLDING_FEATURES, N_HOLDING_FEATURES + 1)
        extra == sf && return (false, sf)
        extra == sf + N_NEWS_FEATURES && return (true, sf)
    end
    error("policy_stock_inputs: unexpected fusion input width ($extra beyond the embedding)")
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
    use_macro  = policy.macro_encoder !== nothing
    use_news, stock_features = policy_stock_inputs(policy)
    price_channels = policy.hourly_encoder === nothing ? 1 : size(policy.hourly_encoder.cell.Wi, 2)
    history_bars   = policy.history_bars
    global_in_fusion = policy.global_in_fusion
    critic_global    = size(policy.critic_head.layers[1].weight, 2) - embed_dim
    BSON.@save path state embed_dim macro_embed_dim attn_heads critic_hidden meta cash_token use_macro use_news price_channels stock_features history_bars global_in_fusion critic_global
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
    use_macro  = get(d, :use_macro, true)     # absent in checkpoints written before game v3
    use_news   = get(d, :use_news, true)
    critic_global    = get(d, :critic_global, 0)       # absent before the critic could see the portfolio directly
    global_in_fusion = get(d, :global_in_fusion, 0)   # absent before the portfolio scalars could be merged into fusion
    history_bars   = get(d, :history_bars, 0)      # absent in checkpoints written before the direct history encoder
    stock_features = get(d, :stock_features, N_HOLDING_FEATURES)
    price_channels = get(d, :price_channels, N_PRICE_CHANNELS)   # absent in checkpoints written before game v3's single-channel window
    policy = ActorCriticPolicy(embed_dim=embed_dim, macro_embed_dim=macro_embed_dim,
                                attn_heads=attn_heads, critic_hidden=critic_hidden,
                                cash_token=cash_token, use_macro=use_macro, use_news=use_news,
                                price_channels=price_channels, stock_features=stock_features,
                                history_bars=history_bars, portfolio_in_fusion=global_in_fusion > 0,
                                portfolio_scalars=max(global_in_fusion, critic_global, 1),
                                portfolio_to_critic=critic_global > 0)
    Flux.loadmodel!(policy, state)
    hyperparams = (embed_dim=embed_dim, macro_embed_dim=macro_embed_dim,
                   attn_heads=attn_heads, critic_hidden=critic_hidden, cash_token=cash_token,
                   use_macro=use_macro, use_news=use_news, price_channels=price_channels,
                   stock_features=stock_features, history_bars=history_bars,
                   global_in_fusion=global_in_fusion, critic_global=critic_global)
    return policy, hyperparams, meta
end
