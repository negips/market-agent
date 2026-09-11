"""
Neural network architecture registry for StockSwingPredictor.

## Adding a new architecture

1. Define a struct subtyping `SwingArchitecture`:
       Base.@kwdef struct MyCNN <: SwingArchitecture ... end

2. Implement `build_model(arch::MyCNN)::SwingPredictor{MyCNN}`.

3. Implement the forward pass:
       function (m::SwingPredictor{MyCNN})(market, hourly, llm) ... end

4. Add the arch to `_arch_to_dict` / `_arch_from_dict` for save/load support.

5. Export the new type from StockSwingPredictor.jl.

## Current architectures

  DUAL_CNN_V1 — two CNN branches (market context + individual stock hourly).
                Market context is mean-pooled across all context companies.

  DUAL_CNN_V2 — same two branches, but replaces mean-pool with cross-attention:
                the target stock's embedding queries the context company embeddings,
                so the model learns which peers matter for each prediction.
"""

using Flux, BSON

# ── Architecture interface ────────────────────────────────────────────────────

"""
Abstract base for all SwingPredictor architectures.
Every concrete subtype must implement `build_model(arch)`.
"""
abstract type SwingArchitecture end

# ── DualCNN v1 ────────────────────────────────────────────────────────────────

"""
Two-branch CNN architecture with mean-pooled market context.

Branch 1 (shared market CNN): processes all N companies' 28-day close+vol series
simultaneously, producing a target embedding and a mean-pooled market context.

Branch 2 (hourly CNN): processes the target stock's 280-bar 60-minute series.

MLP head combines both embeddings with the LLM scalars → N_PRED_HOURS outputs.
"""
Base.@kwdef struct DualCNN <: SwingArchitecture
    name                :: String      = "DualCNN_v1"

    # Market CNN: channel progression from input to embedding.
    # First element must equal N_MARKET_CHANNELS (3).
    market_channels     :: Vector{Int} = [3, 32, 64, 128]
    market_kernel       :: Int         = 3

    # Hourly CNN: channel progression from input to embedding.
    # First element must be 1 (single close channel).
    hourly_channels     :: Vector{Int} = [1, 32, 64, 128, 256]
    hourly_kernel_large :: Int         = 5
    hourly_kernel_small :: Int         = 3

    mlp_hidden          :: Vector{Int} = [512, 256, 128, 64]
    dropout_rate        :: Float64     = 0.3
end

"""The current v1 architecture — mean-pooled market context."""
const DUAL_CNN_V1 = DualCNN()

# ── DualCNN v2 — cross-attention market context ───────────────────────────────

"""
Two-branch CNN architecture with cross-attention market context.

Identical to `DualCNN` except the mean-pool over context companies is replaced
by multi-head cross-attention: the target stock's embedding is the query; the
149 context company embeddings are keys and values. The model learns which peers
are relevant for predicting each target stock on each date.

The `attn_heads` parameter controls the number of attention heads. `embed_dim`
(= `market_channels[end]`) must be divisible by `attn_heads`.

Parameter count vs DualCNN_v1: +65 K (four 128×128 projection matrices).
"""
Base.@kwdef struct DualCNNv2 <: SwingArchitecture
    name                :: String      = "DualCNN_v2"

    market_channels     :: Vector{Int} = [3, 32, 64, 128]
    market_kernel       :: Int         = 3
    attn_heads          :: Int         = 2       # must divide market_channels[end]

    hourly_channels     :: Vector{Int} = [1, 32, 64, 128, 256]
    hourly_kernel_large :: Int         = 5
    hourly_kernel_small :: Int         = 3

    mlp_hidden          :: Vector{Int} = [512, 256, 128, 64]
    dropout_rate        :: Float64     = 0.3
end

"""The current v2 architecture — cross-attention market context."""
const DUAL_CNN_V2 = DualCNNv2()

# ── SwingPredictor container ──────────────────────────────────────────────────

"""
Compiled model container. Parametric on the architecture type so different
forward-pass implementations can dispatch without runtime branching.

`market_attn` is `nothing` for v1 (mean-pool) and a `MultiHeadAttention` layer
for v2 (cross-attention). Flux traverses `nothing` fields safely.
"""
struct SwingPredictor{A <: SwingArchitecture}
    arch        :: A
    market_cnn  :: Chain
    hourly_cnn  :: Chain
    market_attn :: Union{Nothing, MultiHeadAttention}
    mlp_head    :: Chain
end

# ── DualCNN v1 builder ────────────────────────────────────────────────────────

function build_model(arch::DualCNN)::SwingPredictor{DualCNN}
    SwingPredictor(arch,
                   _build_market_cnn(arch),
                   _build_hourly_cnn(arch),
                   nothing,
                   _build_mlp_head(arch))
end

# ── DualCNN v2 builder ────────────────────────────────────────────────────────

"""Build the default architecture (DualCNN_v1)."""
build_model() = build_model(DUAL_CNN_V1)

function build_model(arch::DualCNNv2)::SwingPredictor{DualCNNv2}
    embed_dim = arch.market_channels[end]
    @assert embed_dim % arch.attn_heads == 0 "market_channels[end] ($embed_dim) must be divisible by attn_heads ($(arch.attn_heads))"
    attn = MultiHeadAttention(embed_dim; nheads=arch.attn_heads, bias=false)
    SwingPredictor(arch,
                   _build_market_cnn(arch),
                   _build_hourly_cnn(arch),
                   attn,
                   _build_mlp_head(arch))
end

# ── Shared sub-network builders (duck-typed — work for any arch with same fields) ──

function _build_market_cnn(arch)::Chain
    ch = arch.market_channels
    layers = []
    for i in 1:length(ch)-1
        push!(layers, Conv((arch.market_kernel,), ch[i] => ch[i+1], relu))
    end
    push!(layers, GlobalMeanPool(), Flux.flatten)
    Chain(layers...)
end

function _build_hourly_cnn(arch)::Chain
    ch = arch.hourly_channels
    layers = []
    for i in 1:length(ch)-1
        k = i <= 2 ? arch.hourly_kernel_large : arch.hourly_kernel_small
        push!(layers, Conv((k,), ch[i] => ch[i+1], relu))
    end
    push!(layers, GlobalMeanPool(), Flux.flatten)
    Chain(layers...)
end

function _build_mlp_head(arch)::Chain
    market_embed = arch.market_channels[end]
    hourly_embed = arch.hourly_channels[end]
    mlp_in       = 2 * market_embed + hourly_embed + N_LLM_FEATURES

    sizes  = [mlp_in; arch.mlp_hidden; N_PRED_HOURS]
    n_drop = length(arch.mlp_hidden)

    layers = []
    for i in 1:length(sizes)-1
        push!(layers, Dense(sizes[i] => sizes[i+1], i < length(sizes)-1 ? relu : identity))
        if i < length(sizes)-1 && i <= n_drop
            frac = 1.0 - (i - 1) / max(n_drop - 1, 1)
            rate = arch.dropout_rate * frac
            rate > 0.01 && push!(layers, Dropout(rate))
        end
    end
    Chain(layers...)
end

# ── DualCNN v1 forward pass ───────────────────────────────────────────────────

"""
Forward pass for DualCNN_v1.

`market` — `(N_MARKET_DAYS, N_MARKET_CHANNELS, N_companies, batch)`
`hourly` — `(N_HOURLY_BARS, batch)` normalised 60-min closes.
`llm`    — `(N_LLM_FEATURES, batch)`.

Returns `(N_PRED_HOURS, batch)` predicted log-return trajectories.
"""
function (m::SwingPredictor{DualCNN})(market::Array{Float32,4},
                                      hourly::Matrix{Float32},
                                      llm::Matrix{Float32})
    _, _, N, B = size(market)

    x    = reshape(market, N_MARKET_DAYS, N_MARKET_CHANNELS, N * B)
    embs = m.market_cnn(x)                                     # (embed, N*B)
    embs = reshape(embs, size(embs, 1), N, B)                  # (embed, N, B)

    target_emb = embs[:, 1, :]                                 # (embed, B)
    market_ctx = dropdims(mean(embs[:, 2:end, :], dims=2), dims=2)  # (embed, B)

    h          = reshape(hourly, N_HOURLY_BARS, 1, B)
    hourly_emb = m.hourly_cnn(h)                               # (embed, B)

    combined = vcat(target_emb, market_ctx, hourly_emb, llm)
    return m.mlp_head(combined)
end

# ── DualCNN v2 forward pass ───────────────────────────────────────────────────
# Same contract as v1; mean-pool replaced by multi-head cross-attention.

function (m::SwingPredictor{DualCNNv2})(market::Array{Float32,4},
                                         hourly::Matrix{Float32},
                                         llm::Matrix{Float32})
    _, _, N, B = size(market)

    x    = reshape(market, N_MARKET_DAYS, N_MARKET_CHANNELS, N * B)
    embs = m.market_cnn(x)                             # (embed, N*B)
    embs = reshape(embs, size(embs, 1), N, B)          # (embed, N, B)

    # target_q: (embed, 1, B) — query for cross-attention
    # context:  (embed, N-1, B) — keys and values
    target_q  = embs[:, 1:1, :]
    context   = embs[:, 2:end, :]

    # cross-attention: target queries context peers
    attended, _ = m.market_attn(target_q, context, context)   # (embed, 1, B)
    market_ctx  = dropdims(attended, dims=2)                   # (embed, B)
    target_emb  = dropdims(target_q, dims=2)                   # (embed, B)

    h          = reshape(hourly, N_HOURLY_BARS, 1, B)
    hourly_emb = m.hourly_cnn(h)

    combined = vcat(target_emb, market_ctx, hourly_emb, llm)
    return m.mlp_head(combined)
end

# ── Inference helpers ─────────────────────────────────────────────────────────

"""Run inference on a pre-assembled batch. Returns `(N_PRED_HOURS × batch)` matrix."""
function predict(model::SwingPredictor,
                 market::Array{Float32,4},
                 hourly::Matrix{Float32},
                 llm::Matrix{Float32})::Matrix{Float32}
    Flux.testmode!(model)
    return model(market, hourly, llm)
end

# ── Persistence ───────────────────────────────────────────────────────────────

"""Save model weights, architecture config, universe, and meta to BSON."""
function save_model(model::SwingPredictor, companies::Vector{String},
                    path::String; meta::Dict=Dict())
    state     = Flux.state(model)
    arch_dict = _arch_to_dict(model.arch)
    BSON.@save path state arch_dict companies meta
    @info "Model saved → $path  [arch: $(model.arch.name)]"
end

"""
Load a `SwingPredictor` from BSON.
Returns `(model, companies, meta)`.

Handles checkpoints saved before the `market_attn` field was added to
`SwingPredictor` (DualCNN_v1 runs before DualCNN_v2 was introduced).
"""
function load_model(path::String)
    BSON.@load path state arch_dict companies meta
    arch  = _arch_from_dict(arch_dict)
    model = build_model(arch)
    state = _migrate_state(state, model)
    Flux.loadmodel!(model, state)
    return model, companies, meta
end

# Inserts market_attn=nothing into state NamedTuples that predate the field.
function _migrate_state(state::NamedTuple, model::SwingPredictor)
    hasproperty(state, :market_attn) && return state
    (arch        = state.arch,
     market_cnn  = state.market_cnn,
     hourly_cnn  = state.hourly_cnn,
     market_attn = Flux.state(model.market_attn),  # nothing for v1
     mlp_head    = state.mlp_head)
end

# ── Arch serialisation ────────────────────────────────────────────────────────

function _arch_to_dict(arch::DualCNN)::Dict{String,Any}
    Dict{String,Any}(
        "type"                => "DualCNN",
        "name"                => arch.name,
        "market_channels"     => arch.market_channels,
        "market_kernel"       => arch.market_kernel,
        "hourly_channels"     => arch.hourly_channels,
        "hourly_kernel_large" => arch.hourly_kernel_large,
        "hourly_kernel_small" => arch.hourly_kernel_small,
        "mlp_hidden"          => arch.mlp_hidden,
        "dropout_rate"        => arch.dropout_rate,
    )
end

function _arch_to_dict(arch::DualCNNv2)::Dict{String,Any}
    Dict{String,Any}(
        "type"                => "DualCNNv2",
        "name"                => arch.name,
        "market_channels"     => arch.market_channels,
        "market_kernel"       => arch.market_kernel,
        "attn_heads"          => arch.attn_heads,
        "hourly_channels"     => arch.hourly_channels,
        "hourly_kernel_large" => arch.hourly_kernel_large,
        "hourly_kernel_small" => arch.hourly_kernel_small,
        "mlp_hidden"          => arch.mlp_hidden,
        "dropout_rate"        => arch.dropout_rate,
    )
end

function _arch_from_dict(d::Dict)::SwingArchitecture
    t = d["type"]
    if t == "DualCNN"
        return DualCNN(
            name                = d["name"],
            market_channels     = Vector{Int}(d["market_channels"]),
            market_kernel       = Int(d["market_kernel"]),
            hourly_channels     = Vector{Int}(d["hourly_channels"]),
            hourly_kernel_large = Int(d["hourly_kernel_large"]),
            hourly_kernel_small = Int(d["hourly_kernel_small"]),
            mlp_hidden          = Vector{Int}(d["mlp_hidden"]),
            dropout_rate        = Float64(d["dropout_rate"]),
        )
    elseif t == "DualCNNv2"
        return DualCNNv2(
            name                = d["name"],
            market_channels     = Vector{Int}(d["market_channels"]),
            market_kernel       = Int(d["market_kernel"]),
            attn_heads          = Int(d["attn_heads"]),
            hourly_channels     = Vector{Int}(d["hourly_channels"]),
            hourly_kernel_large = Int(d["hourly_kernel_large"]),
            hourly_kernel_small = Int(d["hourly_kernel_small"]),
            mlp_hidden          = Vector{Int}(d["mlp_hidden"]),
            dropout_rate        = Float64(d["dropout_rate"]),
        )
    end
    error("Unknown architecture type: $t")
end
