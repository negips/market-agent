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
                Named default; all pipeline scripts use this.
"""

using Flux, BSON

# ── Architecture interface ────────────────────────────────────────────────────

"""
Abstract base for all SwingPredictor architectures.
Every concrete subtype must implement `build_model(arch)`.
"""
abstract type SwingArchitecture end

# ── DualCNN — first named architecture ───────────────────────────────────────

"""
Two-branch CNN architecture.

Branch 1 (shared market CNN): processes all N companies' 28-day close+vol series
simultaneously, producing a target embedding and a mean-pooled market context.

Branch 2 (hourly CNN): processes the target stock's 280-bar 60-minute series.

MLP head combines both embeddings with the LLM scalars → N_PRED_HOURS outputs.

All sizes are configurable so width/depth experiments don't require new types —
increment the name field (e.g. "DualCNN_v2") to track variants in saved models.
"""
Base.@kwdef struct DualCNN <: SwingArchitecture
    name                :: String      = "DualCNN_v1"

    # Market CNN: channel progression from input to embedding.
    # First element must equal N_MARKET_CHANNELS (2).
    market_channels     :: Vector{Int} = [2, 32, 64, 128]
    market_kernel       :: Int         = 3

    # Hourly CNN: channel progression from input to embedding.
    # First element must be 1 (single close channel).
    # Larger kernel for the first two layers captures broader intraday patterns.
    hourly_channels     :: Vector{Int} = [1, 32, 64, 128, 256]
    hourly_kernel_large :: Int         = 5   # used for layers 1…2
    hourly_kernel_small :: Int         = 3   # used for layers 3…end

    # MLP head hidden layer sizes (input and output are derived automatically).
    mlp_hidden          :: Vector{Int} = [512, 256, 128, 64]

    # Dropout rate at the first MLP layer; tapers linearly to 0 at the last.
    dropout_rate        :: Float64     = 0.3
end

"""The current default architecture — reference this throughout the codebase."""
const DUAL_CNN_V1 = DualCNN()

# ── SwingPredictor container ──────────────────────────────────────────────────

"""
Compiled model container. Parametric on the architecture type so different
forward-pass implementations can dispatch without runtime branching.
"""
struct SwingPredictor{A <: SwingArchitecture}
    arch       :: A
    market_cnn :: Chain
    hourly_cnn :: Chain
    mlp_head   :: Chain
end

# Only expose the Flux layers to the optimiser — arch is metadata, not parameters.
Flux.@functor SwingPredictor (market_cnn, hourly_cnn, mlp_head)

# ── DualCNN builder ───────────────────────────────────────────────────────────

"""
Build a `SwingPredictor{DualCNN}` from an architecture config.
Defaults to `DUAL_CNN_V1` when called with no arguments.
"""
function build_model(arch::DualCNN = DUAL_CNN_V1)::SwingPredictor{DualCNN}
    SwingPredictor(arch,
                   _build_market_cnn(arch),
                   _build_hourly_cnn(arch),
                   _build_mlp_head(arch))
end

function _build_market_cnn(arch::DualCNN)::Chain
    ch = arch.market_channels
    layers = []
    for i in 1:length(ch)-1
        push!(layers, Conv((arch.market_kernel,), ch[i] => ch[i+1], relu))
    end
    push!(layers, GlobalMeanPool(), Flux.flatten)
    Chain(layers...)
end

function _build_hourly_cnn(arch::DualCNN)::Chain
    ch = arch.hourly_channels
    layers = []
    for i in 1:length(ch)-1
        k = i <= 2 ? arch.hourly_kernel_large : arch.hourly_kernel_small
        push!(layers, Conv((k,), ch[i] => ch[i+1], relu))
    end
    push!(layers, GlobalMeanPool(), Flux.flatten)
    Chain(layers...)
end

function _build_mlp_head(arch::DualCNN)::Chain
    market_embed = arch.market_channels[end]
    hourly_embed = arch.hourly_channels[end]
    mlp_in       = 2 * market_embed + hourly_embed + N_LLM_FEATURES

    sizes  = [mlp_in; arch.mlp_hidden; N_PRED_HOURS]
    n_drop = length(arch.mlp_hidden)   # one dropout after each hidden layer except last

    layers = []
    for i in 1:length(sizes)-1
        push!(layers, Dense(sizes[i] => sizes[i+1], i < length(sizes)-1 ? relu : identity))
        # Dropout tapers from dropout_rate → 0 across the hidden layers.
        if i < length(sizes)-1 && i <= n_drop
            frac = 1.0 - (i - 1) / max(n_drop - 1, 1)
            rate = arch.dropout_rate * frac
            rate > 0.01 && push!(layers, Dropout(rate))
        end
    end
    Chain(layers...)
end

# ── DualCNN forward pass ──────────────────────────────────────────────────────

"""
Forward pass for any DualCNN variant.

`market` — `(N_MARKET_DAYS, N_MARKET_CHANNELS, N_companies, batch)`
            Column 1 along dim 3 is the target stock (set by `assemble_batch`).
`hourly` — `(N_HOURLY_BARS, batch)` normalised 60-min closes.
`llm`    — `(N_LLM_FEATURES, batch)`.

Returns `(N_PRED_HOURS, batch)` predicted log-return trajectories.
"""
function (m::SwingPredictor{DualCNN})(market::Array{Float32,4},
                                       hourly::Matrix{Float32},
                                       llm::Matrix{Float32})
    _, _, N, B = size(market)

    # Shared market CNN — all N companies processed as one large batch
    x    = reshape(market, N_MARKET_DAYS, N_MARKET_CHANNELS, N * B)
    embs = m.market_cnn(x)                                     # (embed, N*B)
    embs = reshape(embs, size(embs, 1), N, B)                  # (embed, N, B)

    target_emb = embs[:, 1, :]                                 # (embed, B)
    market_ctx = dropdims(mean(embs[:, 2:end, :], dims=2), dims=2)  # (embed, B)

    # Hourly CNN — target stock only
    h          = reshape(hourly, N_HOURLY_BARS, 1, B)          # (bars, 1, B)
    hourly_emb = m.hourly_cnn(h)                               # (embed, B)

    combined = vcat(target_emb, market_ctx, hourly_emb, llm)   # (mlp_in, B)
    return m.mlp_head(combined)                                 # (N_PRED_HOURS, B)
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
    state    = Flux.state(model)
    arch_dict = _arch_to_dict(model.arch)
    BSON.@save path state arch_dict companies meta
    @info "Model saved → $path  [arch: $(model.arch.name)]"
end

"""
Load a `SwingPredictor` from BSON.
Returns `(model, companies, meta)`.
"""
function load_model(path::String)
    BSON.@load path state arch_dict companies meta
    arch  = _arch_from_dict(arch_dict)
    model = build_model(arch)
    Flux.loadmodel!(model, state)
    return model, companies, meta
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
    end
    error("Unknown architecture type: $t")
end
