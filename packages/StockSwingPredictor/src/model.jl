"""
Neural network architecture: SwingPredictor.

Two CNN branches feed into a shared MLP head:

  Market branch (shared weights across all N companies):
    Input:  (N_MARKET_DAYS, N_MARKET_CHANNELS, N, B)  — reshaped to (28, 2, N×B)
    CNN:    Conv(3, 2→32) → Conv(3, 32→64) → Conv(3, 64→128) → GlobalMeanPool
    Output: (128, N, B)  split into:
              target_emb  = embedding of the stock being predicted        (128, B)
              market_ctx  = mean of the other N-1 company embeddings      (128, B)

  Hourly branch (target stock only):
    Input:  (N_HOURLY_BARS, 1, B)
    CNN:    Conv(5, 1→32) → Conv(5, 32→64) → Conv(3, 64→128) → Conv(3, 128→256)
            → GlobalMeanPool
    Output: (256, B)

  MLP head:
    Input:  concat(target_emb, market_ctx, hourly_emb, llm)  → (527, B)
    Layers: 527 → 512 → 256 → 128 → 64 → N_PRED_HOURS

The target stock is always placed at column index 1 of the market tensor by
`assemble_batch` so the split is a simple slice rather than an index lookup.
"""

using Flux, BSON

struct SwingPredictor
    market_cnn :: Chain   # shared; processes every company's 28-day series
    hourly_cnn :: Chain   # processes the target stock's 280-bar hourly series
    mlp_head   :: Chain
end
Flux.@functor SwingPredictor

"""
Build a `SwingPredictor`. Dropout rate applies to the MLP head only.
CNN branches use no dropout — they are compact enough to regularise via shared weights.
"""
function build_model(; dropout_rate::Float64=0.3)::SwingPredictor
    market_cnn = Chain(
        Conv((3,), N_MARKET_CHANNELS => 32, relu),
        Conv((3,), 32 => 64, relu),
        Conv((3,), 64 => 128, relu),
        GlobalMeanPool(),
        Flux.flatten,          # (1, 128, batch) → (128, batch)
    )
    hourly_cnn = Chain(
        Conv((5,), 1 => 32, relu),
        Conv((5,), 32 => 64, relu),
        Conv((3,), 64 => 128, relu),
        Conv((3,), 128 => 256, relu),
        GlobalMeanPool(),
        Flux.flatten,
    )
    mlp_in = 128 + 128 + 256 + N_LLM_FEATURES   # 527
    mlp_head = Chain(
        Dense(mlp_in => 512, relu),
        Dropout(dropout_rate),
        Dense(512 => 256, relu),
        Dropout(dropout_rate * 2/3),
        Dense(256 => 128, relu),
        Dropout(dropout_rate / 3),
        Dense(128 => 64, relu),
        Dense(64 => N_PRED_HOURS),
    )
    return SwingPredictor(market_cnn, hourly_cnn, mlp_head)
end

"""
Forward pass.

# Arguments
- `market`: `(N_MARKET_DAYS, N_MARKET_CHANNELS, N_companies, batch)` Float32 array.
  Column 1 along dim 3 is always the target stock (set by `assemble_batch`).
- `hourly`: `(N_HOURLY_BARS, batch)` Float32 matrix — target stock hourly series.
- `llm`:    `(N_LLM_FEATURES, batch)` Float32 matrix.

# Returns
`(N_PRED_HOURS, batch)` Float32 matrix of predicted log-return trajectories.
"""
function (m::SwingPredictor)(market::Array{Float32,4},
                              hourly::Matrix{Float32},
                              llm::Matrix{Float32})
    _, _, N, B = size(market)

    # ── Market CNN: shared across all N companies ─────────────────────────────
    x    = reshape(market, N_MARKET_DAYS, N_MARKET_CHANNELS, N * B)
    embs = m.market_cnn(x)                                    # (128, N*B)
    embs = reshape(embs, 128, N, B)                           # (128, N, B)

    target_emb = embs[:, 1, :]                                # (128, B)
    market_ctx = dropdims(mean(embs[:, 2:end, :], dims=2), dims=2)  # (128, B)

    # ── Hourly CNN: target stock only ─────────────────────────────────────────
    h          = reshape(hourly, N_HOURLY_BARS, 1, B)         # (280, 1, B)
    hourly_emb = m.hourly_cnn(h)                              # (256, B)

    # ── MLP head ──────────────────────────────────────────────────────────────
    combined = vcat(target_emb, market_ctx, hourly_emb, llm)  # (527, B)
    return m.mlp_head(combined)                               # (N_PRED_HOURS, B)
end

# ── Inference helpers ─────────────────────────────────────────────────────────

"""
Run inference on a pre-assembled batch. Returns (N_PRED_HOURS × batch) matrix.
"""
function predict(model::SwingPredictor,
                 market::Array{Float32,4},
                 hourly::Matrix{Float32},
                 llm::Matrix{Float32})::Matrix{Float32}
    Flux.testmode!(model)
    return model(market, hourly, llm)
end

# ── Persistence ───────────────────────────────────────────────────────────────

"""Save model weights, universe metadata, and arbitrary `meta` Dict to BSON."""
function save_model(model::SwingPredictor, companies::Vector{String},
                    path::String; meta::Dict=Dict())
    state = Flux.state(model)
    BSON.@save path state companies meta
    @info "Model saved → $path"
end

"""
Load a `SwingPredictor` from BSON. Returns `(model, companies, meta)`.
`companies` is the fixed universe ordering required at inference time.
"""
function load_model(path::String)
    BSON.@load path state companies meta
    model = build_model()
    Flux.loadmodel!(model, state)
    return model, companies, meta
end
