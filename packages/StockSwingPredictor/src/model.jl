"""
Neural network architecture and inference.

A regularised MLP with dropout. Input is the assembled, normalised feature vector.
Output is a vector of length N_PRED_HOURS (35): predicted hourly log-return trajectory
over the next 5 trading days, each value relative to the reference close price.

Architecture:
  Input(n_features) → Dense(512, relu) → Dropout(0.30)
                    → Dense(256, relu) → Dropout(0.30)
                    → Dense(128, relu) → Dropout(0.20)
                    → Dense(64,  relu) → Dropout(0.15)
                    → Dense(32,  relu) → Dropout(0.10)
                    → Dense(N_PRED_HOURS)

L2 weight regularisation is applied during training (see train.jl).
"""

using Flux, BSON, JSON3

"""
Build the MLP. `n_features` must match the feature vector assembled by `features.jl`.
Output dimension is always `N_PRED_HOURS` (35).
"""
function build_model(n_features::Int; dropout_rate::Float64=0.3)
    Chain(
        Dense(n_features => 512, relu),
        Dropout(dropout_rate),
        Dense(512 => 256, relu),
        Dropout(dropout_rate),
        Dense(256 => 128, relu),
        Dropout(dropout_rate * 2/3),
        Dense(128 => 64, relu),
        Dropout(dropout_rate / 2),
        Dense(64 => 32, relu),
        Dropout(dropout_rate / 3),
        Dense(32 => N_PRED_HOURS),
    )
end

"""
Run inference on a matrix of normalised features.
`X`: (n_features × n_examples) Float32 matrix.
Returns a (N_PRED_HOURS × n_examples) Float32 matrix of predicted trajectories.
"""
function predict(model, X::Matrix{Float32})::Matrix{Float32}
    Flux.testmode!(model)
    return model(X)
end

"""
Run inference on a single normalised feature vector.
Returns a `Vector{Float32}` of length N_PRED_HOURS.
"""
function predict(model, x::Vector{Float32})::Vector{Float32}
    Flux.testmode!(model)
    return vec(model(reshape(x, :, 1)))
end

# ── Persistence ───────────────────────────────────────────────────────────────

"""
Save model weights and metadata to `path` (BSON format).
`meta` is any Dict-serialisable metadata (e.g. n_features, training_date).
"""
function save_model(model, norm_stats::NormStats, sector_vocab::Vector{String},
                    path::String; meta::Dict=Dict())
    state = Flux.state(model)
    BSON.@save path state norm_stats sector_vocab meta
    @info "Model saved: $path"
end

"""
Load model, NormStats, and sector vocabulary from a BSON file.
Returns `(model, norm_stats, sector_vocab, meta)`.
`n_features` must be provided to reconstruct the model architecture.
"""
function load_model(path::String, n_features::Int)
    BSON.@load path state norm_stats sector_vocab meta
    model = build_model(n_features)
    Flux.loadmodel!(model, state)
    return model, norm_stats, sector_vocab, meta
end
