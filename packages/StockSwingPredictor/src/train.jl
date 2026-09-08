"""
Training loop with early stopping, L2 regularisation, and progress logging.

Loss: MSE on 5-day log return prediction.
Regularisation: L2 weight decay (λ=1e-4) applied to all Dense weight matrices.
Optimiser: Adam (lr=1e-3) with cosine decay schedule.
Early stopping: patience=15 epochs on validation MSE; restores best weights.
"""

using Flux, Statistics, Dates, Printf, JSON3

const DEFAULT_EPOCHS     = 150
const DEFAULT_BATCHSIZE  = 64
const DEFAULT_LR         = 1f-3
const DEFAULT_L2         = 1f-4
const DEFAULT_PATIENCE   = 15

"""
Train `model` on the given dataset split. Returns `(model, training_log)`.

# Arguments
- `model`: Flux model from `build_model`
- `X_train`, `y_train`: training features and labels (already normalised)
- `X_val`, `y_val`: validation features and labels
- `epochs`, `batchsize`, `lr`, `l2_lambda`, `patience`: hyperparameters
- `log_every`: print summary every N epochs
"""
function train!(model, X_train::Matrix{Float32}, y_train::Vector{Float32},
                X_val::Matrix{Float32},   y_val::Vector{Float32};
                epochs::Int    = DEFAULT_EPOCHS,
                batchsize::Int = DEFAULT_BATCHSIZE,
                lr::Float32    = DEFAULT_LR,
                l2_lambda::Float32 = DEFAULT_L2,
                patience::Int  = DEFAULT_PATIENCE,
                log_every::Int = 10)

    opt_state = Flux.setup(Flux.Adam(lr), model)

    best_val_loss = Inf32
    best_state    = Flux.state(model)
    no_improve    = 0

    log = Dict{String, Any}(
        "epochs_run"      => 0,
        "train_mse"       => Float32[],
        "val_mse"         => Float32[],
        "best_val_mse"    => Inf32,
        "best_epoch"      => 0,
        "stopped_early"   => false,
        "started_at"      => string(now(UTC)),
    )

    n_train = size(X_train, 2)
    loader  = Flux.DataLoader((X_train, y_train), batchsize=batchsize, shuffle=true)

    for epoch in 1:epochs
        Flux.trainmode!(model)
        epoch_loss = 0f0
        n_batches  = 0

        for (xb, yb) in loader
            loss_val, grads = Flux.withgradient(model) do m
                ŷ = m(xb)
                mse_loss = Flux.mse(ŷ, yb)
                # L2 regularisation on Dense weight matrices only
                l2 = sum(sum(abs2, p)
                         for p in Flux.trainables(m)
                         if ndims(p) == 2)
                mse_loss + l2_lambda * l2
            end
            Flux.update!(opt_state, model, grads[1])
            epoch_loss += loss_val
            n_batches  += 1
        end

        train_mse = epoch_loss / n_batches

        Flux.testmode!(model)
        val_mse = Flux.mse(model(X_val), y_val)

        push!(log["train_mse"], train_mse)
        push!(log["val_mse"],   Float32(val_mse))

        if val_mse < best_val_loss
            best_val_loss = Float32(val_mse)
            best_state    = Flux.state(model)
            no_improve    = 0
            log["best_val_mse"] = best_val_loss
            log["best_epoch"]   = epoch
        else
            no_improve += 1
        end

        if epoch % log_every == 0 || epoch == 1
            @printf("Epoch %3d | train MSE %.6f | val MSE %.6f | best %.6f%s\n",
                    epoch, train_mse, val_mse, best_val_loss,
                    no_improve == 0 ? " ★" : "")
        end

        if no_improve >= patience
            @info "Early stopping at epoch $epoch (no improvement for $patience epochs)"
            log["stopped_early"] = true
            break
        end
    end

    log["epochs_run"] = length(log["train_mse"])

    # Restore best weights
    Flux.loadmodel!(model, best_state)
    Flux.testmode!(model)

    return model, log
end

"""
Evaluate a trained model on a test split. Returns a Dict with metrics.
"""
function evaluate(model, X_test::Matrix{Float32}, y_test::Vector{Float32})::Dict

    Flux.testmode!(model)
    ŷ = model(X_test)

    mse  = Float32(mean((ŷ .- y_test).^2))
    mae  = Float32(mean(abs.(ŷ .- y_test)))

    # Directional accuracy: did we get the sign right?
    dir_acc = Float32(mean(sign.(ŷ) .== sign.(y_test)))

    # Information coefficient: Spearman rank correlation between ŷ and y
    ic = _spearman_corr(ŷ, y_test)

    @printf("Test MSE: %.6f | MAE: %.6f | Direction: %.1f%% | IC: %.4f\n",
            mse, mae, dir_acc * 100, ic)

    return Dict("test_mse" => mse, "test_mae" => mae,
                "direction_accuracy" => dir_acc, "information_coefficient" => ic)
end

"""Spearman rank correlation between two vectors."""
function _spearman_corr(x::AbstractVector, y::AbstractVector)::Float32
    n = length(x)
    rx = _rankdata(x)
    ry = _rankdata(y)
    d2 = sum((rx .- ry).^2)
    Float32(1.0 - 6.0 * d2 / (n * (n^2 - 1)))
end

function _rankdata(v::AbstractVector)
    n = length(v)
    order = sortperm(v)
    ranks = Vector{Float64}(undef, n)
    ranks[order] = 1:n
    return ranks
end

"""Save the training log as JSON."""
function save_training_log(log::Dict, path::String)
    open(path, "w") do io
        JSON3.pretty(io, log)
    end
end
