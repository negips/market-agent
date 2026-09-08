"""
Training loop with early stopping and L2 regularisation.

Loss: MSE over the full N_PRED_HOURS trajectory (all 35 hourly bars).
Direction accuracy and IC are reported on the final bar (end-of-day-5 close),
which is the actionable prediction.

Batches are assembled on-the-fly by `assemble_batch` — the raw market matrices
are sliced and normalised per batch rather than pre-stored, keeping memory usage
proportional to the dataset metadata size (~5 MB) rather than all examples.
"""

using Flux, Statistics, Dates, Printf, JSON3, Random

const DEFAULT_EPOCHS    = 150
const DEFAULT_BATCHSIZE = 32      # smaller than MLP-only — market tensor is large
const DEFAULT_LR        = 1f-3
const DEFAULT_L2        = 1f-4
const DEFAULT_PATIENCE  = 15

"""
Train `model` on `dataset`. Returns `(model, training_log)`.

# Arguments
- `model`: `SwingPredictor` from `build_model()`
- `dataset`: full `Dataset` (train + val + test all included)
- `train_idx`, `val_idx`: index ranges into `dataset.examples`
- `epochs`, `batchsize`, `lr`, `l2_lambda`, `patience`: hyperparameters
- `log_every`: print progress every N epochs
"""
function train!(model::SwingPredictor,
                dataset::Dataset,
                train_idx, val_idx;
                epochs::Int        = DEFAULT_EPOCHS,
                batchsize::Int     = DEFAULT_BATCHSIZE,
                lr::Float32        = DEFAULT_LR,
                l2_lambda::Float32 = DEFAULT_L2,
                patience::Int      = DEFAULT_PATIENCE,
                log_every::Int     = 10)

    opt_state = Flux.setup(Flux.Adam(lr), model)

    best_val_loss = Inf32
    best_state    = Flux.state(model)
    no_improve    = 0

    log = Dict{String, Any}(
        "epochs_run"   => 0,
        "train_mse"    => Float32[],
        "val_mse"      => Float32[],
        "best_val_mse" => Inf32,
        "best_epoch"   => 0,
        "stopped_early"=> false,
        "started_at"   => string(now(UTC)),
    )

    train_ids = collect(train_idx)
    val_ids   = collect(val_idx)

    # Pre-assemble validation batch once (val set is fixed)
    @info "Assembling validation batch ($(length(val_ids)) examples)…"
    val_market, val_hourly, val_llm, val_y = assemble_batch(dataset, val_ids)

    for epoch in 1:epochs
        Flux.trainmode!(model)
        epoch_loss = 0f0
        n_batches  = 0

        shuffled = shuffle(train_ids)
        for start in 1:batchsize:length(shuffled)
            batch_idx = shuffled[start : min(start + batchsize - 1, end)]
            market, hourly, llm, yb = assemble_batch(dataset, batch_idx)

            loss_val, grads = Flux.withgradient(model) do m
                ŷ       = m(market, hourly, llm)
                mse     = Flux.mse(ŷ, yb)
                l2      = sum(sum(abs2, p)
                              for p in Flux.trainables(m)
                              if ndims(p) == 2)
                mse + l2_lambda * l2
            end

            Flux.update!(opt_state, model, grads[1])
            epoch_loss += loss_val
            n_batches  += 1
        end

        train_mse = epoch_loss / n_batches

        Flux.testmode!(model)
        val_ŷ   = model(val_market, val_hourly, val_llm)
        val_mse = Float32(mean((val_ŷ .- val_y).^2))

        push!(log["train_mse"], train_mse)
        push!(log["val_mse"],   val_mse)

        if val_mse < best_val_loss
            best_val_loss        = val_mse
            best_state           = Flux.state(model)
            no_improve           = 0
            log["best_val_mse"]  = best_val_loss
            log["best_epoch"]    = epoch
        else
            no_improve += 1
        end

        if epoch % log_every == 0 || epoch == 1
            @printf("Epoch %3d | train MSE %.6f | val MSE %.6f | best %.6f%s\n",
                    epoch, train_mse, val_mse, best_val_loss,
                    no_improve == 0 ? " ★" : "")
        end

        no_improve >= patience && (@info "Early stop at epoch $epoch"; log["stopped_early"] = true; break)
    end

    log["epochs_run"] = length(log["train_mse"])
    Flux.loadmodel!(model, best_state)
    Flux.testmode!(model)

    return model, log
end

"""
Evaluate a trained model on a set of example indices.
Metrics are computed over the final hourly bar (end-of-day-5 close vs ref).
"""
function evaluate(model::SwingPredictor, dataset::Dataset, test_idx)::Dict
    Flux.testmode!(model)
    test_ids = collect(test_idx)
    market, hourly, llm, y = assemble_batch(dataset, test_ids)
    ŷ = model(market, hourly, llm)

    mse = Float32(mean((ŷ .- y).^2))
    mae = Float32(mean(abs.(ŷ .- y)))

    # Final-bar (eod day 5) metrics
    ŷ_final = ŷ[end, :]
    y_final = y[end, :]
    dir_acc = Float32(mean(sign.(ŷ_final) .== sign.(y_final)))
    ic      = _spearman_corr(ŷ_final, y_final)

    @printf("Test MSE: %.6f | MAE: %.6f | Dir(eod5): %.1f%% | IC(eod5): %.4f\n",
            mse, mae, dir_acc * 100, ic)

    return Dict("test_mse" => mse, "test_mae" => mae,
                "direction_accuracy" => dir_acc, "information_coefficient" => ic)
end

"""Save the training log as JSON."""
function save_training_log(log::Dict, path::String)
    open(path, "w") do io
        JSON3.pretty(io, log)
    end
end

# ── Helpers ───────────────────────────────────────────────────────────────────

function _spearman_corr(x::AbstractVector, y::AbstractVector)::Float32
    n  = length(x)
    rx = _rankdata(x); ry = _rankdata(y)
    d2 = sum((rx .- ry).^2)
    Float32(1.0 - 6.0 * d2 / (n * (n^2 - 1)))
end

function _rankdata(v::AbstractVector)
    n     = length(v)
    order = sortperm(v)
    ranks = Vector{Float64}(undef, n)
    ranks[order] = 1:n
    return ranks
end
