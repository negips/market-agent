"""
Training loop with early stopping and L2 regularisation.

Loss: MSE over the full N_PRED_HOURS trajectory (all 35 hourly bars).
Direction accuracy and IC are reported on the final bar (end-of-day-5 close),
which is the actionable prediction.

## Memory design

Validation and test evaluation are chunked — never more than VAL_CHUNKSIZE
examples are assembled into tensors at once. This keeps peak memory bounded:

  market tensor per chunk = N_MARKET_DAYS × N_MARKET_CHANNELS × N_MARKET_COMPANIES × VAL_CHUNKSIZE × 4 bytes
                          = 28 × 2 × 150 × 256 × 4 ≈ 8.6 MB

Training batches are assembled on-the-fly and GC'd after each step.
Peak memory during training is dominated by CNN gradient intermediates (~200 MB).
"""

using Flux, Statistics, Dates, Printf, JSON3, Random

const DEFAULT_EPOCHS    = 150
const DEFAULT_BATCHSIZE = 32
const DEFAULT_LR        = 1f-3
const DEFAULT_L2        = 1f-4
const DEFAULT_PATIENCE  = 15
const VAL_CHUNKSIZE     = 256   # max examples assembled at once during val/test
const PROGRESS_EVERY    = 50    # print batch progress every N batches

"""
Train `model` on `dataset`. Returns `(model, training_log)`.

Validation loss is computed in chunks of `VAL_CHUNKSIZE` to bound peak memory.

# Arguments
- `model`: `SwingPredictor` from `build_model()`
- `dataset`: full `Dataset` (train + val + test all included)
- `cache`: `InferenceCache` with price matrices
- `train_idx`, `val_idx`: index ranges into `dataset.examples`
- `epochs`, `batchsize`, `lr`, `l2_lambda`, `patience`: hyperparameters
- `log_every`: print progress every N epochs
"""
function train!(model::SwingPredictor,
                dataset::Dataset,
                cache::InferenceCache,
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
        "epochs_run"    => 0,
        "train_mse"     => Float32[],
        "val_mse"       => Float32[],
        "best_val_mse"  => Inf32,
        "best_epoch"    => 0,
        "stopped_early" => false,
        "started_at"    => string(now(UTC)),
    )

    train_ids    = collect(train_idx)
    val_ids      = collect(val_idx)
    n_batches_ep = ceil(Int, length(train_ids) / batchsize)

    @info "Training: $(length(train_ids)) examples | Val: $(length(val_ids)) examples"
    @info "Market context: $(N_MARKET_COMPANIES) companies × $(N_MARKET_DAYS) days"
    @info "Batches per epoch: $n_batches_ep  |  Early stop patience: $patience"
    println()

    train_start = time()

    for epoch in 1:epochs
        epoch_start = time()
        Flux.trainmode!(model)
        epoch_loss = 0f0
        n_batches  = 0

        shuffled = shuffle(train_ids)
        for start in 1:batchsize:length(shuffled)
            batch_idx = shuffled[start : min(start + batchsize - 1, end)]
            market, hourly, llm, yb = assemble_batch(dataset, cache, batch_idx)

            loss_val, grads = Flux.withgradient(model) do m
                ŷ   = m(market, hourly, llm)
                mse = Flux.mse(ŷ, yb)
                l2  = sum(sum(abs2, p) for p in Flux.trainables(m) if ndims(p) == 2)
                mse + l2_lambda * l2
            end

            Flux.update!(opt_state, model, grads[1])
            epoch_loss += loss_val
            n_batches  += 1

            if n_batches % PROGRESS_EVERY == 0
                print("\r  Epoch $epoch | batch $n_batches/$n_batches_ep | loss $(round(loss_val, sigdigits=4))    ")
                flush(stdout)
            end
        end
        print("\r" * " "^72 * "\r")   # clear batch progress line

        train_mse = epoch_loss / n_batches

        Flux.testmode!(model)
        val_mse = _mse_chunked(model, dataset, cache, val_ids)

        push!(log["train_mse"], train_mse)
        push!(log["val_mse"],   val_mse)

        improved = val_mse < best_val_loss
        if improved
            best_val_loss       = val_mse
            best_state          = Flux.state(model)
            no_improve          = 0
            log["best_val_mse"] = best_val_loss
            log["best_epoch"]   = epoch
        else
            no_improve += 1
        end

        epoch_secs  = time() - epoch_start
        avg_secs    = (time() - train_start) / epoch
        eta_secs    = round(Int, avg_secs * (epochs - epoch))
        patience_str = "$no_improve/$patience"

        @printf("Epoch %3d/%d | train %.6f | val %.6f | best %.6f | %s | ETA %s | patience %s%s\n",
                epoch, epochs, train_mse, val_mse, best_val_loss,
                _fmt_duration(round(Int, epoch_secs)),
                _fmt_duration(eta_secs),
                patience_str,
                improved ? " ★" : "")

        no_improve >= patience &&
            (@info "Early stop at epoch $epoch (patience $patience exhausted)";
             log["stopped_early"] = true; break)
    end

    log["epochs_run"] = length(log["train_mse"])
    Flux.loadmodel!(model, best_state)
    Flux.testmode!(model)

    return model, log
end

"""
Evaluate a trained model on a set of example indices.

Inference is chunked to bound memory. Final-bar (eod day-5) predictions
are collected across chunks before computing ranking-based metrics (IC).
"""
function evaluate(model::SwingPredictor, dataset::Dataset, cache::InferenceCache,
                  test_idx)::Dict
    Flux.testmode!(model)
    test_ids = collect(test_idx)

    sum_mse   = 0f0
    sum_mae   = 0f0
    n_batches = 0
    ŷ_finals  = Float32[]
    y_finals  = Float32[]

    for start in 1:VAL_CHUNKSIZE:length(test_ids)
        chunk = test_ids[start : min(start + VAL_CHUNKSIZE - 1, end)]
        market, hourly, llm, y = assemble_batch(dataset, cache, chunk)
        ŷ = model(market, hourly, llm)

        sum_mse   += Float32(mean((ŷ .- y).^2))
        sum_mae   += Float32(mean(abs.(ŷ .- y)))
        n_batches += 1

        append!(ŷ_finals, ŷ[end, :])
        append!(y_finals, y[end, :])
    end

    mse     = sum_mse / n_batches
    mae     = sum_mae / n_batches
    dir_acc = Float32(mean(sign.(ŷ_finals) .== sign.(y_finals)))
    ic      = _spearman_corr(ŷ_finals, y_finals)

    @printf("Test MSE: %.6f | MAE: %.6f | Dir(eod5): %.1f%% | IC(eod5): %.4f\n",
            mse, mae, dir_acc * 100, ic)

    return Dict("test_mse" => mse, "test_mae" => mae,
                "direction_accuracy" => dir_acc, "information_coefficient" => ic)
end

"""Save the training log as JSON."""
function save_training_log(log::Dict, path::String)
    open(path, "w") do io; JSON3.pretty(io, log); end
end

# ── Helpers ───────────────────────────────────────────────────────────────────

"""
Compute mean MSE across `ids` in chunks of `VAL_CHUNKSIZE`.
No gradient tracking — used for validation loss each epoch.
"""
function _mse_chunked(model, dataset, cache, ids)::Float32
    total = 0f0
    n     = 0
    for start in 1:VAL_CHUNKSIZE:length(ids)
        chunk = ids[start : min(start + VAL_CHUNKSIZE - 1, end)]
        market, hourly, llm, y = assemble_batch(dataset, cache, chunk)
        ŷ = model(market, hourly, llm)
        total += Float32(mean((ŷ .- y).^2))
        n     += 1
    end
    return total / n
end

function _fmt_duration(seconds::Int)::String
    seconds < 60   && return "$(seconds)s"
    seconds < 3600 && return "$(seconds ÷ 60)m $(seconds % 60)s"
    return "$(seconds ÷ 3600)h $(seconds % 3600 ÷ 60)m"
end

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
