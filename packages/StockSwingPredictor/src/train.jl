"""
Training loop with early stopping and L2 regularisation.

Loss: weighted MSE over `arch.pred_hours` bars (35 for v1/v2, 70 for v3).
Weighting scheme is per-architecture: JUMP_THRESHOLD quadratic for v1/v2,
Lee-Mykland M² for v3 (see `_batch_weights`).
Direction accuracy and IC are reported on the final bar of each arch's horizon.

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

# Used by v1/v2 (lm_weighting=false): examples above 4% peak return get
# quadratically higher weight; below the threshold all examples share weight 1.
const JUMP_THRESHOLD = 0.04f0

"""
Per-example loss weights for one mini-batch, dispatched on architecture type.

`lm_weighting=false` (v1, v2): quadratic above JUMP_THRESHOLD, flat below.
`lm_weighting=true`  (v3):     Lee-Mykland M² — bipower-variation-normalised
max bar return. Near-constant labels → weight≈0; jump events dominate.
Both variants normalise so mean batch weight ≈ 1.
"""
function _batch_weights(yb::Matrix{Float32}, arch::SwingArchitecture)::Matrix{Float32}
    if arch.lm_weighting
        bar_rets = vcat(yb[1:1, :], diff(yb, dims=1))
        bv = (π/2f0) .* mean(
                 abs.(bar_rets[1:end-1, :]) .* abs.(bar_rets[2:end, :]), dims=1)
        bv   = max.(bv, 1f-10)
        M_lm = maximum(abs.(bar_rets), dims=1) ./ sqrt.(bv)
        w    = M_lm .^ 2
    else
        y_peak = maximum(abs.(yb), dims=1)
        w      = max.(1f0, y_peak ./ JUMP_THRESHOLD) .^ 2
    end
    return w ./ mean(w)
end

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
                epoch_offset::Int  = 0,
                log_every::Int     = 10,
                checkpoint_path::String = "",
                epoch_log_path::String  = "",
                stop_file::String       = "")

    opt_state = Flux.setup(Flux.Adam(lr), model)

    best_val_loss  = Inf32
    best_state     = Flux.state(model)
    no_improve     = 0
    stop_now_fired = false

    log = Dict{String, Any}(
        "epochs_run"      => 0,
        "train_mse"       => Float32[],
        "val_mse"         => Float32[],
        "best_val_mse"    => Inf32,
        "best_epoch"      => 0,
        "stopped_early"   => false,
        "started_at"      => string(now(UTC)),
        "checkpoint_path" => checkpoint_path,
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
        abs_epoch   = epoch + epoch_offset
        epoch_start = time()
        Flux.trainmode!(model)
        epoch_mse = 0f0
        n_batches = 0

        shuffled = shuffle(train_ids)
        for start in 1:batchsize:length(shuffled)
            batch_idx = shuffled[start : min(start + batchsize - 1, end)]
            market, hourly, yb = assemble_batch(dataset, cache, batch_idx)

            # Capture L2 before the update so l2_penalty matches the weights
            # used inside withgradient — lets us strip L2 from the logged MSE.
            l2_penalty = l2_lambda * Float32(
                sum(sum(abs2, p) for p in Flux.trainables(model) if ndims(p) == 2))

            yb_arch = yb[1:model.arch.pred_hours, :]
            w       = _batch_weights(yb_arch, model.arch)

            loss_val, grads = Flux.withgradient(model) do m
                ŷ   = m(market, hourly)
                mse = mean(w .* (ŷ .- yb_arch) .^ 2)
                l2  = sum(sum(abs2, p) for p in Flux.trainables(m) if ndims(p) == 2)
                mse + l2_lambda * l2
            end

            Flux.update!(opt_state, model, grads[1])
            n_batches += 1
            mse_now    = Float32(loss_val) - l2_penalty
            epoch_mse += mse_now

            if n_batches % PROGRESS_EVERY == 0
                print("\r  Epoch $abs_epoch | batch $n_batches/$n_batches_ep | loss $(round(loss_val, sigdigits=4))    ")
                flush(stdout)
                if !isempty(epoch_log_path)
                    open(epoch_log_path, "a") do io
                        JSON3.write(io, (epoch=abs_epoch, batch=n_batches,
                                         train_mse=mse_now,
                                         val_mse=nothing,
                                         n_batches=n_batches_ep,
                                         elapsed_secs=round(time() - train_start, digits=1)))
                        println(io)
                    end
                end
            end

            stop_now_file = isempty(stop_file) ? "" : replace(stop_file, "STOP" => "STOP_NOW")
            if !isempty(stop_now_file) && isfile(stop_now_file)
                rm(stop_now_file)
                @info "STOP_NOW detected — hard interrupt after epoch $abs_epoch batch $n_batches (no save)"
                stop_now_fired = true
                break
            end
        end
        stop_now_fired && break
        print("\r" * " "^72 * "\r")   # clear batch progress line

        train_mse = epoch_mse / n_batches

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
            log["best_epoch"]   = abs_epoch
            # Flush best weights to disk so --resume can recover from a crash.
            isempty(get(log, "checkpoint_path", "")) ||
                _save_checkpoint(log["checkpoint_path"], model, dataset.companies)
        else
            no_improve += 1
        end

        epoch_secs   = time() - epoch_start
        avg_secs     = (time() - train_start) / epoch
        eta_secs     = round(Int, avg_secs * (epochs - epoch))
        patience_str = "$no_improve/$patience"

        @printf("Epoch %3d | train %.6f | val %.6f | best %.6f | %s | ETA %s | patience %s%s\n",
                abs_epoch, train_mse, val_mse, best_val_loss,
                _fmt_duration(round(Int, epoch_secs)),
                _fmt_duration(eta_secs),
                patience_str,
                improved ? " ★" : "")

        if !isempty(epoch_log_path)
            open(epoch_log_path, "a") do io
                JSON3.write(io, (epoch=abs_epoch, batch=nothing,
                                 train_mse=train_mse, val_mse=val_mse,
                                 best_val_mse=best_val_loss, improved=improved,
                                 n_batches=n_batches_ep,
                                 elapsed_secs=round(time() - train_start, digits=1)))
                println(io)
            end
        end

        no_improve >= patience &&
            (@info "Early stop at epoch $abs_epoch (patience $patience exhausted)";
             log["stopped_early"] = true; break)

        if !isempty(stop_file) && isfile(stop_file)
            rm(stop_file)
            @info "Stop file detected — clean interrupt after epoch $abs_epoch"
            break
        end
    end

    log["epochs_run"] = length(log["train_mse"])
    log["stop_now"]   = stop_now_fired
    Flux.loadmodel!(model, best_state)
    Flux.testmode!(model)

    return model, log
end

"""
Evaluate a trained model on a set of example indices.

Inference is chunked to bound memory. Final-bar (eod day-10) predictions
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
        market, hourly, y = assemble_batch(dataset, cache, chunk)
        y_arch = y[1:model.arch.pred_hours, :]
        ŷ = model(market, hourly)

        sum_mse   += Float32(mean((ŷ .- y_arch).^2))
        sum_mae   += Float32(mean(abs.(ŷ .- y_arch)))
        n_batches += 1

        append!(ŷ_finals, ŷ[end, :])
        append!(y_finals, y_arch[end, :])
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

"""Save the training log as JSON. Replaces Inf/NaN with null (JSON-safe)."""
function save_training_log(log::Dict, path::String)
    open(path, "w") do io; JSON3.pretty(io, _sanitize_json(log)); end
end

# ── Helpers ───────────────────────────────────────────────────────────────────

_sanitize_json(x::AbstractFloat)  = isfinite(x) ? x : nothing
_sanitize_json(x::AbstractVector) = [_sanitize_json(v) for v in x]
_sanitize_json(x::Dict)           = Dict(k => _sanitize_json(v) for (k, v) in x)
_sanitize_json(x)                 = x

"""
Compute mean MSE across `ids` in chunks of `VAL_CHUNKSIZE`.
No gradient tracking — used for validation loss each epoch.
"""
function _mse_chunked(model, dataset, cache, ids)::Float32
    total = 0f0
    n     = 0
    for start in 1:VAL_CHUNKSIZE:length(ids)
        chunk  = ids[start : min(start + VAL_CHUNKSIZE - 1, end)]
        market, hourly, y = assemble_batch(dataset, cache, chunk)
        ŷ      = model(market, hourly)
        y_arch = y[1:model.arch.pred_hours, :]
        total += Float32(mean((ŷ .- y_arch).^2))
        n     += 1
    end
    return total / n
end

function _save_checkpoint(path::String, model::SwingPredictor, companies::Vector{String})
    save_model(model, companies, path;
               meta=Dict("checkpoint" => true, "saved_at" => string(now(UTC))))
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
