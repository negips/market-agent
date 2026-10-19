"""
Training and evaluation of the v4 [`Forecaster`](@ref).

Batches are random windows sliced from a [`MinuteCache`](@ref) (no file reads in the loop).
"Epochs" are a fixed number of random batches, because windows overlap and there is no
natural pass over the data. After each epoch the model is scored on every [`EvalSet`](@ref),
and the checkpoint with the best primary evaluation loss is kept.

One-minute returns are close to unpredictable, so every evaluation is reported next to two
baselines on the same masked targets: predict zero, and repeat the previous return.
`skill = 1 - MSE_model / MSE_zero` is positive only when the model beats predicting no move.
"""

using Flux, Statistics, JSON3, Random, Printf, CUDA, Dates

"""Width, in minutes, of the time-of-day buckets of the per-slot report at 1-minute bars (a cache has `bucket_bars` of its own, about 15 minutes)."""
const SLOT_BUCKET_MINUTES = 15

"""Number of time-of-day buckets in the per-slot report at 1-minute bars; see [`n_slot_buckets`](@ref)."""
const N_SLOT_BUCKETS = cld(MINUTES_PER_DAY, SLOT_BUCKET_MINUTES)

"""Tolerances for counting a timestamp prediction as exact: half a bar (applied as `0.5 / bars_per_day`), a quarter of a log-gap unit, half a weekday step."""
const TIME_TOLERANCES = (minute = Float32(0.5 / MINUTES_PER_DAY), gap = 0.25f0, weekday = 0.125f0)

"""
    EvalSet(name, cols, starts)

A fixed set of windows to score: the cache columns `cols` and the start bars `starts`.
"""
struct EvalSet
    name   :: String
    cols   :: Vector{Int}
    starts :: Vector{Int}
end

"""`n` window starts spread evenly over `starts`, non-overlapping where the range allows."""
function spread_starts(starts::UnitRange{Int}, n::Int, window::Int)::Vector{Int}
    isempty(starts) && return Int[]
    n = min(n, length(starts))
    stride = max(window, length(starts) ÷ n)
    s = collect(first(starts):stride:last(starts))
    return length(s) > n ? s[round.(Int, range(1, length(s); length=n))] : s
end

"""Decay of the exponentially weighted volatility baseline, per minute."""
const EWMA_DECAY = 0.94f0

"""
    evaluate_forecaster(model, cache, set, window; batch=32, device=:cpu, dtype=Float32, loss=:student_t, nu=4) -> NamedTuple

Score `model` on every window of `set` with dropout off, run in `dtype`; the error sums are always Float32/Float64.

Point forecast (the location `μ`): `mse`, `mse_zero`, `mse_last` (repeat the previous return), `skill = 1 - mse/mse_zero`,
`skill_last`, `hit_rate` (sign, over targets that moved), `per_stock_skill`, `per_slot_skill` (15-minute buckets).

Timestamp: `minute_acc`, `gap_acc`, `weekday_acc`.

Distribution (when `loss` is `:student_t` or `:gaussian`; `NaN` for `:huber`): `nll` of the targets under the predicted
location and scale, next to two baselines with zero location, `nll_const` (a constant scale per stock) and `nll_ewma`
(a scale from the exponentially weighted recent squared returns, `EWMA_DECAY`); both baseline scales are clamped to the
range the model's own σ can take, so a stock that never moves cannot win by a scale of zero; `cov68`/`cov95`, the fraction of targets
inside the model's own 68%/95% interval (they should be near 0.68/0.95); and `std_pred_slot` against `rms_slot`, the
predicted and realised standard deviation per 15-minute bucket. `loss` (the objective's return part, Huber for
`:huber`, otherwise the unweighted NLL) is reported too.
"""
function evaluate_forecaster(model::Forecaster, cache::MinuteCache, set::EvalSet, window::Int;
                             batch::Int=32, device::Symbol=:cpu, dtype::Type{<:AbstractFloat}=Float32,
                             loss::Symbol=:student_t, nu::Real=STUDENT_T_NU)
    dev = device === :gpu ? Flux.gpu : Flux.cpu
    m = Flux.testmode!(dev(convert_forecaster(model, dtype)))
    Ps = Array{Float32, 3}[]; Ls = Array{Float32, 3}[]; Ys = Array{Float32, 3}[]; Ws = Array{Float32, 3}[]
    Rs = Array{Float32, 3}[]; Sl = Matrix{Int16}[]
    t_min = 0; t_gap = 0; t_wd = 0; t_n = 0
    for chunk in Iterators.partition(set.starts, batch)
        b = make_batch(cache, set.cols, collect(chunk), window)
        mu, logsigma, time = m(dev(dtype.(b.X)))
        push!(Ps, Float32.(Array(mu))); push!(Ls, Float32.(Array(logsigma)))
        push!(Ys, b.Y); push!(Ws, b.W); push!(Rs, b.R); push!(Sl, b.slot)
        td = abs.(Float32.(Array(time)) .- b.T)
        t_min += count(<(0.5f0 / Float32(cache.bars_per_day)), td[1, :, :])
        t_gap += count(<(TIME_TOLERANCES.gap), td[2, :, :])
        t_wd  += count(<(TIME_TOLERANCES.weekday), td[3, :, :])
        t_n += length(td[1, :, :])
    end
    P = cat(Ps...; dims=3); L = cat(Ls...; dims=3); Y = cat(Ys...; dims=3); W = cat(Ws...; dims=3)
    R = cat(Rs...; dims=3); slot = hcat(Sl...)
    r = score_forecasts(P, L, Y, W, R, slot; loss=loss, nu=nu, n_buckets=n_slot_buckets(cache))
    safe(a, b) = b > 0 ? a / b : NaN
    return merge((n_windows=length(set.starts),), r,
                 (minute_acc=safe(t_min, t_n), gap_acc=safe(t_gap, t_n), weekday_acc=safe(t_wd, t_n)))
end

"""
    score_forecasts(P, L, Y, W, R, slot; loss=:student_t, nu=4) -> NamedTuple

Score predicted locations `P` and log-scales `L` (all `(M, N, B)`, like the targets `Y` and weights `W`) against the next
returns, with `R` the input returns (for the repeat-last baseline) and `slot` the `(M, B)` minute of day. This is the
metric part of [`evaluate_forecaster`](@ref), without the timestamp fields, so a forecaster that is not a [`Forecaster`](@ref)
(see [`evaluate_ar_baseline`](@ref)) is scored by exactly the same code.
"""
function score_forecasts(P::AbstractArray, L::AbstractArray, Y::AbstractArray, W::AbstractArray, R::AbstractArray, slot::AbstractMatrix;
                         loss::Symbol=:student_t, nu::Real=STUDENT_T_NU, n_buckets::Int=N_SLOT_BUCKETS)
    N = size(Y, 2)
    dist = loss === :huber ? :none : loss
    sel = W .> 0
    wsum = count(sel)
    safe(a, b) = b > 0 ? a / b : NaN
    skill(a, z) = z > 0 ? 1 - a / z : NaN
    err = P .- Y
    sse = sum(abs2, err[sel]); sse0 = sum(abs2, Y[sel]); ssel = sum(abs2, (R .- Y)[sel])
    moved = count(sel .& (Y .!= 0)); hit = count(sel .& (Y .!= 0) .& (sign.(P) .== sign.(Y)))
    slot_idx = reshape(Int.(slot) .+ 1, size(slot, 1), 1, size(slot, 2))
    slot_idx = repeat(slot_idx, 1, N, 1)
    st_sse = zeros(N); st_sse0 = zeros(N); st_n = zeros(Int, N)
    sl_sse = zeros(n_buckets); sl_sse0 = zeros(n_buckets)
    sl_std = zeros(n_buckets); sl_n = zeros(Int, n_buckets)
    ustd = dist === :none ? 1.0 : unit_std(dist, nu)
    for k in eachindex(Y)
        sel[k] || continue
        ni = (k - 1) ÷ size(Y, 1) % N + 1
        sl = slot_idx[k]
        e2 = err[k]^2; z2 = Y[k]^2
        st_sse[ni] += e2; st_sse0[ni] += z2; st_n[ni] += 1
        sl_sse[sl] += e2; sl_sse0[sl] += z2; sl_n[sl] += 1
        sl_std[sl] += exp(L[k]) * ustd
    end
    hub = loss === :huber ? sum(_huber(err)[sel]) : NaN
    nll = nll_const = nll_ewma = cov68 = cov95 = NaN
    if dist !== :none
        nll = sum(forecast_nll(dist, Y, P, L, nu)[sel])
        rms = [sqrt(st_sse0[i] / max(st_n[i], 1)) for i in 1:N]
        lo, hi = exp(LOGSIGMA_CENTER - LOGSIGMA_HALFRANGE), exp(LOGSIGMA_CENTER + LOGSIGMA_HALFRANGE)
        clamp_sigma(x) = clamp.(x, lo, hi)
        floor_ = reshape(0.1f0 .* Float32.(rms), 1, N, 1)
        sig_const = reshape(Float32.(rms) ./ Float32(ustd), 1, N, 1) .+ zero(Y)
        nll_const = sum(forecast_nll(dist, Y, 0f0, log.(clamp_sigma(sig_const)), nu)[sel])
        v = similar(R); prev = repeat(reshape(Float32.(rms) .^ 2, 1, N, 1), 1, 1, size(R, 3))[1:1, :, :]
        for j in 1:size(R, 1)
            prev = EWMA_DECAY .* prev .+ (1f0 - EWMA_DECAY) .* R[j:j, :, :] .^ 2
            v[j:j, :, :] = prev
        end
        sig_ewma = max.(sqrt.(v), floor_) ./ Float32(ustd)
        nll_ewma = sum(forecast_nll(dist, Y, 0f0, log.(clamp_sigma(sig_ewma)), nu)[sel])
        z = abs.((Y .- P) .* exp.(-L))
        cov68 = count(sel .& (z .<= central_halfwidth(dist, nu, 0.68))) / wsum
        cov95 = count(sel .& (z .<= central_halfwidth(dist, nu, 0.95))) / wsum
        nll /= wsum; nll_const /= wsum; nll_ewma /= wsum
    end
    return (n_targets=wsum,
            loss=loss === :huber ? safe(hub, wsum) : nll,
            mse=safe(sse, wsum), mse_zero=safe(sse0, wsum), mse_last=safe(ssel, wsum),
            skill=skill(sse, sse0), skill_last=skill(ssel, sse0), hit_rate=safe(hit, moved),
            nll=nll, nll_const=nll_const, nll_ewma=nll_ewma, cov68=cov68, cov95=cov95,
            per_stock_skill=[skill(st_sse[i], st_sse0[i]) for i in 1:N],
            per_slot_skill=[skill(sl_sse[i], sl_sse0[i]) for i in 1:n_buckets],
            std_pred_slot=[safe(sl_std[i], sl_n[i]) for i in 1:n_buckets],
            rms_slot=[sqrt(safe(sl_sse0[i], sl_n[i])) for i in 1:n_buckets])
end

"""The scalar fields of an [`evaluate_forecaster`](@ref) result, for the epoch log."""
scalar_metrics(r) = (n_windows=r.n_windows, n_targets=r.n_targets, loss=r.loss, mse=r.mse, mse_zero=r.mse_zero,
                     mse_last=r.mse_last, skill=r.skill, skill_last=r.skill_last, hit_rate=r.hit_rate,
                     minute_acc=r.minute_acc, gap_acc=r.gap_acc, weekday_acc=r.weekday_acc,
                     nll=r.nll, nll_const=r.nll_const, nll_ewma=r.nll_ewma, cov68=r.cov68, cov95=r.cov95)

"""`x` with every non-finite number replaced by `nothing` (JSON has no NaN), through named tuples, dicts and vectors."""
_json_safe(x::AbstractFloat) = isfinite(x) ? x : nothing
_json_safe(x::NamedTuple) = map(_json_safe, x)
_json_safe(x::AbstractDict) = Dict(k => _json_safe(v) for (k, v) in x)
_json_safe(x::AbstractVector) = [_json_safe(v) for v in x]
_json_safe(x) = x

"""Write `obj` as JSON to `path` through a temporary file, so a page polling it never reads half a file."""
function _write_json_atomic(path::String, obj)
    tmp = path * ".tmp"
    open(tmp, "w") do io
        JSON3.write(io, obj)
    end
    mv(tmp, path; force=true)
    return nothing
end

"""Steps between two writes of `live_status.json`."""
const LIVE_EVERY_STEPS = 10

"""True when every gradient array in `g` is finite."""
function grads_finite(g)::Bool
    ok = true
    Flux.fmap(g) do x
        x isa AbstractArray && !all(isfinite, x) && (ok = false)
        x
    end
    return ok
end

"""Learning rate at `step` (1-based) of `total`: linear warm-up, then cosine decay to a tenth of `lr`."""
function scheduled_lr(step::Int, total::Int, lr::Real, warmup::Int)::Float32
    step <= warmup && return Float32(lr * step / max(warmup, 1))
    p = (step - warmup) / max(total - warmup, 1)
    return Float32(lr * (0.1 + 0.9 * 0.5 * (1 + cos(π * min(p, 1.0)))))
end

const _CORE = CUDA.CUDACore

"""True when memory can be read off a GPU memory pool."""
gpu_memory_available() = CUDA.functional()

"""Reset the memory pool's high-water mark to what is in use now, so that [`gpu_pool_peak`](@ref) reports the highest use since this call."""
function reset_gpu_peak!()
    CUDA.synchronize()
    _CORE.attribute!(_CORE.pool_create(CUDA.device()), _CORE.MEMPOOL_ATTR_USED_MEM_HIGH, UInt64(0))
    return nothing
end

"""Highest number of bytes of the memory pool in use since [`reset_gpu_peak!`](@ref). Includes arrays that are garbage but not yet collected."""
gpu_pool_peak() = (CUDA.synchronize(); Int(_CORE.attribute(UInt64, _CORE.pool_create(CUDA.device()), _CORE.MEMPOOL_ATTR_USED_MEM_HIGH)))

"""Bytes of the memory pool in use now, and bytes the pool holds from the device (in use plus kept for reuse)."""
gpu_pool_now() = (CUDA.synchronize(); (used=Int(CUDA.used_memory()), reserved=Int(CUDA.cached_memory())))

const _MIB = 2.0^20

"""One epoch's row of `memory_profile.json` (MiB; the GPU fields are `nothing` without a GPU). `train_peak` is the peak of one probe step run from a collected pool (what a step needs); `train_pool_peak` is the pool's highest use over the epoch's real steps, which includes garbage the collector has not freed yet."""
function memory_row(epoch::Int, steps::Int, train_secs::Real, train_peak, train_pool_peak, train_end, eval_peak, reserved, device_used)
    f(x) = x === nothing ? nothing : round(x / _MIB, digits=1)
    return Dict{String, Any}("epoch" => epoch, "steps" => steps, "secs_per_step" => round(train_secs / max(steps, 1), digits=4),
                             "train_peak_mib" => f(train_peak), "train_pool_peak_mib" => f(train_pool_peak), "train_end_mib" => f(train_end), "eval_peak_mib" => f(eval_peak),
                             "reserved_mib" => f(reserved), "device_used_mib" => f(device_used),
                             "host_rss_peak_mib" => round(Sys.maxrss() / _MIB, digits=1))
end

"""
    memory_summary(rows) -> Dict

Statistics over the epoch rows of a memory profile: min, median, mean and max of the training and evaluation peaks and of the step time, the
highest host resident set, and the change in the training peak from the first to the last five epochs (a steady rise would mean memory is leaking).
Fields the rows do not have (no GPU) are `nothing`.
"""
function memory_summary(rows::AbstractVector)
    col(k) = Float64[r[k] for r in rows if get(r, k, nothing) !== nothing]
    stats(v) = isempty(v) ? nothing : Dict("min" => minimum(v), "median" => median(v), "mean" => mean(v), "max" => maximum(v))
    tp = col("train_peak_mib")
    drift = length(tp) >= 10 ? mean(tp[end-4:end]) - mean(tp[1:5]) : nothing
    return Dict{String, Any}("epochs" => length(rows), "train_peak_mib" => stats(tp), "eval_peak_mib" => stats(col("eval_peak_mib")),
                             "train_pool_peak_mib" => stats(col("train_pool_peak_mib")), "train_end_mib" => stats(col("train_end_mib")), "reserved_mib" => stats(col("reserved_mib")),
                             "device_used_mib" => stats(col("device_used_mib")), "secs_per_step" => stats(col("secs_per_step")),
                             "host_rss_peak_mib" => isempty(col("host_rss_peak_mib")) ? nothing : maximum(col("host_rss_peak_mib")),
                             "train_peak_drift_mib" => drift)
end

"""
    write_memory_profile(path, rows; config) -> path

Write `memory_profile.json`: the per-epoch rows, their [`memory_summary`](@ref), the GPU and the run's `config` (batch, micro-batches, window, …).
The file is replaced atomically. Rows of earlier sessions of a resumed run are passed in by the caller (see [`load_memory_rows`](@ref)).
"""
function write_memory_profile(path::String, rows::AbstractVector; config::AbstractDict=Dict{String, Any}())
    gpu = gpu_memory_available()
    _write_json_atomic(path, _json_safe(Dict{String, Any}(
        "written" => string(now()), "device" => gpu ? "gpu" : "cpu",
        "gpu" => gpu ? CUDA.name(CUDA.device()) : nothing,
        "gpu_total_mib" => gpu ? round(CUDA.total_memory() / _MIB, digits=1) : nothing,
        "config" => config, "summary" => memory_summary(rows), "epochs" => rows)))
    return path
end

"""The rows of an existing `memory_profile.json` for epochs up to `last_epoch` (the earlier sessions of a resumed run); empty if there is none."""
function load_memory_rows(path::String, last_epoch::Int)
    isfile(path) || return Dict{String, Any}[]
    old = try JSON3.read(read(path, String)) catch; return Dict{String, Any}[] end
    return [Dict{String, Any}(string(k) => v for (k, v) in pairs(r)) for r in get(old, :epochs, []) if r.epoch <= last_epoch]
end

"""Largest relative change of `cache.scale` for which a Huber loss still counts as the same quantity (an NLL is converted exactly)."""
const BEST_SCALE_TOLERANCE = 1e-3

"""
    carried_best(meta, comparable, scale, loss) -> Union{Nothing, NamedTuple{(:loss, :epoch)}}

The best validation loss of an earlier session, taken from the checkpoint's `meta`, so that a resumed run only replaces the
checkpoint when it beats it. `nothing` (start the comparison afresh) when the checkpoint has no record, or when `comparable`
(the validation set: companies, dates, windows, objective) differs from the one stored with it.

The loss is in units of `cache.scale`, which changes with the companies in the cache; a negative log-likelihood is converted
exactly (`NLL_x = NLL_scaled + log scale`), while a Huber loss is carried only when the scale is unchanged within
[`BEST_SCALE_TOLERANCE`](@ref).

# Arguments
- `meta`: the `meta` dict of the checkpoint (see [`load_forecaster`](@ref)).
- `comparable`: what must be identical for two validation losses to be comparable.
- `scale`: `cache.scale` of the current run; `loss`: the objective (`:student_t`, `:gaussian` or `:huber`).

# Returns
`(loss, epoch)` in the current run's units, or `nothing`.
"""
function carried_best(meta::AbstractDict, comparable::AbstractDict, scale::Real, loss::Symbol)
    all(k -> haskey(meta, k), ("best_loss", "best_epoch", "scale", "comparable")) || return nothing
    meta["comparable"] == comparable || return nothing
    old_scale = Float64(meta["scale"])
    old_loss = Float64(meta["best_loss"])
    epoch = Int(meta["best_epoch"])
    if loss === :huber
        abs(old_scale / scale - 1) <= BEST_SCALE_TOLERANCE || return nothing
        return (loss=old_loss, epoch=epoch)
    end
    return (loss=old_loss + log(old_scale) - log(Float64(scale)), epoch=epoch)
end

"""
    accumulated_gradient(net, batches, dev, dtype; loss=:student_t, nu=4, beta=0.5, loss_scale=1) -> Union{Nothing, NamedTuple}

The gradient of the training objective over `batches` (a vector of [`ForecastBatch`](@ref)s, e.g. the micro-batches one batch is split
into), computed one micro-batch at a time so that only one micro-batch's activations are in memory, and summed.

Each micro-batch's return loss is weighted by its share of the counted targets (`sum(W)`) and its timestamp loss by its share of the windows,
so the result equals the gradient of [`forecast_loss`](@ref) on the whole batch at once (up to the random dropout masks and rounding).
The gradient is returned in Float32 and already divided by `loss_scale` (the loss is multiplied by it before differentiating, for a Float16 run).

# Arguments
- `net`: the [`Forecaster`](@ref) on the device `dev` (`Flux.gpu` or `Flux.cpu`), in the number type `dtype`.
- `batches`: the micro-batches, built by `make_batch`.

# Returns
`(grads, loss, ret, time)`: the whole batch's total, return and timestamp losses (`time` as in `forecast_loss`'s third value), or `nothing`
when a micro-batch's loss is not finite.
"""
function accumulated_gradient(net, batches::AbstractVector, dev, dtype::Type; loss::Symbol=:student_t, nu::Real=STUDENT_T_NU,
                              beta::Real=NLL_BETA, loss_scale::Real=1)
    n_total = max(sum(sum(b.W) for b in batches), 1f0)
    windows = sum(size(b.Y, 3) for b in batches)
    scale = Float32(loss_scale)
    acc = nothing; total = 0.0; ret_total = 0.0; time_total = 0.0
    for b in batches
        X, Y, W, T = dev(dtype.(b.X)), dev(b.Y), dev(b.W), dev(b.T)
        wr = Float32(sum(b.W) / n_total); wt = Float32(size(b.Y, 3) / windows)
        res, grads = Flux.withgradient(net) do m
            out = forecast_loss(m, X, Y, W, T; loss=loss, nu=nu, beta=beta)
            ret = wr * out[2]; tim = wt * out[3]
            (scaled=(ret + TIME_LOSS_COEF * tim) * scale, loss=ret + TIME_LOSS_COEF * tim, ret=ret, time=tim)
        end
        isfinite(res.loss) || return nothing
        g = Flux.fmap(x -> x isa AbstractArray ? Float32.(x) ./ scale : x, grads[1])
        acc = acc === nothing ? g : Flux.fmap(_add_grads, acc, g)
        total += res.loss; ret_total += res.ret; time_total += res.time
        b === last(batches) || _free_gpu_garbage(X)
    end
    return (grads=acc, loss=total, ret=ret_total, time=time_total)
end

"""Let the garbage collector release the finished micro-batch's activations before the next one is computed. A CuArray's memory goes back to the pool only
when its Julia object is collected, and Julia's collector is driven by host allocation, not by how full the GPU is, so without this the activations of every
micro-batch would pile up and the saving would be lost. Does nothing on the CPU."""
_free_gpu_garbage(::Array) = nothing
_free_gpu_garbage(::AbstractArray) = (GC.gc(false); nothing)

_add_grads(a, b) = a === nothing ? b : (b === nothing ? a : a .+ b)

"""
    train_forecaster!(model, cache, train_cols, train_starts, eval_sets; kwargs...) -> Dict

Train `model` (moved to `device` and back) on random windows of `cache`.

# Arguments
- `train_cols`: cache columns of the training companies; `train_starts`: the allowed start bars (see [`window_starts`](@ref)).
- `eval_sets`: the [`EvalSet`](@ref)s scored after every epoch; the first decides the best checkpoint.
- `epochs`, `steps_per_epoch`, `batch`: the schedule; `run_info`: stored under `run` in every row of `epoch_log.jsonl` (the script passes the session's first epoch, the cache scale, the date windows and the company sets, so a row says which experiment it belongs to); `memory_profile`: record, per epoch, the GPU memory pool's peak during training and during evaluation (and the step time, the pool and the host's peak), returned as `"memory"`; with a `memory_path` the rows (after those of earlier sessions of the file, up to `epoch_offset`) are also written there, with `memory_config`, after every epoch's evaluation (see [`write_memory_profile`](@ref)); `micro_batches`: split each batch into this many pieces, run one after the other with their gradients summed ([`accumulated_gradient`](@ref)), so peak GPU memory follows `batch ÷ micro_batches` while the update is that of the whole batch; `lr`, `weight_decay`, `warmup_steps`, `clip`: AdamW with gradient-norm clipping.
- `loss`, `nu`, `beta`: the return objective (see [`forecast_loss`](@ref)): `:student_t` (default), `:gaussian` or `:huber`; `nu` the Student-t's degrees of freedom, `beta` the β-NLL exponent.
- `dtype`: the number type of the forward/backward pass (`Float32`, `Float16`, `Float64`). A lower-precision run keeps Float32
  master weights (the optimiser updates those; the Float16 copy is rebuilt after every step) and scales the loss
  (dynamic loss scaling: halved and the step skipped on an overflow, doubled after 500 clean steps).
- `out_dir`: where `forecaster.bson`, `epoch_log.jsonl` and `timing.json` go (empty = write nothing).
- `epoch_offset`: numbering continues from here (for `--resume`).
- `prior_best`: `(loss, epoch)` from [`carried_best`](@ref) when resuming: an epoch then counts as best (and rewrites the checkpoint)
  only if it beats that loss; the model's starting weights are that best checkpoint. `nothing` = the first epoch is the best so far.
- `comparable`: stored with the checkpoint with the best loss and scale, so a later resume can tell whether its validation set matches.
- `stop_file`: creating `STOP` ends after the current epoch, `STOP_NOW` right away.

# Returns
A dict with `best_epoch`, `best_loss`, `epochs_run`, `stopped` and the per-epoch history; the model is left holding the best weights.
"""
function train_forecaster!(model::Forecaster, cache::MinuteCache, train_cols::Vector{Int}, train_starts::UnitRange{Int},
                           eval_sets::Vector{EvalSet};
                           epochs::Int=20, steps_per_epoch::Int=500, batch::Int=32, micro_batches::Int=1, memory_profile::Bool=false, run_info::AbstractDict=Dict{String, Any}(),
                           memory_path::String="", memory_config::AbstractDict=Dict{String, Any}(),
                           lr::Real=3e-4, weight_decay::Real=0.01, warmup_steps::Int=200, clip::Real=1.0,
                           device::Symbol=:cpu, dtype::Type{<:AbstractFloat}=Float32,
                           loss::Symbol=:student_t, nu::Real=STUDENT_T_NU, beta::Real=NLL_BETA, rng::AbstractRNG=Random.default_rng(),
                           out_dir::String="", epoch_offset::Int=0, meta::Dict=Dict{String, Any}(),
                           prior_best::Union{Nothing, NamedTuple{(:loss, :epoch)}}=nothing, comparable::Dict=Dict{String, Any}())
    isempty(train_starts) && throw(MinuteCacheError("no training windows: the training date range is shorter than the window"))
    isempty(eval_sets) && throw(ArgumentError("train_forecaster! needs at least one EvalSet"))
    1 <= micro_batches <= batch || throw(ArgumentError("micro_batches must be between 1 and batch ($batch), got $micro_batches"))
    cfg = forecaster_config(model)
    dev = device === :gpu ? Flux.gpu : Flux.cpu
    master = dev(convert_forecaster(model, Float32))
    low = sizeof(dtype) < 4
    as_dtype(m) = dtype === Float32 ? m : convert_forecaster(m, dtype)
    net = as_dtype(master)
    opt = Flux.setup(Flux.OptimiserChain(Flux.ClipNorm(Float32(clip)), Flux.AdamW(Float32(lr), (0.9f0, 0.999f0), Float32(weight_decay))), master)
    loss_scale = low ? 1024f0 : 1f0
    clean_steps = 0; skipped = 0
    total = epochs * steps_per_epoch
    ckpt = isempty(out_dir) ? "" : joinpath(out_dir, "forecaster.bson")
    log_path = isempty(out_dir) ? "" : joinpath(out_dir, "epoch_log.jsonl")
    timing_path = isempty(out_dir) ? "" : joinpath(out_dir, "timing.json")
    live_path = isempty(out_dir) ? "" : joinpath(out_dir, "live_status.json")
    stop_file = isempty(out_dir) ? "" : joinpath(out_dir, "STOP")
    stop_now_file = isempty(out_dir) ? "" : joinpath(out_dir, "STOP_NOW")
    isempty(out_dir) || mkpath(out_dir)

    best_loss = prior_best === nothing ? Inf : prior_best.loss
    best_epoch = prior_best === nothing ? 0 : prior_best.epoch
    best_state = deepcopy(Flux.state(cpu(master)))
    history = Any[]; stopped = ""; step = 0; epochs_run = 0
    track_gpu = memory_profile && device === :gpu && gpu_memory_available()
    memory_rows = Dict{String, Any}[]
    prior_memory = memory_profile && !isempty(memory_path) ? load_memory_rows(memory_path, epoch_offset) : Dict{String, Any}[]
    t_start = time(); started = string(now()); n_good = 0; loss_sum = 0.0
    function live!(state::String, e::Int, i::Int)
        isempty(live_path) && return nothing
        el = time() - t_start
        _write_json_atomic(live_path, _json_safe(Dict{String, Any}(
            "state" => state, "epoch" => epoch_offset + e, "epochs_total" => epoch_offset + epochs,
            "step_in_epoch" => i, "steps_per_epoch" => steps_per_epoch, "step" => step, "steps_total" => total,
            "running_train_loss" => n_good > 0 ? loss_sum / n_good : nothing,
            "lr" => scheduled_lr(max(step, 1), total, lr, warmup_steps), "loss_scale" => loss_scale, "skipped_steps" => skipped,
            "dtype" => string(dtype), "objective" => string(loss), "device" => string(device),
            "elapsed_secs" => el, "secs_per_step" => step > 0 ? el / step : nothing,
            "eta_secs" => step > 0 ? el / step * (total - step) : nothing,
            "best_epoch" => best_epoch, "best_loss" => best_loss, "started" => started, "updated" => string(now()))))
        return nothing
    end
    live!("starting", 1, 0)
    for e in 1:epochs
        t_epoch = time()
        timer = PhaseTimer()
        Flux.trainmode!(net)
        track_gpu && reset_gpu_peak!()
        loss_sum = 0.0; ret_sum = 0.0; time_sum = 0.0; n_good = 0
        for i in 1:steps_per_epoch
            if !isempty(stop_now_file) && isfile(stop_now_file)
                rm(stop_now_file); stopped = "STOP_NOW"; break
            end
            step += 1
            (i == 1 || i % LIVE_EVERY_STEPS == 0) && live!("training", e, i - 1)
            t0 = tic(timer)
            chunks = Iterators.partition(rand(rng, train_starts, batch), cld(batch, micro_batches))
            bs = [make_batch(cache, train_cols, collect(c), cfg.window) for c in chunks]
            t0 = toc!(timer, "batch", t0)
            acc = accumulated_gradient(net, bs, dev, dtype; loss=loss, nu=nu, beta=beta, loss_scale=loss_scale)
            t0 = toc!(timer, "forward_backward", t0)
            if acc === nothing || !grads_finite(acc.grads)
                low && (loss_scale = max(loss_scale / 2, 1f0))
                clean_steps = 0; skipped += 1
                skipped == 1 && @warn "non-finite loss or gradient at step $step; skipping the update (further skips are only counted)"
                toc!(timer, "optimizer", t0)
                continue
            end
            g32 = acc.grads
            Flux.Optimisers.adjust!(opt; eta=scheduled_lr(step, total, lr, warmup_steps))
            Flux.update!(opt, master, g32)
            if dtype !== Float32
                net = as_dtype(master)
                Flux.trainmode!(net)
            end
            if low && (clean_steps += 1) % 500 == 0
                loss_scale = min(loss_scale * 2, 32768f0)
            end
            toc!(timer, "optimizer", t0)
            loss_sum += acc.loss; ret_sum += acc.ret; time_sum += acc.time; n_good += 1
        end
        stopped == "STOP_NOW" && break
        live!("evaluating", e, steps_per_epoch)
        train_secs = time() - t_epoch
        mem_train = track_gpu ? (pool_peak=gpu_pool_peak(), now=gpu_pool_now().used, peak=nothing) : (pool_peak=nothing, now=nothing, peak=nothing)
        if track_gpu
            GC.gc(true); reset_gpu_peak!()
            probe = [make_batch(cache, train_cols, collect(c), cfg.window)
                     for c in Iterators.partition(rand(MersenneTwister(epoch_offset + e), train_starts, batch), cld(batch, micro_batches))]
            accumulated_gradient(net, probe, dev, dtype; loss=loss, nu=nu, beta=beta, loss_scale=loss_scale)
            mem_train = merge(mem_train, (peak=gpu_pool_peak(),))
            probe = nothing; GC.gc(true); reset_gpu_peak!()
        end
        t0 = tic(timer)
        n_steps = steps_per_epoch
        results = Dict(es.name => evaluate_forecaster(cpu(master), cache, es, cfg.window; batch=batch, device=device, dtype=dtype, loss=loss, nu=nu) for es in eval_sets)
        toc!(timer, "eval", t0)
        if memory_profile
            pool = track_gpu ? gpu_pool_now() : (used=nothing, reserved=nothing)
            free_total = track_gpu ? CUDA.memory_info() : nothing
            push!(memory_rows, memory_row(epoch_offset + e, steps_per_epoch, train_secs, mem_train.peak, mem_train.pool_peak, mem_train.now,
                                          track_gpu ? gpu_pool_peak() : nothing, pool.reserved,
                                          free_total === nothing ? nothing : free_total[2] - free_total[1]))
            isempty(memory_path) || write_memory_profile(memory_path, vcat(prior_memory, memory_rows); config=memory_config)
        end
        primary = results[first(eval_sets).name]
        epochs_run = e
        epoch = epoch_offset + e
        improved = isfinite(primary.loss) && primary.loss < best_loss
        if improved
            best_loss = primary.loss; best_epoch = epoch
            best_state = deepcopy(Flux.state(cpu(master)))
            if !isempty(ckpt)
                Flux.loadmodel!(model, best_state)
                save_forecaster(model, ckpt; meta=merge(meta, Dict{String, Any}(
                    "best_loss" => best_loss, "best_epoch" => best_epoch, "scale" => Float64(cache.scale), "comparable" => comparable)))
            end
        end
        row = (epoch=epoch, step=step, lr=scheduled_lr(step, total, lr, warmup_steps), secs=round(time() - t_epoch, digits=2),
               train_loss=loss_sum / n_steps, train_return_loss=ret_sum / n_steps, train_time_loss=time_sum / n_steps,
               best=improved, run=run_info, dtype=string(dtype), objective=string(loss), loss_scale=loss_scale, skipped_steps=skipped, eval=Dict(k => scalar_metrics(v) for (k, v) in results))
        push!(history, row)
        if !isempty(log_path)
            open(log_path, "a") do io
                JSON3.write(io, _json_safe(row)); println(io)
            end
            _upsert_run_row(timing_path, (iteration=epoch, iter_secs=row.secs, phases=phase_rows(timer)))
        end
        @info @sprintf("epoch %d  train %.4f  %s loss %.4f skill %+.4f (repeat-last %+.4f)  hit %.3f%s%s",
                       epoch, row.train_loss, first(eval_sets).name, primary.loss, primary.skill, primary.skill_last,
                       primary.hit_rate, loss === :huber ? "" : @sprintf("  NLL gain vs const σ %+.4f, vs EWMA σ %+.4f  cov68 %.3f cov95 %.3f",
                           primary.nll_const - primary.nll, primary.nll_ewma - primary.nll, primary.cov68, primary.cov95),
                       improved ? "  *" : "")
        if !isempty(stop_file) && isfile(stop_file)
            rm(stop_file); stopped = "STOP"; break
        end
    end
    live!(isempty(stopped) ? "finished" : "stopped", max(epochs_run, 1), steps_per_epoch)
    Flux.loadmodel!(model, best_state)
    return Dict("best_epoch" => best_epoch, "best_loss" => best_loss, "epochs_run" => epochs_run,
                "stopped" => stopped, "history" => history, "memory" => memory_rows)
end
