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

using Flux, Statistics, JSON3, Random, Printf

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

"""
    train_forecaster!(model, cache, train_cols, train_starts, eval_sets; kwargs...) -> Dict

Train `model` (moved to `device` and back) on random windows of `cache`.

# Arguments
- `train_cols`: cache columns of the training companies; `train_starts`: the allowed start bars (see [`window_starts`](@ref)).
- `eval_sets`: the [`EvalSet`](@ref)s scored after every epoch; the first decides the best checkpoint.
- `epochs`, `steps_per_epoch`, `batch`: the schedule; `lr`, `weight_decay`, `warmup_steps`, `clip`: AdamW with gradient-norm clipping.
- `loss`, `nu`, `beta`: the return objective (see [`forecast_loss`](@ref)): `:student_t` (default), `:gaussian` or `:huber`; `nu` the Student-t's degrees of freedom, `beta` the β-NLL exponent.
- `dtype`: the number type of the forward/backward pass (`Float32`, `Float16`, `Float64`). A lower-precision run keeps Float32
  master weights (the optimiser updates those; the Float16 copy is rebuilt after every step) and scales the loss
  (dynamic loss scaling: halved and the step skipped on an overflow, doubled after 500 clean steps).
- `out_dir`: where `forecaster.bson`, `epoch_log.jsonl` and `timing.json` go (empty = write nothing).
- `epoch_offset`: numbering continues from here (for `--resume`).
- `stop_file`: creating `STOP` ends after the current epoch, `STOP_NOW` right away.

# Returns
A dict with `best_epoch`, `best_loss`, `epochs_run`, `stopped` and the per-epoch history; the model is left holding the best weights.
"""
function train_forecaster!(model::Forecaster, cache::MinuteCache, train_cols::Vector{Int}, train_starts::UnitRange{Int},
                           eval_sets::Vector{EvalSet};
                           epochs::Int=20, steps_per_epoch::Int=500, batch::Int=32,
                           lr::Real=3e-4, weight_decay::Real=0.01, warmup_steps::Int=200, clip::Real=1.0,
                           device::Symbol=:cpu, dtype::Type{<:AbstractFloat}=Float32,
                           loss::Symbol=:student_t, nu::Real=STUDENT_T_NU, beta::Real=NLL_BETA, rng::AbstractRNG=Random.default_rng(),
                           out_dir::String="", epoch_offset::Int=0, meta::Dict=Dict{String, Any}())
    isempty(train_starts) && throw(MinuteCacheError("no training windows: the training date range is shorter than the window"))
    isempty(eval_sets) && throw(ArgumentError("train_forecaster! needs at least one EvalSet"))
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

    best_loss = Inf; best_epoch = 0; best_state = deepcopy(Flux.state(cpu(master)))
    history = Any[]; stopped = ""; step = 0; epochs_run = 0
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
        loss_sum = 0.0; ret_sum = 0.0; time_sum = 0.0; n_good = 0
        for i in 1:steps_per_epoch
            if !isempty(stop_now_file) && isfile(stop_now_file)
                rm(stop_now_file); stopped = "STOP_NOW"; break
            end
            step += 1
            (i == 1 || i % LIVE_EVERY_STEPS == 0) && live!("training", e, i - 1)
            t0 = tic(timer)
            b = make_batch(cache, train_cols, rand(rng, train_starts, batch), cfg.window)
            X, Y, W, T = dev(dtype.(b.X)), dev(b.Y), dev(b.W), dev(b.T)
            t0 = toc!(timer, "batch", t0)
            res, grads = Flux.withgradient(net) do m
                out = forecast_loss(m, X, Y, W, T; loss=loss, nu=nu, beta=beta)
                (scaled=out[1] * loss_scale, loss=out[1], ret=out[2], time=out[3])
            end
            t0 = toc!(timer, "forward_backward", t0)
            g32 = Flux.fmap(x -> x isa AbstractArray ? Float32.(x) ./ loss_scale : x, grads[1])
            if !isfinite(res.loss) || !grads_finite(g32)
                low && (loss_scale = max(loss_scale / 2, 1f0))
                clean_steps = 0; skipped += 1
                skipped == 1 && @warn "non-finite loss or gradient at step $step; skipping the update (further skips are only counted)"
                toc!(timer, "optimizer", t0)
                continue
            end
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
            loss_sum += res.loss; ret_sum += res.ret; time_sum += res.time; n_good += 1
        end
        stopped == "STOP_NOW" && break
        live!("evaluating", e, steps_per_epoch)
        t0 = tic(timer)
        n_steps = steps_per_epoch
        results = Dict(es.name => evaluate_forecaster(cpu(master), cache, es, cfg.window; batch=batch, device=device, dtype=dtype, loss=loss, nu=nu) for es in eval_sets)
        toc!(timer, "eval", t0)
        primary = results[first(eval_sets).name]
        epochs_run = e
        epoch = epoch_offset + e
        improved = isfinite(primary.loss) && primary.loss < best_loss
        if improved
            best_loss = primary.loss; best_epoch = epoch
            best_state = deepcopy(Flux.state(cpu(master)))
            if !isempty(ckpt)
                Flux.loadmodel!(model, best_state)
                save_forecaster(model, ckpt; meta=meta)
            end
        end
        row = (epoch=epoch, step=step, lr=scheduled_lr(step, total, lr, warmup_steps), secs=round(time() - t_epoch, digits=2),
               train_loss=loss_sum / n_steps, train_return_loss=ret_sum / n_steps, train_time_loss=time_sum / n_steps,
               best=improved, dtype=string(dtype), objective=string(loss), loss_scale=loss_scale, skipped_steps=skipped, eval=Dict(k => scalar_metrics(v) for (k, v) in results))
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
                "stopped" => stopped, "history" => history)
end
