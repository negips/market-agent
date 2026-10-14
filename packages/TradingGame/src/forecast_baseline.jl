"""
A linear autoregressive baseline for the v4 forecaster.

The question it answers: how much of the transformer's skill and likelihood gain does a plain linear model on the
last few returns of the *same stock* already get? It is fitted and scored on exactly the windows, targets and metrics of
[`evaluate_forecaster`](@ref) (through [`score_forecasts`](@ref)), so the numbers sit side by side.

- **Location**: `μ_t = Σ_{l<k} a_l r_{t-l}`, ridge regression on the `k` latest returns, one set of coefficients shared by all
  stocks (the same weight-sharing as the transformer). Lags before the window's first bar are zero, as the transformer
  also sees nothing before its window.
- **Scale**: each stock has a level `m`, its mean `|r|` over the previous trading day divided by its traded fraction (the mean
  `|r|` of a traded minute). `E|y - μ| / m` is regressed on the 25 minute-of-day buckets (the intraday volatility pattern), the `k`
  latest `|r|/m` and an exponentially weighted `|r|/m`, and the prediction times `m` is turned into the distribution's σ
  (`E|e| = c(ν)·σ`), so the baseline gets the same Student-t or Gaussian likelihood as the network. Volatility is multiplicative
  across stocks, which is why the regression is on the scaled quantities.

It has no cross-stock term and no non-linearity; whatever the network gains beyond it comes from those, from longer memory,
or from the time features it can use better than 25 buckets.
"""

using Random

"""Decay of the exponentially weighted `|r|` feature of the scale regression, per minute."""
const AR_EWMA_DECAY = 0.94f0

"""A stock's traded fraction is floored here when forming its level, so a stock that barely traded does not get an enormous one."""
const AR_MIN_TRADED_FRACTION = 0.02f0

"""The stock level (mean `|r|` of a traded minute, in cache return units) is kept inside this range."""
const AR_LEVEL_RANGE = (0.05f0, 10f0)

"""Default number of lagged returns."""
const AR_DEFAULT_LAGS = 10

"""A fitted baseline: `mean_coef` (`lags`) for the location and `scale_coef` for the scale regression."""
struct ARBaseline
    lags       :: Int
    dist       :: Symbol
    nu         :: Float64
    mean_coef  :: Vector{Float64}
    scale_coef :: Vector{Float64}
end

"""`E|e| / σ` of a standardised Student-t (`:student_t`) or Gaussian (`:gaussian`) with scale 1."""
function mean_abs_factor(dist::Symbol, ν::Real)::Float64
    dist === :gaussian && return sqrt(2 / π)
    dist === :student_t || throw(ForecastLossError("the AR baseline needs a likelihood (gaussian or student_t), got '$dist'"))
    ν > 2 || throw(ForecastLossError("a Student-t needs ν > 2 for a finite variance, got $ν"))
    return exp(log(2) + 0.5 * log(ν) + _loggamma((ν + 1) / 2) - 0.5 * log(π) - log(ν - 1) - _loggamma(ν / 2))
end

"""Return `x` shifted `l` minutes later along the first axis (the first `l` rows are zero)."""
function _lagged(x::AbstractArray{<:Real, 3}, l::Int)
    out = zeros(Float32, size(x))
    l < size(x, 1) && (out[l+1:end, :, :] = x[1:end-l, :, :])
    return out
end

"""
Features of every `(minute, stock, window)` position of `b`, flattened in the order of `vec(b.Y)`: the `(lags, S)` location
features, the `(buckets + lags + 1, S)` scale features (already divided by the stock level) and the `(S,)` stock level `m`.
"""
function _ar_features(b::ForecastBatch, lags::Int, nb::Int=N_SLOT_BUCKETS)
    M, N, B = size(b.Y)
    S = M * N * B
    Fm = zeros(Float32, lags, S)
    absr = abs.(b.R)
    Fs = zeros(Float32, nb + lags + 1, S)
    ma = reshape(b.X[N_TIME_FEATURES + 2, :, 1, :], 1, N, B)
    tf = reshape(b.X[N_TIME_FEATURES + 3, :, 1, :], 1, N, B)
    level = clamp.(ma ./ max.(tf, AR_MIN_TRADED_FRACTION), AR_LEVEL_RANGE[1], AR_LEVEL_RANGE[2])
    m = vec(repeat(level, M, 1, 1))
    for l in 0:lags-1
        Fm[l+1, :] = vec(_lagged(b.R, l))
        Fs[nb+l+1, :] = vec(_lagged(absr, l)) ./ m
    end
    bucket = reshape(Int.(b.slot) .+ 1, M, 1, B)
    @inbounds for (s, bk) in enumerate(vec(repeat(bucket, 1, N, 1)))
        Fs[bk, s] = 1f0
    end
    ewma = similar(absr)
    prev = ma .+ zeros(Float32, 1, N, B)
    for j in 1:M
        prev = AR_EWMA_DECAY .* prev .+ (1f0 - AR_EWMA_DECAY) .* absr[j:j, :, :]
        ewma[j:j, :, :] = prev
    end
    Fs[nb+lags+1, :] = vec(ewma) ./ m
    return Fm, Fs, m
end

"""Solve the ridge normal equations `(XtX + λ·mean(diag)·I) c = Xty`."""
function _ridge_solve(XtX::Matrix{Float64}, Xty::Vector{Float64}, ridge::Real)
    A = copy(XtX)
    λ = ridge * sum(A[i, i] for i in 1:size(A, 1)) / size(A, 1)
    for i in 1:size(A, 1)
        A[i, i] += λ
    end
    return A \ Xty
end

"""
    fit_ar_baseline(cache, cols, starts, window; lags=10, loss=:student_t, nu=4, windows=3000, batch=32, ridge=1e-4, rng) -> ARBaseline

Fit the baseline on `windows` random windows drawn from `starts` (the training range, see [`window_starts`](@ref)) over the cache
columns `cols`. Only counted targets (traded, not the first bar of a day) enter the regressions, as in the network's loss.

# Arguments
- `lags`: how many latest returns the location and the scale regressions see.
- `loss`, `nu`: the likelihood the scale is converted for, `:student_t` (default) or `:gaussian`.
- `ridge`: the ridge penalty relative to the mean diagonal of `XᵀX` (location and scale).

# Returns
An [`ARBaseline`](@ref).

# Example
```julia
cache = load_minute_cache("website/data/minute_cache.bson")
b = fit_ar_baseline(cache, cols, window_starts(cache, Date(2022, 1, 3), Date(2024, 12, 30), 96), 96)
evaluate_ar_baseline(b, cache, EvalSet("val", val_cols, val_starts), 96).skill
```
"""
function fit_ar_baseline(cache::MinuteCache, cols::AbstractVector{Int}, starts::AbstractVector{Int}, window::Int;
                         lags::Int=AR_DEFAULT_LAGS, loss::Symbol=:student_t, nu::Real=STUDENT_T_NU, windows::Int=3000,
                         batch::Int=32, ridge::Real=1e-4, rng::AbstractRNG=Random.default_rng())
    lags >= 1 || throw(ForecastLossError("lags must be at least 1, got $lags"))
    isempty(starts) && throw(MinuteCacheError("no training windows for the AR baseline"))
    c = mean_abs_factor(loss, nu)
    nb = n_slot_buckets(cache)
    nf = nb + lags + 1
    chosen = rand(rng, starts, windows)
    XtX_m = zeros(lags, lags); Xty_m = zeros(lags)
    for chunk in Iterators.partition(chosen, batch)
        b = make_batch(cache, collect(cols), collect(chunk), window)
        sel = vec(b.W) .> 0
        Fm, _, _ = _ar_features(b, lags, nb)
        X = Float64.(Fm[:, sel]); y = Float64.(vec(b.Y)[sel])
        XtX_m .+= X * X'; Xty_m .+= X * y
    end
    mean_coef = _ridge_solve(XtX_m, Xty_m, ridge)
    XtX_s = zeros(nf, nf); Xty_s = zeros(nf)
    for chunk in Iterators.partition(chosen, batch)
        b = make_batch(cache, collect(cols), collect(chunk), window)
        sel = vec(b.W) .> 0
        Fm, Fs, m = _ar_features(b, lags, nb)
        μ = vec(Float32.(mean_coef)' * Fm)
        X = Float64.(Fs[:, sel]); y = Float64.((abs.(vec(b.Y) .- μ) ./ m)[sel])
        XtX_s .+= X * X'; Xty_s .+= X * y
    end
    scale_coef = _ridge_solve(XtX_s, Xty_s, ridge) ./ c
    return ARBaseline(lags, loss, Float64(nu), mean_coef, scale_coef)
end

"""`(P, L)`, the predicted location and log-scale of every position of `b`, each `(M, N, B)`."""
function predict_ar(m::ARBaseline, b::ForecastBatch)
    Fm, Fs, level = _ar_features(b, m.lags, length(m.scale_coef) - m.lags - 1)
    P = reshape(vec(Float32.(m.mean_coef)' * Fm), size(b.Y))
    σ = reshape(level .* vec(Float32.(m.scale_coef)' * Fs), size(b.Y))
    lo, hi = exp(LOGSIGMA_CENTER - LOGSIGMA_HALFRANGE), exp(LOGSIGMA_CENTER + LOGSIGMA_HALFRANGE)
    return P, log.(clamp.(σ, Float32(lo), Float32(hi)))
end

"""
    evaluate_ar_baseline(model, cache, set, window; batch=32) -> NamedTuple

Score `model` on `set` with the same code and fields as [`evaluate_forecaster`](@ref); the timestamp fields are `NaN`
(the baseline does not predict a timestamp).
"""
function evaluate_ar_baseline(m::ARBaseline, cache::MinuteCache, set::EvalSet, window::Int; batch::Int=32)
    Ps = Array{Float32, 3}[]; Ls = Array{Float32, 3}[]; Ys = Array{Float32, 3}[]; Ws = Array{Float32, 3}[]
    Rs = Array{Float32, 3}[]; Sl = Matrix{Int16}[]
    for chunk in Iterators.partition(set.starts, batch)
        b = make_batch(cache, set.cols, collect(chunk), window)
        P, L = predict_ar(m, b)
        push!(Ps, P); push!(Ls, L); push!(Ys, b.Y); push!(Ws, b.W); push!(Rs, b.R); push!(Sl, b.slot)
    end
    r = score_forecasts(cat(Ps...; dims=3), cat(Ls...; dims=3), cat(Ys...; dims=3), cat(Ws...; dims=3), cat(Rs...; dims=3), hcat(Sl...);
                        loss=m.dist, nu=m.nu, n_buckets=n_slot_buckets(cache))
    return merge((n_windows=length(set.starts),), r, (minute_acc=NaN, gap_acc=NaN, weekday_acc=NaN))
end
