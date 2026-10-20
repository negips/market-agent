"""
legendre_projection.jl

Projects one company's price series onto the first N Legendre polynomials and reports how faithful the reconstruction is.
Used by the ops server (`/api/legendre`, the Custom Plots page) and runnable by itself:

    julia scripts/tools/legendre_projection.jl --symbol RELIANCE --n 20 [--granularity 15min] [--from 2024-07-01] [--to 2025-06-30] [--kind price|logprice]

The series is indexed by bar (position 1…M along the cache's trading-day axis), mapped linearly onto [-1, 1], and
fitted by least squares on P_0 … P_{N-1}. On a uniform grid that is the discrete orthogonal projection onto the span
of those polynomials. The fit goes through a QR factorisation of the Legendre design matrix, so the coefficients stay
accurate for N in the hundreds, and one factorisation also gives the error for every smaller N.
"""

using LinearAlgebra, Statistics, Dates, BSON

const MAX_LEGENDRE_N = 500
const MAX_FIT_POINTS = 20_000
const MAX_PLOT_POINTS = 1_500
const CURVE_MAX_N = 200

"""Raised when the series cannot be projected (too few points, N out of range, …)."""
struct LegendreError <: Exception
    msg::String
end
Base.showerror(io::IO, e::LegendreError) = print(io, "LegendreError: ", e.msg)

"""
    legendre_design(x, n) -> Matrix{Float64}

The `length(x) × n` matrix whose column `k` is `P_{k-1}(x)`, built with the three-term recurrence
`(k+1) P_{k+1} = (2k+1) x P_k − k P_{k−1}`.

# Arguments
- `x`: points in `[-1, 1]`.
- `n`: number of polynomials (degrees `0 … n-1`).

# Example
```julia
A = legendre_design(range(-1, 1; length=5), 3)   # columns 1, x, (3x²−1)/2
```
"""
function legendre_design(x::AbstractVector{<:Real}, n::Integer)
    n >= 1 || throw(LegendreError("N must be at least 1"))
    A = Matrix{Float64}(undef, length(x), n)
    A[:, 1] .= 1.0
    n >= 2 && (A[:, 2] .= x)
    for k in 1:n-2
        @views A[:, k + 2] .= ((2k + 1) .* x .* A[:, k + 1] .- k .* A[:, k]) ./ (k + 1)
    end
    return A
end

"""
    legendre_project(y, n; curve_max_n=CURVE_MAX_N) -> NamedTuple

Least-squares projection of `y` (sampled at equally spaced positions) onto `P_0 … P_{n-1}`.

# Arguments
- `y`: the series (finite values only).
- `n`: dimension of the basis, `1 ≤ n < length(y)` and `n ≤ MAX_LEGENDRE_N`.

# Returns
`(fit, coefficients, rmse_by_n)`: the reconstruction at every point of `y`, the coefficients on `P_0 … P_{n-1}`
(in the unnormalised basis), and the RMS error after keeping the first `1, 2, …` basis vectors (up to `max(n, curve_max_n)`,
capped by the series length), from the same factorisation.

# Example
```julia
r = legendre_project(cumsum(randn(500)), 20)
r.fit, r.coefficients, r.rmse_by_n[20]
```
"""
function legendre_project(y::AbstractVector{<:Real}, n::Integer; curve_max_n::Integer=CURVE_MAX_N)
    m = length(y)
    m >= 3 || throw(LegendreError("need at least 3 points, got $m"))
    1 <= n <= MAX_LEGENDRE_N || throw(LegendreError("N must be between 1 and $MAX_LEGENDRE_N"))
    n < m || throw(LegendreError("N ($n) must be smaller than the number of points ($m)"))
    all(isfinite, y) || throw(LegendreError("the series holds non-finite values"))
    x = range(-1.0, 1.0; length=m)
    kmax = min(m - 1, max(n, curve_max_n))
    A = legendre_design(x, kmax)
    F = qr(A)
    Q = Matrix(F.Q)
    qty = Q' * collect(Float64, y)
    coefficients = (F.R[1:n, 1:n]) \ qty[1:n]
    fit = A[:, 1:n] * coefficients
    total = sum(abs2, y)
    residual = [max(total - sum(abs2, @view qty[1:k]), 0.0) for k in 1:kmax]
    rmse_by_n = sqrt.(residual ./ m)
    return (fit=fit, coefficients=coefficients, rmse_by_n=rmse_by_n)
end

"""
    legendre_report(price, n; kind="price", plot_points=MAX_PLOT_POINTS) -> Dict

Projects a price series and describes the result.

# Arguments
- `price`: closes in rupees, one per bar; `NaN` (no price yet) and non-positive values are dropped from the front.
- `n`: dimension of the Legendre basis.
- `kind`: `"price"` fits the price itself, `"logprice"` fits `ln(price)` and exponentiates the fit back.
- `plot_points`: at most this many points are returned for plotting.

# Returns
A `Dict` with `index` (0-based positions in `price` of the plotted points), `actual`, `fit`, `coefficients`, `metrics`
and `error_curve` (RMS error as a percentage of the mean price after keeping `1…K` terms). `metrics` holds `r2`,
`rmse`, `rmse_pct`, `max_abs`, `max_abs_pct`, `return_variance_kept` (variance of the fit's step log-returns
over the variance of the actual ones; sampled steps when the series was thinned), `step_corr` (correlation between those step returns),
`points` (bars in the range), `fit_points` (bars used in the fit) and `stride`.

# Example
```julia
legendre_report(100 .* exp.(cumsum(0.01 .* randn(500))), 12)["metrics"]["r2"]
```
"""
function legendre_report(price::AbstractVector{<:Real}, n::Integer; kind::AbstractString="price", plot_points::Integer=MAX_PLOT_POINTS)
    kind in ("price", "logprice") || throw(LegendreError("kind must be price or logprice, not $kind"))
    first_ok = findfirst(p -> isfinite(p) && p > 0, price)
    first_ok === nothing && throw(LegendreError("the range holds no price"))
    p = Float64.(price[first_ok:end])
    any(q -> !isfinite(q) || q <= 0, p) && throw(LegendreError("the price series has gaps after its first value"))
    stride = cld(length(p), MAX_FIT_POINTS)
    pos = collect(1:stride:length(p))
    y = kind == "price" ? p[pos] : log.(p[pos])
    r = legendre_project(y, n)
    actual = p[pos]
    fit = kind == "price" ? r.fit : exp.(r.fit)
    err = actual .- fit
    scale = mean(actual)
    sstot = sum(abs2, y .- mean(y))
    fity = kind == "price" ? fit : log.(fit)
    sse = sum(abs2, y .- fity)
    steps_a, steps_f = diff(log.(actual)), diff(log.(max.(fit, eps())))
    va = var(steps_a)
    kept = va > 0 ? var(steps_f) / va : nothing
    sc = std(steps_a) > 0 && std(steps_f) > 0 ? cor(steps_a, steps_f) : nothing
    thin = cld(length(pos), plot_points)
    sel = collect(1:thin:length(pos))
    curve = [(k, 100 * r.rmse_by_n[k] / (kind == "price" ? scale : 1.0)) for k in 1:length(r.rmse_by_n)]
    metrics = Dict{String,Any}("r2" => sstot > 0 ? 1 - sse / sstot : nothing, "rmse" => sqrt(mean(abs2, err)),
                               "rmse_pct" => 100 * sqrt(mean(abs2, err)) / scale, "max_abs" => maximum(abs, err),
                               "max_abs_pct" => 100 * maximum(abs, err) / scale, "return_variance_kept" => kept, "step_corr" => sc,
                               "points" => length(p), "fit_points" => length(pos), "stride" => stride)
    return Dict{String,Any}("index" => (first_ok - 1) .+ (pos[sel] .- 1), "actual" => actual[sel], "fit" => fit[sel],
                            "coefficients" => r.coefficients, "metrics" => metrics, "kind" => kind, "n" => n,
                            "error_curve" => [[k, e] for (k, e) in curve])
end

function _main(args)
    if "--help" in args || "-h" in args
        println("""
        Usage: julia scripts/tools/legendre_projection.jl --symbol SYM --n N [--granularity 15min] [--from DATE] [--to DATE] [--kind price|logprice]

        Projects the symbol's close-price series (from website/data/forecaster_cache_G.bson) onto the first N Legendre
        polynomials and prints the reconstruction error. The ops server serves the same computation at /api/legendre.""")
        return
    end
    known = ("--symbol", "--n", "--granularity", "--from", "--to", "--kind")
    for a in args
        startswith(a, "--") && !(a in known) && (println(stderr, "ERROR: unknown flag $a; known: $(join(known, ", "))"); exit(2))
    end
    opt(k, d) = (i = findfirst(==(k), args); i === nothing || i == length(args) ? d : args[i + 1])
    sym = opt("--symbol", ""); isempty(sym) && (println(stderr, "ERROR: --symbol is required"); exit(2))
    gran = opt("--granularity", "15min")
    d = BSON.load(joinpath(@__DIR__, "..", "..", "website", "data", "forecaster_cache_$(gran).bson"))
    k = findfirst(==(sym), Vector{String}(d[:symbols])); k === nothing && (println(stderr, "ERROR: $sym is not in the $gran cache"); exit(2))
    isempty(d[:price]) && (println(stderr, "ERROR: this cache holds no prices; rebuild it"); exit(2))
    dates = Vector{Date}(d[:dates]); bpd = size(d[:logret], 1) ÷ length(dates)
    from, to = opt("--from", ""), opt("--to", "")
    keep = [(isempty(from) || string(dates[(i - 1) ÷ bpd + 1]) >= from) && (isempty(to) || string(dates[(i - 1) ÷ bpd + 1]) <= to) for i in 1:size(d[:price], 1)]
    rep = legendre_report(Float64.(d[:price][keep, k]), parse(Int, opt("--n", "20")); kind=opt("--kind", "price"))
    for (key, v) in sort(collect(rep["metrics"])); println(rpad(key, 22), v); end
    println("first coefficients: ", round.(rep["coefficients"][1:min(6, end)]; digits=3))
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    _main(ARGS)
end
