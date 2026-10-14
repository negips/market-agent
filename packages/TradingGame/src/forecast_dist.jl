"""
The predictive distribution of the v4 forecaster: for every stock and minute the network gives a
location `μ` (the predicted next return, in units of the cache's return scale) and `log σ`, and the
return is modelled as Student-t (heavy tails, the default) or Gaussian with that location and scale.

For a Student-t with `ν` degrees of freedom the standard deviation is `σ·sqrt(ν/(ν-2))`, so `ν` must
exceed 2. This file holds the likelihoods and the interval widths used by the loss and by the
calibration report; none of it depends on the network.
"""

"""Loss/distribution names accepted by `[forecaster] loss` and `--loss`."""
const FORECAST_LOSSES = (:huber, :gaussian, :student_t)

"""Default degrees of freedom of the Student-t likelihood."""
const STUDENT_T_NU = 4f0

"""Default exponent of the β-NLL weight `σ^(2β)` (0 = plain NLL, 1 = the mean is fitted as under squared error)."""
const NLL_BETA = 0.5f0

"""`log σ` is `LOGSIGMA_CENTER + LOGSIGMA_HALFRANGE · tanh(s)`, so σ stays between about 0.05 and 7 return units."""
const LOGSIGMA_CENTER = -0.5f0
const LOGSIGMA_HALFRANGE = 2.5f0

"""Thrown for an unknown `loss` name or an invalid Student-t `ν` (must exceed 2)."""
struct ForecastLossError <: Exception
    msg::String
end
Base.showerror(io::IO, e::ForecastLossError) = print(io, "ForecastLossError: ", e.msg)

"""`:huber`, `:gaussian` or `:student_t` from a string or symbol; throws [`ForecastLossError`](@ref) otherwise."""
function forecast_loss_kind(name)::Symbol
    s = Symbol(lowercase(strip(string(name))))
    s in FORECAST_LOSSES || throw(ForecastLossError("unknown loss '$name'; expected one of: $(join(FORECAST_LOSSES, ", "))"))
    return s
end

"""`log Γ(x)` for `x > 0` (shift above 10, then Stirling's series; accurate to about 1e-10)."""
function _loggamma(x::Real)
    x > 0 || throw(DomainError(x, "log Γ needs x > 0"))
    shift = 0.0
    while x < 10
        shift += log(x)
        x += 1
    end
    inv = 1 / x
    series = inv / 12 - inv^3 / 360 + inv^5 / 1260 - inv^7 / 1680
    return (x - 0.5) * log(x) - x + 0.5 * log(2π) + series - shift
end

"""Constant of the Student-t log-density, `log Γ(ν/2) - log Γ((ν+1)/2) + ½ log(νπ)`."""
student_t_const(ν::Real) = _loggamma(ν / 2) - _loggamma((ν + 1) / 2) + 0.5 * log(ν * π)

"""
    forecast_nll(dist, y, μ, logσ, ν) -> array

Elementwise negative log-likelihood of `y` under `dist` (`:student_t` or `:gaussian`) with location `μ`,
log-scale `logσ` and degrees of freedom `ν` (ignored for the Gaussian). Works on CPU and GPU arrays.
"""
function forecast_nll(dist::Symbol, y, μ, logσ, ν::Real)
    z2 = ((y .- μ) .* exp.(-logσ)) .^ 2
    if dist === :student_t
        c = Float32(student_t_const(ν))
        return logσ .+ (0.5f0 * (Float32(ν) + 1f0)) .* log1p.(z2 ./ Float32(ν)) .+ c
    elseif dist === :gaussian
        return logσ .+ 0.5f0 .* z2 .+ Float32(0.5 * log(2π))
    end
    throw(ForecastLossError("no likelihood for '$dist'"))
end

"""Standard deviation of the distribution with scale 1 (`sqrt(ν/(ν-2))` for the Student-t, 1 for the Gaussian)."""
function unit_std(dist::Symbol, ν::Real)
    dist === :student_t || return 1.0
    ν > 2 || throw(ForecastLossError("a Student-t needs ν > 2 for a finite variance, got $ν"))
    return sqrt(ν / (ν - 2))
end

function _density(dist::Symbol, ν::Real)
    if dist === :student_t
        c = exp(-student_t_const(ν))
        return x -> c * (1 + x^2 / ν)^(-(ν + 1) / 2)
    end
    return x -> exp(-x^2 / 2) / sqrt(2π)
end

"""
    central_halfwidth(dist, ν, p) -> Float64

The `h` with `P(|Z| ≤ h) = p` for a unit-scale `dist`, by numerical integration of its density and bisection.
Used to count how often a target falls inside the model's own `p` interval.
"""
function central_halfwidth(dist::Symbol, ν::Real, p::Real)
    f = _density(dist, ν)
    mass(h) = begin
        n = 2000
        dx = h / n
        s = f(0.0) + f(h)
        for i in 1:n - 1
            s += (isodd(i) ? 4 : 2) * f(i * dx)
        end
        2 * s * dx / 3
    end
    lo, hi = 0.0, 200.0
    for _ in 1:60
        mid = (lo + hi) / 2
        mass(mid) < p ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end
