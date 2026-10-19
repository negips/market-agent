"""
The v4 minute forecaster: a supervised transformer that reads the last `M` one-minute
states of `N` stocks and predicts, at every minute, the next minute's return of each stock
and the next timestamp. The layer-by-layer mathematics is on `website/mathematics.html`.

    X (8, N, M, B)
      → stock embedding        Dense(8 → d), shared by every stock and minute
      → stock mixing           attention over the N stocks, per minute (no stock positions)
      → fuse                   + W_s · mean over stocks + learned minute position
      → L causal time blocks   attention over the M minutes, per stock (pre-norm, GELU FFN)
      → final LayerNorm
      → return head            Dense(d → 1), shared by every stock
      → time head              Dense(d → 3) on the mean over stocks

Because no weight depends on which company a column holds, the training and validation
companies may differ, and so may `N`.
"""

using Flux, Statistics, BSON, Random

"""
    ForecasterConfig(; window=96, d=64, heads=4, blocks=4, dropout=0.1, ffn_mult=4, bars_per_day=375)

Architecture hyperparameters. `d` must be a multiple of `heads`. `bars_per_day` (375, 75, 25, 7 or 1) is the bar size's number of bars in a
session: the calendar guess of the next timestamp needs it, and no weight depends on it.
"""
Base.@kwdef struct ForecasterConfig
    window   :: Int     = 96
    d        :: Int     = 64
    heads    :: Int     = 4
    blocks   :: Int     = 4
    dropout  :: Float32 = 0.1f0
    ffn_mult :: Int     = 4
    bars_per_day :: Int = 375
end

"""Thrown for an invalid [`ForecasterConfig`](@ref), e.g. a width `d` that is not a multiple of `heads`, a window longer than the model's positions, an unknown `dtype`, or a missing checkpoint."""
struct ForecasterConfigError <: Exception
    msg::String
end
Base.showerror(io::IO, e::ForecasterConfigError) = print(io, "ForecasterConfigError: ", e.msg)

"""One pre-norm causal block: `z + MHA(LN z)`, then `z + FFN(LN z)` with a tanh-GELU."""
struct TimeBlock
    ln1  :: LayerNorm
    attn :: MultiHeadAttention
    ln2  :: LayerNorm
    ff1  :: Dense
    ff2  :: Dense
    drop :: Dropout
end

Flux.@layer TimeBlock

function TimeBlock(cfg::ForecasterConfig)
    d = cfg.d
    return TimeBlock(LayerNorm(d), MultiHeadAttention(d; nheads=cfg.heads, bias=true, dropout_prob=cfg.dropout),
                     LayerNorm(d), Dense(d => cfg.ffn_mult * d), Dense(cfg.ffn_mult * d => d), Dropout(cfg.dropout))
end

"""Added to the attention logits of the time blocks: 0 where a minute may be attended, -1e4 for later minutes
(large enough that `exp` underflows to 0, small enough for Float16, whose largest value is 65,504).
A constant additive bias differentiates about three times faster than `MultiHeadAttention`'s boolean `mask`."""
const CAUSAL_BIAS = -1f4

"""`y` in the number type of `x`. `Dropout` and `LayerNorm` carry Float32 constants (the scale factor, `ϵ`), which would
promote a Float16 activation to Float32."""
_like(y::AbstractArray, x::AbstractArray) = eltype(y) === eltype(x) ? y : convert(AbstractArray{eltype(x)}, y)

_drop(d::Dropout, x::AbstractArray) = _like(d(x), x)
_norm(l::LayerNorm, x::AbstractArray) = _like(l(x), x)

const CRC = Flux.Zygote.ChainRulesCore

"""
    attention_probs(q, k, bias) -> α

Softmax over the keys of the scores `kᵀq (+ bias)`: `q`, `k` are `(dh, L, heads·batch)` (`q` already divided by `√dh`), `bias` is `(L_k, L_q)` or
`nothing`, `α` is `(L_k, L_q, heads·batch)`. It has its own reverse rule, which keeps only `α`, `q` and `k` for the backward pass; differentiating the
steps one by one (scores, bias add, softmax, dropout) also keeps the intermediate score tensors, several times the size of `α`.
"""
function attention_probs(q::AbstractArray{<:Real, 3}, k::AbstractArray{<:Real, 3}, bias)
    scores = Flux.NNlib.batched_mul(Flux.NNlib.batched_transpose(k), q)
    bias === nothing || (scores .+= bias)
    return Flux.NNlib.softmax(scores; dims=1)
end

function CRC.rrule(::typeof(attention_probs), q::AbstractArray{<:Real, 3}, k::AbstractArray{<:Real, 3}, bias)
    α = attention_probs(q, k, bias)
    function attention_probs_pullback(Δ)
        Δs = Flux.NNlib.∇softmax_data(CRC.unthunk(Δ), α; dims=1)
        dq = Flux.NNlib.batched_mul(k, Δs)
        dk = Flux.NNlib.batched_mul(q, Flux.NNlib.batched_transpose(Δs))
        return CRC.NoTangent(), dq, dk, CRC.NoTangent()
    end
    return α, attention_probs_pullback
end

"""
    _attend(attn, x, bias=nothing)

Self-attention over dim 2 of `x = (d, L, rest…)` (the rest are folded into the batch), with the weights of a `MultiHeadAttention` but not its code: the
score tensor is `L × L` per head and per sequence, by far the largest thing in memory, so [`attention_probs`](@ref) keeps one copy of it for the
backward pass instead of three or four, and there is no dropout on the attention probabilities (the residual dropout after the attention stays).
`bias` is added to the scores (`(L_k, L_q)`, e.g. [`CAUSAL_BIAS`](@ref) above the diagonal).
"""
function _attend(attn::MultiHeadAttention, x::AbstractArray, bias=nothing)
    sz = size(x)
    x3 = reshape(x, sz[1], sz[2], :)
    d, L, B = size(x3)
    H = attn.nheads
    dh = d ÷ H
    split(t) = reshape(permutedims(reshape(t, dh, H, L, B), (1, 3, 2, 4)), dh, L, H * B)
    q = split(attn.q_proj(x3)) .* convert(eltype(x3), inv(sqrt(dh)))
    k = split(attn.k_proj(x3))
    v = split(attn.v_proj(x3))
    α = attention_probs(q, k, bias)
    o = Flux.NNlib.batched_mul(v, α)
    o = reshape(permutedims(reshape(o, dh, L, H, B), (1, 3, 2, 4)), d, L, B)
    return reshape(attn.out_proj(o), sz)
end

function (b::TimeBlock)(z, bias)
    y = _attend(b.attn, _norm(b.ln1, z), bias)
    z = z .+ _drop(b.drop, y)
    return z .+ _drop(b.drop, b.ff2(Flux.gelu_tanh.(b.ff1(_norm(b.ln2, z)))))
end

"""The forecaster network; build it with `Forecaster(cfg)`."""
struct Forecaster
    embed     :: Dense
    mix_ln    :: LayerNorm
    mix_attn  :: MultiHeadAttention
    fuse      :: Dense
    pos       :: AbstractMatrix{<:AbstractFloat}
    blocks    :: Vector{TimeBlock}
    final_ln  :: LayerNorm
    ret_head  :: Dense
    time_head :: Dense
    drop      :: Dropout
    bars_per_day :: Int
end

Flux.@layer Forecaster

function Forecaster(cfg::ForecasterConfig; seed::Union{Int, Nothing}=nothing)
    cfg.d % cfg.heads == 0 ||
        throw(ForecasterConfigError("d = $(cfg.d) must be a multiple of heads = $(cfg.heads)"))
    seed === nothing || Random.seed!(seed)
    d = cfg.d
    return Forecaster(Dense(N_FORECAST_FEATURES => d), LayerNorm(d),
                      MultiHeadAttention(d; nheads=cfg.heads, bias=true, dropout_prob=cfg.dropout),
                      Dense(d => d), 0.02f0 .* randn(Float32, d, cfg.window),
                      [TimeBlock(cfg) for _ in 1:cfg.blocks], LayerNorm(d),
                      Dense(d => 2), Dense(d => N_TIME_TARGETS; init=Flux.zeros32), Dropout(cfg.dropout), cfg.bars_per_day)
end

"""The hyperparameters a built network was made with, read back from its layers."""
function forecaster_config(m::Forecaster)::ForecasterConfig
    d = size(m.pos, 1)
    return ForecasterConfig(window=size(m.pos, 2), d=d, heads=m.mix_attn.nheads, blocks=length(m.blocks),
                            dropout=Float32(m.drop.p), ffn_mult=size(m.blocks[1].ff1.weight, 1) ÷ d, bars_per_day=m.bars_per_day)
end

"""Number of trainable parameters."""
n_parameters(m) = sum(length, Flux.trainables(m))

"""
    (model::Forecaster)(X) -> (mu, logsigma, time)

`X` is `(8, N, M, B)` (see [`ForecastBatch`](@ref)) with `M` at most the configured window.
`mu` and `logsigma` are `(M, N, B)`: the location and log-scale of the predicted next-minute return of each stock at each minute
(`logsigma` is bounded, see [`LOGSIGMA_CENTER`](@ref)).
`time` is `(3, M, B)`: the predicted description of the next bar's timestamp, as the calendar guess ([`next_time_prior`](@ref)) plus the head's correction (the bar-of-day part in units of one bar).
Position `j` only depends on minutes `1 … j`.
"""
function (m::Forecaster)(X::AbstractArray{<:Real, 4})
    d, M = size(m.pos, 1), size(X, 3)
    M <= size(m.pos, 2) || throw(ForecasterConfigError("window of $M minutes exceeds the model's $(size(m.pos, 2))"))
    E = m.embed(X)
    E = E .+ _drop(m.drop, _attend(m.mix_attn, _norm(m.mix_ln, E)))
    s = mean(E; dims=2)
    Z = E .+ m.fuse(s) .+ reshape(m.pos[:, 1:M], d, 1, M, 1)
    Z = permutedims(Z, (1, 3, 2, 4))
    bias = Flux.Zygote.ignore_derivatives() do
        ifelse.(Flux.NNlib.make_causal_mask(Z; dims=2), zero(eltype(Z)), eltype(Z)(CAUSAL_BIAS))
    end
    for blk in m.blocks
        Z = blk(Z, bias)
    end
    H = _norm(m.final_ln, Z)
    out = m.ret_head(H)
    mu = dropdims(out[1:1, :, :, :]; dims=1)
    logsigma = LOGSIGMA_CENTER .+ LOGSIGMA_HALFRANGE .* tanh.(dropdims(out[2:2, :, :, :]; dims=1))
    corr = dropdims(m.time_head(mean(H; dims=3)); dims=3)
    time = next_time_prior(X, m.bars_per_day) .+ vcat(corr[1:1, :, :] ./ Float32(m.bars_per_day), corr[2:3, :, :])
    return mu, logsigma, time
end

"""
The calendar's own guess at the next bar's timestamp description, `(3, M, B)`, read off the time features in `X`
(bar-of-day fraction, sin/cos of the weekday): one bar on within a day; at the day's last bar (every bar, for daily bars), the
next weekday with a gap of one day (three after a Friday). Holidays are the only thing the guess misses, so the
timestamp head learns a small correction on top of it, and the minute is exact by construction.
"""
function next_time_prior(X::AbstractArray{<:Real, 4}, bars_per_day::Int=MINUTES_PER_DAY)
    return Flux.Zygote.ignore_derivatives() do
        Xt   = Float32.(X[2:4, :, :, :])
        frac = dropdims(mean(Xt[1:1, :, :, :]; dims=2); dims=2)
        sn   = dropdims(mean(Xt[2:2, :, :, :]; dims=2); dims=2)
        cs   = dropdims(mean(Xt[3:3, :, :, :]; dims=2); dims=2)
        m    = round.(frac .* bars_per_day)
        last = m .>= bars_per_day - 1
        wd   = mod.(round.(atan.(sn, cs) .* (5f0 / Float32(2π))), 5f0)
        next_frac = ifelse.(last, 0f0, (m .+ 1f0) ./ bars_per_day)
        next_wd   = ifelse.(last, mod.(wd .+ 1f0, 5f0), wd) ./ 4f0
        next_gap  = ifelse.(last, log1p.(ifelse.(wd .== 4f0, 3f0, 1f0)), 0f0)
        cat(next_frac, next_gap, next_wd; dims=1)
    end
end

"""Weight of the timestamp loss next to the return loss."""
const TIME_LOSS_COEF = 0.1f0

"""Huber threshold, in units of the return scale (one RMS move)."""
const HUBER_DELTA = 1f0

_huber(r) = ifelse.(abs.(r) .<= HUBER_DELTA, 0.5f0 .* r .^ 2, HUBER_DELTA .* (abs.(r) .- 0.5f0 * HUBER_DELTA))

"""
    forecast_loss(model, X, Y, W, T; loss=:student_t, nu=4, beta=0.5) -> (total, return_loss, time_loss)

The training objective, counted only where `W` is 1 (so forward-filled minutes teach nothing).

- `loss = :student_t` or `:gaussian`: the negative log-likelihood of the next return under the predicted
  location and scale, each term weighted by `σ^(2β)` held constant (β-NLL: without it the model can lower the
  loss by inflating σ where the mean fits badly). `nu` is the Student-t's degrees of freedom.
- `loss = :huber`: Huber loss on the predicted location alone; the scale head is not trained.

`total` adds `TIME_LOSS_COEF` times the mean squared error of the timestamp description (the minute fraction
measured in minutes).
"""
function forecast_loss(model::Forecaster, X, Y, W, T; loss::Symbol=:student_t, nu::Real=STUDENT_T_NU, beta::Real=NLL_BETA)
    mu, logsigma, time = model(X)
    eltype(mu) === Float32 || (mu = Float32.(mu); logsigma = Float32.(logsigma))
    eltype(time) === Float32 || (time = Float32.(time))
    n = max(sum(W), 1f0)
    lr = if loss === :huber
        sum(_huber(mu .- Y) .* W) / n
    else
        nll = forecast_nll(loss, Y, mu, logsigma, nu)
        w = Flux.Zygote.dropgrad(exp.(2f0 * Float32(beta) .* logsigma))
        sum(nll .* w .* W) / n
    end
    e1 = (time[1:1, :, :] .- T[1:1, :, :]) .* Float32(model.bars_per_day)
    e23 = time[2:3, :, :] .- T[2:3, :, :]
    lt = (mean(e1 .^ 2) + 2 * mean(e23 .^ 2)) / 3
    return lr + TIME_LOSS_COEF * lt, lr, lt
end

"""Number types a forecaster can run in, by the name used in `[forecaster] dtype` and `--dtype`."""
const FORECAST_DTYPES = Dict("float32" => Float32, "float16" => Float16, "float64" => Float64)

"""
    forecast_dtype(name) -> Type

The floating-point type called `name` (`"float32"`, `"float16"` or `"float64"`, case-insensitive).
Throws [`ForecasterConfigError`](@ref) for anything else (BFloat16 would need the BFloat16s package).
"""
function forecast_dtype(name::AbstractString)
    haskey(FORECAST_DTYPES, lowercase(strip(name))) ||
        throw(ForecasterConfigError("unknown dtype '$name'; expected one of: $(join(sort(collect(keys(FORECAST_DTYPES))), ", "))"))
    return FORECAST_DTYPES[lowercase(strip(name))]
end

"""
    convert_forecaster(model, T) -> Forecaster

A copy of `model` with every floating-point weight converted to `T` (on whatever device it lives on).
"""
convert_forecaster(model::Forecaster, ::Type{T}) where {T <: AbstractFloat} =
    Flux.fmap(x -> x isa AbstractArray{<:AbstractFloat} ? T.(x) : x, model)

"""Save the weights (CPU, always Float32) and the architecture to BSON."""
function save_forecaster(model::Forecaster, path::String; meta::Dict=Dict{String, Any}())
    mkpath(dirname(path))
    state = Flux.state(convert_forecaster(cpu(model), Float32))
    cfg = forecaster_config(model)
    config = Dict("window" => cfg.window, "d" => cfg.d, "heads" => cfg.heads, "blocks" => cfg.blocks,
                  "dropout" => Float64(cfg.dropout), "ffn_mult" => cfg.ffn_mult, "bars_per_day" => cfg.bars_per_day)
    BSON.@save path state config meta
    return path
end

"""
    warm_start_forecaster(cfg, path; seed=nothing) -> Forecaster

A fresh [`Forecaster`](@ref) built from `cfg` (so its dropout, seed and bar size are the ones asked for now) whose weights are
copied from the checkpoint at `path`. The weights only fit one architecture, so the checkpoint's `window`, `d`, `heads`,
`blocks` and `ffn_mult` must equal `cfg`'s; `dropout` may differ.

# Arguments
- `cfg`: the [`ForecasterConfig`](@ref) of the new run.
- `path`: a checkpoint written by [`save_forecaster`](@ref).
- `seed`: seeds the layers before the weights are overwritten (it only matters for anything the checkpoint does not hold).

# Returns
The warm-started model. Throws [`ForecasterConfigError`](@ref) when the checkpoint is missing or its architecture differs.

# Example
```julia
model = warm_start_forecaster(ForecasterConfig(d=32, dropout=0.2f0), "website/data/forecaster/forecaster.bson")
```
"""
function warm_start_forecaster(cfg::ForecasterConfig, path::String; seed::Union{Int, Nothing}=nothing)
    source, scfg, _ = load_forecaster(path)
    diff = [string(f, " = ", getfield(scfg, f), " in the checkpoint, ", getfield(cfg, f), " asked for")
            for f in (:window, :d, :heads, :blocks, :ffn_mult) if getfield(scfg, f) != getfield(cfg, f)]
    isempty(diff) || throw(ForecasterConfigError("cannot warm-start from $path: " * join(diff, "; ")))
    model = Forecaster(cfg; seed=seed)
    _load_state!(model, Flux.state(source))
    return model
end

_load_state!(model::Forecaster, state::NamedTuple) = Flux.loadmodel!(model,
    haskey(state, :bars_per_day) ? state : merge(state, (bars_per_day=model.bars_per_day,)))

"""Load a [`Forecaster`](@ref) saved by [`save_forecaster`](@ref); returns `(model, config, meta)`."""
function load_forecaster(path::String)
    isfile(path) || throw(ForecasterConfigError("not found: $path"))
    d = BSON.load(path)
    c = d[:config]
    cfg = ForecasterConfig(window=c["window"], d=c["d"], heads=c["heads"], blocks=c["blocks"],
                           dropout=Float32(c["dropout"]), ffn_mult=c["ffn_mult"], bars_per_day=get(c, "bars_per_day", MINUTES_PER_DAY))
    model = Forecaster(cfg)
    _load_state!(model, d[:state])
    return model, cfg, get(d, :meta, Dict{String, Any}())
end
