"""
Per-decision-step diagnostics for a rollout — one entry for every 15-minute (or
hourly) decision bar, whether or not any trade happened on it. Written for
held-out validation runs (`train_policy!`'s `val_steps_dir`) so the policy's
behaviour can be studied bar by bar: what it believed (probabilities, value),
what it chose, what the rules did with it, and the book state it was looking at.

Arrays are laid out candidate-major `(N, T)` / `(3, N, T)` with `T` decision bars
in time order and candidates in `env.candidate_order` (`symbols`). Probabilities
and weights are stored as `Float16` to keep a year of 15-minute bars to a few MB.
"""

using BSON, JSON3

"""One rollout's per-step record. `filled` is how many of the `T` columns are
populated (all of them after a completed episode)."""
mutable struct StepLog
    symbols         :: Vector{String}
    datetimes       :: Vector{String}        # start time of each decision bar
    probs           :: Array{Float16, 3}     # (3, N, T) HOLD/SELL/BUY probabilities, AFTER the pre-sampling mask
    p_sell_raw      :: Matrix{Float16}       # (N, T)   SELL probability BEFORE the mask (what it "wanted")
    buy_weight      :: Matrix{Float16}       # (N, T)   sigmoid(buy-weight logit)
    action          :: Matrix{Int8}          # (N, T)   chosen action: 1 HOLD, 2 SELL, 3 BUY
    sell_ok         :: Matrix{Bool}          # (N, T)   false = SELL was masked out for this stock
    held            :: Matrix{Bool}          # (N, T)   stock held when the decision was made
    price           :: Matrix{Float32}       # (N, T)   price the decision would fill at
    value           :: Vector{Float32}       # (T)      critic's V(s)
    reward          :: Vector{Float32}       # (T)      reward received for the step (incl. penalties)
    illegal_penalty :: Vector{Float32}       # (T)      the part of it charged for illegal moves
    portfolio_value :: Vector{Float64}       # (T)      after the step
    cash            :: Vector{Float64}       # (T)      after the step
    stocks_value    :: Vector{Float64}       # (T)      after the step
    n_holdings      :: Vector{Int16}         # (T)      distinct stocks held after the step
    n_trades        :: Vector{Int16}         # (T)      executed trades on the step
    filled          :: Int
end

function StepLog(symbols::Vector{String}, T::Int)
    N = length(symbols)
    StepLog(symbols, fill("", T), zeros(Float16, 3, N, T), zeros(Float16, N, T), zeros(Float16, N, T),
            zeros(Int8, N, T), trues(N, T) |> Matrix{Bool}, falses(N, T) |> Matrix{Bool}, zeros(Float32, N, T),
            zeros(Float32, T), zeros(Float32, T), zeros(Float32, T), zeros(Float64, T), zeros(Float64, T),
            zeros(Float64, T), zeros(Int16, T), zeros(Int16, T), 0)
end

"""Write column `t` of `log` (everything known once step `t` has been taken)."""
function record_step!(log::StepLog, t::Int; datetime::String, probs::AbstractMatrix, p_sell_raw::AbstractVector,
                       buy_weight::AbstractVector, action::AbstractVector{<:Integer}, sell_ok::AbstractVector{Bool},
                       held::AbstractVector, price::AbstractVector, value::Real, reward::Real, illegal_penalty::Real,
                       portfolio_value::Real, cash::Real, stocks_value::Real, n_holdings::Integer, n_trades::Integer)
    log.datetimes[t]       = datetime
    log.probs[:, :, t]    .= probs
    log.p_sell_raw[:, t]  .= p_sell_raw
    log.buy_weight[:, t]  .= buy_weight
    log.action[:, t]      .= action
    log.sell_ok[:, t]     .= sell_ok
    log.held[:, t]        .= held .> 0.5f0
    log.price[:, t]       .= price
    log.value[t]           = value
    log.reward[t]          = reward
    log.illegal_penalty[t] = illegal_penalty
    log.portfolio_value[t] = portfolio_value
    log.cash[t]            = cash
    log.stocks_value[t]    = stocks_value
    log.n_holdings[t]      = n_holdings
    log.n_trades[t]        = n_trades
    log.filled             = t
    return nothing
end

"""Save `log` (its filled columns) to `path` as BSON, with `meta` alongside
(iteration, validation return, …). Read it back with `load_val_steps`."""
function save_step_log(path::String, log::StepLog; meta::Dict=Dict())
    T = log.filled
    mkpath(dirname(path))
    BSON.bson(path, Dict{Symbol, Any}(
        :meta => meta, :symbols => log.symbols, :datetimes => log.datetimes[1:T],
        :probs => log.probs[:, :, 1:T], :p_sell_raw => log.p_sell_raw[:, 1:T], :buy_weight => log.buy_weight[:, 1:T],
        :action => log.action[:, 1:T], :sell_ok => log.sell_ok[:, 1:T], :held => log.held[:, 1:T],
        :price => log.price[:, 1:T], :value => log.value[1:T], :reward => log.reward[1:T],
        :illegal_penalty => log.illegal_penalty[1:T], :portfolio_value => log.portfolio_value[1:T],
        :cash => log.cash[1:T], :stocks_value => log.stocks_value[1:T], :n_holdings => log.n_holdings[1:T],
        :n_trades => log.n_trades[1:T]))
    return path
end

"""
Load one validation run's per-step log written by `train_policy!`
(`<val_steps_dir>/iter_NNNNN.bson`).

# Returns
A `NamedTuple` with `meta`, `symbols`, `datetimes` and the arrays described on
`StepLog` — `probs` is `(3, N, T)` (HOLD/SELL/BUY, after the sell mask), the
others `(N, T)` or `(T,)`.

# Example
```julia
s = load_val_steps("website/data/trading_game/val_steps/iter_00042.bson")
mean(s.probs[3, :, :])                       # average P(buy) over every stock and bar
s.datetimes[findall(s.n_trades .> 0)]         # bars on which anything was traded
```
"""
function load_val_steps(path::String)
    d = BSON.load(path)
    return (; (k => d[k] for k in (:meta, :symbols, :datetimes, :probs, :p_sell_raw, :buy_weight, :action,
                                   :sell_ok, :held, :price, :value, :reward, :illegal_penalty, :portfolio_value,
                                   :cash, :stocks_value, :n_holdings, :n_trades))...)
end

"""Per-bar summary of `log` as column arrays, small enough for the website
(`tradinggamelive.html` reads it to fill in the bars on which nothing was traded):
for every decision bar the mean HOLD/SELL/BUY probability over the stocks (`ph`/
`ps`/`pb`, after the sell mask), the most likely BUY and SELL candidates (`mb`/
`mbs`, `ms`/`mss`: probability and 1-based index into `symbols`), how many stocks
the policy picked BUY/SELL for (`n_pick_buy`/`n_pick_sell` — picks the rules may
then have refused), V(s) (`v`), reward (`r`), the illegal-move charge (`pen`), the
book (`pv` portfolio value, `cash`, `nh` holdings) and trades executed (`nt`).
Rounded to 4 decimals; `t` are the bar start times, in the same format as trade
events' `t`."""
function step_summary(log::StepLog; meta::Dict=Dict())
    T = log.filled
    r4(x) = round.(Float64.(x); digits=4)
    pb = Float32.(log.probs[3, :, 1:T]); ps = Float32.(log.probs[2, :, 1:T]); ph = Float32.(log.probs[1, :, 1:T])
    mb_i = [argmax(view(pb, :, t)) for t in 1:T]; ms_i = [argmax(view(ps, :, t)) for t in 1:T]
    return (meta = meta, symbols = log.symbols, t = log.datetimes[1:T],
            ph = r4(vec(sum(ph; dims=1)) ./ size(ph, 1)), ps = r4(vec(sum(ps; dims=1)) ./ size(ps, 1)),
            pb = r4(vec(sum(pb; dims=1)) ./ size(pb, 1)),
            mb = r4([pb[mb_i[t], t] for t in 1:T]), mbs = mb_i, ms = r4([ps[ms_i[t], t] for t in 1:T]), mss = ms_i,
            n_pick_buy = vec(sum(log.action[:, 1:T] .== 3; dims=1)), n_pick_sell = vec(sum(log.action[:, 1:T] .== 2; dims=1)),
            v = r4(log.value[1:T]), r = r4(log.reward[1:T]), pen = r4(log.illegal_penalty[1:T]),
            pv = round.(log.portfolio_value[1:T]; digits=2), cash = round.(log.cash[1:T]; digits=2),
            nh = Int.(log.n_holdings[1:T]), nt = Int.(log.n_trades[1:T]))
end

"""Write `step_summary(log)` as JSON to `path` (via a temp file, so the website never reads a half-written one)."""
function save_step_summary(path::String, log::StepLog; meta::Dict=Dict())
    mkpath(dirname(path))
    tmp = path * ".tmp"
    open(tmp, "w") do io
        JSON3.write(io, step_summary(log; meta=meta))
    end
    mv(tmp, path; force=true)
    return path
end
