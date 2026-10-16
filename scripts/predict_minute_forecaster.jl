"""
predict_minute_forecaster.jl

Run a trained v4 forecaster on one section of the minute cache and write what it predicts, with
the prices that actually followed, for website/predict.html (Forecaster v4 tab).

The section is the `window` (96) one-minute bars ending at --date/--time, for the chosen stocks.
The model predicts, at every bar, the next minute's return as a distribution; the page shows the
last prediction (the *next* price vector: mean and a 95% interval for every stock) and, because
the model is causal, its prediction at each earlier bar of the section too. The actual closes
come from the 1-minute CSVs, forward-filled the same way the cache is.

Usage:
  julia --project=packages/TradingGame scripts/predict_minute_forecaster.jl --date 2025-06-12 --time 12:00
  julia --project=packages/TradingGame scripts/predict_minute_forecaster.jl --date 2025-06-12 --symbols RECLTD,RVNL,INDIGO
  julia --project=packages/TradingGame scripts/predict_minute_forecaster.jl --config configs/bse_v3.toml --date 2025-06-12 --device gpu

Options:
  --config PATH       TOML run file; reads [forecaster_cache] (granularity, exchange, out) and [forecaster] (device)
  --date DATE         Day of the last input bar (yyyy-mm-dd); must be a day on the cache's axis (required)
  --time HH:MM        Time of the last input bar, 09:15 to 15:29 (default 12:00); it picks the bar that contains it (ignored for daily)
  --symbols A,B,...   Stocks to forecast together (default: the first --n validation companies in the cache)
  --n N               How many validation companies when --symbols is not given (default 8)
  --granularity G     1min (default), 5min, 15min, hourly or daily: picks the cache forecaster_cache_G.bson and the checkpoint
                      website/data/forecaster[_G]/forecaster.bson unless --cache/--checkpoint are given
  --checkpoint PATH   Forecaster checkpoint (default website/data/forecaster/forecaster.bson)
  --out PATH          Result file (default website/data/forecast_prediction.json for 1min, forecast_prediction_{granularity}.json otherwise)
  --exchange nse|bse  Which ohlcv/{exchange} tree holds the CSVs (default bse)
  --cache PATH        Forecaster cache (default website/data/forecaster_cache_{granularity}.bson)
  --device cpu|gpu    Falls back to cpu when CUDA is not functional (default cpu)
  --horizon Y         Also forecast the next Y bars (default 1 = the next bar only; at most 2880 minutes = 48 hours of bars,
                      about 7.7 trading days, and no further than the minute cache reaches). Each step is generated from the
                      previous one: a return is drawn from the predicted distribution, appended to the window, and the model
                      is run again. The model was not trained on overnight moves, so the first minute of each later day is a
                      draw from that stock's own past overnight gaps (the last 250 days before the section) instead
  --paths S           Sample paths for the multi-step forecast (default 200); the bands are their quantiles
  --save-paths K      How many of the sample paths to write to the file for the page to draw (default 50; the page shows as many as you pick)
  --seed N            Random seed of the sample paths
  --levels P,Q,...    Central interval probabilities to compute (default 0.68,0.95)

In the Julia REPL, include the file (it only runs `main` when called as a script) and keep the cache and model loaded:

  julia --project=packages/TradingGame
  julia> include("scripts/predict_minute_forecaster.jl")
  julia> s = load_session(device="gpu")
  julia> r = predict_section(s; date=Date(2025, 6, 12), time="12:00", symbols=["RECLTD", "RVNL"])
  julia> show_next(r)               # the next-bar vector as a table
  julia> save_prediction(r)         # write website/data/forecast_prediction.json for the page

All stocks of one run are forecast in the same window, as in training (the model mixes across them),
so the result depends on which stocks you list together.
"""

using TradingGame, Dates, JSON3, CUDA, Flux, Random, Statistics

include(joinpath(@__DIR__, "run_config.jl"))
using .RunConfigs

const REPO_ROOT     = normpath(joinpath(@__DIR__, ".."))
const DATA_DIR      = joinpath(REPO_ROOT, "website", "data")
const UNIVERSE_FILE = joinpath(DATA_DIR, "trading_game", "universe_latest.json")
const DEFAULT_TIME  = "12:00"
const DEFAULT_N     = 8
const CSV_MARGIN_DAYS = 4
const DEFAULT_PATHS = 200
const SAVED_PATHS   = 50
const MAX_HORIZON   = 48 * 60
const GAP_HISTORY_DAYS = 250
const PROGRESS_EVERY   = 250
const N_CTX_ROWS    = (2 + TradingGame.N_TIME_FEATURES):(1 + TradingGame.N_TIME_FEATURES + TradingGame.N_CONTEXT_FEATURES)

"""Raised for an unusable section: a date or time off the cache's axis, or a window that runs out of history."""
struct PredictSectionError <: Exception
    msg::String
end
Base.showerror(io::IO, e::PredictSectionError) = print(io, "PredictSectionError: ", e.msg)

"""The settings every entry point starts from (the command line and the keyword arguments of [`predict_section`](@ref) override them)."""
default_cache_path(g::String) = normpath(joinpath(DATA_DIR, "forecaster_cache_$g.bson"))
default_out_path(g::String) = normpath(joinpath(DATA_DIR, g == "1min" ? "forecast_prediction.json" : "forecast_prediction_$g.json"))
default_checkpoint_path(g::String) = normpath(joinpath(DATA_DIR, g == "1min" ? "forecaster" : "forecaster_$g", "forecaster.bson"))

default_opts() = Dict{String, Any}(
    "exchange" => "bse", "granularity" => "1min", "cache" => nothing, "device" => "cpu",
    "checkpoint" => nothing,
    "out" => nothing,
    "date" => nothing, "time" => DEFAULT_TIME, "symbols" => String[], "n" => DEFAULT_N, "levels" => [0.68, 0.95], "horizon" => 1, "paths" => DEFAULT_PATHS, "save_paths" => SAVED_PATHS, "seed" => nothing)

function parse_args()
    config_path = RunConfigs.take_config_arg!(ARGS)
    cfg = config_path === nothing ? nothing : RunConfigs.load_or_exit(config_path)
    opts = default_opts()
    mc = RunConfigs.section_opts(cfg, "forecaster_cache")
    haskey(mc, "granularity") && (opts["granularity"] = mc["granularity"])
    haskey(mc, "exchange") && (opts["exchange"] = mc["exchange"])
    haskey(mc, "out") && (opts["cache"] = abspath(joinpath(REPO_ROOT, mc["out"])))
    fc = RunConfigs.section_opts(cfg, "forecaster")
    haskey(fc, "device") && (opts["device"] = fc["device"])
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a in ("-h", "--help")
            println(match(r"\"\"\"\n(.*?)\"\"\""s, read(@__FILE__, String)).captures[1])
            exit(0)
        elseif a == "--date";       opts["date"] = Date(ARGS[i+1]); i += 2
        elseif a == "--time";       opts["time"] = ARGS[i+1]; i += 2
        elseif a == "--symbols";    opts["symbols"] = String.(strip.(split(uppercase(ARGS[i+1]), ','; keepempty=false))); i += 2
        elseif a == "--n";          opts["n"] = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--checkpoint"; opts["checkpoint"] = abspath(ARGS[i+1]); i += 2
        elseif a == "--out";        opts["out"] = abspath(ARGS[i+1]); i += 2
        elseif a == "--granularity"; opts["granularity"] = lowercase(ARGS[i+1]); i += 2
        elseif a == "--exchange";   opts["exchange"] = lowercase(ARGS[i+1]); i += 2
        elseif a == "--cache";      opts["cache"] = abspath(ARGS[i+1]); i += 2
        elseif a == "--device";     opts["device"] = ARGS[i+1]; i += 2
        elseif a == "--horizon";    opts["horizon"] = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--paths";      opts["paths"] = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--save-paths"; opts["save_paths"] = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--seed";       opts["seed"] = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--levels";     opts["levels"] = parse.(Float64, split(ARGS[i+1], ','; keepempty=false)); i += 2
        else; error("Unknown argument: $a")
        end
    end
    opts["date"] === nothing && error("--date is required (a trading day in the minute cache, yyyy-mm-dd)")
    opts["exchange"] in ("nse", "bse") || error("--exchange must be nse or bse")
    haskey(GRANULARITIES, opts["granularity"]) || error("--granularity must be one of $(join(sort(collect(keys(GRANULARITIES))), ", "))")
    opts["cache"] === nothing && (opts["cache"] = default_cache_path(opts["granularity"]))
    opts["checkpoint"] === nothing && (opts["checkpoint"] = default_checkpoint_path(opts["granularity"]))
    opts["out"] === nothing && (opts["out"] = default_out_path(opts["granularity"]))
    opts["device"] in ("cpu", "gpu") || error("--device must be cpu or gpu")
    opts["horizon"] >= 1 || error("--horizon must be at least 1")
    opts["paths"] >= 2 || error("--paths must be at least 2")
    !isempty(opts["levels"]) && all(0 .< opts["levels"] .< 1) || error("--levels must be probabilities between 0 and 1, e.g. 0.68,0.95")
    return opts
end

"""Bar of the day (0 = the 09:15 bar) containing an `HH:MM` string, for the cache's bar size; 0 for daily."""
function minute_index(s::String, cache::MinuteCache)::Int
    m = match(r"^(\d{1,2}):(\d{2})$", s)
    m === nothing && throw(PredictSectionError("--time must look like 12:00, got \"$s\""))
    idx = parse(Int, m[1]) * 60 + parse(Int, m[2]) - TradingGame.SESSION_OPEN_MINUTE
    0 <= idx < TradingGame.MINUTES_PER_DAY ||
        throw(PredictSectionError("--time $s is outside the session 09:15 to 15:29"))
    return cache.bars_per_day == 1 ? 0 : idx ÷ cache.bar_minutes
end

"""Wall-clock label of bar `t` (1-based) on the cache's axis."""
function bar_datetime(cache::MinuteCache, t::Int)::DateTime
    di, m = divrem(t - 1, cache.bars_per_day)
    return DateTime(cache.dates[di + 1]) + Minute(TradingGame.SESSION_OPEN_MINUTE + m * cache.bar_minutes)
end

"""Forward-filled closes of `symbol` for bars `t0:t1` of the cache's axis (NaN before the first bar on disk)."""
function closes_on_axis(cache::MinuteCache, csv::String, t0::Int, t1::Int)::Vector{Float64}
    d0 = cache.dates[(t0 - 1) ÷ cache.bars_per_day + 1]
    d1 = cache.dates[(t1 - 1) ÷ cache.bars_per_day + 1]
    days, mins, closes, _ = TradingGame._read_bar_csv(csv, Dates.value(d0 - Day(CSV_MARGIN_DAYS)), Dates.value(d1), GRANULARITIES[cache.granularity])
    di_of = Dict(d => i for (i, d) in enumerate(cache.dates))
    px = fill(NaN, t1 - t0 + 1)
    last = NaN
    started = false
    seen = Dict{Int, Float64}()
    for r in eachindex(days)
        di = get(di_of, Date(Dates.UTD(Int(days[r]))), 0)
        di == 0 && continue
        seen[(di - 1) * cache.bars_per_day + Int(mins[r]) + 1] = Float64(closes[r])
    end
    prior = [t for t in keys(seen) if t < t0]
    isempty(prior) || (last = seen[maximum(prior)])
    for t in t0:t1
        haskey(seen, t) && (last = seen[t])
        px[t - t0 + 1] = last
    end
    return px
end


"""A Gamma(a, 1) draw (Marsaglia–Tsang)."""
function _rand_gamma(rng::AbstractRNG, a::Real)
    a < 1 && return _rand_gamma(rng, a + 1) * rand(rng)^(1 / a)
    d = a - 1 / 3; c = 1 / sqrt(9d)
    while true
        x = randn(rng); v = (1 + c * x)^3
        v <= 0 && continue
        log(rand(rng)) < x^2 / 2 + d - d * v + d * log(v) && return d * v
    end
end

"""A `(N, S)` matrix of unit-scale draws from the model's distribution (standard normal, or Student-t with `nu` degrees of freedom)."""
function _draw_unit(rng::AbstractRNG, loss::Symbol, nu::Real, N::Int, S::Int)::Matrix{Float32}
    z = randn(rng, Float32, N, S)
    loss === :student_t || return z
    chi2 = Float32[2 * _rand_gamma(rng, nu / 2) for _ in 1:N, _ in 1:S]
    return z ./ sqrt.(chi2 ./ Float32(nu))
end

"""
    rollout_returns(net, dev, cache, X0, t_end, horizon, paths, loss, nu, rng) -> Array{Float32, 3}

Generate `horizon` minutes ahead, one at a time. `X0` is the `(F, N, M, 1)` input of the window ending at bar `t_end`. For each of `paths`
independent copies of the window the model's last position gives a distribution for the next return; one return is drawn from it (clipped
like the training returns), appended as the newest minute together with that bar's known time features, the oldest minute is dropped, and the model
runs again. The per-stock context features (mean |r|, traded fraction of the 375 bars before the window) stay at their values for the real window.
The model has no target for the first minute of a day (the overnight gap), so when the bar being generated opens a day its return is
drawn from `gaps`, the stock's own earlier overnight returns (`(N)` vectors in units of `cache.scale`), instead of from the model.
Returns the drawn returns in percent, `(N, paths, horizon)`.
"""
function rollout_returns(net, dev, cache::MinuteCache, X0::Array{Float32, 4}, t_end::Int, horizon::Int, paths::Int,
                         loss::Symbol, nu::Real, rng::AbstractRNG, gaps::Vector{Vector{Float32}})
    F, N, M, _ = size(X0)
    ctx = reshape(X0[N_CTX_ROWS, :, end, 1], length(N_CTX_ROWS), N, 1)
    X = dev(repeat(X0, 1, 1, 1, paths))
    clip = cache.ret_clip / cache.scale
    out = zeros(Float32, N, paths, horizon)
    for h in 1:horizon
        mu, logsigma, _ = net(X)
        μ = Float32.(Array(mu[end, :, :])); σ = exp.(Float32.(Array(logsigma[end, :, :])))
        y = clamp.(μ .+ σ .* _draw_unit(rng, loss, nu, N, paths), -clip, clip)
        if cache.bars_per_day > 1 && (t_end + h - 1) % cache.bars_per_day == 0
            for k in 1:N
                y[k, :] = rand(rng, gaps[k], paths)
            end
        end
        out[:, :, h] = y .* cache.scale
        h % PROGRESS_EVERY == 0 && @info "  rollout minute $h of $horizon"
        h == horizon && break
        col = zeros(Float32, F, N, 1, paths)
        col[1, :, 1, :] = y
        for k in 1:TradingGame.N_TIME_FEATURES
            col[1 + k, :, 1, :] .= cache.tau[t_end + h, k]
        end
        col[N_CTX_ROWS, :, 1, :] .= ctx
        X = cat(X[:, :, 2:end, :], dev(col); dims=3)
    end
    return out
end

_num(x) = x isa Real && isfinite(x) ? round(Float64(x); sigdigits=8) : nothing

"""
    load_session(; checkpoint, cache, device) -> NamedTuple

Load the minute cache and the checkpoint once, so that several [`predict_section`](@ref) calls in a REPL do not
reload them. Fields: `cache`, `model`, `cfg` (the `ForecasterConfig`), `loss`, `nu`, `device` (`:cpu`/`:gpu`), `checkpoint`.
"""
function load_session(; granularity::String="1min", checkpoint::String=default_checkpoint_path(granularity),
                      cache::String=default_cache_path(granularity), device::String="cpu")
    mc = load_minute_cache(cache)
    model, mcfg, meta = load_forecaster(checkpoint)
    loss = forecast_loss_kind(string(get(meta, "loss", "student_t")))
    nu = Float64(get(meta, "nu", TradingGame.STUDENT_T_NU))
    loss === :huber && throw(PredictSectionError("this checkpoint was trained with loss = huber: it predicts a location only, so there is no interval to draw"))
    gpu = device == "gpu" && CUDA.functional()
    device == "gpu" && !gpu && @warn "device = gpu requested but CUDA.functional() is false; using cpu"
    return (cache=mc, model=gpu ? Flux.gpu(model) : model, cfg=mcfg, loss=loss, nu=nu, device=gpu ? :gpu : :cpu, checkpoint=checkpoint)
end

"""
    predict_section(session; date, time="12:00", symbols=String[], n=8, levels=[0.68, 0.95], exchange="bse", horizon=1, paths=200, save_paths=50, seed=nothing) -> Dict

Forecast the `window` bars of the stocks `symbols` (default: the first `n` validation companies in the cache) that end at
`date`/`time`, and return the result that the page reads (`forecast_prediction.json`) as a `Dict`:
`"stocks"` holds, per stock, `close` (the `window + 1` actual closes), `pred_mean`/`pred_lo`/`pred_hi` (the price the
model predicted at each of the `window` bars for the bar after it, with a central interval for each of `levels`, under `"bands"` keyed by the percentage) and the return and σ in percent.
With `horizon = Y > 1` the result also has a `"rollout"` (top level: `datetimes` of the Y bars; per stock: `actual`, `mean`, `median`,
`sd_pct`, `bands` and the first `save_paths` of the `sample_paths`): each of `paths` sample paths is generated one minute at a time, feeding every drawn return back
in as the newest input, and the mean and bands are taken over the paths. Throws [`PredictSectionError`](@ref) for a date, time or symbol that does not fit the cache.

# Example
```julia
include("scripts/predict_minute_forecaster.jl")
s = load_session(device="gpu")
r = predict_section(s; date=Date(2025, 6, 12), time="12:00", symbols=["RECLTD", "RVNL"])
show_next(r)
```
"""
function predict_section(session; date::Date, time::String=DEFAULT_TIME, symbols::Vector{String}=String[], n::Int=DEFAULT_N,
                         levels::Vector{Float64}=[0.68, 0.95], exchange::String="bse",
                         horizon::Int=1, paths::Int=DEFAULT_PATHS, save_paths::Int=SAVED_PATHS, seed::Union{Nothing, Int}=nothing)
    cache, mcfg, loss, nu = session.cache, session.cfg, session.loss, session.nu
    M = mcfg.window
    symbols = uppercase.(symbols)
    if isempty(symbols)
        universe = load_universe_snapshot(UNIVERSE_FILE)
        have = Set(cache.symbols)
        symbols = first([s for s in universe.val if s in have], n)
    end
    idx = Dict(s => i for (i, s) in enumerate(cache.symbols))
    gone = [s for s in symbols if !haskey(idx, s)]
    isempty(gone) || throw(PredictSectionError("not in the minute cache: $(join(gone, ", ")); the cache holds the universe's train and val companies"))
    isempty(symbols) && throw(PredictSectionError("no symbols to forecast"))
    cols = [idx[s] for s in symbols]

    di = searchsortedfirst(cache.dates, date)
    (di <= length(cache.dates) && cache.dates[di] == date) ||
        throw(PredictSectionError("$date is not a day on the minute cache's axis ($(first(cache.dates)) .. $(last(cache.dates)), $(length(cache.dates)) days); " *
                                  "nearest: $(cache.dates[clamp(di, 1, length(cache.dates))])"))
    t_end = (di - 1) * cache.bars_per_day + minute_index(time, cache) + 1
    s = t_end - M + 1
    s > cache.context_bars ||
        throw(PredictSectionError("not enough history before $(bar_datetime(cache, t_end)): the window needs $(M) bars plus $(cache.context_bars) bars of context"))
    t_end + 1 <= TradingGame.n_bars(cache) ||
        throw(PredictSectionError("$(bar_datetime(cache, t_end)) is the last bar of the cache, so there is no next price to compare with"))

    max_h = max(1, MAX_HORIZON ÷ cache.bar_minutes)
    1 <= horizon <= max_h ||
        throw(PredictSectionError("--horizon must be between 1 and $max_h $(cache.granularity) bars (48 hours of bar time), got $horizon"))
    t_end + horizon <= TradingGame.n_bars(cache) ||
        throw(PredictSectionError("--horizon $horizon runs past the end of the minute cache (last bar $(bar_datetime(cache, TradingGame.n_bars(cache)))): from $time on $date at most $(TradingGame.n_bars(cache) - t_end) minutes fit"))

    dev = session.device === :gpu ? Flux.gpu : Flux.cpu
    net = Flux.testmode!(session.model)
    b = make_batch(cache, cols, [s], M)
    mu, logsigma, _ = net(dev(b.X))
    mu = Float32.(Array(mu))[:, :, 1]; logsigma = Float32.(Array(logsigma))[:, :, 1]
    hs = [central_halfwidth(loss, nu, l) for l in levels]
    to_pct = cache.scale

    csv_dir = joinpath(DATA_DIR, "ohlcv", exchange, GRANULARITIES[cache.granularity].dir)
    stocks = Dict{String, Any}[]
    rolled = nothing
    if horizon > 1
        rng = seed === nothing ? Random.default_rng() : Random.MersenneTwister(seed)
        day_end = (t_end - 1) ÷ cache.bars_per_day + 1
        first_days = max(2, day_end - GAP_HISTORY_DAYS):day_end
        gaps = [Float32[cache.logret[(d - 1) * cache.bars_per_day + 1, c] / cache.scale for d in first_days] for c in cols]
        rolled = rollout_returns(net, dev, cache, b.X, t_end, horizon, paths, loss, nu, rng, gaps)
    end
    for (k, sym) in enumerate(symbols)
        px = closes_on_axis(cache, joinpath(csv_dir, "$sym.csv"), s, t_end + max(1, horizon))
        mu_r = mu[:, k] .* to_pct
        sig_r = exp.(logsigma[:, k]) .* to_pct
        base = px[1:M]
        pred = base .* exp.(mu_r ./ 100)
        actual_next = px[2:M + 1]
        counted = [cache.traded[s + j, cols[k]] && (cache.bars_per_day == 1 || (s + j - 1) % cache.bars_per_day != 0) for j in 1:M]
        bands = Dict{String, Any}()
        for (l, h) in zip(levels, hs)
            lo = base .* exp.((mu_r .- h .* sig_r) ./ 100); hi = base .* exp.((mu_r .+ h .* sig_r) ./ 100)
            inside = [counted[j] && lo[j] <= actual_next[j] <= hi[j] for j in 1:M]
            bands[string(round(Int, 100 * l))] = Dict("level" => l, "lo" => _num.(lo), "hi" => _num.(hi),
                                                       "coverage_in_section" => count(counted) > 0 ? _num(count(inside) / count(counted)) : nothing)
        end
        st = Dict{String, Any}(
            "symbol" => sym,
            "close" => _num.(px[1:M + 1]),
            "traded" => [cache.traded[s + j - 1, cols[k]] for j in 1:M + 1],
            "pred_mean" => _num.(pred), "bands" => bands,
            "pred_return_pct" => _num.(mu_r), "pred_sigma_pct" => _num.(sig_r),
            "counted" => counted)
        if rolled !== nothing && !isfinite(px[M])
            @warn "$sym has no price at $(bar_datetime(cache, t_end)) (not listed yet, or no bar on disk): no iterated forecast for it"
        elseif rolled !== nothing
            cum = cumsum(rolled[k, :, :]; dims=2)
            price = px[M] .* exp.(cum ./ 100)
            rb = Dict{String, Any}()
            for l in levels
                q = [(1 - l) / 2, (1 + l) / 2]
                lohi = [quantile(price[:, h], q) for h in 1:horizon]
                rb[string(round(Int, 100 * l))] = Dict("level" => l, "lo" => _num.(first.(lohi)), "hi" => _num.(last.(lohi)))
            end
            st["rollout"] = Dict{String, Any}(
                "actual" => _num.(px[M + 1:M + horizon]),
                "traded" => [cache.traded[t_end + h, cols[k]] for h in 1:horizon],
                "mean" => _num.(vec(mean(price; dims=1))), "median" => _num.([median(price[:, h]) for h in 1:horizon]),
                "sd_pct" => _num.(vec(std(cum; dims=1))), "bands" => rb,
                "sample_paths" => [_num.(price[i, :]) for i in 1:min(save_paths, paths)])
        end
        push!(stocks, st)
    end
    result = Dict{String, Any}(
        "model" => Dict("checkpoint" => session.checkpoint, "window" => M, "d" => mcfg.d, "blocks" => mcfg.blocks, "heads" => mcfg.heads,
                        "loss" => string(loss), "nu" => nu),
        "levels" => levels, "return_scale_percent" => _num(cache.scale),
        "exchange" => exchange, "symbols" => symbols, "granularity" => cache.granularity, "bar_minutes" => cache.bar_minutes,
        "first_input" => string(bar_datetime(cache, s)), "last_input" => string(bar_datetime(cache, t_end)),
        "next" => string(bar_datetime(cache, t_end + 1)),
        "datetimes" => [string(bar_datetime(cache, t)) for t in s:t_end + 1],
        "stocks" => stocks, "generated" => string(now()))
    horizon > 1 && (result["rollout"] = Dict("horizon" => horizon, "paths" => paths, "seed" => seed,
                                              "datetimes" => [string(bar_datetime(cache, t_end + h)) for h in 1:horizon]))
    return result
end

"""Print the next-bar vector of a [`predict_section`](@ref) result: last close, predicted mean, every interval, the actual next close and whether each interval held it."""
function show_next(r::Dict)
    keys_ = [string(round(Int, 100 * l)) for l in r["levels"]]
    println("Section ", r["first_input"], " .. ", r["last_input"], "  ->  next bar ", r["next"])
    println(rpad("stock", 14), lpad("last", 11), lpad("pred", 11), join([lpad("lo$k", 11) * lpad("hi$k", 11) for k in keys_]), lpad("actual", 11), "  inside")
    f(x) = x === nothing ? lpad("—", 11) : lpad(string(round(x; digits=2)), 11)
    for st in r["stocks"]
        k = length(st["pred_mean"])
        last, actual = st["close"][end - 1], st["close"][end]
        hits = String[]
        cols = String[]
        for key in keys_
            lo, hi = st["bands"][key]["lo"][k], st["bands"][key]["hi"][k]
            push!(cols, f(lo) * f(hi))
            push!(hits, key * "%:" * (!st["counted"][k] || actual === nothing || lo === nothing ? "-" : lo <= actual <= hi ? "yes" : "no"))
        end
        println(rpad(st["symbol"], 14), f(last), f(st["pred_mean"][k]), join(cols), f(actual), "  ", join(hits, " "))
    end
end

"""Write a [`predict_section`](@ref) result to `path` (the file `website/predict.html` reads)."""
function save_prediction(r::Dict, path::String=default_out_path("1min"))
    mkpath(dirname(path))
    open(io -> JSON3.pretty(io, r), path, "w")
    return path
end

function main()
    opts = parse_args()
    session = load_session(; granularity=opts["granularity"], checkpoint=opts["checkpoint"], cache=opts["cache"], device=opts["device"])
    r = predict_section(session; date=opts["date"], time=opts["time"], symbols=opts["symbols"], n=opts["n"],
                        levels=opts["levels"], exchange=opts["exchange"],
                        horizon=opts["horizon"], paths=opts["paths"], save_paths=opts["save_paths"], seed=opts["seed"])
    save_prediction(r, opts["out"])
    println("Wrote $(opts["out"]): $(length(r["symbols"])) stocks, bars $(r["first_input"]) .. $(r["last_input"]), next bar $(r["next"])")
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        main()
    catch e
        e isa PredictSectionError || rethrow()
        println(stderr, sprint(showerror, e))
        exit(1)
    end
end
