"""
Return cache for the v4 forecaster, at any bar size from one minute to one day.

`build_forecaster_cache` reads each symbol's `ohlcv/{exchange}/{granularity}/{SYMBOL}.csv`
(`granularity` is one of `1min`, `5min`, `15min`, `hourly`, `daily`), puts all symbols on one
shared axis of trading days × `bars_per_day` bars (375, 75, 25, 7 or 1 for the 375-minute
09:15–15:29 session), forward-fills bars with no row, and keeps two matrices: the log-return
of the close from one bar to the next, and a `traded` flag (a row with volume > 0).
The forecaster never sees a price level, only returns, so the cache holds no prices.
`build_minute_cache` is the 1-minute case of the same function.

Everything the model needs besides the returns is derived from the axis alone: the
five time features of each bar (`tau`) and the three-number description of the
bar's timestamp the model is asked to predict (`tdesc`).
"""

using BSON, Dates, Printf, Statistics

"""Session minutes per trading day, 09:15 to 15:29 inclusive."""
const MINUTES_PER_DAY = 375

"""Minutes after midnight of the first session bar (09:15)."""
const SESSION_OPEN_MINUTE = 9 * 60 + 15

"""Returns are stored in percent and clipped to ± this; larger moves are almost always data errors or splits."""
const MINUTE_RET_CLIP = 10f0

"""Time features per bar: minute-of-day fraction, sin/cos weekday, log(1 + calendar-day gap), first-bar flag."""
const N_TIME_FEATURES = 5

"""Per-stock context features: mean |return| and traded fraction over the `cache.context_bars` bars before the window."""
const N_CONTEXT_FEATURES = 2

"""Bars of history behind a window used for the per-stock context features at 1-minute bars (one session); see [`GranularitySpec`](@ref) for the others."""
const CONTEXT_BARS = MINUTES_PER_DAY

"""Context bars are at least this many (a day of 5-minute bars is 75, but a day of 15-minute, hourly or daily bars is too few to average over)."""
const MIN_CONTEXT_BARS = 60

"""Evaluation reports group the bars of a session into buckets of about this many minutes (a bucket is at least one bar)."""
const SLOT_BUCKET_MINUTES_TARGET = 15

"""Numbers describing a bar's timestamp that the time head predicts: minute fraction, log(1 + gap), weekday/4."""
const N_TIME_TARGETS = 3

"""Per-token input width: the return, the time features and the context features."""
const N_FORECAST_FEATURES = 1 + N_TIME_FEATURES + N_CONTEXT_FEATURES

"""A day joins the axis when at least this fraction of the symbols have `MIN_DAY_BARS` (1-minute bars; scaled down for coarser bars) in the session that day."""
const MIN_DAY_SYMBOL_FRACTION = 0.2

"""Bars a symbol needs in a day to count as active that day for the day filter."""
const MIN_DAY_BARS = 10

"""
    GranularitySpec

What a bar size means: `dir` (the folder under `ohlcv/{exchange}/`), `bar_minutes` (a daily bar counts as the whole 375-minute session),
`ret_clip` (returns are stored in percent and clipped to ± this; larger one-bar moves are almost always data errors or splits, and
a day can move more than a minute can).
"""
struct GranularitySpec
    dir         :: String
    bar_minutes :: Int
    ret_clip    :: Float32
end

"""The bar sizes a forecaster cache can be built at, by the name used in `[forecaster_cache] granularity` and `--granularity`."""
const GRANULARITIES = Dict(
    "1min"   => GranularitySpec("1min",   1,   10f0),
    "5min"   => GranularitySpec("5min",   5,   12f0),
    "15min"  => GranularitySpec("15min",  15,  15f0),
    "hourly" => GranularitySpec("hourly", 60,  20f0),
    "daily"  => GranularitySpec("daily",  MINUTES_PER_DAY, 30f0))

"""
    MinuteCache

Aligned bar data for a set of symbols, at the bar size named by `granularity` (the name is historical: it holds any granularity).

# Fields
- `symbols`: column order of `logret` and `traded`.
- `dates`: the trading days on the axis; bar `t` is bar `(t-1) % bars_per_day` of day `(t-1) ÷ bars_per_day + 1`.
- `logret`: `T × N` percent log-returns of the close, clipped at `±ret_clip`; 0 where a bar has no earlier price.
- `traded`: `T × N`, true where the minute had a row with volume > 0.
- `scale`: root-mean-square of the traded returns (leaving out each day's first bar when a day has more than one bar); inputs and targets are divided by it.
- `tau`: `T × 5` time features of each bar, derived from `dates`.
- `tdesc`: `T × 3` timestamp description of each bar (the time head's target), derived from `dates`.
- `granularity`, `bars_per_day`, `bar_minutes`, `ret_clip`: the bar size (see [`GRANULARITIES`](@ref)).
- `context_bars`: bars before a window used for the per-stock context features (a session of bars, at least `MIN_CONTEXT_BARS`).
- `bucket_bars`: bars per bucket of the per-slot evaluation reports (about `SLOT_BUCKET_MINUTES_TARGET` minutes, at least one bar).
"""
struct MinuteCache
    symbols      :: Vector{String}
    dates        :: Vector{Date}
    logret       :: Matrix{Float32}
    traded       :: Matrix{Bool}
    scale        :: Float32
    tau          :: Matrix{Float32}
    tdesc        :: Matrix{Float32}
    granularity  :: String
    bars_per_day :: Int
    bar_minutes  :: Int
    ret_clip     :: Float32
    context_bars :: Int
    bucket_bars  :: Int
end

"""The `tau`/`tdesc` matrices of an axis of `dates` × `bars_per_day` bars."""
function minute_time_features(dates::Vector{Date}, bars_per_day::Int=MINUTES_PER_DAY)
    T = length(dates) * bars_per_day
    tau   = zeros(Float32, T, N_TIME_FEATURES)
    tdesc = zeros(Float32, T, N_TIME_TARGETS)
    for (di, date) in enumerate(dates)
        gap = di == 1 ? 1 : Dates.value(date - dates[di - 1])
        wd  = dayofweek(date) - 1
        for m in 0:bars_per_day - 1
            t = (di - 1) * bars_per_day + m + 1
            frac = Float32(m / bars_per_day)
            first = m == 0
            lg = first ? Float32(log1p(gap)) : 0f0
            tau[t, 1] = frac
            tau[t, 2] = Float32(sin(2π * wd / 5))
            tau[t, 3] = Float32(cos(2π * wd / 5))
            tau[t, 4] = lg
            tau[t, 5] = first ? 1f0 : 0f0
            tdesc[t, 1] = frac
            tdesc[t, 2] = lg
            tdesc[t, 3] = Float32(wd / 4)
        end
    end
    return tau, tdesc
end

"""
    MinuteCache(symbols, dates, logret, traded; granularity="1min")

Assemble a cache from its primary arrays; the scale, the time matrices and the bar-size fields are computed here.
"""
function MinuteCache(symbols::Vector{String}, dates::Vector{Date}, logret::AbstractMatrix{<:Real}, traded::AbstractMatrix{Bool};
                     granularity::AbstractString="1min")
    haskey(GRANULARITIES, granularity) ||
        throw(MinuteCacheError("unknown granularity '$granularity'; expected one of: $(join(sort(collect(keys(GRANULARITIES))), ", "))"))
    spec = GRANULARITIES[granularity]
    bpd = cld(MINUTES_PER_DAY, spec.bar_minutes)
    size(logret) == size(traded) == (length(dates) * bpd, length(symbols)) ||
        throw(DimensionMismatch("logret/traded must be $(length(dates) * bpd) × $(length(symbols)), got $(size(logret)) and $(size(traded))"))
    sq, n = 0.0, 0
    @inbounds for i in axes(logret, 2), t in 2:size(logret, 1)
        if traded[t, i] && (bpd == 1 || (t - 1) % bpd != 0)
            sq += Float64(logret[t, i])^2
            n += 1
        end
    end
    scale = n == 0 ? 1f0 : Float32(max(sqrt(sq / n), 1e-6))
    tau, tdesc = minute_time_features(dates, bpd)
    bucket_bars = max(1, SLOT_BUCKET_MINUTES_TARGET ÷ spec.bar_minutes)
    return MinuteCache(symbols, dates, Matrix{Float32}(logret), Matrix{Bool}(traded), scale, tau, tdesc,
                       String(granularity), bpd, spec.bar_minutes, spec.ret_clip, max(bpd, MIN_CONTEXT_BARS), bucket_bars)
end

"""Number of buckets of the per-slot evaluation reports (a session of bars cut into `bucket_bars`-bar pieces)."""
n_slot_buckets(c::MinuteCache) = cld(c.bars_per_day, c.bucket_bars)

"""Number of bars on the axis."""
n_bars(c::MinuteCache) = size(c.logret, 1)

"""First bar index of `date`, or of the first later day on the axis."""
first_bar_on_or_after(c::MinuteCache, date::Date) = (searchsortedfirst(c.dates, date) - 1) * c.bars_per_day + 1

"""Last bar index of `date`, or of the last earlier day on the axis (0 if none)."""
last_bar_on_or_before(c::MinuteCache, date::Date) = searchsortedlast(c.dates, date) * c.bars_per_day

"""Thrown when a forecaster cache cannot be built or read: an unknown granularity, no usable days, a missing CSV or cache file, or a training date range shorter than one window."""
struct MinuteCacheError <: Exception
    msg::String
end
Base.showerror(io::IO, e::MinuteCacheError) = print(io, "MinuteCacheError: ", e.msg)

"""
    _read_bar_csv(path, from, to, spec) -> (days, bars, closes, vols)

One symbol's rows inside `[from, to]` (day numbers): the day, the bar of the day (0-based; a row belongs to the bar that contains its
timestamp, counted from 09:15 in steps of `spec.bar_minutes`; a daily row is bar 0), the close and the volume. Intraday rows outside
the 375-minute session are dropped.
"""
function _read_bar_csv(path::String, from::Int, to::Int, spec::GranularitySpec)
    days = Int32[]; bars = Int16[]; closes = Float32[]; vols = Float32[]
    daily = spec.dir == "daily"
    bad = 0
    open(path) do io
        readline(io)
        for line in eachline(io)
            length(line) < (daily ? 10 : 20) && continue
            try
                y = parse(Int, SubString(line, 1, 4)); mo = parse(Int, SubString(line, 6, 7)); d = parse(Int, SubString(line, 9, 10))
                b = 0
                if !daily
                    hh = parse(Int, SubString(line, 12, 13)); mm = parse(Int, SubString(line, 15, 16))
                    m = hh * 60 + mm - SESSION_OPEN_MINUTE
                    (0 <= m < MINUTES_PER_DAY) || continue
                    b = m ÷ spec.bar_minutes
                end
                dn = Dates.value(Date(y, mo, d))
                (from <= dn <= to) || continue
                f = split(line, ','; limit=7)
                c = parse(Float32, f[5])
                c > 0 || continue
                v = parse(Float32, f[6])
                push!(days, Int32(dn)); push!(bars, Int16(b)); push!(closes, c); push!(vols, v)
            catch e
                e isa Union{ArgumentError, BoundsError} || e isa MethodError || rethrow()
                bad += 1
            end
        end
    end
    bad > 0 && @warn "$path: skipped $bad unreadable rows (corrupt bytes, e.g. from an interrupted write)"
    return days, bars, closes, vols
end

"""One symbol's 1-minute rows; see [`_read_bar_csv`](@ref)."""
_read_minute_csv(path::String, from::Int, to::Int) = _read_bar_csv(path, from, to, GRANULARITIES["1min"])

"""
    build_forecaster_cache(ohlcv_dir, symbols; granularity, from, to) -> MinuteCache

Read `ohlcv_dir/{SYMBOL}.csv` for every symbol and align them (see the file docstring).

# Arguments
- `ohlcv_dir`: the `website/data/ohlcv/{exchange}/{granularity folder}` directory (`1min`, `5min`, `15min`, `hourly` or `daily`).
- `symbols`: the columns, in this order.
- `granularity`: one of the keys of [`GRANULARITIES`](@ref) (default `"1min"`).
- `from`, `to`: only days in this inclusive range are kept.
- `min_day_fraction`, `min_day_bars`: a day joins the axis when this fraction of the symbols have this many bars that day
  (`min_day_bars` defaults to `MIN_DAY_BARS` 1-minute bars scaled to the bar size, at least 1).

# Returns
A [`MinuteCache`](@ref). Throws [`MinuteCacheError`](@ref) for an unknown granularity, a missing CSV, or when no day qualifies.

# Example
```julia
cache = build_forecaster_cache("website/data/ohlcv/bse/15min", ["RELIANCE", "TCS"]; granularity="15min", from=Date(2022, 1, 3), to=Date(2025, 12, 30))
```
"""
function build_forecaster_cache(ohlcv_dir::String, symbols::Vector{String}; granularity::AbstractString="1min", from::Date, to::Date,
                                min_day_fraction::Float64=MIN_DAY_SYMBOL_FRACTION, min_day_bars::Union{Int, Nothing}=nothing,
                                verbose::Bool=true)
    haskey(GRANULARITIES, granularity) ||
        throw(MinuteCacheError("unknown granularity '$granularity'; expected one of: $(join(sort(collect(keys(GRANULARITIES))), ", "))"))
    spec = GRANULARITIES[granularity]
    bpd = cld(MINUTES_PER_DAY, spec.bar_minutes)
    min_day_bars === nothing && (min_day_bars = max(1, round(Int, MIN_DAY_BARS * bpd / MINUTES_PER_DAY)))
    missing_files = [s for s in symbols if !isfile(joinpath(ohlcv_dir, "$s.csv"))]
    isempty(missing_files) ||
        throw(MinuteCacheError("no $granularity CSV in $ohlcv_dir for: $(join(first(missing_files, 10), ", "))" *
                               (length(missing_files) > 10 ? " … ($(length(missing_files)) in all)" : "")))
    from_n, to_n = Dates.value(from), Dates.value(to)
    N = length(symbols)
    rows = Vector{Any}(undef, N)
    Threads.@threads for k in 1:N
        rows[k] = _read_bar_csv(joinpath(ohlcv_dir, "$(symbols[k]).csv"), from_n, to_n, spec)
        verbose && @info "  read $(symbols[k]) ($(length(rows[k][1])) bars)"
    end

    active = Dict{Int32, Int}()
    for (days, _, _, _) in rows
        counts = Dict{Int32, Int}()
        for d in days
            counts[d] = get(counts, d, 0) + 1
        end
        for (d, n) in counts
            n >= min_day_bars && (active[d] = get(active, d, 0) + 1)
        end
    end
    need = max(1, ceil(Int, min_day_fraction * N))
    day_nums = sort!([d for (d, n) in active if n >= need])
    isempty(day_nums) && throw(MinuteCacheError("no trading day between $from and $to has $need of $N symbols active"))
    day_index = Dict(d => i for (i, d) in enumerate(day_nums))
    dates = [Date(Dates.UTD(Int(d))) for d in day_nums]
    T = length(dates) * bpd

    logret = zeros(Float32, T, N)
    traded = zeros(Bool, T, N)
    Threads.@threads for k in 1:N
        days, bars, closes, vols = rows[k]
        px = fill(NaN32, T)
        for r in eachindex(days)
            di = get(day_index, days[r], 0)
            di == 0 && continue
            t = (di - 1) * bpd + Int(bars[r]) + 1
            px[t] = closes[r]
            vols[r] > 0 && (traded[t, k] = true)
        end
        last = NaN32
        for t in 1:T
            if isnan(px[t])
                px[t] = last
            else
                last = px[t]
            end
        end
        for t in 2:T
            (isnan(px[t]) || isnan(px[t - 1])) && continue
            logret[t, k] = clamp(Float32(100 * log(px[t] / px[t - 1])), -spec.ret_clip, spec.ret_clip)
        end
        rows[k] = nothing
    end
    return MinuteCache(symbols, dates, logret, traded; granularity=granularity)
end

"""
    build_minute_cache(ohlcv_dir, symbols; from, to) -> MinuteCache

[`build_forecaster_cache`](@ref) at 1-minute bars.
"""
build_minute_cache(ohlcv_dir::String, symbols::Vector{String}; kw...) = build_forecaster_cache(ohlcv_dir, symbols; granularity="1min", kw...)

"""
    save_minute_cache(cache, path)

Write the primary arrays to BSON (the derived time matrices are rebuilt on load).
"""
function save_minute_cache(cache::MinuteCache, path::String)
    mkpath(dirname(path))
    symbols, dates, logret, traded = cache.symbols, cache.dates, cache.logret, cache.traded
    granularity = cache.granularity
    BSON.@save path symbols dates logret traded granularity
    return path
end

"""
    load_minute_cache(path) -> MinuteCache

Read a cache written by [`save_minute_cache`](@ref).
"""
function load_minute_cache(path::String)
    isfile(path) || throw(MinuteCacheError("not found: $path — run scripts/forecaster/build_forecaster_cache.jl"))
    d = BSON.load(path)
    return MinuteCache(Vector{String}(d[:symbols]), Vector{Date}(d[:dates]), Matrix{Float32}(d[:logret]), Matrix{Bool}(d[:traded]);
                       granularity=String(get(d, :granularity, "1min")))
end

# ── Windows ──────────────────────────────────────────────────────────────────

"""
    window_starts(cache, first_date, last_date, window) -> UnitRange{Int}

Every start bar `s` whose input bars `s … s+window-1` and target bars `s+1 … s+window`
lie inside `[first_date, last_date]`, and which has `cache.context_bars` bars of history behind
it (for the per-stock context features). Empty when the range is too short.
"""
function window_starts(cache::MinuteCache, first_date::Date, last_date::Date, window::Int)::UnitRange{Int}
    lo = max(first_bar_on_or_after(cache, first_date), cache.context_bars + 1)
    hi = min(last_bar_on_or_before(cache, last_date), n_bars(cache)) - window
    return lo:hi
end

"""
    ForecastBatch

One batch of windows. `X` is `(F, N, M, B)` with `F = 8`: `[return; time features; context]` per stock and minute.
`Y` and `W` are `(M, N, B)`: the next-minute return in units of `cache.scale`, and 1 where that target counts
(the stock traded in that minute and it is not the first bar of a day). `T` is `(3, M, B)`: the next bar's timestamp
description. `R` is the input return `(M, N, B)`, kept for the repeat-last baseline. `slot` is the report bucket (0-based, see [`n_slot_buckets`](@ref)) of the next bar's
minute of day `(M, B)`.
"""
struct ForecastBatch
    X    :: Array{Float32, 4}
    Y    :: Array{Float32, 3}
    W    :: Array{Float32, 3}
    T    :: Array{Float32, 3}
    R    :: Array{Float32, 3}
    slot :: Matrix{Int16}
end

"""
    make_batch(cache, cols, starts, window) -> ForecastBatch

Assemble the windows beginning at each bar in `starts`, for the columns `cols` of the cache.
The context features of a window come only from the `cache.context_bars` bars before it, so no
token sees a later bar.
"""
function make_batch(cache::MinuteCache, cols::AbstractVector{Int}, starts::AbstractVector{Int}, window::Int)::ForecastBatch
    N, B, M = length(cols), length(starts), window
    bpd = cache.bars_per_day
    X = zeros(Float32, N_FORECAST_FEATURES, N, M, B)
    Y = zeros(Float32, M, N, B); W = zeros(Float32, M, N, B); R = zeros(Float32, M, N, B)
    Tt = zeros(Float32, N_TIME_TARGETS, M, B)
    slot = zeros(Int16, M, B)
    inv = 1f0 / cache.scale
    for (b, s) in enumerate(starts)
        for (ni, c) in enumerate(cols)
            ma = 0f0; tf = 0f0
            @inbounds for t in s - cache.context_bars:s - 1
                ma += abs(cache.logret[t, c])
                tf += cache.traded[t, c]
            end
            ma = ma * inv / cache.context_bars; tf /= cache.context_bars
            @inbounds for j in 1:M
                t = s + j - 1
                r = cache.logret[t, c] * inv
                X[1, ni, j, b] = r
                for k in 1:N_TIME_FEATURES
                    X[1 + k, ni, j, b] = cache.tau[t, k]
                end
                X[2 + N_TIME_FEATURES, ni, j, b] = ma
                X[3 + N_TIME_FEATURES, ni, j, b] = tf
                R[j, ni, b] = r
                Y[j, ni, b] = cache.logret[t + 1, c] * inv
                W[j, ni, b] = (cache.traded[t + 1, c] && (bpd == 1 || t % bpd != 0)) ? 1f0 : 0f0
            end
        end
        @inbounds for j in 1:M
            t = s + j - 1
            for k in 1:N_TIME_TARGETS
                Tt[k, j, b] = cache.tdesc[t + 1, k]
            end
            slot[j, b] = Int16((t % bpd) ÷ cache.bucket_bars)
        end
    end
    return ForecastBatch(X, Y, W, Tt, R, slot)
end
