"""
Inference cache: aligned market matrices loaded once at session start.

All feature assembly — for both live inference and training batch assembly —
is done by O(1) array slices into the in-memory matrices. No CSV reads happen
after the cache is loaded.

## Layout

`closes[i, j]`    — daily close for company j on trading date i (forward-filled).
`vols[i, j]`      — daily (high-low)/close for company j on date i.
`rel_vols[i, j]`  — volume[i] / trailing-20-day-avg-volume[i] (relative volume).
                    0.0 on non-trading days; 1.0 when history is unavailable.
`hourly_closes[h, j]` — intraday close for company j at bar h (forward-filled). The
                    bar length is `bar_minutes`: 60 for the original hourly cache, 15 for a
                    15-minute cache (`build_inference_cache(...; granularity="15min")`). The
                    field keeps its historical `hourly_` name — it is "the intraday matrix",
                    whatever the bar length. `bar_minutes` and `exchange` ("nse"/"bse") are
                    stored with the cache; old files without them load as 60 / "nse".

Column ordering in all matrices matches `companies` exactly. Row ordering
matches `dates` (daily) and `hourly_datetimes` (hourly), both sorted ascending.

## Typical sizes (5 years, ~942 companies)

  closes + vols + rel_vols : 3 × 1250 × 942 × 4 bytes ≈ 14 MB
  hourly_closes             :     8750 × 942 × 4 bytes ≈ 33 MB
  Total on disk             :                          ≈ 47 MB
"""

using CSV, DataFrames, Dates, BSON, Printf

# ── Struct ────────────────────────────────────────────────────────────────────

"""
Pre-loaded market data for fast inference and training.

`date_index` and `sym_index` are derived on load and not serialised —
they provide O(1) lookups from Date / symbol string to matrix row/column.
"""
struct InferenceCache
    closes           :: Matrix{Float32}     # (n_dates,  n_companies)
    vols             :: Matrix{Float32}     # (n_dates,  n_companies) — (H-L)/C
    rel_vols         :: Matrix{Float32}     # (n_dates,  n_companies) — volume / 20d avg
    hourly_closes    :: Matrix{Float32}     # (n_hourly, n_companies)
    dates            :: Vector{Date}
    hourly_datetimes :: Vector{DateTime}
    companies        :: Vector{String}
    date_index       :: Dict{Date,   Int}   # derived on load
    sym_index        :: Dict{String, Int}   # derived on load
    bar_minutes      :: Int                 # length of one intraday bar: 60 (hourly) or 15
    exchange         :: String              # "nse" or "bse" — which OHLCV tree the cache came from
    history_closes   :: Matrix{Float32}     # (n_history, n_companies) hourly closes; 0 rows = no history axis
    history_datetimes :: Vector{DateTime}   # start time of each hourly history bar
    history_last_bar :: Vector{DateTime}    # derived: start of the LAST intraday bar inside each history bar
    history_end_idx  :: Vector{Int}         # derived: per intraday bar, newest history bar already complete (0 = none)
end

"""Constructor for a cache with no separate history axis: `bar_minutes`/`exchange`
as given and zero history rows. The intraday matrix is then the only price series."""
InferenceCache(closes, vols, rel_vols, hourly_closes, dates, hourly_datetimes, companies,
               date_index, sym_index, bar_minutes::Int, exchange::String) =
    InferenceCache(closes, vols, rel_vols, hourly_closes, dates, hourly_datetimes, companies,
                   date_index, sym_index, bar_minutes, exchange,
                   zeros(Float32, 0, length(companies)), DateTime[], DateTime[], Int[])

"""Backwards-compatible constructor: an hourly NSE cache, as before `bar_minutes`
and `exchange` existed."""
InferenceCache(closes, vols, rel_vols, hourly_closes, dates, hourly_datetimes, companies,
               date_index, sym_index) =
    InferenceCache(closes, vols, rel_vols, hourly_closes, dates, hourly_datetimes, companies,
                   date_index, sym_index, 60, "nse")

"""Whether `cache` carries a separate hourly history axis (built with `history=`)."""
has_history(cache::InferenceCache) = !isempty(cache.history_datetimes)

"""Start of the 60-minute history slot (9:15, 10:15, …, 15:15) that an intraday
bar starting at `dt` falls in."""
function history_slot(dt::DateTime)::DateTime
    k = fld(Dates.value(Time(dt) - SESSION_FIRST_BAR), 3_600_000_000_000)   # whole hours since 9:15, in ns
    return DateTime(Date(dt)) + Hour(9) + Minute(15) + Hour(k)
end

"""Start of the last 15-minute bar inside the history slot beginning at `slot`
(the 15:15 slot holds only the 15:15 bar). A history bar is complete — its close
known — exactly when the intraday clock has reached this bar."""
history_last_bar_start(slot::DateTime)::DateTime = min(slot + Minute(45), DateTime(Date(slot)) + Hour(15) + Minute(15))

"""For each intraday bar start in `clock`, the index of the newest history bar
whose last constituent bar has started by then (so its close is known); 0 when
none is. This is what keeps the history window free of look-ahead."""
history_end_index(clock::Vector{DateTime}, last_bar::Vector{DateTime})::Vector{Int} =
    [searchsortedlast(last_bar, t) for t in clock]

"""Intraday granularity folder name → bar length in minutes."""
const INTRADAY_BAR_MINUTES = Dict("hourly" => 60, "15min" => 15)

"""Regular-session window for an intraday bar's *start* time. Drops BSE's
Muhurat-evening bars (18:15–19:00 on Diwali) and any other out-of-session
prints, which would otherwise add phantom bars to the shared time axis."""
const SESSION_FIRST_BAR = Time(9, 15)
const SESSION_LAST_BAR  = Time(15, 15)

# ── Build ─────────────────────────────────────────────────────────────────────

"""
Build the inference cache from cached OHLCV CSVs and save to `path`.

Reads all `{SYMBOL}.csv` in `ohlcv_dir/daily/` and `ohlcv_dir/{granularity}/` for
the given `companies` (the per-granularity subfolder layout — see
`kite_data.jl`'s module docstring), aligns them onto a shared time axis,
forward-fills missing bars, and writes a single BSON file (~42 MB for 942
companies at hourly granularity; a 15-minute cache is ~3.6× larger per company).
Intraday bars whose start time falls outside 09:15–15:15 are dropped.

# Arguments
- `ohlcv_dir`: exchange directory containing `daily/` and the intraday subfolder
- `companies`: symbols in the desired column order (sets the universe)
- `path`: output BSON file path
- `granularity`: intraday folder, `"hourly"` (default) or `"15min"`
- `exchange`: `"nse"` (default) or `"bse"`; recorded in the cache
- `history`: a second, coarser price axis for the policy's look-back window.
  `"none"` (default): no separate axis. `"resample"`: hourly bars aggregated from
  the 15-minute bars (slots 9:15 … 15:15, close = close of the slot's last
  15-minute bar — identical to Kite's hourly close, and available for every
  company that has 15-minute data). `"hourly"`: read `ohlcv_dir/hourly/` instead.
  Needs `granularity="15min"`. The intraday matrix stays the decision clock and
  the instantaneous price; history bars enter an observation only once complete.
"""
function build_inference_cache(ohlcv_dir::String, companies::Vector{String},
                                path::String; granularity::String="hourly",
                                exchange::String="nse",
                                history::String="none")::InferenceCache
    history in ("none", "resample", "hourly") ||
        error("build_inference_cache: history must be none, resample or hourly, got '$history'")
    history != "none" && granularity != "15min" &&
        error("build_inference_cache: a history axis needs granularity=\"15min\" as the clock")
    haskey(INTRADAY_BAR_MINUTES, granularity) ||
        error("build_inference_cache: granularity must be one of $(sort(collect(keys(INTRADAY_BAR_MINUTES)))), got '$granularity'")
    bar_minutes = INTRADAY_BAR_MINUTES[granularity]
    n_comp = length(companies)
    daily_dir  = joinpath(ohlcv_dir, "daily")
    hourly_dir = joinpath(ohlcv_dir, granularity)

    @info "Building inference cache — $(n_comp) companies"

    # ── Daily matrices ────────────────────────────────────────────────────────

    all_dates = Set{Date}()
    for sym in companies
        p = joinpath(daily_dir, "$sym.csv")
        isfile(p) || continue
        df = CSV.read(p, DataFrame; select=[:date], types=Dict(:date => Date))
        union!(all_dates, df.date)
    end
    dates    = sort(collect(all_dates))
    n_dates  = length(dates)
    date_idx = Dict(d => i for (i, d) in enumerate(dates))

    closes      = fill(NaN32, n_dates, n_comp)
    vols        = fill(NaN32, n_dates, n_comp)
    raw_volumes = zeros(Float32, n_dates, n_comp)

    for (j, sym) in enumerate(companies)
        p = joinpath(daily_dir, "$sym.csv")
        isfile(p) || continue
        df = CSV.read(p, DataFrame; types=Dict(:date => Date))

        for row in eachrow(df)
            i = get(date_idx, row.date, 0)
            i == 0 && continue
            closes[i, j]      = Float32(row.close)
            vols[i, j]        = (row.high > row.low && row.close > 0) ?
                                 Float32((row.high - row.low) / row.close) : 0f0
            raw_volumes[i, j] = Float32(row.volume)
        end

        last_c = NaN32; last_v = 0f0
        for i in 1:n_dates
            if !isnan(closes[i, j])
                last_c = closes[i, j]; last_v = vols[i, j]
            elseif !isnan(last_c)
                closes[i, j] = last_c; vols[i, j] = last_v
            end
        end
    end

    # ── Relative volume: volume[i] / trailing-20-day average ─────────────────
    # Uses only the 20 trading days *before* day i to avoid lookahead bias.
    # Non-trading days (raw_volumes == 0) get rel_vol = 0.
    # Days with insufficient history default to 1.0 (neutral).
    rel_vols = zeros(Float32, n_dates, n_comp)
    for j in 1:n_comp
        for i in 1:n_dates
            raw_volumes[i, j] == 0f0 && continue
            lo = max(1, i - 20); hi = i - 1
            if hi < lo
                rel_vols[i, j] = 1f0
                continue
            end
            nonzero = [raw_volumes[k, j] for k in lo:hi if raw_volumes[k, j] > 0f0]
            rel_vols[i, j] = isempty(nonzero) ? 1f0 :
                              Float32(raw_volumes[i, j] / mean(nonzero))
        end
    end

    @info "  Daily: $n_dates trading dates × $n_comp companies"

    # ── Intraday matrix ───────────────────────────────────────────────────────

    in_session(dt) = SESSION_FIRST_BAR <= Time(dt) <= SESSION_LAST_BAR

    all_dts = Set{DateTime}()
    for sym in companies
        p = joinpath(hourly_dir, "$sym.csv")
        isfile(p) || continue
        df = CSV.read(p, DataFrame; select=[:datetime], types=Dict(:datetime => DateTime))
        union!(all_dts, filter(in_session, df.datetime))
    end
    hourly_dts = sort(collect(all_dts))
    n_hourly   = length(hourly_dts)
    dt_idx     = Dict(dt => i for (i, dt) in enumerate(hourly_dts))

    hourly_closes = fill(NaN32, n_hourly, n_comp)

    # ── History axis (hourly), if requested ───────────────────────────────────
    history_dir = history == "hourly" ? joinpath(ohlcv_dir, "hourly") : hourly_dir
    hist_dts = DateTime[]
    if history != "none"
        slots = Set{DateTime}()
        for sym in companies
            p = joinpath(history_dir, "$sym.csv")
            isfile(p) || continue
            df = CSV.read(p, DataFrame; select=[:datetime], types=Dict(:datetime => DateTime))
            union!(slots, history_slot.(filter(in_session, df.datetime)))
        end
        hist_dts = sort(collect(slots))
    end
    hist_idx    = Dict(dt => i for (i, dt) in enumerate(hist_dts))
    hist_closes = fill(NaN32, length(hist_dts), n_comp)

    for (j, sym) in enumerate(companies)
        p = joinpath(hourly_dir, "$sym.csv")
        isfile(p) || continue
        df = CSV.read(p, DataFrame; select=[:datetime, :close],
                      types=Dict(:datetime => DateTime))

        for (dt, c) in zip(df.datetime, df.close)
            i = get(dt_idx, dt, 0)
            i == 0 && continue
            hourly_closes[i, j] = Float32(c)
        end

        if history == "resample"
            for (dt, c) in zip(df.datetime, df.close)         # rows ascend: the slot's last bar wins
                in_session(dt) || continue
                hist_closes[hist_idx[history_slot(dt)], j] = Float32(c)
            end
        elseif history == "hourly"
            hp = joinpath(history_dir, "$sym.csv")
            if isfile(hp)
                hdf = CSV.read(hp, DataFrame; select=[:datetime, :close], types=Dict(:datetime => DateTime))
                for (dt, c) in zip(hdf.datetime, hdf.close)
                    in_session(dt) || continue
                    hist_closes[hist_idx[history_slot(dt)], j] = Float32(c)
                end
            end
        end
        if history != "none"
            last_h = NaN32
            for i in 1:length(hist_dts)
                if !isnan(hist_closes[i, j]); last_h = hist_closes[i, j]
                elseif !isnan(last_h); hist_closes[i, j] = last_h
                end
            end
        end

        last_h = NaN32
        for i in 1:n_hourly
            if !isnan(hourly_closes[i, j])
                last_h = hourly_closes[i, j]
            elseif !isnan(last_h)
                hourly_closes[i, j] = last_h
            end
        end

        j % 200 == 0 && @info "  [$j/$n_comp] $granularity loaded"
    end

    @info "  Intraday ($granularity, $(bar_minutes)-min): $n_hourly bars × $n_comp companies"
    history != "none" && @info "  History ($history, hourly): $(length(hist_dts)) bars × $n_comp companies"

    # ── Save ──────────────────────────────────────────────────────────────────

    mkpath(dirname(path))
    history_closes    = hist_closes
    history_datetimes = hist_dts
    BSON.@save path closes vols rel_vols hourly_closes dates hourly_dts companies bar_minutes exchange history_closes history_datetimes
    mb = (sizeof(closes) + sizeof(vols) + sizeof(rel_vols) + sizeof(hourly_closes) + sizeof(hist_closes)) / 1e6
    @info "Cache saved → $path  ($(@sprintf("%.1f", mb)) MB)"

    last_bar = history_last_bar_start.(hist_dts)
    return InferenceCache(closes, vols, rel_vols, hourly_closes, dates, hourly_dts, companies,
                          date_idx, Dict(s => i for (i, s) in enumerate(companies)),
                          bar_minutes, exchange, hist_closes, hist_dts, last_bar,
                          history_end_index(hourly_dts, last_bar))
end

# ── Load ──────────────────────────────────────────────────────────────────────

"""
Load a pre-built inference cache from `path`.

The index dicts are re-derived on load (they are not serialised).
Typical load time: 1–3 seconds for a ~42 MB file.
"""
function load_inference_cache(path::String)::InferenceCache
    isfile(path) || error("Cache not found: $path\nRun: julia --project=packages/StockSwingPredictor scripts/build_cache.jl")
    d             = BSON.load(path)
    closes        = d[:closes]
    vols          = d[:vols]
    hourly_closes = d[:hourly_closes]
    dates         = d[:dates]
    hourly_dts    = d[:hourly_dts]
    companies     = d[:companies]
    # rel_vols added in a later version — neutral fallback for old cache files
    rel_vols      = get(d, :rel_vols, ones(Float32, size(closes)))
    hist_closes = get(d, :history_closes, zeros(Float32, 0, length(companies)))
    hist_dts    = get(d, :history_datetimes, DateTime[])
    last_bar    = history_last_bar_start.(hist_dts)
    InferenceCache(
        closes, vols, rel_vols, hourly_closes, dates, hourly_dts, companies,
        Dict(dt => i for (i, dt) in enumerate(dates)),
        Dict(s  => i for (i, s)  in enumerate(companies)),
        get(d, :bar_minutes, 60), get(d, :exchange, "nse"),
        hist_closes, hist_dts, last_bar, history_end_index(hourly_dts, last_bar),
    )
end

# ── Lookup helpers ────────────────────────────────────────────────────────────

"""
Index of the last hourly bar at or before `dt`. Returns 0 if none exists.
Used at inference time to find the current position in the hourly matrix.
"""
function find_hourly_end(cache::InferenceCache, dt::DateTime)::Int
    searchsortedlast(cache.hourly_datetimes, dt)
end

"""
Index of `d` in `cache.dates`. Returns 0 if the date is not a trading day.
"""
function find_date(cache::InferenceCache, d::Date)::Int
    get(cache.date_index, d, 0)
end
