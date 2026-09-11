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
`hourly_closes[h, j]` — 60-min close for company j at hourly bar h (forward-filled).

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
end

# ── Build ─────────────────────────────────────────────────────────────────────

"""
Build the inference cache from cached OHLCV CSVs and save to `path`.

Reads all `{SYMBOL}_daily.csv` and `{SYMBOL}_hourly.csv` in `ohlcv_dir`
for the given `companies`, aligns them onto a shared time axis, forward-fills
missing bars, and writes a single BSON file (~42 MB for 942 companies).

# Arguments
- `ohlcv_dir`: directory containing the OHLCV CSVs
- `companies`: symbols in the desired column order (sets the universe)
- `path`: output BSON file path
"""
function build_inference_cache(ohlcv_dir::String, companies::Vector{String},
                                path::String)::InferenceCache
    n_comp = length(companies)

    @info "Building inference cache — $(n_comp) companies"

    # ── Daily matrices ────────────────────────────────────────────────────────

    all_dates = Set{Date}()
    for sym in companies
        p = joinpath(ohlcv_dir, "$(sym)_daily.csv")
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
        p = joinpath(ohlcv_dir, "$(sym)_daily.csv")
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

    # ── Hourly matrix ─────────────────────────────────────────────────────────

    all_dts = Set{DateTime}()
    for sym in companies
        p = joinpath(ohlcv_dir, "$(sym)_hourly.csv")
        isfile(p) || continue
        df = CSV.read(p, DataFrame; select=[:datetime], types=Dict(:datetime => DateTime))
        union!(all_dts, df.datetime)
    end
    hourly_dts = sort(collect(all_dts))
    n_hourly   = length(hourly_dts)
    dt_idx     = Dict(dt => i for (i, dt) in enumerate(hourly_dts))

    hourly_closes = fill(NaN32, n_hourly, n_comp)

    for (j, sym) in enumerate(companies)
        p = joinpath(ohlcv_dir, "$(sym)_hourly.csv")
        isfile(p) || continue
        df = CSV.read(p, DataFrame; types=Dict(:datetime => DateTime))

        for row in eachrow(df)
            i = get(dt_idx, row.datetime, 0)
            i == 0 && continue
            hourly_closes[i, j] = Float32(row.close)
        end

        last_h = NaN32
        for i in 1:n_hourly
            if !isnan(hourly_closes[i, j])
                last_h = hourly_closes[i, j]
            elseif !isnan(last_h)
                hourly_closes[i, j] = last_h
            end
        end

        j % 200 == 0 && @info "  [$j/$n_comp] hourly loaded"
    end

    @info "  Hourly: $n_hourly bars × $n_comp companies"

    # ── Save ──────────────────────────────────────────────────────────────────

    mkpath(dirname(path))
    BSON.@save path closes vols rel_vols hourly_closes dates hourly_dts companies
    mb = (sizeof(closes) + sizeof(vols) + sizeof(rel_vols) + sizeof(hourly_closes)) / 1e6
    @info "Cache saved → $path  ($(@sprintf("%.1f", mb)) MB)"

    return InferenceCache(closes, vols, rel_vols, hourly_closes, dates, hourly_dts, companies,
                          date_idx, Dict(s => i for (i, s) in enumerate(companies)))
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
    InferenceCache(
        closes, vols, rel_vols, hourly_closes, dates, hourly_dts, companies,
        Dict(dt => i for (i, dt) in enumerate(dates)),
        Dict(s  => i for (i, s)  in enumerate(companies)),
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
