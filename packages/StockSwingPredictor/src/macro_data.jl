"""
Macro instrument historical OHLCV fetching.

Two sources:
  - Yahoo Finance (no auth) — global indices and USD/INR spot
  - Kite Connect (session required) — India VIX (NSE index) and INR-denominated
    commodities via MCX continuous futures (history from ~2015)

Output format: `out_dir/{NAME}_daily.csv` — same schema as equity OHLCV (date, open,
high, low, close, volume). Compatible with `load_cached_ohlcv`.
"""

# ── Instrument catalogue ──────────────────────────────────────────────────────

"""Yahoo Finance ticker mappings for global macro instruments."""
const YAHOO_MACRO_INSTRUMENTS = [
    ("SP500",    "^GSPC"),    # S&P 500
    ("US_VIX",   "^VIX"),    # CBOE Volatility Index
    ("USD_INR",  "USDINR=X"), # USD/INR spot (full history; CDS continuous only from 2022)
]

"""
Kite Connect macro instruments (INR-denominated, continuous futures).

MCX continuous futures history starts ~2015. The `tradingsymbol` field is the
exact commodity name — token resolution matches `SYMBOL` followed by a 2-digit
year to avoid mini/micro variants (e.g. GOLDM, CRUDEOILM, SILVERMIC).
"""
const KITE_MACRO_INSTRUMENTS = [
    (name="INDIA_VIX",   exchange="NSE", tradingsymbol="INDIA VIX",  continuous=false),
    (name="CRUDE_OIL",   exchange="MCX", tradingsymbol="CRUDEOIL",   continuous=true),
    (name="GOLD",        exchange="MCX", tradingsymbol="GOLD",        continuous=true),
    (name="SILVER",      exchange="MCX", tradingsymbol="SILVER",      continuous=true),
    (name="NATURAL_GAS", exchange="MCX", tradingsymbol="NATURALGAS",  continuous=true),
    (name="COPPER",      exchange="MCX", tradingsymbol="COPPER",      continuous=true),
]

const YAHOO_CHART_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"
const _YAHOO_HEADERS = [
    "User-Agent" => "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Accept"     => "application/json",
]

# ── Yahoo Finance fetcher ─────────────────────────────────────────────────────

"""
Fetch daily OHLCV from Yahoo Finance.

# Arguments
- `ticker`: Yahoo Finance symbol (e.g. "^GSPC", "CL=F", "FCPO.KLS")
- `from_date`, `to_date`: inclusive date range

# Returns
DataFrame with columns: date, open, high, low, close, volume.
Rows with no close price (futures rollover gaps, non-trading days) are dropped.
Returns empty DataFrame on any failure.
"""
function fetch_yahoo_ohlcv(ticker::String, from_date::Date, to_date::Date)::DataFrame
    p1  = string(Int(Dates.datetime2unix(DateTime(from_date))))
    p2  = string(Int(Dates.datetime2unix(DateTime(to_date, Time(23, 59, 59)))))
    # Encode ^ and other special chars without pulling in URI deps
    safe_ticker = replace(replace(ticker, "^" => "%5E"), "=" => "%3D")
    url = "$YAHOO_CHART_BASE/$safe_ticker?interval=1d&period1=$p1&period2=$p2"

    resp = try
        HTTP.get(url; headers=_YAHOO_HEADERS, request_timeout=20, status_exception=false)
    catch e
        @warn "Yahoo fetch failed ($ticker): $(sprint(showerror, e))"
        return DataFrame()
    end

    resp.status != 200 && begin
        @warn "Yahoo HTTP $(resp.status) for $ticker"
        return DataFrame()
    end

    raw = try JSON3.read(resp.body) catch
        @warn "Yahoo JSON parse failed for $ticker"
        return DataFrame()
    end

    chart  = get(raw,   :chart,  nothing); isnothing(chart)  && return DataFrame()
    result = get(chart, :result, nothing)
    (isnothing(result) || isempty(result)) && return DataFrame()

    r          = result[1]
    timestamps = get(r, :timestamp, nothing); isnothing(timestamps) && return DataFrame()
    inds       = get(r, :indicators, nothing); isnothing(inds)      && return DataFrame()
    qlist      = get(inds, :quote, nothing)
    (isnothing(qlist) || isempty(qlist)) && return DataFrame()
    q = qlist[1]

    _f(v) = (isnothing(v) || ismissing(v)) ? NaN   : Float64(v)
    _v(v) = (isnothing(v) || ismissing(v)) ? 0.0   : Float64(v)

    dates  = [Date(Dates.unix2datetime(Float64(t))) for t in timestamps]
    opens  = [_f(v) for v in get(q, :open,   [])]
    highs  = [_f(v) for v in get(q, :high,   [])]
    lows   = [_f(v) for v in get(q, :low,    [])]
    closes = [_f(v) for v in get(q, :close,  [])]
    vols   = [_v(v) for v in get(q, :volume, [])]

    n  = minimum(length, (dates, opens, highs, lows, closes, vols))
    df = DataFrame(date   = dates[1:n],  open  = opens[1:n],
                   high   = highs[1:n],  low   = lows[1:n],
                   close  = closes[1:n], volume = vols[1:n])
    filter!(row -> !isnan(row.close), df)
    return sort!(df, :date)
end

# ── Kite macro instrument helpers ─────────────────────────────────────────────

"""
Load Kite instruments master for any exchange segment.
Caches alongside `INSTRUMENTS_CACHE`; refreshes once per calendar day.

# Arguments
- `session`: Kite session named tuple
- `exchange`: "NSE", "CDS", "MCX", etc.
"""
function load_instruments_for_exchange(session; exchange::String)::DataFrame
    cache_path = joinpath(dirname(INSTRUMENTS_CACHE),
                          "_instruments_$(exchange)_cache.csv")
    if isfile(cache_path)
        cache_date = Date(Dates.unix2datetime(stat(cache_path).mtime))
        cache_date == today() && return CSV.read(cache_path, DataFrame)
    end

    resp = HTTP.get("$KITE_BASE/instruments/$exchange";
                    headers=_kite_headers(session), request_timeout=30)
    resp.status == 200 || error("Kite instruments/$exchange returned HTTP $(resp.status)")

    df = CSV.read(IOBuffer(resp.body), DataFrame)
    mkpath(dirname(cache_path))
    CSV.write(cache_path, df)
    return df
end

"""
Resolve Kite instrument tokens for all `KITE_MACRO_INSTRUMENTS`.

- NSE: exact tradingsymbol match on INDICES type (India VIX).
- MCX: regex `^SYMBOL\\d{2}` match on FUT type — the digit guard excludes mini/micro
  variants (GOLDM, CRUDEOILM, SILVERMIC, etc.). Picks nearest non-expired contract;
  Kite stitches the continuous series regardless of which contract token is used.

# Returns
Dict mapping name → (token::Int, continuous::Bool)
"""
function build_macro_kite_tokens(session)::Dict{String, Tuple{Int, Bool}}
    result    = Dict{String, Tuple{Int, Bool}}()
    today_str = string(today())

    for inst in KITE_MACRO_INSTRUMENTS
        df = try
            load_instruments_for_exchange(session; exchange=inst.exchange)
        catch e
            @warn "Could not load $(inst.exchange) instruments: $(sprint(showerror, e))"
            continue
        end

        if inst.exchange == "NSE"
            # India VIX has instrument_type="EQ" but segment="INDICES"
            found = filter(r -> string(r.tradingsymbol) == inst.tradingsymbol &&
                                string(r.segment) == "INDICES", df)
            isempty(found) && begin
                @warn "$(inst.name) not found in NSE instruments"; continue
            end
            result[inst.name] = (Int(found[1, :instrument_token]), inst.continuous)

        else  # MCX — match SYMBOL immediately followed by 2-digit year then a letter.
              # e.g. GOLD26OCT… matches; GOLDM26…, SILVER10026… do not.
            pat = Regex("^$(inst.tradingsymbol)\\d{2}[A-Z]")
            found = filter(r -> !isnothing(match(pat, string(r.tradingsymbol))) &&
                                string(r.instrument_type) == "FUT" &&
                                string(get(r, :expiry, "")) >= today_str, df)
            if isempty(found)
                found = filter(r -> !isnothing(match(pat, string(r.tradingsymbol))) &&
                                    string(r.instrument_type) == "FUT", df)
            end
            isempty(found) && begin
                @warn "$(inst.name) ($(inst.tradingsymbol)) not found in MCX instruments"; continue
            end
            # Lowest instrument_token = oldest listed contract = deepest continuous history
            sorted = sort(found, :instrument_token)
            result[inst.name] = (Int(sorted[1, :instrument_token]), inst.continuous)
        end
    end

    return result
end

"""
Fetch daily OHLCV from Kite for a macro instrument.

Automatically chunks requests into 1800-day windows to stay under Kite's
2000-day-per-request limit for daily bars.

# Arguments
- `token`: Kite instrument token
- `from_date`, `to_date`: inclusive date range
- `session`: Kite session
- `continuous`: use Kite continuous futures stitching (set true for FUT instruments)
"""
function fetch_kite_macro_ohlcv(token::Int, from_date::Date, to_date::Date,
                                  session; continuous::Bool=false)::DataFrame
    cont       = continuous ? 1 : 0
    chunk_days = 1800   # stay under Kite's 2000-day daily-bar limit
    all_chunks = DataFrame[]

    chunk_start = from_date
    while chunk_start <= to_date
        chunk_end = min(chunk_start + Day(chunk_days), to_date)
        from_s = Dates.format(chunk_start, "yyyy-mm-dd") * "+09:00:00"
        to_s   = Dates.format(chunk_end,   "yyyy-mm-dd") * "+15:30:00"
        url    = "$KITE_BASE/instruments/historical/$token/day" *
                 "?from=$from_s&to=$to_s&continuous=$cont&oi=0"

        resp = try
            HTTP.get(url; headers=_kite_headers(session), request_timeout=30,
                     status_exception=false)
        catch e
            @warn "Kite macro fetch failed (token $token, $chunk_start…$chunk_end): $(sprint(showerror, e))"
            chunk_start = chunk_end + Day(1); sleep(0.35); continue
        end

        if resp.status == 200
            raw = try JSON3.read(resp.body) catch; nothing end
            if !isnothing(raw)
                cd   = get(raw, :data,    nothing)
                carr = isnothing(cd) ? nothing : get(cd, :candles, nothing)
                if !isnothing(carr) && !isempty(carr)
                    rows = [(
                        date   = Date(string(c[1])[1:10]),
                        open   = Float64(c[2]),
                        high   = Float64(c[3]),
                        low    = Float64(c[4]),
                        close  = Float64(c[5]),
                        volume = Float64(c[6]),
                    ) for c in carr]
                    push!(all_chunks, DataFrame(rows))
                end
            end
        else
            @warn "Kite macro HTTP $(resp.status) for token $token ($chunk_start…$chunk_end)"
        end

        chunk_start = chunk_end + Day(1)
        sleep(0.35)
    end

    isempty(all_chunks) && return DataFrame()
    return sort!(vcat(all_chunks...), :date)
end

# ── 5-minute macro fetcher ───────────────────────────────────────────────────

"""
Fetch 5-minute OHLCV from Kite for a macro instrument.

Uses a wide time window (09:00–23:59) to cover both NSE (closes 15:30) and
MCX (closes 23:30) trading hours. Requests are chunked into 90-day windows.

Note: Kite does not support `continuous=1` for intraday intervals — 5-min
data always uses the specific contract (continuous=0). For MCX, this means
the current front-month contract; contract rollovers create a brief gap
which the daily update fills naturally as the active token changes.

# Arguments
- `token`: Kite instrument token (front-month contract from `build_macro_kite_tokens`)
- `from_date`, `to_date`: inclusive date range (within 100-day retention window)
- `session`: Kite session
"""
function fetch_kite_macro_5min(token::Int, from_date::Date, to_date::Date,
                                 session)::DataFrame
    cont       = 0   # continuous=1 is invalid for intraday intervals on Kite
    chunk_days = 90
    all_chunks = DataFrame[]

    chunk_start = from_date
    while chunk_start <= to_date
        chunk_end = min(chunk_start + Day(chunk_days), to_date)
        from_s = Dates.format(chunk_start, "yyyy-mm-dd") * "+09:00:00"
        to_s   = Dates.format(chunk_end,   "yyyy-mm-dd") * "+23:59:00"
        url    = "$KITE_BASE/instruments/historical/$token/5minute" *
                 "?from=$from_s&to=$to_s&continuous=$cont&oi=0"

        resp = try
            HTTP.get(url; headers=_kite_headers(session), request_timeout=30,
                     status_exception=false)
        catch e
            @warn "Macro 5min fetch failed (token $token, $chunk_start…$chunk_end): $(sprint(showerror, e))"
            chunk_start = chunk_end + Day(1); sleep(0.35); continue
        end

        if resp.status == 200
            raw = try JSON3.read(resp.body) catch; nothing end
            if !isnothing(raw)
                cd   = get(raw, :data,    nothing)
                carr = isnothing(cd) ? nothing : get(cd, :candles, nothing)
                if !isnothing(carr) && !isempty(carr)
                    rows = [(
                        datetime = DateTime(string(c[1])[1:19], "yyyy-mm-ddTHH:MM:SS"),
                        open     = Float64(c[2]),
                        high     = Float64(c[3]),
                        low      = Float64(c[4]),
                        close    = Float64(c[5]),
                        volume   = Float64(c[6]),
                    ) for c in carr]
                    push!(all_chunks, DataFrame(rows))
                end
            end
        else
            @warn "Macro 5min HTTP $(resp.status) for token $token ($chunk_start…$chunk_end)"
        end

        chunk_start = chunk_end + Day(1)
        sleep(0.35)
    end

    isempty(all_chunks) && return DataFrame()
    return sort!(vcat(all_chunks...), :datetime)
end

"""
Fetch and cache 5-minute OHLCV for all `KITE_MACRO_INSTRUMENTS`.

Only covers the 100-day Kite retention window — call daily to keep current.
Skips instruments already cached unless `refresh=true`.

# Arguments
- `session`: Kite session from `load_kite_session`
- `out_dir`: output directory (default: `website/data/ohlcv/macro`)
- `from_date`: start date (default: 99 days ago; Kite retains only 100 days)
- `to_date`: end date (default: yesterday)
- `refresh`: re-fetch even if file exists
"""
function collect_macro_5min(session;
                             out_dir::String = MACRO_OHLCV_DIR,
                             from_date::Date = today() - Day(99),
                             to_date::Date   = today() - Day(1),
                             refresh::Bool   = false)
    mkpath(out_dir)
    ok = skipped = failed = 0

    token_map = try build_macro_kite_tokens(session) catch e
        @warn "Token resolution failed: $(sprint(showerror, e))"
        Dict{String, Tuple{Int, Bool}}()
    end

    for inst in KITE_MACRO_INSTRUMENTS
        path = joinpath(out_dir, "$(inst.name)_5min.csv")
        if !refresh && isfile(path)
            @info "  $(inst.name) 5min — cached, skipping"
            skipped += 1; continue
        end

        entry = get(token_map, inst.name, nothing)
        if isnothing(entry)
            @warn "  $(inst.name) — token not found, skipping"
            failed += 1; continue
        end
        token, continuous = entry

        df = fetch_kite_macro_5min(token, from_date, to_date, session)
        if isempty(df)
            @warn "  $(inst.name) 5min — no data from Kite"
            failed += 1; continue
        end
        CSV.write(path, df)
        @info "  $(inst.name) 5min — $(nrow(df)) bars  $(df.datetime[1]) → $(df.datetime[end])"
        ok += 1
        sleep(0.35)
    end

    @info "Macro 5min done: $ok fetched, $skipped skipped, $failed failed"
end

"""
Load cached 5-minute macro OHLCV for a named instrument.

# Arguments
- `name`: instrument name (e.g. "CRUDE_OIL", "INDIA_VIX")
- `out_dir`: directory to read from (default: `website/data/ohlcv/macro`)
"""
function load_macro_5min(name::String;
                          out_dir::String = MACRO_OHLCV_DIR)::DataFrame
    path = joinpath(out_dir, "$(name)_5min.csv")
    isfile(path) || return DataFrame()
    return CSV.read(path, DataFrame; types=Dict(:datetime => DateTime))
end

# ── Collection ────────────────────────────────────────────────────────────────

const MACRO_OHLCV_DIR = joinpath(@__DIR__, "..", "..", "..", "website", "data",
                                  "ohlcv", "macro")

"""
Fetch and cache all macro instrument OHLCV.

Yahoo instruments are fetched first (no auth). Kite instruments follow (session
required for token resolution). Instruments already cached are skipped unless
`refresh=true`.

# Arguments
- `session`: Kite session from `load_kite_session`
- `out_dir`: output directory (default: `website/data/ohlcv/macro`)
- `from_date`: history start (default: 2010-01-01)
- `to_date`: history end (default: yesterday)
- `refresh`: re-fetch all even if file exists

# Example
```julia
session = load_kite_session(pwd())
collect_macro_ohlcv(session)
```
"""
function collect_macro_ohlcv(session;
                              out_dir::String   = MACRO_OHLCV_DIR,
                              from_date::Date   = Date(2010, 1, 1),
                              to_date::Date     = today() - Day(1),
                              refresh::Bool     = false)
    mkpath(out_dir)
    ok = skipped = failed = 0

    # ── Yahoo ─────────────────────────────────────────────────────────────────
    @info "Yahoo Finance — fetching $(length(YAHOO_MACRO_INSTRUMENTS)) instruments"
    for (name, ticker) in YAHOO_MACRO_INSTRUMENTS
        path = joinpath(out_dir, "$(name)_daily.csv")
        if !refresh && isfile(path)
            @info "  $name — cached, skipping"
            skipped += 1; continue
        end
        df = fetch_yahoo_ohlcv(ticker, from_date, to_date)
        if isempty(df)
            @warn "  $name ($ticker) — no data returned"
            failed += 1; continue
        end
        CSV.write(path, df)
        @info "  $name ($ticker) — $(nrow(df)) days"
        ok += 1
        sleep(0.5)   # rate-limit Yahoo requests
    end

    # ── Kite ──────────────────────────────────────────────────────────────────
    @info "Kite — resolving $(length(KITE_MACRO_INSTRUMENTS)) macro tokens"
    token_map = try
        build_macro_kite_tokens(session)
    catch e
        @warn "Token resolution failed: $(sprint(showerror, e))"
        Dict{String, Tuple{Int, Bool}}()
    end

    for inst in KITE_MACRO_INSTRUMENTS
        path = joinpath(out_dir, "$(inst.name)_daily.csv")
        if !refresh && isfile(path)
            @info "  $(inst.name) — cached, skipping"
            skipped += 1; continue
        end
        entry = get(token_map, inst.name, nothing)
        if isnothing(entry)
            @warn "  $(inst.name) — token not found, skipping"
            failed += 1; continue
        end
        token, continuous = entry
        df = fetch_kite_macro_ohlcv(token, from_date, to_date, session;
                                     continuous=continuous)
        if isempty(df)
            @warn "  $(inst.name) — no data from Kite"
            failed += 1; continue
        end
        CSV.write(path, df)
        @info "  $(inst.name) — $(nrow(df)) days"
        ok += 1
        sleep(0.35)
    end

    @info "Macro OHLCV done: $ok fetched, $skipped skipped, $failed failed"
end

"""
Load cached macro OHLCV for a named instrument.

# Arguments
- `name`: instrument name (e.g. "SP500", "INDIA_VIX", "CRUDE_OIL")
- `out_dir`: directory to read from (default: `website/data/ohlcv/macro`)

# Returns
DataFrame with columns: date, open, high, low, close, volume.
Returns empty DataFrame if not cached.
"""
function load_macro_ohlcv(name::String;
                           out_dir::String = MACRO_OHLCV_DIR)::DataFrame
    path = joinpath(out_dir, "$(name)_daily.csv")
    isfile(path) || return DataFrame()
    return CSV.read(path, DataFrame; types=Dict(:date => Date))
end
