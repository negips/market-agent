"""
Kite Connect historical OHLCV data fetching.

Loads the NSE instrument list, resolves symbols to instrument tokens, and
fetches daily OHLCV candles via the Kite Connect historical data API.

Rate limiting: Kite allows ~3 requests/second for historical data. All public
functions that make multiple requests add a 400 ms inter-request sleep.
"""

using HTTP, JSON3, CSV, DataFrames, Dates

const KITE_BASE             = "https://api.kite.trade"
const NSE_INSTRUMENTS_CACHE = joinpath(@__DIR__, "..", "..", "..", "website", "data",
                                       "ohlcv", "_instruments_nse_cache.csv")
const BSE_INSTRUMENTS_CACHE = joinpath(@__DIR__, "..", "..", "..", "website", "data",
                                       "ohlcv", "_instruments_bse_cache.csv")

# Known NSE index tradingsymbols → looked up from instruments list at runtime.
const NSE_INDICES = [
    "NIFTY 50",
    "NIFTY BANK",
    "NIFTY IT",
    "NIFTY AUTO",
    "NIFTY PHARMA",
    "NIFTY FMCG",
    "NIFTY METAL",
    "NIFTY ENERGY",
    "NIFTY REALTY",
    "NIFTY MIDCAP 100",
]

# Maps Tijori sector names (from nse_companies_latest.json) to NSE index names.
const SECTOR_TO_INDEX = Dict(
    "Information Technology" => "NIFTY IT",
    "Banking"                => "NIFTY BANK",
    "Financial Services"     => "NIFTY BANK",
    "Automobile"             => "NIFTY AUTO",
    "Auto Ancillaries"       => "NIFTY AUTO",
    "Pharmaceutical"         => "NIFTY PHARMA",
    "Healthcare"             => "NIFTY PHARMA",
    "FMCG"                   => "NIFTY FMCG",
    "Consumer Goods"         => "NIFTY FMCG",
    "Metal"                  => "NIFTY METAL",
    "Mining"                 => "NIFTY METAL",
    "Oil & Gas"              => "NIFTY ENERGY",
    "Energy"                 => "NIFTY ENERGY",
    "Realty"                 => "NIFTY REALTY",
    "Real Estate"            => "NIFTY REALTY",
)

"""
Read the Kite session from kite_session.json.
Returns `(api_key, access_token)` or throws if not found / expired.
"""
function load_kite_session(repo_root::String)
    path = joinpath(repo_root, "sidecar", "kite_session.json")
    isfile(path) || error("No Kite session found. Run: node sidecar/kite_login.js")
    s = JSON3.read(read(path, String))
    date_str = string(get(s, :date, ""))
    if date_str != string(today())
        @warn "Kite session is from $date_str — token may be stale. " *
              "Run: node sidecar/kite_login.js"
    end
    return (api_key=string(s.api_key), access_token=string(s.access_token))
end

function _kite_headers(session)
    ["X-Kite-Version" => "3",
     "Authorization"  => "token $(session.api_key):$(session.access_token)"]
end

"""
Download the instrument list from Kite for a given exchange and cache it locally.
Returns a DataFrame with columns: instrument_token, tradingsymbol, name,
instrument_type, segment, exchange.

# Arguments
- `session`: named tuple from `load_kite_session`
- `exchange`: `"NSE"` (default) or `"BSE"`
- `refresh`: force re-download even if cache exists (default false)
"""
function load_instruments(session; exchange::String="NSE", refresh::Bool=false)::DataFrame
    cache = exchange == "BSE" ? BSE_INSTRUMENTS_CACHE : NSE_INSTRUMENTS_CACHE
    if !refresh && isfile(cache)
        return CSV.read(cache, DataFrame)
    end

    resp = HTTP.get("$KITE_BASE/instruments/$exchange";
                    headers=_kite_headers(session), request_timeout=30)
    resp.status == 200 || error("Instruments endpoint returned HTTP $(resp.status)")

    df = CSV.read(IOBuffer(resp.body), DataFrame)

    mkpath(dirname(cache))
    CSV.write(cache, df)
    return df
end

"""
Build a symbol → instrument_token lookup dict for EQ stocks (and NSE INDICES).

# Arguments
- `instruments`: DataFrame from `load_instruments`
- `exchange`: `"NSE"` (default) or `"BSE"`
"""
function build_token_map(instruments::DataFrame; exchange::String="NSE")::Dict{String, Int}
    map = Dict{String, Int}()
    for row in eachrow(instruments)
        type = string(get(row, :instrument_type, ""))
        exch = string(get(row, :exchange, ""))
        exch == exchange || continue
        if type == "EQ" || (exchange == "NSE" && type == "INDICES")
            sym = string(row.tradingsymbol)
            map[sym] = Int(row.instrument_token)
        end
    end
    return map
end

"""
Fetch daily OHLCV candles from Kite for one instrument.

# Arguments
- `token`: Kite instrument_token (integer)
- `from_date`, `to_date`: inclusive date range
- `session`: named tuple with api_key, access_token

# Returns
DataFrame with columns: date, open, high, low, close, volume
Sorted ascending by date. Returns empty DataFrame on failure.
"""
function fetch_ohlcv(token::Int, from_date::Date, to_date::Date, session)::DataFrame
    from_s = Dates.format(from_date, "yyyy-mm-dd") * "+09:15:00"
    to_s   = Dates.format(to_date,   "yyyy-mm-dd") * "+15:30:00"
    url = "$KITE_BASE/instruments/historical/$token/day?from=$from_s&to=$to_s&continuous=0&oi=0"

    resp = try
        HTTP.get(url; headers=_kite_headers(session), request_timeout=20,
                 status_exception=false)
    catch e
        @warn "OHLCV fetch failed for token $token: $(sprint(showerror, e))"
        return DataFrame()
    end

    if resp.status != 200
        @warn "OHLCV HTTP $(resp.status) for token $token"
        return DataFrame()
    end

    raw = try
        JSON3.read(resp.body)
    catch
        @warn "Could not parse OHLCV response for token $token"
        return DataFrame()
    end

    candles = get(raw, :data, nothing)
    candles === nothing && return DataFrame()
    candles_arr = get(candles, :candles, nothing)
    (candles_arr === nothing || isempty(candles_arr)) && return DataFrame()

    rows = [(
        date   = Date(string(c[1])[1:10]),
        open   = Float64(c[2]),
        high   = Float64(c[3]),
        low    = Float64(c[4]),
        close  = Float64(c[5]),
        volume = Float64(c[6]),
    ) for c in candles_arr]

    return sort!(DataFrame(rows), :date)
end

"""
Fetch and save OHLCV for a list of symbols. Skips symbols already cached
unless `refresh=true`. Saves each symbol to `out_dir/{SYMBOL}_daily.csv`.

# Arguments
- `symbols`: NSE tradingsymbols
- `token_map`: from `build_token_map`
- `session`: Kite session
- `out_dir`: directory for CSV output
- `from_date`, `to_date`: date range
- `refresh`: re-fetch even if file exists
"""
function collect_ohlcv(symbols::Vector{String}, token_map::Dict{String,Int},
                       session, out_dir::String,
                       from_date::Date, to_date::Date;
                       refresh::Bool=false)
    mkpath(out_dir)
    ok = skipped = failed = 0
    total = length(symbols)

    for (i, sym) in enumerate(symbols)
        path = joinpath(out_dir, "$(sym)_daily.csv")
        if !refresh && isfile(path)
            skipped += 1
            continue
        end

        token = get(token_map, sym, nothing)
        if isnothing(token)
            @warn "[$i/$total] No instrument token for $sym — skipping"
            failed += 1
            continue
        end

        df = fetch_ohlcv(token, from_date, to_date, session)
        if isempty(df)
            failed += 1
            @warn "[$i/$total] $sym — empty response"
            continue
        end

        CSV.write(path, df)
        ok += 1
        @info "[$i/$total] $sym — $(nrow(df)) days"
        sleep(0.35)   # ~3 req/s rate limit
    end

    @info "OHLCV collection done: $ok fetched, $skipped skipped, $failed failed"
end

"""
Load a cached OHLCV CSV for a symbol. Returns empty DataFrame if not found.
"""
function load_cached_ohlcv(symbol::String, out_dir::String)::DataFrame
    path = joinpath(out_dir, "$(symbol)_daily.csv")
    isfile(path) || return DataFrame()
    return CSV.read(path, DataFrame; types=Dict(:date => Date))
end

"""
Map a Tijori sector string to the closest NSE sector index name.
Returns "NIFTY 50" as the default when no specific mapping exists.
"""
function sector_index_name(sector::String)::String
    get(SECTOR_TO_INDEX, sector, "NIFTY 50")
end

# ── Hourly (60-minute) OHLCV ─────────────────────────────────────────────────

"""
Fetch 60-minute OHLCV candles from Kite for one instrument.

Kite limits intraday historical data to 60-day windows per request. This
function automatically chunks the date range and concatenates the results.

# Returns
DataFrame with columns: datetime, open, high, low, close, volume.
Sorted ascending by datetime. Returns empty DataFrame on failure.
"""
function fetch_ohlcv_hourly(token::Int, from_date::Date, to_date::Date,
                             session)::DataFrame
    chunk_days = 59   # stay under the 60-day Kite limit
    all_chunks = DataFrame[]

    chunk_start = from_date
    while chunk_start <= to_date
        chunk_end = min(chunk_start + Day(chunk_days), to_date)
        from_s = Dates.format(chunk_start, "yyyy-mm-dd") * "+09:00:00"
        to_s   = Dates.format(chunk_end,   "yyyy-mm-dd") * "+15:30:00"
        url = "$KITE_BASE/instruments/historical/$token/60minute?from=$from_s&to=$to_s&continuous=0&oi=0"

        resp = try
            HTTP.get(url; headers=_kite_headers(session), request_timeout=30,
                     status_exception=false)
        catch e
            @warn "Hourly fetch error for token $token ($chunk_start…$chunk_end): $(sprint(showerror, e))"
            chunk_start = chunk_end + Day(1)
            sleep(0.35)
            continue
        end

        if resp.status == 200
            raw = try JSON3.read(resp.body) catch; nothing end
            if !isnothing(raw)
                candles_data = get(raw, :data, nothing)
                candles_arr  = isnothing(candles_data) ? nothing :
                               get(candles_data, :candles, nothing)
                if !isnothing(candles_arr) && !isempty(candles_arr)
                    rows = [(
                        datetime = DateTime(string(c[1])[1:19], "yyyy-mm-ddTHH:MM:SS"),
                        open     = Float64(c[2]),
                        high     = Float64(c[3]),
                        low      = Float64(c[4]),
                        close    = Float64(c[5]),
                        volume   = Float64(c[6]),
                    ) for c in candles_arr]
                    push!(all_chunks, DataFrame(rows))
                end
            end
        else
            @warn "Hourly HTTP $(resp.status) for token $token ($chunk_start…$chunk_end)"
        end

        chunk_start = chunk_end + Day(1)
        sleep(0.35)
    end

    isempty(all_chunks) && return DataFrame()
    return sort!(vcat(all_chunks...), :datetime)
end

"""
Fetch and cache 60-minute OHLCV for a list of symbols.
Output: `out_dir/{SYMBOL}_hourly.csv`.
"""
function collect_ohlcv_hourly(symbols::Vector{String}, token_map::Dict{String,Int},
                               session, out_dir::String,
                               from_date::Date, to_date::Date;
                               refresh::Bool=false)
    mkpath(out_dir)
    ok = skipped = failed = 0
    total = length(symbols)

    for (i, sym) in enumerate(symbols)
        path = joinpath(out_dir, "$(sym)_hourly.csv")
        if !refresh && isfile(path)
            skipped += 1
            continue
        end

        token = get(token_map, sym, nothing)
        if isnothing(token)
            @warn "[$i/$total] No token for $sym — skipping hourly"
            failed += 1
            continue
        end

        df = fetch_ohlcv_hourly(token, from_date, to_date, session)
        if isempty(df)
            failed += 1
            @warn "[$i/$total] $sym — empty hourly response"
            continue
        end

        CSV.write(path, df)
        ok += 1
        @info "[$i/$total] $sym hourly — $(nrow(df)) bars"
    end

    @info "Hourly OHLCV done: $ok fetched, $skipped skipped, $failed failed"
end

"""
Load cached 60-minute OHLCV for a symbol. Returns empty DataFrame if not found.
"""
function load_cached_ohlcv_hourly(symbol::String, out_dir::String)::DataFrame
    path = joinpath(out_dir, "$(symbol)_hourly.csv")
    isfile(path) || return DataFrame()
    return CSV.read(path, DataFrame; types=Dict(:datetime => DateTime))
end

# ── 5-minute OHLCV ───────────────────────────────────────────────────────────

"""
Fetch 5-minute OHLCV candles from Kite for one instrument.

Kite retains 5-minute bars for 100 days. Requests are chunked into 90-day
windows to stay safely within any per-request limits.

# Returns
DataFrame with columns: datetime, open, high, low, close, volume.
Sorted ascending by datetime. Returns empty DataFrame on failure.
"""
function fetch_ohlcv_5min(token::Int, from_date::Date, to_date::Date,
                           session)::DataFrame
    chunk_days = 90
    all_chunks = DataFrame[]

    chunk_start = from_date
    while chunk_start <= to_date
        chunk_end = min(chunk_start + Day(chunk_days), to_date)
        from_s = Dates.format(chunk_start, "yyyy-mm-dd") * "+09:15:00"
        to_s   = Dates.format(chunk_end,   "yyyy-mm-dd") * "+15:30:00"
        url = "$KITE_BASE/instruments/historical/$token/5minute" *
              "?from=$from_s&to=$to_s&continuous=0&oi=0"

        resp = try
            HTTP.get(url; headers=_kite_headers(session), request_timeout=30,
                     status_exception=false)
        catch e
            @warn "5min fetch error for token $token ($chunk_start…$chunk_end): $(sprint(showerror, e))"
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
            @warn "5min HTTP $(resp.status) for token $token ($chunk_start…$chunk_end)"
        end

        chunk_start = chunk_end + Day(1)
        sleep(0.35)
    end

    isempty(all_chunks) && return DataFrame()
    return sort!(vcat(all_chunks...), :datetime)
end

"""
Fetch and cache 5-minute OHLCV for a list of symbols.
Output: `out_dir/{SYMBOL}_5min.csv`.
"""
function collect_ohlcv_5min(symbols::Vector{String}, token_map::Dict{String,Int},
                              session, out_dir::String,
                              from_date::Date, to_date::Date;
                              refresh::Bool=false)
    mkpath(out_dir)
    ok = skipped = failed = 0
    total = length(symbols)

    for (i, sym) in enumerate(symbols)
        path = joinpath(out_dir, "$(sym)_5min.csv")
        if !refresh && isfile(path)
            skipped += 1; continue
        end

        token = get(token_map, sym, nothing)
        if isnothing(token)
            @warn "[$i/$total] No token for $sym — skipping 5min"
            failed += 1; continue
        end

        df = fetch_ohlcv_5min(token, from_date, to_date, session)
        if isempty(df)
            failed += 1
            @warn "[$i/$total] $sym — empty 5min response"
            continue
        end

        CSV.write(path, df)
        ok += 1
        @info "[$i/$total] $sym 5min — $(nrow(df)) bars"
    end

    @info "5min OHLCV done: $ok fetched, $skipped skipped, $failed failed"
end

"""
Load cached 5-minute OHLCV for a symbol. Returns empty DataFrame if not found.
"""
function load_cached_ohlcv_5min(symbol::String, out_dir::String)::DataFrame
    path = joinpath(out_dir, "$(symbol)_5min.csv")
    isfile(path) || return DataFrame()
    return CSV.read(path, DataFrame; types=Dict(:datetime => DateTime))
end

# ── 15-minute bars ────────────────────────────────────────────────────────────

"""
Fetch 15-minute OHLCV bars for a single NSE instrument token.

Kite retains 15-minute bars for 200 days. Requests are chunked into 175-day
windows to stay safely within that limit.

# Arguments
- `token`: Kite instrument token
- `from_date`, `to_date`: inclusive date range (within 200-day retention window)
- `session`: Kite session

# Returns
DataFrame with columns: datetime, open, high, low, close, volume.
Sorted ascending by datetime. Returns empty DataFrame on failure.
"""
function fetch_ohlcv_15min(token::Int, from_date::Date, to_date::Date,
                            session)::DataFrame
    chunk_days = 175
    all_chunks = DataFrame[]

    chunk_start = from_date
    while chunk_start <= to_date
        chunk_end = min(chunk_start + Day(chunk_days), to_date)
        from_s = Dates.format(chunk_start, "yyyy-mm-dd") * "+09:15:00"
        to_s   = Dates.format(chunk_end,   "yyyy-mm-dd") * "+15:30:00"
        url = "$KITE_BASE/instruments/historical/$token/15minute" *
              "?from=$from_s&to=$to_s&continuous=0&oi=0"

        resp = try
            HTTP.get(url; headers=_kite_headers(session), request_timeout=30,
                     status_exception=false)
        catch e
            @warn "15min fetch error for token $token ($chunk_start…$chunk_end): $(sprint(showerror, e))"
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
            @warn "15min HTTP $(resp.status) for token $token ($chunk_start…$chunk_end)"
        end

        chunk_start = chunk_end + Day(1)
        sleep(0.35)
    end

    isempty(all_chunks) && return DataFrame()
    return sort!(vcat(all_chunks...), :datetime)
end

"""
Fetch and cache 15-minute OHLCV for a list of symbols.
Output: `out_dir/{SYMBOL}_15min.csv`.
"""
function collect_ohlcv_15min(symbols::Vector{String}, token_map::Dict{String,Int},
                               session, out_dir::String,
                               from_date::Date, to_date::Date;
                               refresh::Bool=false)
    mkpath(out_dir)
    ok = skipped = failed = 0
    total = length(symbols)

    for (i, sym) in enumerate(symbols)
        path = joinpath(out_dir, "$(sym)_15min.csv")
        if !refresh && isfile(path)
            skipped += 1; continue
        end

        token = get(token_map, sym, nothing)
        if isnothing(token)
            @warn "[$i/$total] No token for $sym — skipping 15min"
            failed += 1; continue
        end

        df = fetch_ohlcv_15min(token, from_date, to_date, session)
        if isempty(df)
            failed += 1
            @warn "[$i/$total] $sym — empty 15min response"
            continue
        end

        CSV.write(path, df)
        ok += 1
        @info "[$i/$total] $sym 15min — $(nrow(df)) bars"
    end

    @info "15min OHLCV done: $ok fetched, $skipped skipped, $failed failed"
end

"""
Load cached 15-minute OHLCV for a symbol. Returns empty DataFrame if not found.
"""
function load_cached_ohlcv_15min(symbol::String, out_dir::String)::DataFrame
    path = joinpath(out_dir, "$(symbol)_15min.csv")
    isfile(path) || return DataFrame()
    return CSV.read(path, DataFrame; types=Dict(:datetime => DateTime))
end
