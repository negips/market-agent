const _NSE_BASE = "https://www.nseindia.com"

const _HEADERS = [
    "User-Agent"       => "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
    "Accept"           => "application/json, text/plain, */*",
    "Accept-Language"  => "en-US,en;q=0.9",
    "Referer"          => "https://www.nseindia.com/",
    "X-Requested-With" => "XMLHttpRequest",
]

# NSE blocks direct API calls without a prior homepage hit that sets session cookies.
function _nse_headers_with_cookies()::Vector{Pair{String,String}}
    resp = try
        HTTP.get(_NSE_BASE; headers=_HEADERS, redirect=true,
                 status_exception=false, request_timeout=15)
    catch
        return _HEADERS
    end

    cookie_parts = String[]
    for (k, v) in resp.headers
        lowercase(k) == "set-cookie" || continue
        push!(cookie_parts, split(v, ";")[1])   # name=value only
    end

    isempty(cookie_parts) && return _HEADERS
    return [_HEADERS..., "Cookie" => join(cookie_parts, "; ")]
end

function _parse_nse_date(s::AbstractString)::Union{Date, Nothing}
    s = strip(s)
    (isempty(s) || s == "-") && return nothing
    for fmt in (dateformat"dd-u-yyyy", dateformat"dd-mm-yyyy")
        try return Date(s, fmt) catch end
    end
    return nothing
end

function _is_earnings(purpose::AbstractString)::Bool
    p = lowercase(purpose)
    occursin("result", p) || occursin("board meeting", p)
end

"""
    fetch_earnings_calendar(from::Date, to::Date) -> Vector{EarningsEvent}

Fetch all earnings-related NSE corporate events between `from` and `to` inclusive.
Results are sorted by date ascending.

Hits `nseindia.com/api/event-calendar` — no sidecar required.

# Arguments
- `from` — start date
- `to`   — end date (inclusive)

# Returns
`Vector{EarningsEvent}` sorted by date. Returns an empty vector if no events match.

# Throws
- `NSEFetchError` on HTTP error or unparseable response.

# Example
```julia
events = fetch_earnings_calendar(Date(2026, 10, 1), Date(2026, 10, 31))
```
"""
function fetch_earnings_calendar(from::Date, to::Date)::Vector{EarningsEvent}
    from_s = Dates.format(from, dateformat"dd-mm-yyyy")
    to_s   = Dates.format(to,   dateformat"dd-mm-yyyy")
    url    = "$_NSE_BASE/api/event-calendar?index=equities&from_date=$from_s&to_date=$to_s"

    headers = _nse_headers_with_cookies()

    resp = try
        HTTP.get(url; headers=headers, request_timeout=20, status_exception=false)
    catch e
        throw(NSEFetchError("Request failed: $(sprint(showerror, e))"))
    end

    resp.status == 200 ||
        throw(NSEFetchError("HTTP $(resp.status) from NSE event-calendar API"))

    raw = try
        JSON3.read(resp.body)
    catch
        throw(NSEFetchError("Could not parse NSE response as JSON"))
    end

    events = EarningsEvent[]
    for item in raw
        # NSE uses :purpose on event-calendar, :subject on some other endpoints
        purpose = string(get(item, :purpose, get(item, :subject, "")))
        _is_earnings(purpose) || continue

        # Date field name varies across NSE endpoints
        date_str = string(get(item, :date,
                    get(item, :nd_startDate,
                    get(item, :bm_date, ""))))
        dt = _parse_nse_date(date_str)
        isnothing(dt) && continue

        symbol  = strip(string(get(item, :symbol, "")))
        company = strip(string(get(item, :company, get(item, :companyName, ""))))
        isempty(symbol) && continue

        push!(events, EarningsEvent(symbol, company, dt, purpose))
    end

    sort!(events, by = e -> e.date)
    return events
end

"""
    upcoming_earnings(days::Int=30; from::Date=today()) -> Vector{EarningsEvent}

Fetch NSE earnings announcements for the next `days` calendar days starting from `from`.

# Arguments
- `days` — window length in calendar days (default: 30)
- `from` — start date (default: today)

# Example
```julia
events = upcoming_earnings()        # next 30 days
events = upcoming_earnings(7)       # next week only
events = upcoming_earnings(90; from=Date(2026, 10, 1))
```
"""
function upcoming_earnings(days::Int=30; from::Date=today())::Vector{EarningsEvent}
    days > 0 || throw(ArgumentError("days must be positive, got $days"))
    fetch_earnings_calendar(from, from + Day(days - 1))
end
