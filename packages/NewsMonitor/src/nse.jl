"""
NSE corporate announcements fetcher.

Polls the NSE JSON API for corporate announcements. This is the primary
high-signal source for sharp single-day moves: board meeting outcomes,
merger/acquisition disclosures, QIPs, buybacks, and results filings.

The API provides roughly 20 years of history (full density from 2004–05)
and includes `attchmntText` summaries suitable for direct LLM classification.
No authentication is required; a valid Referer and XMLHttpRequest header are
needed to avoid 403 responses.
"""

using HTTP, JSON3, Dates

const NSE_ANN_URL = "https://www.nseindia.com/api/corporate-announcements"
const NSE_HEADERS = [
    "Referer"          => "https://www.nseindia.com/",
    "User-Agent"       => "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Accept"           => "application/json, text/plain, */*",
    "X-Requested-With" => "XMLHttpRequest",
]

"""
Fetch corporate announcements from NSE for the given date range.

# Arguments
- `from_date`, `to_date`: inclusive date range (default: today)
- `index`: NSE segment — `"equities"` (default) or `"sme"`

# Returns
`Vector{NewsItem}` — empty on any fetch or parse failure.
"""
function fetch_nse_announcements(; from_date::Date=today(),
                                   to_date::Date=today(),
                                   index::String="equities")::Vector{NewsItem}
    from_s = Dates.format(from_date, "dd-mm-yyyy")
    to_s   = Dates.format(to_date,   "dd-mm-yyyy")
    url    = "$NSE_ANN_URL?index=$index&from_date=$from_s&to_date=$to_s"

    resp = try
        HTTP.get(url; headers=NSE_HEADERS, request_timeout=15, status_exception=false)
    catch e
        @warn "NSE announcements fetch failed: $(sprint(showerror, e))"
        return NewsItem[]
    end

    resp.status != 200 && begin
        @warn "NSE API returned HTTP $(resp.status)"
        return NewsItem[]
    end

    rows = try JSON3.read(resp.body) catch
        @warn "NSE response parse failed"
        return NewsItem[]
    end

    # API returns a plain array; an error response is an object
    rows isa AbstractVector || return NewsItem[]

    items = NewsItem[]
    for row in rows
        symbol   = string(get(row, :symbol, ""))
        file_url = string(get(row, :attchmntFile, ""))

        # Use PDF filename as guid (includes timestamp → unique per filing).
        # Fall back to symbol+datetime when there is no attachment.
        guid = isempty(file_url) ?
               "NSE:$(symbol):$(get(row, :an_dt, ""))" :
               "NSE:" * basename(file_url)

        push!(items, NewsItem(
            guid         = guid,
            source       = "NSE",
            headline     = string(get(row, :desc, "")),
            body         = string(get(row, :attchmntText, "")),
            url          = file_url,
            published_at = _parse_nse_dt(string(get(row, :an_dt, ""))),
            nse_symbol   = symbol,
        ))
    end
    return items
end

function _parse_nse_dt(s::String)::DateTime
    isempty(s) && return now(UTC)
    try
        # NSE format: "2026-09-09 10:30:00" (IST, no TZ offset marker)
        return DateTime(s[1:min(19, length(s))], "yyyy-mm-dd HH:MM:SS")
    catch
        return now(UTC)
    end
end
