"""
BSE corporate announcements fetcher.

Polls the BSE JSON API for corporate announcements. This is the highest-signal
source for sharp single-day moves: board meeting outcomes, merger/acquisition
disclosures, QIPs, buybacks, and surprise results all appear here first.

The endpoint is BSE's internal API used by bseindia.com; no authentication is
required but a Referer header is needed to avoid 403 responses.
"""

using HTTP, JSON3, Dates

const BSE_ANN_URL = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
const BSE_HEADERS = [
    "Referer"    => "https://www.bseindia.com/",
    "User-Agent" => "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Accept"     => "application/json, text/plain, */*",
]

"""
Fetch corporate announcements from BSE for the given date range.

# Arguments
- `from_date`, `to_date`: inclusive date range (default: today)

# Returns
`Vector{NewsItem}` — empty on any fetch or parse failure.
"""
function fetch_bse_announcements(; from_date::Date=today(),
                                   to_date::Date=today())::Vector{NewsItem}
    from_s = Dates.format(from_date, "dd/mm/yyyy")
    to_s   = Dates.format(to_date,   "dd/mm/yyyy")
    url = BSE_ANN_URL *
          "?pageno=1&category=-1&subcategory=-1&scripcode=" *
          "&strdate=$from_s&enddate=$to_s&bcategory=-1"

    resp = try
        HTTP.get(url; headers=BSE_HEADERS, request_timeout=15, status_exception=false)
    catch e
        @warn "BSE announcements fetch failed: $(sprint(showerror, e))"
        return NewsItem[]
    end

    resp.status != 200 && begin
        @warn "BSE API returned HTTP $(resp.status)"
        return NewsItem[]
    end

    raw = try JSON3.read(resp.body) catch
        @warn "BSE response parse failed"
        return NewsItem[]
    end

    table = get(raw, :Table, nothing)
    (isnothing(table) || isempty(table)) && return NewsItem[]

    items = NewsItem[]
    for row in table
        news_id = string(get(row, :NEWSID, ""))
        isempty(news_id) && continue

        dt = _parse_bse_dt(string(get(row, :NEWS_DT, "")))
        push!(items, NewsItem(
            guid         = "BSE:$news_id",
            source       = "BSE",
            headline     = string(get(row, :HEADLINE, "")),
            body         = string(get(row, :SUBCATNAME, "")),
            url          = string(get(row, :NSURL, "")),
            published_at = dt,
            bse_code     = string(get(row, :SCRIP_CD, "")),
        ))
    end
    return items
end

function _parse_bse_dt(s::String)::DateTime
    isempty(s) && return now(UTC)
    try
        # BSE format: "2026-09-09T10:30:00" (IST, no offset marker)
        return DateTime(s[1:min(19, length(s))], "yyyy-mm-ddTHH:MM:SS")
    catch
        return now(UTC)
    end
end
