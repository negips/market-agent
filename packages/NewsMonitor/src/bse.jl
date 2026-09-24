"""
BSE corporate announcements fetcher.

Polls the BSE RSS feed for today's corporate announcements. No authentication
is required. The feed provides a `<scripcode>` (BSE numeric code) alongside
each filing description, enabling reliable LLM classification with known
company identity.

Historical data is not available via this endpoint — BSE's JSON APIs require
authentication. For historical filings on cross-listed companies, use
`fetch_nse_announcements` which has 20+ years of history.
"""

using HTTP, Dates

const BSE_RSS_URL = "https://www.bseindia.com/data/xml/announcements.aspx"
const BSE_RSS_HEADERS = [
    "Referer"    => "https://www.bseindia.com/",
    "User-Agent" => "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Accept"     => "application/rss+xml, application/xml, text/xml, */*",
]

const _BSE_MONTHS = Dict("Jan"=>1,"Feb"=>2,"Mar"=>3,"Apr"=>4,"May"=>5,"Jun"=>6,
                         "Jul"=>7,"Aug"=>8,"Sep"=>9,"Oct"=>10,"Nov"=>11,"Dec"=>12)

"""
Fetch today's corporate announcements from the BSE RSS feed.

Note: the BSE RSS endpoint always returns the current day's filings regardless
of date arguments. The `from_date`/`to_date` parameters are accepted for API
consistency with `fetch_nse_announcements` but are ignored.

# Arguments
- `from_date`, `to_date`: accepted for interface symmetry; BSE RSS is today-only

# Returns
`Vector{NewsItem}` — empty on any fetch or parse failure.
"""
function fetch_bse_announcements(; from_date::Date=today(),
                                   to_date::Date=today())::Vector{NewsItem}
    resp = try
        HTTP.get(BSE_RSS_URL; headers=BSE_RSS_HEADERS, request_timeout=15,
                 status_exception=false)
    catch e
        @warn "BSE RSS fetch failed: $(sprint(showerror, e))"
        return NewsItem[]
    end

    resp.status != 200 && begin
        @warn "BSE RSS returned HTTP $(resp.status)"
        return NewsItem[]
    end

    return _parse_bse_rss(String(resp.body))
end

function _parse_bse_rss(xml::String)::Vector{NewsItem}
    items = NewsItem[]
    for m in eachmatch(r"<item[^>]*>(.*?)</item>"s, xml)
        block     = m[1]
        title     = _bse_tag(block, "title")       # "Company Name (scripcode)"
        desc      = _bse_tag(block, "description") # filing description text
        link      = _bse_tag(block, "link")
        scripcode = _bse_tag(block, "scripcode")
        pubdate   = _bse_tag(block, "pubDate")

        isempty(link) && isempty(scripcode) && continue

        # PDF filename contains a UUID → globally unique per filing
        guid = isempty(link) ? "BSE:$(scripcode):$pubdate" :
                               "BSE:" * basename(link)

        push!(items, NewsItem(
            guid         = guid,
            source       = "BSE",
            headline     = _bse_decode(desc),  # event description first for LLM
            body         = title,              # "Company Name (scripcode)" for symbol resolution
            url          = link,
            published_at = _parse_bse_rss_dt(pubdate),
            bse_code     = scripcode,
        ))
    end
    return items
end

function _bse_tag(xml::String, name::String)::String
    m = match(Regex("<$name[^>]*><!\\[CDATA\\[(.*?)\\]\\]></$name>", "s"), xml)
    !isnothing(m) && return strip(m[1])
    m = match(Regex("<$name[^>]*>(.*?)</$name>", "s"), xml)
    !isnothing(m) && return strip(m[1])
    return ""
end

function _bse_decode(s::String)::String
    s = replace(s, "&amp;"  => "&")
    s = replace(s, "&lt;"   => "<")
    s = replace(s, "&gt;"   => ">")
    s = replace(s, "&apos;" => "'")
    s = replace(s, "&#39;"  => "'")
    s = replace(s, "&nbsp;" => " ")
    return strip(s)
end

function _parse_bse_rss_dt(s::String)::DateTime
    isempty(s) && return now(UTC)
    # BSE RSS date format: "23-Sep-2026 22:31:38" (IST, no TZ offset)
    m = match(r"(\d{1,2})-([A-Za-z]{3})-(\d{4})\s+(\d{2}):(\d{2}):(\d{2})", s)
    isnothing(m) && return now(UTC)
    try
        day   = parse(Int, m[1])
        month = get(_BSE_MONTHS, m[2], 1)
        year  = parse(Int, m[3])
        h, mn, sc = parse(Int, m[4]), parse(Int, m[5]), parse(Int, m[6])
        return DateTime(year, month, day, h, mn, sc)
    catch
        return now(UTC)
    end
end
