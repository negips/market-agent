"""
RSS feed fetcher and parser.

Parses RSS 2.0 feeds without an XML library dependency — the format is
regular enough that regex extraction is reliable for the fields we need.
"""

using HTTP, Dates

const RSS_HEADERS = [
    "User-Agent" => "Mozilla/5.0 (compatible; market-agent/1.0)",
    "Accept"     => "application/rss+xml, application/xml, text/xml, */*",
]

const _MONTH_ABBR = Dict("Jan"=>1,"Feb"=>2,"Mar"=>3,"Apr"=>4,"May"=>5,"Jun"=>6,
                         "Jul"=>7,"Aug"=>8,"Sep"=>9,"Oct"=>10,"Nov"=>11,"Dec"=>12)

"""
Fetch and parse an RSS 2.0 feed.

# Arguments
- `url`: feed URL
- `label`: short source label used in `NewsItem.source` (e.g. "ET")

# Returns
`Vector{NewsItem}` — empty on failure.
"""
function fetch_rss(url::String, label::String)::Vector{NewsItem}
    resp = try
        HTTP.get(url; headers=RSS_HEADERS, request_timeout=15, status_exception=false)
    catch e
        @warn "RSS fetch failed ($label): $(sprint(showerror, e))"
        return NewsItem[]
    end

    resp.status != 200 && begin
        @warn "RSS HTTP $(resp.status) for $label"
        return NewsItem[]
    end

    return _parse_rss(String(resp.body), label)
end

function _parse_rss(xml::String, label::String)::Vector{NewsItem}
    items = NewsItem[]
    for m in eachmatch(r"<item[^>]*>(.*?)</item>"s, xml)
        block = m[1]
        title = _tag(block, "title")
        desc  = _strip_html(_tag(block, "description"))
        link  = _tag(block, "link")
        guid  = _tag(block, "guid")
        isempty(guid) && (guid = link)
        isempty(guid) && continue
        pub   = _tag(block, "pubDate")
        dt    = _parse_rss_date(pub)

        push!(items, NewsItem(
            guid         = "RSS:$label:$guid",
            source       = "RSS:$label",
            headline     = title,
            body         = desc,
            url          = link,
            published_at = dt,
        ))
    end
    return items
end

function _tag(xml::String, name::String)::String
    m = match(Regex("<$name[^>]*><!\\[CDATA\\[(.*?)\\]\\]></$name>", "s"), xml)
    !isnothing(m) && return strip(m[1])
    m = match(Regex("<$name[^>]*>(.*?)</$name>", "s"), xml)
    !isnothing(m) && return strip(m[1])
    return ""
end

function _strip_html(s::String)::String
    s = replace(s, r"<[^>]+>" => " ")
    s = replace(s, r"&amp;"   => "&")
    s = replace(s, r"&lt;"    => "<")
    s = replace(s, r"&gt;"    => ">")
    s = replace(s, r"&nbsp;"  => " ")
    s = replace(s, r"\s+"     => " ")
    return strip(s)
end

function _parse_rss_date(s::String)::DateTime
    isempty(s) && return now(UTC)
    # RFC 822: "Wed, 09 Sep 2026 10:30:00 +0530"
    m = match(r"(\d{1,2})\s+([A-Za-z]{3})\s+(\d{4})\s+(\d{2}):(\d{2}):(\d{2})", s)
    isnothing(m) && return now(UTC)
    try
        day   = parse(Int, m[1])
        month = get(_MONTH_ABBR, m[2], 1)
        year  = parse(Int, m[3])
        h, mn, sc = parse(Int, m[4]), parse(Int, m[5]), parse(Int, m[6])
        return DateTime(year, month, day, h, mn, sc)
    catch
        return now(UTC)
    end
end
