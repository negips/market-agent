"""
Company search, overview, and knowledge base functions.
"""

# ── Search ────────────────────────────────────────────────────────────────────

"""
    search_company(query) -> Vector{SearchResult}

Search Tijori Finance for companies matching `query`. Returns up to ~10 results
ordered by relevance.

# Example
```julia
results = search_company("HDFC Bank")
slug = results[1].slug   # "hdfc-bank-limited"
```
"""
function search_company(query::String)::Vector{SearchResult}
    isempty(strip(query)) && error("query must not be empty")
    raw = _get("/search"; q=query)
    return [SearchResult(string(r.name), string(r.slug)) for r in raw]
end

# ── Overview ──────────────────────────────────────────────────────────────────

"""
    get_overview(slug) -> CompanyOverview

Fetch a company's key ratios, market cap, P/E, and Tijori's forensics score.

`slug` is the identifier returned by `search_company` (e.g. "hdfc-bank-limited").

# Example
```julia
ov = get_overview("hdfc-bank-limited")
println(ov.forensics)        # Tijori's fraud risk assessment
println(ov.ratios["ROE"])    # "18.4%"
```
"""
function get_overview(slug::String)::CompanyOverview
    raw = _get("/overview"; slug=slug)

    ratios = OrderedDict{String, String}()
    if haskey(raw, :ratios)
        for (k, v) in pairs(raw.ratios)
            ratios[string(k)] = string(v)
        end
    end

    forensics = haskey(raw, :quick_look) && !isnothing(raw.quick_look) ?
        Dict{String, Any}(string(k) => v for (k, v) in pairs(raw.quick_look)) :
        nothing

    return CompanyOverview(
        slug,
        string(get(raw, :company, slug)),
        _str_or_nothing(get(raw, :shortname, nothing)),
        _str_or_nothing(get(raw, :symbol, nothing)),
        _int_or_nothing(get(raw, :company_id, nothing)),
        _str_or_nothing(get(raw, :ind_code, nothing)),
        Bool(get(raw, :is_banking, false)),
        _f64_or_nothing(get(raw, :mcap, nothing)),
        _f64_or_nothing(get(raw, :pe, nothing)),
        ratios,
        forensics,
    )
end

# ── Knowledge base ────────────────────────────────────────────────────────────

"""
    get_knowledge_base(slug) -> KnowledgeBase

List all investor documents available for a company: annual reports, earnings
releases, investor presentations, and conference call transcripts.

URLs in the returned `Document` objects are authenticated Tijori CDN links.
Pass them to `fetch_document` to retrieve the full text.

# Example
```julia
kb = get_knowledge_base("hdfc-bank-limited")
# Most recent annual report
ar = kb.annual_reports[end]
println(ar.period)  # "FY24"
text = fetch_document(ar.url)
```
"""
function get_knowledge_base(slug::String)::KnowledgeBase
    raw = _get("/knowledge"; slug=slug)

    parse_docs(key) = [
        Document(string(d.period), string(d.url))
        for d in get(raw, key, [])
    ]

    return KnowledgeBase(
        slug,
        parse_docs(:annual_reports),
        parse_docs(:earnings_releases),
        parse_docs(:investor_presentations),
        parse_docs(:conference_calls),
    )
end

# ── ID resolution ─────────────────────────────────────────────────────────────

"""
    resolve_id(slug) -> Int

Resolve a company slug to its numeric Tijori company_id.
Required for `get_fund_flow`, which takes a numeric ID.

# Example
```julia
id = resolve_id("hdfc-bank-limited")   # e.g. 1234
ff = get_fund_flow(id, 5)
```
"""
function resolve_id(slug::String)::Int
    raw = _get("/resolve"; slug=slug)
    id = get(raw, :company_id, nothing)
    isnothing(id) && error("Could not resolve company_id for slug: $slug")
    return Int(id)
end

# ── Internal helpers ──────────────────────────────────────────────────────────

_str_or_nothing(x) = isnothing(x) ? nothing : string(x)
_f64_or_nothing(x) = isnothing(x) ? nothing : _parse_number(x)
_int_or_nothing(x) = isnothing(x) ? nothing : Int(x)
