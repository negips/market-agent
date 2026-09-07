"""
Stock screener: query Tijori's 5,000+ company database by financial metrics.

Three modes:
  screen_companies(query)           — free-form query string
  screen_companies(preset="...")    — pre-built Tijori screen by name
  list_screens()                    — browse available pre-built screens
  search_fields(query)              — discover ~1,500 available metric names
"""

"""
    screen_companies(query; kwargs...) -> ScreenResult
    screen_companies(; preset, kwargs...) -> ScreenResult

Screen 5,000+ NSE/BSE listed companies by financial metrics.

# Positional form (free-form query string):
```julia
screen_companies("( ROCE > 20 ) and ( Market Capitalization > 1000 )")
screen_companies("( ROE > 15 ) and ( Debt to Equity Ratio < 1 ) and ( 3yr Growth Net Sales > 15 )")
```

# Keyword-only forms:
```julia
screen_companies(preset="Monopoly Companies")
screen_companies(alternate="market share > 50")
screen_companies(alternate="revenue from Defence > 50", filters="Market Capitalization > 500")
```

# Keyword arguments:
- `preset`              — name of a Tijori pre-built screen (see `list_screens()`)
- `alternate`           — business-data query (market share, revenue by segment)
- `latest_results_only` — only companies that have filed the latest quarter
- `superstar_investors` — only companies held by prominent investors
- `sme`                 — search the SME-listed universe instead
- `offset`              — starting row for pagination (default 0)
- `limit`               — max rows returned (default 50)

Use `search_fields` to discover valid metric names for free-form queries.

# Example
```julia
r = screen_companies("( ROCE > 25 ) and ( Market Capitalization > 5000 )")
println(r.total_results, " companies match")
r.data   # DataFrame with one row per company
```
"""
function screen_companies(
    query::String = "";
    preset::Union{String, Nothing}=nothing,
    alternate::Union{String, Nothing}=nothing,
    latest_results_only::Bool=false,
    superstar_investors::Bool=false,
    sme::Bool=false,
    offset::Int=0,
    limit::Int=50,
)::ScreenResult

    payload = Dict{String, Any}(
        "latest_results_only" => latest_results_only,
        "superstar_investors" => superstar_investors,
        "sme"                 => sme,
        "offset"              => offset,
        "limit"               => limit,
    )
    if !isnothing(preset)
        payload["preset"] = preset
    elseif !isempty(query)
        payload["filters"] = query
    end
    isnothing(alternate) || (payload["alternate"] = alternate)

    raw = _post("/screen", payload)
    return _parse_screen_result(raw, isnothing(preset) ? (isempty(query) ? alternate : query) : preset)
end

# ── Pre-built screens ─────────────────────────────────────────────────────────

"""
    list_screens() -> DataFrame

List all of Tijori's pre-built stock screens, grouped by category.

Returns a DataFrame with columns:
- `category`    — screen category (e.g. "Value", "Quality", "Growth")
- `name`        — screen name (use this as `preset` in `screen_companies`)
- `description` — what the screen looks for

# Example
```julia
screens = list_screens()
filter(row -> row.category == "Quality", screens)
# Then run one:
screen_companies(preset="Monopoly Companies")
```
"""
function list_screens()::DataFrame
    raw = _get("/screens")
    isempty(raw) && return DataFrame(category=String[], name=String[], description=String[])
    return DataFrame(
        category    = [string(get(s, :category, "")) for s in raw],
        name        = [string(get(s, :name, "")) for s in raw],
        description = [string(get(s, :description, "")) for s in raw],
    )
end

# ── Field catalog ─────────────────────────────────────────────────────────────

"""
    search_fields(query) -> DataFrame

Search Tijori's ~1,500-metric field catalog to find exact metric names for use
in `screen_companies` queries.

Returns a DataFrame with columns `name`, `type`, `unit`.

# Example
```julia
search_fields("roce")      # find all ROCE variants
search_fields("npa")       # NPA-related metrics for banks
search_fields("promoter")  # promoter holding metrics
```
"""
function search_fields(query::String)::DataFrame
    isempty(strip(query)) && error("query must not be empty")
    raw = _get("/fields"; q=query)
    fields = get(raw, :fields, raw)
    isempty(fields) && return DataFrame(name=String[], type=String[], unit=String[])
    return DataFrame(
        name = [string(f.name) for f in fields],
        type = [string(get(f, :type, "")) for f in fields],
        unit = [string(get(f, :unit, "")) for f in fields],
    )
end

# ── Market and sector data ────────────────────────────────────────────────────

"""
    get_markets(; type=nothing) -> DataFrame

Fetch index performance data.

`type` controls which index group is returned:
- `nothing` (default) — main indices (Nifty 50, Bank Nifty, etc.)
- `"niche"`            — Tijori's niche sector indices
- `"conglomerates"`    — business group indices

Returns a DataFrame with index names and return columns.

# Example
```julia
get_markets()                      # Nifty, Bank Nifty, ...
get_markets(type="niche")          # niche sector indices + their tjiids
```
"""
function get_markets(; type::Union{String, Nothing}=nothing)::DataFrame
    raw = isnothing(type) ? _get("/markets") : _get("/markets"; type=type)
    _raw_to_df(raw)
end

"""
    get_sector_stocks(tjiid) -> DataFrame

All stocks inside a Tijori niche sector index. `tjiid` comes from
`get_markets(type="niche")`.

Returns a DataFrame with slug, market-cap weight, and price return columns.
"""
function get_sector_stocks(tjiid::String)::DataFrame
    _raw_to_df(_get("/sector"; tjiid=tjiid))
end

"""
    get_conglomerate_stocks(tjiid) -> DataFrame

All companies inside a business group (e.g. Tata, Reliance). `tjiid` comes
from `get_markets(type="conglomerates")`.
"""
function get_conglomerate_stocks(tjiid::String)::DataFrame
    _raw_to_df(_get("/conglomerate"; tjiid=tjiid))
end

"""
    get_macro_indicators() -> Any

India macro indicators: credit growth, IIP, GST collections, auto sales,
GDP, and trade data as returned by Tijori.
"""
function get_macro_indicators()
    _get("/macro")
end

"""
    get_raw_materials() -> Any

Commodity price performance: chemicals, metal spreads, crude. Useful for
assessing input cost pressure on material-intensive companies.
"""
function get_raw_materials()
    _get("/rawmaterials")
end

# ── Internal helpers ──────────────────────────────────────────────────────────

function _parse_screen_result(raw, query)::ScreenResult
    results = get(raw, :results, [])
    df = if isempty(results)
        DataFrame()
    else
        all_keys = unique(vcat([:slug, :name], [k for r in results for k in keys(r)]))
        d = DataFrame()
        for k in all_keys
            col_vals = [get(r, k, nothing) for r in results]
            # Use String for text fields, try Float64 for numerics
            if all(v -> isnothing(v) || v isa Number, col_vals)
                d[!, k] = [isnothing(v) ? missing : Float64(v) for v in col_vals]
            else
                d[!, k] = [isnothing(v) ? "" : string(v) for v in col_vals]
            end
        end
        d
    end

    return ScreenResult(
        Int(get(raw, :total_results, length(results))),
        Int(get(raw, :returned, length(results))),
        Int(get(raw, :offset, 0)),
        isnothing(query) ? nothing : string(query),
        df,
    )
end

function _raw_to_df(raw)::DataFrame
    raw isa Vector || return DataFrame()
    isempty(raw) && return DataFrame()
    all_keys = unique([k for r in raw for k in keys(r)])
    d = DataFrame()
    for k in all_keys
        vals = [get(r, k, nothing) for r in raw]
        if all(v -> isnothing(v) || v isa Number, vals)
            d[!, k] = [isnothing(v) ? missing : Float64(v) for v in vals]
        else
            d[!, k] = [isnothing(v) ? "" : string(v) for v in vals]
        end
    end
    return d
end
