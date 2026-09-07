"""
Financial statement and operational metrics functions.

All financial data is returned as DataFrames with `metric` as the first column
and fiscal period labels (e.g. "FY24", "Sep 24") as subsequent columns.
Numeric values are Float64; unavailable values are `nothing` (not `missing`).
"""

# ── Financial statements ──────────────────────────────────────────────────────

"""
    get_financials(slug, type) -> DataFrame

Fetch a financial statement for `slug`.

`type` must be one of:
- `:pl`        — Profit & Loss (annual)
- `:bs`        — Balance Sheet (annual)
- `:cf`        — Cash Flow Statement (annual)
- `:ratios`    — Key financial ratios (annual)
- `:quarterly` — Quarterly P&L results

Returns a DataFrame where each row is a line item (e.g. "Net Revenue") and
each column (after `metric`) is a fiscal period.

# Example
```julia
pl = get_financials("hdfc-bank-limited", :pl)
# metric column holds row labels; FY20..FY24 hold values in ₹ Cr
cfo = get_financials("hdfc-bank-limited", :cf)
```
"""
function get_financials(slug::String, type::Symbol)::DataFrame
    valid = (:pl, :bs, :cf, :ratios, :quarterly)
    type in valid || error("type must be one of: $(join(valid, ", ")). Got: $type")
    raw = _get("/financials"; slug=slug, type=string(type))
    return _financials_to_df(raw)
end

# ── Operational metrics ───────────────────────────────────────────────────────

"""
    get_operational_metrics(slug) -> Vector{MetricSeries}

Fetch all operational KPIs for a company. Each `MetricSeries` contains the
metric name, unit, latest value, and full historical time series as a DataFrame.

Operational KPIs are company-specific (e.g. "Passenger Load Factor" for
airlines, "GNPA Ratio" for banks, "Revenue per Room" for hotels).

# Example
```julia
metrics = get_operational_metrics("interglobe-aviation-limited")
for m in metrics
    println(m.name, " (", m.unit, "): ", m.latest_value)
end
```
"""
function get_operational_metrics(slug::String)::Vector{MetricSeries}
    raw = _get("/metrics"; slug=slug)
    return [
        MetricSeries(
            string(m.name),
            _str_or_nothing(get(m, :unit, nothing)),
            _f64_or_nothing(get(m, :latest_value, nothing)),
            _str_or_nothing(get(m, :latest_date, nothing)),
            _history_to_df(get(m, :history, [])),
        )
        for m in raw.metrics
    ]
end

# ── Fund flow ─────────────────────────────────────────────────────────────────

"""
    get_fund_flow(company_id, years) -> NamedTuple

Capital allocation breakdown: where money came from and where it went over the
specified number of years.

`years` must be one of: 1, 3, 5, 7, 10.
`company_id` is Tijori's numeric ID — obtain it from `resolve_id(slug)`.

Returns a NamedTuple with fields:
- `sources` — capital inflows
- `uses`    — capital outflows

# Example
```julia
id = resolve_id("tata-consultancy-services-limited")
ff = get_fund_flow(id, 5)
```
"""
function get_fund_flow(company_id::Int, years::Int)::NamedTuple
    valid = (1, 3, 5, 7, 10)
    years in valid || error("years must be one of: $(join(valid, ", ")). Got: $years")
    raw = _get("/fundflow"; company_id=company_id, years=years)
    sources = [Dict{String, Any}(string(k) => v for (k, v) in pairs(s)) for s in get(raw, :sources, [])]
    uses    = [Dict{String, Any}(string(k) => v for (k, v) in pairs(u)) for u in get(raw, :uses,    [])]
    return (sources=sources, uses=uses, company_id=company_id, years=years)
end

# ── Revenue mix ───────────────────────────────────────────────────────────────

"""
    get_revenue_mix(slug) -> Vector{NamedTuple}

Segment revenue breakdown with historical trend per segment.
Each element represents one revenue segment chart (e.g. "Geography Mix",
"Product Mix"). Each segment has a `history` DataFrame with [:date, :value].

# Example
```julia
mix = get_revenue_mix("titan-company-limited")
for chart in mix
    println(chart.title)
    println(chart.latest_breakdown)
end
```
"""
function get_revenue_mix(slug::String)::Vector{NamedTuple}
    raw = _get("/revenuemix"; slug=slug)
    return [
        (
            title=string(get(c, :title, "")),
            chart_id=get(c, :chart_id, nothing),
            latest_breakdown=[
                (name=string(s.name), pct=Float64(s.pct))
                for s in get(c, :latest_breakdown, [])
            ],
            segments=[
                (
                    name=string(s.name),
                    history=_history_to_df(get(s, :history, []))
                )
                for s in get(c, :segments, [])
            ],
        )
        for c in raw.charts
    ]
end

# ── Market share ──────────────────────────────────────────────────────────────

"""
    get_market_share(slug) -> DataFrame

Market share % per metric (e.g. "Volume Market Share", "Value Market Share")
with the as-of date. Returns an empty DataFrame if no market share data exists
(most non-market-leader companies have no data here).

# Example
```julia
ms = get_market_share("asian-paints-limited")
```
"""
function get_market_share(slug::String)::DataFrame
    raw = _get("/marketshare"; slug=slug)
    metrics = get(raw, :metrics, [])
    isempty(metrics) && return DataFrame(metric=String[], value=String[], as_of=String[])
    return DataFrame(
        metric = [string(m.metric) for m in metrics],
        value  = [string(get(m, :value, "")) for m in metrics],
        as_of  = [string(get(m, :as_of, "")) for m in metrics],
    )
end

# ── Internal helpers ──────────────────────────────────────────────────────────

function _history_to_df(history)::DataFrame
    isempty(history) && return DataFrame(date=String[], value=Union{Float64, Missing}[])
    dates  = [string(get(h, :date, "")) for h in history]
    values = [_f64_or_nothing(get(h, :value, nothing)) for h in history]
    return DataFrame(date=dates, value=values)
end
