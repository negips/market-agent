"""
Cash-flow-vs-earnings divergence check.

Compares Net Income (from P&L) with Cash Flow from Operations (from CF
statement) over multiple fiscal years. Persistent CFO < NI is the hallmark
of accrual-based earnings inflation.
"""

# ── Line item candidates ──────────────────────────────────────────────────────

const _CF_NET_INCOME = String["profit after tax", "net profit", "pat",
                               "profit for the year", "net income", "net earnings"]
const _CF_CFO        = String["cash from operations", "net cash from operating",
                               "operating cash flow", "cash generated from operations",
                               "cash flow from operations"]

# ── Main function ─────────────────────────────────────────────────────────────

"""
    cashflow_check(pl, cf) -> CashflowResult

Compare Net Income vs Cash Flow from Operations for each available fiscal year.

`pl` and `cf` are DataFrames from `TijoriData.get_financials`. Common year
columns between the two statements are matched by column name. Up to 5 years
are checked.

Returns a `CashflowResult` with `is_flagged=true` when CFO < NI in 3 or more
of at least 4 comparable years.

# Example
```julia
pl = get_financials("suzlon-energy-limited", :pl)
cf = get_financials("suzlon-energy-limited", :cf)
r  = cashflow_check(pl, cf)
r.years_cfo_lt_ni   # e.g. 4 out of 5 years → flagged
r.avg_accrual_ratio # positive → earnings > cash consistently
```
"""
function cashflow_check(pl::DataFrame, cf::DataFrame)::CashflowResult
    pl_years = _sorted_year_cols(pl)
    cf_years = _sorted_year_cols(cf)

    # Match years by column name; fall back to matching by position (year index)
    common_years = _match_years(pl_years, cf_years)
    if isempty(common_years)
        return CashflowResult(0, 0, nothing, false)
    end

    ni_row  = _find_metric(pl, _CF_NET_INCOME)
    cfo_row = _find_metric(cf, _CF_CFO)

    if isnothing(ni_row) || isnothing(cfo_row)
        return CashflowResult(0, 0, nothing, false)
    end

    years_checked   = 0
    years_cfo_lt_ni = 0
    accrual_ratios  = Float64[]

    for (pl_col, cf_col) in first(common_years, 5)
        ni  = _val(ni_row,  pl_col)
        cfo = _val(cfo_row, cf_col)
        (isnothing(ni) || isnothing(cfo)) && continue

        years_checked += 1
        cfo < ni && (years_cfo_lt_ni += 1)
        abs(ni) > 0 && push!(accrual_ratios, (ni - cfo) / abs(ni))
    end

    avg_accrual = isempty(accrual_ratios) ? nothing : Statistics.mean(accrual_ratios)
    is_flagged  = years_checked >= 4 && years_cfo_lt_ni >= 3

    return CashflowResult(years_checked, years_cfo_lt_ni, avg_accrual, is_flagged)
end

# ── Internal ──────────────────────────────────────────────────────────────────

# Returns pairs of (pl_col, cf_col) for matching years.
# First tries exact column name match, then matches by sorted position.
function _match_years(pl_years::Vector{Symbol}, cf_years::Vector{Symbol})::Vector{Tuple{Symbol,Symbol}}
    exact = [(y, y) for y in pl_years if y in cf_years]
    !isempty(exact) && return exact

    # Fall back: match by descending order (both sorted newest-first)
    pairs = Tuple{Symbol,Symbol}[]
    for i in 1:min(length(pl_years), length(cf_years))
        push!(pairs, (pl_years[i], cf_years[i]))
    end
    return pairs
end
