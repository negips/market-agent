"""
Shared utilities for analyzing TijoriData financial DataFrames.

All functions here are internal (not exported). They are used by beneish.jl,
cashflow.jl, and pledging.jl to locate and extract numeric values from the
DataFrames produced by TijoriData.get_financials and get_shareholding.
"""

# ── DataFrame navigation ──────────────────────────────────────────────────────

"""
    _sorted_year_cols(df) -> Vector{Symbol}

Return all non-metric column names sorted descending by year (most recent
first). Handles column naming conventions from Tijori: "FY24", "Mar 24",
"Q4FY24", "Dec 24", etc. — all are sorted by the trailing 2-digit year number.
"""
function _sorted_year_cols(df::DataFrame)::Vector{Symbol}
    cols = [c for c in propertynames(df) if c != :metric]
    isempty(cols) && return cols
    key(c) = (m = match(r"(\d{2,4})$", string(c)); isnothing(m) ? 0 : parse(Int, m.captures[1]) % 100)
    sort(cols, by=key, rev=true)
end

"""
    _find_metric(df, candidates) -> Union{DataFrameRow, Nothing}

Search the `metric` column of `df` for the first row whose text contains any
of the `candidates` (case-insensitive substring match). Returns nothing if
no row matches.
"""
function _find_metric(df::DataFrame, candidates::Vector{String})::Union{DataFrameRow, Nothing}
    for cand in candidates
        cl = lowercase(cand)
        for row in eachrow(df)
            occursin(cl, lowercase(string(row.metric))) && return row
        end
    end
    return nothing
end

"""
    _val(row, col) -> Union{Float64, Nothing}

Extract a numeric value from a DataFrameRow at column `col`.
Returns nothing if the column is absent, missing, or non-numeric.
"""
function _val(row::DataFrameRow, col::Symbol)::Union{Float64, Nothing}
    hasproperty(row, col) || return nothing
    v = row[col]
    (isnothing(v) || ismissing(v)) && return nothing
    v isa Number ? Float64(v) : nothing
end

"""
    _extract(df, candidates, col, label, missing_items) -> Union{Float64, Nothing}

Find a metric row by candidate names and return its value at `col`.
Appends `label` to `missing_items` if the row is not found or the value is absent.
"""
function _extract(df::DataFrame, candidates::Vector{String}, col::Symbol, label::String, missing_items::Vector{String})::Union{Float64, Nothing}
    row = _find_metric(df, candidates)
    if isnothing(row)
        push!(missing_items, label)
        return nothing
    end
    v = _val(row, col)
    if isnothing(v)
        push!(missing_items, "$label (no data for $col)")
    end
    return v
end

# ── Nil-propagating arithmetic ────────────────────────────────────────────────

_sdiv(a, b) = (isnothing(a) || isnothing(b) || b == 0.0) ? nothing : a / b
_sadd(a, b) = (isnothing(a) || isnothing(b)) ? nothing : a + b
_ssub(a, b) = (isnothing(a) || isnothing(b)) ? nothing : a - b
