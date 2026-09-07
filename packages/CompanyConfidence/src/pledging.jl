"""
Promoter pledging trend analysis.

Promoters pledging their shares as collateral signals personal or group-level
financial stress. Rising pledging — especially above 25% — warrants extra
scrutiny before taking a position.
"""

"""
    pledging_check(sh) -> PledgingResult

Analyse the promoter pledging trend from a shareholding DataFrame.

`sh` is a DataFrame as returned by `TijoriData.get_shareholding`. Requires a
"Promoter Pledged" column; returns `:unknown` trend if that column is absent
or all-nothing.

Trend is determined from the change over the last 4 quarters (oldest to
most recent in the window). Rows are assumed to be oldest-first (as documented
in TijoriData).

Flagging rules:
- latest_pct > 25% → flagged
- change over last 4 quarters > +5 pp → flagged

# Example
```julia
sh = get_shareholding("yes-bank-limited")
r  = pledging_check(sh)
r.latest_pct   # e.g. 78.3
r.trend        # :rising
r.is_flagged   # true
```
"""
function pledging_check(sh::DataFrame)::PledgingResult
    pledge_col = Symbol("Promoter Pledged")
    if !(pledge_col in propertynames(sh))
        return PledgingResult(nothing, :unknown, nothing, 0, false)
    end

    # Collect non-nothing values in order (oldest-first per TijoriData docs)
    raw = [v for v in sh[!, pledge_col] if !isnothing(v) && !ismissing(v)]
    if isempty(raw)
        return PledgingResult(nothing, :unknown, nothing, 0, false)
    end

    pledging = Float64.(raw)
    n        = length(pledging)
    latest   = last(pledging)

    # Change over last 4 quarters
    change_4q = n >= 2 ? latest - pledging[max(1, n - 3)] : nothing

    trend = if isnothing(change_4q)
        :unknown
    elseif change_4q > 2.0
        :rising
    elseif change_4q < -2.0
        :falling
    else
        :stable
    end

    is_flagged = latest > 25.0 || (!isnothing(change_4q) && change_4q > 5.0)

    return PledgingResult(latest, trend, change_4q, n, is_flagged)
end
