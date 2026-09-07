"""
Tijori Finance forensics / quick-look assessment parser.

`CompanyOverview.forensics` is a Dict{String,Any} from Tijori's quick_look
section. We scan keys and values for known red-flag keywords associated with
earnings manipulation, audit concerns, and related-party risks.
"""

const _FORENSICS_RED_FLAG_KEYWORDS = String[
    "risk", "fraud", "concern", "manipulat", "warning", "adverse",
    "qualified", "emphasis of matter", "going concern", "related party",
    "pledge", "diversion", "misappropriation", "delay", "negative",
    "deteriorat", "default", "restructur",
]

"""
    forensics_check(forensics) -> ForensicsResult

Parse Tijori's forensics dict for red-flag keywords.

`forensics` is `CompanyOverview.forensics` (a Dict or nothing). Each key-value
pair whose combined text contains a red-flag keyword is added to `flags`.

Returns `ForensicsResult(nothing, [], false)` if `forensics` is nothing.

# Example
```julia
ov = get_overview("dewan-housing-finance-corporation-limited")
r  = forensics_check(ov.forensics)
r.flags      # ["audit_opinion: qualified", ...]
r.is_flagged # true
```
"""
function forensics_check(forensics::Union{Dict{String, Any}, Nothing})::ForensicsResult
    isnothing(forensics) && return ForensicsResult(nothing, String[], false)

    flags = String[]
    for (k, v) in forensics
        text = lowercase("$(k) $(v)")
        if any(kw -> occursin(kw, text), _FORENSICS_RED_FLAG_KEYWORDS)
            push!(flags, "$(k): $(v)")
        end
    end
    unique!(flags)

    return ForensicsResult(forensics, flags, !isempty(flags))
end
