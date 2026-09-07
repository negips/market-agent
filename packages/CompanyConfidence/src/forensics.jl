"""
Tijori Finance forensics / quick-look assessment parser.

Tijori's `quick_look` object has a structured format:
  {
    "count": { "green": N, "red": N, "neutral": N, "total": N },
    "data": [
      {
        "name": "Accounting & Shareholding",
        "factories": [
          { "name": "...", "sentence": "...", "explanation": "...", "flag": 1|2|3 }
        ]
      }, ...
    ]
  }

Flag values: 1 = green (good), 2 = neutral, 3 = red (concern).

If the dict doesn't follow this structure (future Tijori changes), we fall
back to keyword scanning.
"""

const _FORENSICS_RED_FLAG_KEYWORDS = String[
    "risk", "fraud", "concern", "manipulat", "warning", "adverse",
    "qualified", "going concern", "related party",
    "pledge", "diversion", "misappropriation", "delay",
    "negative", "deteriorat", "default", "restructur", "solvency",
]

"""
    forensics_check(forensics) -> ForensicsResult

Parse Tijori's forensics dict for red flags.

Handles the structured `flag: 1/2/3` format (primary) and falls back to
keyword scanning for legacy or unexpected structures.

Each factory item with `flag == 3` becomes one entry in `ForensicsResult.flags`,
formatted as `"Category › Check name: What Tijori says about it"`.

# Example
```julia
ov = get_overview("dewan-housing-finance-corporation-limited")
r  = forensics_check(ov.forensics)
r.flags       # ["Accounting & Shareholding › Depreciation Effect: ...", ...]
length(r.flags)  # e.g. 9 red flags
```
"""
function forensics_check(forensics::Union{Dict{String, Any}, Nothing})::ForensicsResult
    isnothing(forensics) && return ForensicsResult(nothing, String[], false)

    # ── Structured Tijori format ───────────────────────────────────────────────
    data_val = get(forensics, "data", nothing)
    if !isnothing(data_val)
        flags = String[]
        try
            for cat in data_val
                cat_name  = string(get(cat, :name, ""))
                factories = get(cat, :factories, [])
                for factory in factories
                    Int(get(factory, :flag, 0)) == 3 || continue
                    f_name     = string(get(factory, :name, ""))
                    f_sentence = string(get(factory, :sentence, ""))
                    push!(flags, "$cat_name › $f_name: $f_sentence")
                end
            end
        catch
            # Parsing failed — fall through to keyword scan
        end
        if !isempty(flags)
            return ForensicsResult(forensics, flags, true)
        end
    end

    # ── Keyword scan fallback ─────────────────────────────────────────────────
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
