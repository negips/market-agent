"""
Feature helpers for StockSwingPredictor.

The CNN branches learn their own features from raw normalised prices, so this
file is intentionally minimal — just the LLM scalar serialisation and a few
shared utilities used in dataset assembly.
"""

using DataFrames, Dates

# ── LLM scalar serialisation ──────────────────────────────────────────────────

"""Flatten `LLMFeatures` to a plain `Vector{Float32}` (length N_LLM_FEATURES)."""
function llm_to_vec(f::LLMFeatures)::Vector{Float32}
    Float32[
        f.management_tone, f.guidance_direction, f.guidance_specificity,
        f.demand_outlook, f.margin_commentary, f.competitive_pressure,
        f.new_wins_announced, f.capex_expansion, f.buyback_or_dividend,
        f.mgmt_language_hedging, f.auditor_concerns, f.related_party_flags,
        f.contingent_liability_flag, f.extraction_confidence, f.doc_age_days,
    ]
end

"""Feature names matching `llm_to_vec` output order."""
function llm_feature_names()::Vector{String}
    ["llm_mgmt_tone", "llm_guidance_dir", "llm_guidance_spec",
     "llm_demand", "llm_margin", "llm_competitive",
     "llm_new_wins", "llm_capex", "llm_buyback",
     "llm_hedging", "llm_auditor", "llm_related_party",
     "llm_contingent", "llm_confidence", "llm_doc_age"]
end

# ── Date lookup ───────────────────────────────────────────────────────────────

"""
Return the largest-key entry from `cache` whose key ≤ `date`, or `default`.
Used to find the most recent LLM document published before a training date.
"""
function latest_before(cache::Dict{K,V}, date::Date, default::V)::V where {K,V}
    best = nothing
    for k in keys(cache)
        (k <= date) && (isnothing(best) || k > best) && (best = k)
    end
    isnothing(best) ? default : cache[best]
end

"""Find the index of the last Date ≤ `target` in a sorted Date vector."""
function find_date_index(dates::AbstractVector{Date}, target::Date)::Int
    searchsortedlast(dates, target)
end
