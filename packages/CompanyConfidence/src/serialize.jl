"""
    report_to_dict(report) -> Dict{String,Any}

Flatten a `ConfidenceReport` into the JSON-friendly dict stored under each
company's `"confidence"` key in `nse_companies_latest.json` /
`bse_companies_latest.json` (used by `run_confidence_checks.jl` and
`run_bse_confidence_checks.jl`, and read by the website and TradingGame's
universe filter).

# Arguments
- `report`: a `ConfidenceReport` from `analyze`

# Returns
- `Dict{String,Any}`: `score`, `pass`, `checked_at` and one sub-dict per signal
"""
function report_to_dict(r::ConfidenceReport)::Dict{String,Any}
    Dict{String,Any}(
        "score"       => r.score,
        "pass"        => r.pass,
        "checked_at"  => string(r.generated_at),
        "beneish"     => Dict{String,Any}(
            "applicable"    => r.beneish.applicable,
            "m_score"       => r.beneish.m_score,
            "is_flagged"    => r.beneish.is_flagged,
            "year_t"        => r.beneish.year_t,
            "year_t1"       => r.beneish.year_t1,
            "dsri"          => r.beneish.dsri,
            "gmi"           => r.beneish.gmi,
            "aqi"           => r.beneish.aqi,
            "sgi"           => r.beneish.sgi,
            "depi"          => r.beneish.depi,
            "sgai"          => r.beneish.sgai,
            "lvgi"          => r.beneish.lvgi,
            "tata"          => r.beneish.tata,
            "missing_items" => r.beneish.missing_items,
        ),
        "cashflow"    => Dict{String,Any}(
            "years_checked"   => r.cashflow.years_checked,
            "years_cfo_lt_ni" => r.cashflow.years_cfo_lt_ni,
            "avg_accrual_ratio" => r.cashflow.avg_accrual_ratio,
            "is_flagged"      => r.cashflow.is_flagged,
        ),
        "pledging"    => Dict{String,Any}(
            "latest_pct"    => r.pledging.latest_pct,
            "trend"         => string(r.pledging.trend),
            "change_4q"     => r.pledging.change_4q,
            "quarters_used" => r.pledging.quarters_used,
            "is_flagged"    => r.pledging.is_flagged,
        ),
        "forensics"   => Dict{String,Any}(
            "flag_count" => length(r.forensics.flags),
            "flags"      => r.forensics.flags,
            "is_flagged" => r.forensics.is_flagged,
        ),
        "surveillance" => Dict{String,Any}(
            "on_asm"    => r.surveillance.on_asm,
            "on_gsm"    => r.surveillance.on_gsm,
            "checked"   => r.surveillance.checked,
            "is_flagged" => r.surveillance.is_flagged,
        ),
    )
end
