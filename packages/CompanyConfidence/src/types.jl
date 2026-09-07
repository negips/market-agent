"""
Result types for all CompanyConfidence signal checks.

Each struct is the output of one checker function. All five are combined into
a `ConfidenceReport` by `analyze`.
"""

# ── Beneish M-Score ───────────────────────────────────────────────────────────

"""
    BeneishResult

Output of the Beneish M-Score earnings manipulation check.

The 8-factor Beneish (1999) probit model detects manipulation via accrual-based
accounting distortions across two consecutive fiscal years. Not applicable to
banking/NBFC companies (their balance sheets use different conventions).

Fields:
- `applicable`     — false for banking/NBFC (Beneish doesn't apply)
- `m_score`        — computed M-Score; nothing if insufficient data
- `is_manipulator` — true if m_score > BENEISH_THRESHOLD (-1.78)
- `dsri`  — Days Sales Receivable Index  (>1.031 → concerning)
- `gmi`   — Gross Margin Index           (>1.014 → concerning)
- `aqi`   — Asset Quality Index          (>1.040 → concerning)
- `sgi`   — Sales Growth Index           (>1.134 → concerning)
- `depi`  — Depreciation Index           (>1.001 → concerning)
- `sgai`  — SGA Expenses Index           (>1.054 → concerning)
- `lvgi`  — Leverage Index               (>1.111 → concerning)
- `tata`  — Total Accruals to Total Assets (>0.031 → concerning)
- `is_flagged`     — true if is_manipulator is true
- `year_t`         — most recent fiscal year used (e.g. "FY24")
- `year_t1`        — prior fiscal year used (e.g. "FY23")
- `missing_items`  — line items that could not be matched in the financials
"""
struct BeneishResult
    applicable::Bool
    m_score::Union{Float64, Nothing}
    is_manipulator::Union{Bool, Nothing}
    is_flagged::Bool
    dsri::Union{Float64, Nothing}
    gmi::Union{Float64, Nothing}
    aqi::Union{Float64, Nothing}
    sgi::Union{Float64, Nothing}
    depi::Union{Float64, Nothing}
    sgai::Union{Float64, Nothing}
    lvgi::Union{Float64, Nothing}
    tata::Union{Float64, Nothing}
    year_t::Union{String, Nothing}
    year_t1::Union{String, Nothing}
    missing_items::Vector{String}
end

# ── Cash flow quality ─────────────────────────────────────────────────────────

"""
    CashflowResult

Output of the cash-flow-vs-earnings divergence check.

Persistent CFO < Net Income signals accrual inflation: the company reports
profits that are not backed by actual cash generation. Three or more years
of divergence in a 4-year window is treated as a red flag.

Fields:
- `years_checked`   — number of fiscal years where both NI and CFO were found
- `years_cfo_lt_ni` — years where CFO < Net Income
- `avg_accrual_ratio` — mean (NI − CFO) / |NI| across years (positive = concerning)
- `is_flagged`      — true if years_cfo_lt_ni ≥ 3 with years_checked ≥ 4
"""
struct CashflowResult
    years_checked::Int
    years_cfo_lt_ni::Int
    avg_accrual_ratio::Union{Float64, Nothing}
    is_flagged::Bool
end

# ── Promoter pledging ─────────────────────────────────────────────────────────

"""
    PledgingResult

Output of the promoter pledging trend analysis.

Pledging of promoter shares as collateral for personal/business loans signals
financial stress. Rising pledging or high absolute levels are red flags.

Fields:
- `latest_pct`    — most recent quarter's promoter pledged %; nothing if not reported
- `trend`         — `:rising`, `:stable`, `:falling`, or `:unknown`
- `change_4q`     — change in pledging over last 4 quarters (percentage points)
- `quarters_used` — number of quarters included in the analysis
- `is_flagged`    — true if latest_pct > 25% or rising by > 5 pp in 4 quarters
"""
struct PledgingResult
    latest_pct::Union{Float64, Nothing}
    trend::Symbol
    change_4q::Union{Float64, Nothing}
    quarters_used::Int
    is_flagged::Bool
end

# ── Tijori forensics ──────────────────────────────────────────────────────────

"""
    ForensicsResult

Output of parsing Tijori Finance's quick-look forensics assessment.

Tijori's forensics dict (from `get_overview`) is scanned for keywords
associated with accounting fraud, audit concerns, and related-party risks.

Fields:
- `raw`        — the raw forensics dict from `CompanyOverview.forensics`; nothing if absent
- `flags`      — list of key: value entries containing red-flag keywords
- `is_flagged` — true if any flags were found
"""
struct ForensicsResult
    raw::Union{Dict{String, Any}, Nothing}
    flags::Vector{String}
    is_flagged::Bool
end

# ── Surveillance lists ────────────────────────────────────────────────────────

"""
    SurveillanceResult

Output of checking NSE's ASM and GSM surveillance lists.

- ASM (Additional Surveillance Measure): unusual price/volume movements
- GSM (Graded Surveillance Measure): weak fundamentals — more severe

Fields:
- `on_asm`   — true if stock symbol appears on the NSE ASM list
- `on_gsm`   — true if stock symbol appears on the NSE GSM list
- `checked`  — false if the check could not run (network unavailable, NSE blocked)
- `is_flagged` — true if on_asm || on_gsm
"""
struct SurveillanceResult
    on_asm::Bool
    on_gsm::Bool
    checked::Bool
    is_flagged::Bool
end

# ── Aggregate report ──────────────────────────────────────────────────────────

"""
    ConfidenceReport

Full reliability assessment for one company. Returned by `analyze(slug)`.

A score below `PASS_THRESHOLD` (40) means the company is too risky to spend
effort predicting earnings swings on.

Fields:
- `slug`         — company slug passed to `analyze`
- `company`      — full company name from Tijori
- `score`        — aggregate confidence score 0–100 (100 = maximum confidence)
- `pass`         — true if score ≥ PASS_THRESHOLD
- `beneish`      — BeneishResult
- `cashflow`     — CashflowResult
- `pledging`     — PledgingResult
- `forensics`    — ForensicsResult
- `surveillance` — SurveillanceResult
- `generated_at` — UTC timestamp when the report was produced
"""
struct ConfidenceReport
    slug::String
    company::String
    score::Float64
    pass::Bool
    beneish::BeneishResult
    cashflow::CashflowResult
    pledging::PledgingResult
    forensics::ForensicsResult
    surveillance::SurveillanceResult
    generated_at::DateTime
end
