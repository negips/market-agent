"""
Aggregate scoring and the top-level `analyze` function.

Scoring model (all deductions additive; result clamped to [0, 100]):

  Beneish M-Score (non-banking only):
    Cannot compute (missing data):        −5
    M-Score in gray zone (−2.22, −1.78]: −15
    M-Score > −1.78 (manipulator zone):  −30

  Cash flow quality:
    2 of ≥4 years with CFO < NI:         −10
    ≥3 of ≥4 years with CFO < NI:        −20

  Promoter pledging:
    latest > 10%:                         −10
    latest > 25%:                         −20 (replaces −10)
    latest > 50%:                         −30 (replaces −20)
    rising trend (> +5 pp in 4 quarters): additional −10

  Tijori forensics:
    Each distinct red flag:               −8 (capped at −25 total)

  NSE surveillance (checked only):
    On ASM list:                          −25
    On GSM list:                          −35
"""

# ── Score constants ───────────────────────────────────────────────────────────

const _BENEISH_GRAY_ZONE = -2.22   # M-Score between gray and manipulator

# ── Top-level public function ─────────────────────────────────────────────────

"""
    analyze(slug; check_surveillance=true) -> ConfidenceReport

Run all confidence checks for the company identified by `slug` and return a
`ConfidenceReport`. Requires the TijoriData sidecar to be running.

Set `check_surveillance=false` to skip the NSE ASM/GSM HTTP check (e.g. in
offline testing or when you know NSE is unavailable).

The returned `report.pass` is `true` when `report.score ≥ PASS_THRESHOLD` (40).

# Example
```julia
using CompanyConfidence, TijoriData
TijoriData.start!()

report = analyze("infosys-limited")
report.score    # e.g. 85.0
report.pass     # true
report.beneish.m_score   # e.g. -2.41 (non-manipulator)
report.pledging.latest_pct  # e.g. 0.0

# Skip NSE check (faster; useful in tests)
report = analyze("yes-bank-limited"; check_surveillance=false)
```
"""
function analyze(slug::String; check_surveillance::Bool=true)::ConfidenceReport
    @info "CompanyConfidence: fetching data for $slug"

    ov = TijoriData.get_overview(slug)
    pl = TijoriData.get_financials(slug, :pl)
    bs = TijoriData.get_financials(slug, :bs)
    cf = TijoriData.get_financials(slug, :cf)
    sh = TijoriData.get_shareholding(slug)

    @info "CompanyConfidence: running checks"

    b = beneish_score(pl, bs, cf; is_banking=ov.is_banking)
    c = cashflow_check(pl, cf)
    p = pledging_check(sh)
    f = forensics_check(ov.forensics)
    s = check_surveillance ?
            surveillance_check(ov.symbol) :
            SurveillanceResult(false, false, false, false)

    score = _aggregate(b, c, p, f, s)

    return ConfidenceReport(
        slug,
        ov.company,
        score,
        score >= PASS_THRESHOLD,
        b, c, p, f, s,
        Dates.now(Dates.UTC),
    )
end

# ── Scoring model ─────────────────────────────────────────────────────────────

function _aggregate(b::BeneishResult, c::CashflowResult, p::PledgingResult,
                    f::ForensicsResult, s::SurveillanceResult)::Float64
    score = 100.0

    # ── Beneish ───────────────────────────────────────────────────────────────
    if b.applicable
        if isnothing(b.m_score)
            score -= 5.0
        elseif b.m_score > BENEISH_THRESHOLD
            score -= 30.0
        elseif b.m_score > _BENEISH_GRAY_ZONE
            score -= 15.0
        end
    end

    # ── Cash flow ─────────────────────────────────────────────────────────────
    if c.years_checked >= 4
        if c.years_cfo_lt_ni >= 3
            score -= 20.0
        elseif c.years_cfo_lt_ni >= 2
            score -= 10.0
        end
    end

    # ── Pledging ──────────────────────────────────────────────────────────────
    if !isnothing(p.latest_pct)
        if p.latest_pct > 50.0
            score -= 30.0
        elseif p.latest_pct > 25.0
            score -= 20.0
        elseif p.latest_pct > 10.0
            score -= 10.0
        end
        if !isnothing(p.change_4q) && p.change_4q > 5.0
            score -= 10.0
        end
    end

    # ── Forensics ─────────────────────────────────────────────────────────────
    score -= min(8.0 * length(f.flags), 25.0)

    # ── Surveillance (only penalise if check actually ran) ────────────────────
    if s.checked
        s.on_asm && (score -= 25.0)
        s.on_gsm && (score -= 35.0)
    end

    return clamp(score, 0.0, 100.0)
end
