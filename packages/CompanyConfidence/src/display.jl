"""
REPL display methods for CompanyConfidence result types.
"""

using Printf

# ── ConfidenceReport ──────────────────────────────────────────────────────────

function Base.show(io::IO, ::MIME"text/plain", r::ConfidenceReport)
    verdict = r.pass ? "PASS ✓" : "FAIL ✗"
    bar     = _score_bar(r.score)
    println(io, "═"^62)
    println(io, "  CompanyConfidence: $(r.company)")
    println(io, "  Slug: $(r.slug)")
    println(io, "═"^62)
    @printf(io, "  Score: %5.1f / 100   [%s]   %s\n", r.score, bar, verdict)
    println(io)
    println(io, "  Signal summary:")
    _show_signal(io, "Beneish M-Score", r.beneish)
    _show_signal(io, "Cash flow quality", r.cashflow)
    _show_signal(io, "Promoter pledging", r.pledging)
    _show_signal(io, "Tijori forensics", r.forensics)
    _show_signal(io, "NSE surveillance", r.surveillance)
    println(io, "─"^62)
    println(io, "  Generated: $(r.generated_at) UTC")
    print(io,   "  (Use report.beneish, .cashflow, .pledging, .forensics, .surveillance for details)")
end

function Base.show(io::IO, r::ConfidenceReport)
    verdict = r.pass ? "PASS" : "FAIL"
    print(io, "ConfidenceReport($(r.company), score=$(round(r.score, digits=1)), $verdict)")
end

# ── BeneishResult ─────────────────────────────────────────────────────────────

function Base.show(io::IO, ::MIME"text/plain", b::BeneishResult)
    println(io, "BeneishResult")
    println(io, "─"^40)
    if !b.applicable
        println(io, "  Not applicable (banking/NBFC company)")
        return
    end
    if isnothing(b.m_score)
        println(io, "  M-Score: cannot compute")
    else
        zone = b.m_score > BENEISH_THRESHOLD ? "MANIPULATOR ZONE" :
               b.m_score > -2.22             ? "gray zone"        : "safe"
        @printf(io, "  M-Score: %.3f  [%s]\n", b.m_score, zone)
    end
    isnothing(b.year_t) || println(io, "  Years: $(b.year_t) vs $(b.year_t1)")
    println(io)
    println(io, "  Index      Value   Benchmark")
    println(io, "  ─────────  ──────  ─────────")
    _show_index(io, "DSRI",  b.dsri,  1.031)
    _show_index(io, "GMI",   b.gmi,   1.014)
    _show_index(io, "AQI",   b.aqi,   1.040)
    _show_index(io, "SGI",   b.sgi,   1.134)
    _show_index(io, "DEPI",  b.depi,  1.001)
    _show_index(io, "SGAI",  b.sgai,  1.054)
    _show_index(io, "LVGI",  b.lvgi,  1.111)
    _show_index(io, "TATA",  b.tata,  0.031)
    if !isempty(b.missing_items)
        println(io)
        println(io, "  Missing: ", join(b.missing_items, ", "))
    end
end

function Base.show(io::IO, b::BeneishResult)
    !b.applicable && return print(io, "BeneishResult(N/A — banking)")
    s = isnothing(b.m_score) ? "M=N/A" : @sprintf("M=%.3f", b.m_score)
    flag = b.is_manipulator === true ? " ⚠" : ""
    print(io, "BeneishResult($s$flag)")
end

# ── CashflowResult ────────────────────────────────────────────────────────────

function Base.show(io::IO, ::MIME"text/plain", c::CashflowResult)
    println(io, "CashflowResult")
    println(io, "─"^40)
    println(io, "  Years checked:    $(c.years_checked)")
    println(io, "  CFO < NI:         $(c.years_cfo_lt_ni) years")
    if !isnothing(c.avg_accrual_ratio)
        @printf(io, "  Avg accrual ratio: %.3f\n", c.avg_accrual_ratio)
    end
    println(io, "  Flagged:          $(c.is_flagged)")
end

function Base.show(io::IO, c::CashflowResult)
    flag = c.is_flagged ? " ⚠" : ""
    print(io, "CashflowResult($(c.years_cfo_lt_ni)/$(c.years_checked) years CFO<NI$flag)")
end

# ── PledgingResult ────────────────────────────────────────────────────────────

function Base.show(io::IO, ::MIME"text/plain", p::PledgingResult)
    println(io, "PledgingResult")
    println(io, "─"^40)
    pct_str = isnothing(p.latest_pct) ? "not reported" : @sprintf("%.1f%%", p.latest_pct)
    println(io, "  Latest pledging:  $pct_str")
    println(io, "  Trend (4 quarters): $(p.trend)")
    if !isnothing(p.change_4q)
        @printf(io, "  Change (4Q):      %+.1f pp\n", p.change_4q)
    end
    println(io, "  Quarters used:    $(p.quarters_used)")
    println(io, "  Flagged:          $(p.is_flagged)")
end

function Base.show(io::IO, p::PledgingResult)
    pct = isnothing(p.latest_pct) ? "N/A" : @sprintf("%.1f%%", p.latest_pct)
    flag = p.is_flagged ? " ⚠" : ""
    print(io, "PledgingResult($pct, $(p.trend)$flag)")
end

# ── ForensicsResult ───────────────────────────────────────────────────────────

function Base.show(io::IO, ::MIME"text/plain", f::ForensicsResult)
    println(io, "ForensicsResult")
    println(io, "─"^40)
    if isnothing(f.raw)
        println(io, "  No Tijori forensics data available")
        return
    end
    println(io, "  Red flags found: $(length(f.flags))")
    for flag in f.flags
        println(io, "    • $flag")
    end
    if isempty(f.flags)
        println(io, "  Raw forensics:")
        for (k, v) in f.raw
            @printf(io, "    %-28s %s\n", string(k) * ":", string(v))
        end
    end
end

function Base.show(io::IO, f::ForensicsResult)
    flag = f.is_flagged ? " ⚠" : ""
    n = length(f.flags)
    print(io, "ForensicsResult($n flag(s)$flag)")
end

# ── SurveillanceResult ────────────────────────────────────────────────────────

function Base.show(io::IO, ::MIME"text/plain", s::SurveillanceResult)
    println(io, "SurveillanceResult")
    println(io, "─"^40)
    if !s.checked
        println(io, "  Check did not complete (NSE unreachable)")
        return
    end
    println(io, "  On ASM list: $(s.on_asm)")
    println(io, "  On GSM list: $(s.on_gsm)")
    println(io, "  Flagged:     $(s.is_flagged)")
end

function Base.show(io::IO, s::SurveillanceResult)
    !s.checked && return print(io, "SurveillanceResult(unchecked)")
    flag = s.is_flagged ? " ⚠" : ""
    tags = filter(!isempty, [s.on_asm ? "ASM" : "", s.on_gsm ? "GSM" : ""])
    label = isempty(tags) ? "clean" : join(tags, "+")
    print(io, "SurveillanceResult($label$flag)")
end

# ── Helpers ───────────────────────────────────────────────────────────────────

function _score_bar(score::Float64)::String
    filled = round(Int, score / 5)   # 20-char bar
    empty  = 20 - filled
    ("█"^filled) * ("░"^empty)
end

function _show_signal(io::IO, label::String, result)
    flag = result.is_flagged ? " ⚠" : ""
    @printf(io, "    %-22s %s\n", label * ":", _signal_summary(result) * flag)
end

_signal_summary(b::BeneishResult) =
    !b.applicable ? "N/A (bank)" :
    isnothing(b.m_score) ? "cannot compute" :
    @sprintf("M=%.2f", b.m_score)

_signal_summary(c::CashflowResult) =
    c.years_checked == 0 ? "no data" :
    "$(c.years_cfo_lt_ni)/$(c.years_checked) years CFO<NI"

_signal_summary(p::PledgingResult) =
    isnothing(p.latest_pct) ? "not reported" :
    @sprintf("%.1f%% (%s)", p.latest_pct, p.trend)

_signal_summary(f::ForensicsResult) =
    isnothing(f.raw) ? "no data" : "$(length(f.flags)) flag(s)"

_signal_summary(s::SurveillanceResult) =
    !s.checked ? "unchecked" :
    s.on_asm && s.on_gsm ? "ASM + GSM" :
    s.on_asm ? "ASM" :
    s.on_gsm ? "GSM" : "clean"

function _show_index(io::IO, name::String, val, benchmark::Float64)
    if isnothing(val)
        @printf(io, "  %-9s  %6s  > %.3f\n", name, "N/A", benchmark)
    else
        flag = val > benchmark ? " ↑" : ""
        @printf(io, "  %-9s  %6.3f  > %.3f%s\n", name, val, benchmark, flag)
    end
end
