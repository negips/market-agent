"""
CompanyConfidence test suite.

Tests are split into two groups:
  - Unit tests: run without a live sidecar (test each checker with synthetic DataFrames)
  - Integration tests: require a running sidecar (TIJORI_SIDECAR_DIR must be set)

Run unit tests only:
  julia --project=. test/runtests.jl

Run integration tests:
  TIJORI_SIDECAR_DIR=/path/to/sidecar julia --project=. test/runtests.jl
"""

using Test
using CompanyConfidence
using DataFrames
using Dates

# ── Helpers ───────────────────────────────────────────────────────────────────

# Build a minimal P&L DataFrame for Beneish testing with known values.
function make_pl(; rev_t, rev_t1, cogs_t, cogs_t1, sga_t, sga_t1,
                   dep_t, dep_t1, ni_t)
    DataFrame(
        metric = ["Net Revenue", "Cost of Revenue", "Selling and Distribution",
                   "Depreciation", "Profit After Tax"],
        FY24   = [rev_t,  cogs_t,  sga_t,  dep_t,  ni_t],
        FY23   = [rev_t1, cogs_t1, sga_t1, dep_t1, nothing],
    )
end

function make_bs(; rec_t, rec_t1, ca_t, ca_t1, ppe_t, ppe_t1,
                   ta_t, ta_t1, ltd_t, ltd_t1, cl_t, cl_t1)
    DataFrame(
        metric = ["Trade Receivables", "Total Current Assets", "Net Fixed Assets",
                   "Total Assets", "Long Term Borrowings", "Total Current Liabilities"],
        FY24   = [rec_t, ca_t, ppe_t, ta_t, ltd_t, cl_t],
        FY23   = [rec_t1, ca_t1, ppe_t1, ta_t1, ltd_t1, cl_t1],
    )
end

function make_cf(; cfo_t)
    DataFrame(
        metric = ["Cash from Operations"],
        FY24   = [cfo_t],
        FY23   = [nothing],
    )
end

# ── Unit tests ────────────────────────────────────────────────────────────────

@testset "CompanyConfidence" begin

    @testset "Beneish — non-banking" begin
        pl = make_pl(rev_t=1200., rev_t1=1000., cogs_t=700., cogs_t1=600.,
                     sga_t=100.,  sga_t1=80.,   dep_t=50.,  dep_t1=45.,
                     ni_t=150.)
        bs = make_bs(rec_t=180., rec_t1=140., ca_t=400., ca_t1=350.,
                     ppe_t=300., ppe_t1=280., ta_t=900., ta_t1=800.,
                     ltd_t=100., ltd_t1=90.,  cl_t=150., cl_t1=130.)
        cf = make_cf(cfo_t=120.)

        r = beneish_score(pl, bs, cf; is_banking=false)
        @test r.applicable
        @test r.year_t  == "FY24"
        @test r.year_t1 == "FY23"
        @test !isnothing(r.m_score)
        @test r.m_score isa Float64

        # Each index should be computed
        @test !isnothing(r.dsri)
        @test !isnothing(r.sgi)
        @test !isnothing(r.tata)

        # TATA = (NI - CFO) / TA = (150 - 120) / 900 = 0.0333
        @test isapprox(r.tata, 30.0 / 900.0, atol=1e-6)

        # SGI = 1200 / 1000 = 1.2
        @test isapprox(r.sgi, 1.2, atol=1e-6)

        # is_manipulator reflects BENEISH_THRESHOLD
        @test r.is_manipulator == (r.m_score > BENEISH_THRESHOLD)
    end

    @testset "Beneish — banking company" begin
        r = beneish_score(DataFrame(), DataFrame(), DataFrame(); is_banking=true)
        @test !r.applicable
        @test isnothing(r.m_score)
    end

    @testset "Beneish — insufficient data" begin
        r = beneish_score(DataFrame(metric=String[], FY24=Float64[]),
                          DataFrame(), DataFrame())
        @test r.applicable
        @test isnothing(r.m_score)
        @test !isempty(r.missing_items)
    end

    @testset "Cash flow check" begin
        # 4 years where CFO < NI in 3 → should flag
        pl = DataFrame(
            metric = ["Profit After Tax"],
            FY24   = [100.], FY23 = [90.], FY22 = [80.], FY21 = [70.],
        )
        cf = DataFrame(
            metric = ["Cash from Operations"],
            FY24   = [80.], FY23 = [70.], FY22 = [90.], FY21 = [60.],
        )
        r = cashflow_check(pl, cf)
        @test r.years_checked == 4
        @test r.years_cfo_lt_ni == 3
        @test r.is_flagged

        # 2 divergence years → not flagged (FY23: 70<90, FY22: 70<80)
        cf2 = DataFrame(
            metric = ["Cash from Operations"],
            FY24   = [110.], FY23 = [70.], FY22 = [70.], FY21 = [80.],
        )
        r2 = cashflow_check(pl, cf2)
        @test r2.years_cfo_lt_ni == 2
        @test !r2.is_flagged
    end

    @testset "Pledging check" begin
        sh_rising = DataFrame(
            :period             => ["Sep 23", "Dec 23", "Mar 24", "Jun 24", "Sep 24"],
            :Promoter           => [50., 50., 50., 50., 50.],
            Symbol("Promoter Pledged") => [5., 8., 12., 18., 28.],
        )
        r = pledging_check(sh_rising)
        @test r.latest_pct == 28.0
        @test r.trend == :rising
        @test r.is_flagged       # 28% > 25%

        # No pledging column
        sh_none = DataFrame(:period => ["Sep 24"], :Promoter => [60.])
        r2 = pledging_check(sh_none)
        @test r2.trend == :unknown
        @test !r2.is_flagged

        # Zero pledging
        sh_zero = DataFrame(
            :period => ["Sep 23", "Sep 24"],
            Symbol("Promoter Pledged") => [0., 0.],
        )
        r3 = pledging_check(sh_zero)
        @test r3.latest_pct == 0.0
        @test !r3.is_flagged
    end

    @testset "Forensics check" begin
        f1 = forensics_check(nothing)
        @test !f1.is_flagged
        @test isempty(f1.flags)

        f2 = forensics_check(Dict{String,Any}("audit_opinion" => "Qualified"))
        @test f2.is_flagged
        @test length(f2.flags) == 1

        f3 = forensics_check(Dict{String,Any}("overall_rating" => "Excellent",
                                               "revenue_trend"  => "Positive"))
        @test !f3.is_flagged
    end

    @testset "Score aggregation" begin
        # Clean company → high score
        b_clean = BeneishResult(true, -2.5, false, false,
                                1.0, 1.0, 1.0, 1.1, 1.0, 1.0, 1.0, 0.01,
                                "FY24", "FY23", String[])
        c_clean = CashflowResult(5, 0, -0.1, false)
        p_clean = PledgingResult(0.0, :stable, 0.0, 8, false)
        f_clean = ForensicsResult(nothing, String[], false)
        s_clean = SurveillanceResult(false, false, true, false)

        score = CompanyConfidence._aggregate(b_clean, c_clean, p_clean, f_clean, s_clean)
        @test score == 100.0

        # Manipulator Beneish → -30
        b_manip = BeneishResult(true, -1.5, true, true,
                                1.2, 1.1, 1.1, 1.3, 1.0, 1.1, 1.0, 0.05,
                                "FY24", "FY23", String[])
        score2 = CompanyConfidence._aggregate(b_manip, c_clean, p_clean, f_clean, s_clean)
        @test score2 == 70.0

        # ASM + GSM → -60 additional (clamped to 0)
        s_listed = SurveillanceResult(true, true, true, true)
        # 100 - 30(beneish) - 25(asm) - 35(gsm) = 10
        score3 = CompanyConfidence._aggregate(b_manip, c_clean, p_clean, f_clean, s_listed)
        @test score3 == 10.0

        # Bank → Beneish N/A, no deduction
        b_bank = BeneishResult(false, nothing, nothing, false,
                               nothing, nothing, nothing, nothing,
                               nothing, nothing, nothing, nothing,
                               nothing, nothing, String[])
        score4 = CompanyConfidence._aggregate(b_bank, c_clean, p_clean, f_clean, s_clean)
        @test score4 == 100.0
    end

    @testset "ConfidenceReport pass/fail threshold" begin
        b = BeneishResult(true, -1.5, true, true, 1.2, 1.1, 1.1, 1.3, 1.0, 1.1, 1.0, 0.05, "FY24", "FY23", String[])
        c = CashflowResult(4, 3, 0.2, true)
        p = PledgingResult(30.0, :rising, 8.0, 8, true)
        f = ForensicsResult(Dict{String,Any}("x" => "concern"), ["x: concern"], true)
        s = SurveillanceResult(false, false, true, false)
        score = CompanyConfidence._aggregate(b, c, p, f, s)
        # 100 - 30 - 20 - 20 - 10 - 8 = 12
        @test score < PASS_THRESHOLD
    end

    @testset "Display — smoke test" begin
        b = BeneishResult(true, -2.5, false, false, 1.0, 1.0, 1.0, 1.1, 1.0, 1.0, 1.0, 0.01, "FY24", "FY23", String[])
        c = CashflowResult(5, 1, -0.05, false)
        p = PledgingResult(0.0, :stable, 0.0, 8, false)
        f = ForensicsResult(nothing, String[], false)
        s = SurveillanceResult(false, false, true, false)
        report = ConfidenceReport("test-company", "Test Company Ltd", 95.0, true, b, c, p, f, s, Dates.now(Dates.UTC))
        buf = IOBuffer()
        show(buf, MIME"text/plain"(), report)
        out = String(take!(buf))
        @test occursin("Test Company", out)
        @test occursin("95.0", out)
        @test occursin("PASS", out)
    end

    # ── Integration tests ─────────────────────────────────────────────────────

    sidecar_dir = get(ENV, "TIJORI_SIDECAR_DIR", "")

    if !isempty(sidecar_dir)
        using TijoriData

        @testset "Integration (live sidecar)" begin
            TijoriData.start!(sidecar_dir)
            @test TijoriData.is_running()

            @testset "analyze — Infosys (should pass)" begin
                r = analyze("infosys-limited"; check_surveillance=false)
                @test r isa ConfidenceReport
                @test r.score >= 0 && r.score <= 100
                @test r.beneish.applicable
                @test r.cashflow.years_checked > 0
                @test r.pledging.quarters_used > 0
                # Infosys is a clean company; should pass
                @test r.pass
            end

            @testset "analyze — result types" begin
                r = analyze("infosys-limited"; check_surveillance=false)
                @test r.beneish isa BeneishResult
                @test r.cashflow isa CashflowResult
                @test r.pledging isa PledgingResult
                @test r.forensics isa ForensicsResult
                @test r.surveillance isa SurveillanceResult
            end
        end
    else
        @info "Skipping integration tests (set TIJORI_SIDECAR_DIR to enable)"
    end

end
