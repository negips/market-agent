"""
TijoriData test suite.

Tests are split into two groups:
  - Unit tests: run without a live sidecar (test parsing helpers)
  - Integration tests: require a running sidecar (TIJORI_SIDECAR_DIR must be set)

Run unit tests only:
  julia --project=. test/runtests.jl

Run integration tests:
  TIJORI_SIDECAR_DIR=/path/to/sidecar julia --project=. test/runtests.jl
"""

using Test
using TijoriData
using DataFrames

# ── Unit tests (no sidecar required) ─────────────────────────────────────────

@testset "TijoriData" begin

    @testset "Number parsing" begin
        @test TijoriData._parse_number("1,23,456.78") == 123456.78
        @test TijoriData._parse_number("18.4%")       == 18.4
        @test TijoriData._parse_number("—")           === nothing
        @test TijoriData._parse_number("")             === nothing
        @test TijoriData._parse_number(nothing)        === nothing
        @test TijoriData._parse_number(42.0)           == 42.0
    end

    @testset "Financials DataFrame conversion" begin
        fake_raw = (
            type = "pl",
            headers = ["metric", "FY23", "FY24"],
            rows = [
                (metric="Net Revenue", var"FY23"="1,000.00", var"FY24"="1,200.00"),
                (metric="PAT",         var"FY23"="100.00",   var"FY24"="—"),
            ]
        )
        df = TijoriData._financials_to_df(fake_raw)
        @test df isa DataFrame
        @test "metric" in names(df)
        @test df[1, :metric] == "Net Revenue"
        @test df[1, Symbol("FY24")] == 1200.0
        @test ismissing(df[2, Symbol("FY24")]) || isnothing(df[2, Symbol("FY24")])
    end

    @testset "Shareholding DataFrame conversion" begin
        fake_raw = (
            quarters = [
                (period="Sep 24", Promoter=26.3, var"Promoter Pledged"=5.1, FII=54.0, DII=12.0, Public=7.7),
                (period="Dec 24", Promoter=26.1, var"Promoter Pledged"=4.8, FII=54.5, DII=12.2, Public=7.2),
            ]
        )
        df = TijoriData._shareholding_to_df(fake_raw)
        @test df isa DataFrame
        @test "period" in names(df)
        @test df[1, :period] == "Sep 24"
        @test df[2, Symbol("Promoter Pledged")] == 4.8
    end

    @testset "Type constructors" begin
        err = TijoriError("test error")
        @test err.message == "test error"
        @test err isa Exception

        doc = Document("FY24", "https://files.tijorifinance.com/test.pdf")
        @test doc.period == "FY24"

        dt = DocumentText("https://example.com/test.pdf", 10, "hello world")
        @test dt.pages == 10
        @test dt.text == "hello world"
    end

    # ── Integration tests ─────────────────────────────────────────────────────

    sidecar_dir = get(ENV, "TIJORI_SIDECAR_DIR", "")

    if !isempty(sidecar_dir)
        @testset "Integration (live sidecar)" begin
            TijoriData.start!(sidecar_dir)
            @test is_running()

            @testset "search_company" begin
                results = search_company("Infosys")
                @test !isempty(results)
                @test any(r -> occursin("Infosys", r.name), results)
            end

            @testset "get_overview" begin
                slug = first(filter(r -> occursin("Infosys", r.name),
                                    search_company("Infosys"))).slug
                ov = get_overview(slug)
                @test ov.company != ""
                @test !isnothing(ov.company_id)
                @test ov.ratios isa Dict
            end

            @testset "get_financials" begin
                slug = "infosys-limited"
                for type in (:pl, :bs, :cf)
                    df = get_financials(slug, type)
                    @test df isa DataFrame
                    @test "metric" in names(df)
                    @test nrow(df) > 0
                end
            end

            @testset "get_shareholding" begin
                df = get_shareholding("infosys-limited")
                @test df isa DataFrame
                @test "period" in names(df)
                @test nrow(df) > 0
            end

            @testset "screen_companies" begin
                r = screen_companies("( ROCE > 25 ) and ( Market Capitalization > 10000 )")
                @test r isa ScreenResult
                @test r.total_results >= 0
                @test r.data isa DataFrame
            end
        end
    else
        @info "Skipping integration tests (set TIJORI_SIDECAR_DIR to enable)"
    end

end
