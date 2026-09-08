using Test, EarningsCalendar, Dates

@testset "EarningsCalendar" begin

    @testset "_parse_nse_date" begin
        f = EarningsCalendar._parse_nse_date
        @test f("17-Sep-2026") == Date(2026, 9, 17)
        @test f("01-10-2026")  == Date(2026, 10, 1)
        @test f("31-Dec-2025") == Date(2025, 12, 31)
        @test f("-")           === nothing
        @test f("")            === nothing
        @test f("   ")         === nothing
    end

    @testset "_is_earnings" begin
        f = EarningsCalendar._is_earnings
        @test f("Quarterly Results")
        @test f("Financial Results")
        @test f("Annual Results")
        @test f("Half Yearly Results")
        @test f("Board Meeting")
        @test f("Board Meeting to consider Quarterly Results")
        @test f("quarterly results")        # case-insensitive
        @test !f("Dividend")
        @test !f("Stock Split")
        @test !f("Buy Back")
        @test !f("Bonus")
        @test !f("")
    end

    @testset "upcoming_earnings argument validation" begin
        @test_throws ArgumentError upcoming_earnings(0)
        @test_throws ArgumentError upcoming_earnings(-5)
    end

    @testset "EarningsEvent construction" begin
        e = EarningsEvent("INFY", "Infosys Limited", Date(2026, 10, 17), "Quarterly Results")
        @test e.symbol  == "INFY"
        @test e.company == "Infosys Limited"
        @test e.date    == Date(2026, 10, 17)
        @test e.purpose == "Quarterly Results"
    end

    @testset "display" begin
        e = EarningsEvent("TCS", "Tata Consultancy Services", Date(2026, 10, 9), "Quarterly Results")
        @test contains(sprint(show, e), "TCS")
        @test contains(sprint(show, e), "Quarterly Results")

        events = [e]
        txt = sprint(show, MIME"text/plain"(), events)
        @test contains(txt, "TCS")

        empty_txt = sprint(show, MIME"text/plain"(), EarningsEvent[])
        @test contains(empty_txt, "no events")
    end

    # Integration test — requires live NSE access
    if get(ENV, "NSE_INTEGRATION", "0") == "1"
        @testset "live NSE fetch" begin
            events = upcoming_earnings(30)
            @test events isa Vector{EarningsEvent}
            @test !isempty(events)
            @test all(e -> e.date >= today(), events)
            @test all(e -> !isempty(e.symbol), events)
            @test issorted(events, by = e -> e.date)
        end
    end

end
