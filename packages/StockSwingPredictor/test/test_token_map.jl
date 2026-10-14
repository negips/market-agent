# ── Tests: build_token_map keeps operating companies, drops fund vehicles and debt ─────

@testset "build_token_map: instrument filters" begin
    rows(spec) = DataFrame(
        instrument_token = 1:length(spec), exchange = fill("NSE", length(spec)),
        tradingsymbol = first.(spec), name = last.(spec), tick_size = fill(0.05, length(spec)),
        instrument_type = fill("EQ", length(spec)))
    kept = ["RELIANCE" => "RELIANCE INDUSTRIES", "FIRSTCRY" => "BRAINBEES SOLUTIONS", "JETFREIGHT" => "JET FREIGHT LOGISTICS",
            "HDFCAMC" => "HDFC ASSET MANAGEMENT COMPANY", "AAKAAR-SM" => "AAKAAR"]
    dropped = ["MAFSETINAV" => "MAFSETF INAV", "AUTOBEINAV" => "AUTOBEES INAV", "AXISGOINAV" => "AXISGOLD NAV",
               "AB10BKINAV" => "BIRLASLAMC - AB10BKINAV", "AONESILVER" => "AONEAMC - AONESILVER",
               "ABSLBANETF" => "ABSLBANETF", "NIFTYBEES" => "NIPPON INDIA ETF NIFTY BEES",
               "BHARATBOND-APR30" => "NSE INDEX BHARATBOND-APR30", "SOMEFUND" => "SOME LIQUID FUND"]
    tm = build_token_map(rows([kept; dropped]); exchange="NSE")
    @test Set(keys(tm)) == Set(first.(kept))
end
