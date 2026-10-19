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

@testset "excluded_symbol: symbol-only rules" begin
    junk = ["772SBIPER", "960EFSLA", "D60MS264DG", "D60MS264DP", "EFSL241024", "EFSLZC26C", "IIFLZC26", "IPRU3533",
            "KFL150425", "KFLZC26D", "MFL26B", "MFL26D", "MFL26Z", "SFMP51GR"]
    @test all(s -> excluded_symbol(s; exchange="BSE"), junk)
    @test all(excluded_symbol, ["CENTEXT-RE", "INDRA-RE", "MANOJ-RE", "NATCO-RE", "SALEM-RE"])
    @test all(s -> excluded_symbol(s; exchange="BSE"), ["EMBASSY-RR", "NIFTYINAV", "GS12DEC2034", "64GJ30A"])
    real = ["RELIANCE", "360ONE", "BODHTREE", "FRONTCAP", "GFSTEELS", "NOBPOL", "RAJINFRA", "RGIL", "SHENTRA", "3MINDIA", "TCS"]
    @test !any(s -> excluded_symbol(s; exchange="BSE"), real)
    @test !any(s -> excluded_symbol(s; exchange="NSE"), real)
    @test !excluded_symbol("MFL26B"; exchange="NSE")
end
