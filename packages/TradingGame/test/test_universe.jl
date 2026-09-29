# ── Tests: candidate-universe construction (universe.jl) ──────────────────────────
#
# Offline only — a synthetic nse_companies_latest.json-shaped fixture, no
# sidecar/network calls. Reuses make_test_cache from test_rules.jl for the
# InferenceCache side of the intersection.

function write_companies_fixture(path::String, rows::Vector)
    open(path, "w") do io
        JSON3.write(io, (companies = rows,))
    end
end

@testset "Candidate universe" begin

    @testset "Filters by confidence, market cap presence, and cache membership" begin
        cache = make_test_cache(symbols=["AAA", "BBB", "CCC"])
        path  = tempname() * ".json"
        write_companies_fixture(path, [
            (symbol="AAA", name="Alpha",   market_cap_cr=500, confidence=(score=65, pass=true)),
            (symbol="BBB", name="Beta",    market_cap_cr=300, confidence=(score=20, pass=false)),  # below threshold
            (symbol="CCC", name="Gamma",   market_cap_cr=800),                                      # no confidence key
            (symbol="ZZZ", name="Zed",     market_cap_cr=900, confidence=(score=90, pass=true)),     # not in cache
            (symbol="DDD", name="Delta",   confidence=(score=70, pass=true)),                        # no market cap
        ])

        entries = build_candidate_universe(cache, path)
        @test [e.symbol for e in entries] == ["AAA"]
        @test entries[1].market_cap_cr    == 500.0
        @test entries[1].confidence_score == 65.0
    end

    @testset "Sorted by market cap descending and capped at n_candidates" begin
        symbols = ["AAA", "BBB", "CCC", "DDD"]
        cache = make_test_cache(symbols=symbols)
        path  = tempname() * ".json"
        write_companies_fixture(path, [
            (symbol="AAA", name="Alpha", market_cap_cr=100, confidence=(score=50, pass=true)),
            (symbol="BBB", name="Beta",  market_cap_cr=400, confidence=(score=50, pass=true)),
            (symbol="CCC", name="Gamma", market_cap_cr=300, confidence=(score=50, pass=true)),
            (symbol="DDD", name="Delta", market_cap_cr=200, confidence=(score=50, pass=true)),
        ])

        full = build_candidate_universe(cache, path; n_candidates=100)
        @test [e.symbol for e in full] == ["BBB", "CCC", "DDD", "AAA"]

        capped = build_candidate_universe(cache, path; n_candidates=2)
        @test [e.symbol for e in capped] == ["BBB", "CCC"]
    end

    @testset "min_confidence is configurable" begin
        cache = make_test_cache(symbols=["AAA", "BBB"])
        path  = tempname() * ".json"
        write_companies_fixture(path, [
            (symbol="AAA", name="Alpha", market_cap_cr=100, confidence=(score=45, pass=true)),
            (symbol="BBB", name="Beta",  market_cap_cr=200, confidence=(score=55, pass=true)),
        ])

        @test length(build_candidate_universe(cache, path; min_confidence=40.0)) == 2
        @test length(build_candidate_universe(cache, path; min_confidence=50.0)) == 1
        @test length(build_candidate_universe(cache, path; min_confidence=60.0)) == 0
    end

    @testset "save_universe_snapshot / load_universe_snapshot round-trip" begin
        entries = [
            UniverseEntry("BBB", "Beta",  400.0, 65.0),
            UniverseEntry("CCC", "Gamma", 300.0, 55.0),
        ]
        out = joinpath(mktempdir(), "universe_latest.json")   # exercises mkpath(dirname(path))
        save_universe_snapshot(entries, out)
        @test isfile(out)
        @test load_universe_snapshot(out) == ["BBB", "CCC"]
    end

    @testset "load_universe_snapshot errors with a clear message when missing" begin
        @test_throws ErrorException load_universe_snapshot(tempname() * "_does_not_exist.json")
    end

end
