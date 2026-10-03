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
        train = [UniverseEntry("BBB", "Beta",  400.0, 65.0), UniverseEntry("CCC", "Gamma", 300.0, 55.0)]
        val   = [UniverseEntry("DDD", "Delta", 250.0, 60.0)]
        out = joinpath(mktempdir(), "universe_latest.json")   # exercises mkpath(dirname(path))
        save_universe_snapshot(SharedTopMarketCap(n=2), train, val, out)
        @test isfile(out)
        loaded = load_universe_snapshot(out)
        @test loaded.train == ["BBB", "CCC"]
        @test loaded.val   == ["DDD"]

        raw = JSON3.read(read(out, String))
        @test raw.strategy == "SharedTopMarketCap"
        @test raw.strategy_params.n == 2
        @test raw.n_candidates_train == 2
        @test raw.n_candidates_val   == 1
    end

    @testset "load_universe_snapshot errors with a clear message when missing" begin
        @test_throws ErrorException load_universe_snapshot(tempname() * "_does_not_exist.json")
    end

    # ── UniverseStrategy subtypes ───────────────────────────────────────────────

    function make_pool(n::Int; base_mcap::Float64=1000.0)
        # Descending market cap, like eligible_candidates' own sort — rank i has
        # the i-th largest cap, so bin/strata tests can reason about rank position.
        return [UniverseEntry("S" * lpad(i, 3, '0'), "Company $i", base_mcap - i, 50.0 + i % 10)
                for i in 1:n]
    end

    @testset "SharedTopMarketCap: identical top-n list for both roles" begin
        pool = make_pool(10)
        (; train, val) = build_universes(SharedTopMarketCap(n=4), pool)
        @test train === val
        @test [e.symbol for e in train] == ["S001", "S002", "S003", "S004"]
    end

    @testset "DisjointTopMarketCap: exact sizes, disjoint, reproducible by seed" begin
        pool = make_pool(40)
        (; train, val) = build_universes(DisjointTopMarketCap(n_train=12, n_val=8, seed=1), pool)
        @test length(train) == 12
        @test length(val)   == 8
        @test isempty(intersect(Set(e.symbol for e in train), Set(e.symbol for e in val)))
        @test issubset(Set(e.symbol for e in train) ∪ Set(e.symbol for e in val),
                        Set(e.symbol for e in pool[1:20]))

        repeat = build_universes(DisjointTopMarketCap(n_train=12, n_val=8, seed=1), pool)
        @test Set(e.symbol for e in train) == Set(e.symbol for e in repeat.train)
        @test Set(e.symbol for e in val)   == Set(e.symbol for e in repeat.val)
    end

    @testset "DisjointTopMarketCap: caps combined pool at pool length" begin
        pool = make_pool(10)
        (; train, val) = build_universes(DisjointTopMarketCap(n_train=8, n_val=8, seed=2), pool)
        @test length(train) + length(val) == 10
    end

    @testset "RandomUniverse: shared draw by default, disjoint when requested" begin
        pool = make_pool(30)
        (; train, val) = build_universes(RandomUniverse(n=10, seed=3), pool)
        @test train === val
        @test length(train) == 10

        disjoint_result = build_universes(RandomUniverse(n=10, disjoint=true, seed=3), pool)
        @test length(disjoint_result.train) == 10
        @test length(disjoint_result.val)   == 10
        @test isempty(intersect(Set(e.symbol for e in disjoint_result.train), Set(e.symbol for e in disjoint_result.val)))
    end

    @testset "RandomUniverse: errors when pool too small for disjoint draw" begin
        pool = make_pool(15)
        @test_throws ErrorException build_universes(RandomUniverse(n=10, disjoint=true), pool)
    end

    @testset "BucketedRandom: respects band membership and quotas" begin
        pool = [UniverseEntry("LOW1", "Low 1", 1000.0, 50.0), UniverseEntry("LOW2", "Low 2", 2000.0, 50.0),
                UniverseEntry("LOW3", "Low 3", 3000.0, 50.0), UniverseEntry("HIGH1", "High 1", 8000.0, 50.0),
                UniverseEntry("HIGH2", "High 2", 9000.0, 50.0)]
        strategy = BucketedRandom(quotas=[(0.0, 5000.0) => 2, (5000.0, Inf) => 2], seed=4)
        (; train, val) = build_universes(strategy, pool)
        @test train === val
        @test length(train) == 4
        @test Set(e.symbol for e in train if e.market_cap_cr <= 5000.0) ⊆ Set(["LOW1", "LOW2", "LOW3"])
        @test length(Set(e.symbol for e in train if e.market_cap_cr > 5000.0)) == 2
    end

    @testset "BucketedRandom: errors when a band's quota exceeds its membership" begin
        pool = [UniverseEntry("LOW1", "Low 1", 1000.0, 50.0)]
        strategy = BucketedRandom(quotas=[(0.0, 5000.0) => 5])
        @test_throws ErrorException build_universes(strategy, pool)
    end

    @testset "BucketedRandom: disjoint draws independent train/val per band" begin
        pool = [UniverseEntry("L" * lpad(i, 2, '0'), "Low $i", 1000.0 - i, 50.0) for i in 1:6]
        strategy = BucketedRandom(quotas=[(0.0, 5000.0) => 3], disjoint=true, seed=5)
        (; train, val) = build_universes(strategy, pool)
        @test length(train) == 3
        @test length(val)   == 3
        @test isempty(intersect(Set(e.symbol for e in train), Set(e.symbol for e in val)))
    end

end
