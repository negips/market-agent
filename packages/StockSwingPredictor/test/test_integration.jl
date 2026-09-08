"""
Integration tests — require real OHLCV data on disk.

Run with:
  SSP_DATA_DIR=/path/to/website/data julia --project=. test/runtests.jl
"""

ohlcv_dir = joinpath(data_dir, "ohlcv")
llm_dir   = joinpath(data_dir, "llm_features")

@testset "integration — build_market_matrices" begin

    isdir(ohlcv_dir) || (@warn "ohlcv_dir not found: $ohlcv_dir"; return)

    csvs = filter(f -> endswith(f, "_daily.csv"), readdir(ohlcv_dir))
    isempty(csvs) && (@warn "No daily CSVs in $ohlcv_dir"; return)

    # Use first 5 companies found on disk
    companies = first([replace(f, "_daily.csv" => "") for f in csvs], 5)

    closes, vols, dates = build_market_matrices(ohlcv_dir, companies)

    @test length(dates) > 0
    @test size(closes) == (length(dates), length(companies))
    @test size(vols)   == (length(dates), length(companies))

    # After forward-fill, no column should be entirely NaN
    for j in 1:length(companies)
        if !all(isnan, closes[:, j])
            @test !all(isnan, closes[:, j])
        end
    end

    # Vols should be non-negative
    valid_vols = filter(!isnan, vols)
    @test all(>=(0), valid_vols)

end

@testset "integration — generate_company_examples" begin

    isdir(ohlcv_dir) || return

    csvs = filter(f -> endswith(f, "_daily.csv"), readdir(ohlcv_dir))
    isempty(csvs) && return

    sym = replace(first(csvs), "_daily.csv" => "")

    daily_path  = joinpath(ohlcv_dir, "$(sym)_daily.csv")
    hourly_path = joinpath(ohlcv_dir, "$(sym)_hourly.csv")
    (isfile(daily_path) && isfile(hourly_path)) || return

    daily  = sort!(CSV.read(daily_path,  DataFrame; types=Dict(:date => Date)), :date)
    hourly = sort!(CSV.read(hourly_path, DataFrame; types=Dict(:datetime => DateTime)), :datetime)

    # Build a minimal master calendar from the daily data
    master_dates = daily.date

    examples = generate_company_examples(sym, 1, daily, hourly,
                                          master_dates, Dict{Date, LLMFeatures}())

    @test !isempty(examples)

    ex = examples[1]
    @test length(ex.hourly) == N_HOURLY_BARS
    @test length(ex.llm)    == N_LLM_FEATURES
    @test length(ex.label)  == N_PRED_HOURS
    @test all(isfinite, ex.hourly)
    @test all(isfinite, ex.label)
    @test ex.hourly[1] ≈ 1f0   # normalised: first bar = 1.0

end
