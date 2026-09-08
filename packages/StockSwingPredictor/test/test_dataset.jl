using CSV

# ── Helpers shared across dataset tests ──────────────────────────────────────

function _make_daily(n_days::Int; base_close=100.0)
    start = Date(2023, 1, 2)
    dates = [start + Day(i) for i in 0:n_days-1]
    DataFrame(
        date   = dates,
        open   = base_close .+ randn(n_days),
        high   = base_close .+ abs.(randn(n_days)) .+ 1.0,
        low    = base_close .- abs.(randn(n_days)) .- 1.0,
        close  = base_close .+ cumsum(randn(n_days) .* 0.5),
        volume = fill(1e6, n_days),
    )
end

function _make_hourly(daily::DataFrame)
    rows = []
    for d in daily.date
        for h in 1:N_HOURS_PER_DAY
            push!(rows, (
                datetime = DateTime(d, Time(8 + h, 15)),
                open     = 100.0 + randn(),
                high     = 102.0,
                low      = 98.0,
                close    = 100.0 + randn(),
                volume   = 1e5,
            ))
        end
    end
    DataFrame(rows)
end

function _make_dataset(n_companies::Int, n_dates::Int)
    companies = ["SYM$j" for j in 1:n_companies]
    dates     = [Date(2023, 1, 2) + Day(i) for i in 0:n_dates-1]
    closes    = Float32.(100 .+ randn(n_dates, n_companies))
    # Ensure no NaN and positive prices
    closes    = abs.(closes) .+ 10f0
    vols      = Float32.(abs.(randn(n_dates, n_companies)) .* 0.02)

    # Create a few dummy examples spread across the date range
    examples = TrainingExample[]
    for j in 1:n_companies
        t = N_MARKET_DAYS + 5
        t + N_PRED_DAYS <= n_dates || continue
        push!(examples, TrainingExample(
            dates[t], "SYM$j", t, j,
            ones(Float32, N_HOURLY_BARS),
            zeros(Float32, N_LLM_FEATURES),
            zeros(Float32, N_PRED_HOURS),
        ))
    end

    Dataset(closes, vols, dates, companies, examples)
end

# ── Label computation ─────────────────────────────────────────────────────────

@testset "label_5d_hourly" begin

    n_days = 20
    daily  = _make_daily(n_days)
    hourly = _make_hourly(daily)

    @testset "returns length N_PRED_HOURS" begin
        label = label_5d_hourly(hourly, daily, 5)
        @test !isnothing(label)
        @test length(label) == N_PRED_HOURS
        @test eltype(label) == Float32
    end

    @testset "returns nothing when near end of data" begin
        label = label_5d_hourly(hourly, daily, n_days - 2)
        @test isnothing(label)
    end

    @testset "values are finite log-returns" begin
        label = label_5d_hourly(hourly, daily, 5)
        @test !isnothing(label)
        @test all(isfinite, label)
    end

    @testset "empty hourly → nothing" begin
        label = label_5d_hourly(DataFrame(), daily, 5)
        @test isnothing(label)
    end

end

# ── assemble_batch ────────────────────────────────────────────────────────────

@testset "assemble_batch" begin

    N, n_ex = 6, 3
    dataset = _make_dataset(N, 60)
    isempty(dataset.examples) && @warn "No examples generated — increase n_dates"

    if !isempty(dataset.examples)
        idx = 1:min(n_ex, length(dataset.examples))
        market, hourly, llm, y = assemble_batch(dataset, idx)

        @testset "output shapes" begin
            B = length(idx)
            @test size(market) == (N_MARKET_DAYS, N_MARKET_CHANNELS, N, B)
            @test size(hourly) == (N_HOURLY_BARS, B)
            @test size(llm)    == (N_LLM_FEATURES, B)
            @test size(y)      == (N_PRED_HOURS, B)
        end

        @testset "normalisation: first day of each company = 1.0" begin
            # market[:, 1, :, b] is the close channel (normalised)
            # First row (day 1 of window) should be 1.0 for all companies
            @test all(market[1, 1, :, :] .≈ 1f0)
        end

        @testset "target company at column 1" begin
            # The target company's sym_idx should match position 1.
            # We can verify indirectly: market[:, 1, 1, b] should correspond
            # to the target stock's closes (same normalised series).
            ex   = dataset.examples[first(idx)]
            t    = ex.date_idx
            k    = ex.sym_idx
            raw  = dataset.closes[t-N_MARKET_DAYS+1:t, k]
            norm = Float32.(raw ./ raw[1])
            @test market[:, 1, 1, 1] ≈ norm
        end

        @testset "no NaN or Inf in output" begin
            @test all(isfinite, market)
            @test all(isfinite, hourly)
            @test all(isfinite, llm)
        end
    end

end

# ── time_split ────────────────────────────────────────────────────────────────

@testset "time_split" begin

    dataset = _make_dataset(10, 200)

    # Pad to a known number of examples for deterministic split
    fill_ex = dataset.examples[1]
    while length(dataset.examples) < 100
        push!(dataset.examples, fill_ex)
    end

    train_idx, val_idx, test_idx = time_split(dataset)

    @testset "sizes sum to total" begin
        @test length(train_idx) + length(val_idx) + length(test_idx) == length(dataset.examples)
    end

    @testset "no overlap between splits" begin
        @test isempty(intersect(train_idx, val_idx))
        @test isempty(intersect(train_idx, test_idx))
        @test isempty(intersect(val_idx,   test_idx))
    end

    @testset "time ordering preserved (train < val < test)" begin
        @test maximum(train_idx) < minimum(val_idx)
        @test maximum(val_idx)   < minimum(test_idx)
    end

    @testset "approximate 80/10/10 proportions" begin
        n = length(dataset.examples)
        @test length(train_idx) ≈ n * 0.80  atol=2
        @test length(val_idx)   ≈ n * 0.10  atol=2
    end

end

# ── Dataset save/load ─────────────────────────────────────────────────────────

@testset "Dataset save/load roundtrip" begin

    dataset = _make_dataset(5, 80)
    isempty(dataset.examples) && return

    tmp  = mktempdir()
    path = joinpath(tmp, "test_dataset.bson")

    save_dataset(dataset, path)
    dataset2 = load_dataset(path)

    @test dataset2.companies == dataset.companies
    @test dataset2.dates     == dataset.dates
    @test size(dataset2.closes) == size(dataset.closes)
    @test size(dataset2.vols)   == size(dataset.vols)
    @test length(dataset2.examples) == length(dataset.examples)

    # Check a specific example survived roundtrip
    ex  = dataset.examples[1]
    ex2 = dataset2.examples[1]
    @test ex2.symbol   == ex.symbol
    @test ex2.date     == ex.date
    @test ex2.date_idx == ex.date_idx
    @test ex2.sym_idx  == ex.sym_idx
    @test ex2.hourly   ≈ ex.hourly
    @test ex2.llm      ≈ ex.llm
    @test ex2.label    ≈ ex.label

end
