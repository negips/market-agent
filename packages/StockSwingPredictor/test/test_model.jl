@testset "model — architecture config" begin

    @testset "DUAL_CNN_V1 is a DualCNN" begin
        @test DUAL_CNN_V1 isa DualCNN
        @test DUAL_CNN_V1 isa SwingArchitecture
        @test DUAL_CNN_V1.name == "DualCNN_v1"
    end

    @testset "DualCNN arch serialisation roundtrip" begin
        arch  = DUAL_CNN_V1
        d     = StockSwingPredictor._arch_to_dict(arch)
        arch2 = StockSwingPredictor._arch_from_dict(d)

        @test arch2.name                == arch.name
        @test arch2.market_channels     == arch.market_channels
        @test arch2.market_kernel       == arch.market_kernel
        @test arch2.hourly_channels     == arch.hourly_channels
        @test arch2.hourly_kernel_large == arch.hourly_kernel_large
        @test arch2.hourly_kernel_small == arch.hourly_kernel_small
        @test arch2.mlp_hidden          == arch.mlp_hidden
        @test arch2.dropout_rate        == arch.dropout_rate
    end

    @testset "custom DualCNN variant" begin
        arch = DualCNN(name="DualCNN_v2", market_channels=[2, 64, 128, 256])
        @test arch.name == "DualCNN_v2"
        @test arch.market_channels == [2, 64, 128, 256]
        # Unspecified fields keep defaults
        @test arch.hourly_channels == DUAL_CNN_V1.hourly_channels
    end

end

@testset "model — forward pass shapes" begin

    # Use a small N (not 500) for speed — the reshape logic is what matters
    N_TEST = 8

    model = build_model(DUAL_CNN_V1)
    Flux.testmode!(model)

    for B in [1, 4]   # single example and a real batch
        @testset "B=$B" begin
            market = rand(Float32, N_MARKET_DAYS, N_MARKET_CHANNELS, N_TEST, B)
            hourly = rand(Float32, N_HOURLY_BARS, B)
            llm    = rand(Float32, N_LLM_FEATURES, B)

            out = model(market, hourly, llm)
            @test size(out) == (N_PRED_HOURS, B)
            @test eltype(out) == Float32
            @test all(isfinite, out)
        end
    end

end

@testset "model — gradient flow" begin

    N_TEST = 8
    B      = 2
    model  = build_model(DUAL_CNN_V1)

    market = rand(Float32, N_MARKET_DAYS, N_MARKET_CHANNELS, N_TEST, B)
    hourly = rand(Float32, N_HOURLY_BARS, B)
    llm    = rand(Float32, N_LLM_FEATURES, B)
    y      = rand(Float32, N_PRED_HOURS, B)

    loss, grads = Flux.withgradient(model) do m
        Flux.mse(m(market, hourly, llm), y)
    end

    @test isfinite(loss)
    @test !isnothing(grads[1])

    # At least some parameters should have non-zero gradients
    any_nonzero = any(any(!iszero, g) for g in Flux.trainables(grads[1])
                      if g isa AbstractArray)
    @test any_nonzero

end

@testset "model — save/load roundtrip" begin

    N_TEST    = 8
    model     = build_model(DUAL_CNN_V1)
    companies = ["SYM$i" for i in 1:N_TEST]

    tmp  = mktempdir()
    path = joinpath(tmp, "test_model.bson")

    save_model(model, companies, path; meta=Dict("test" => true))
    model2, companies2, meta2 = load_model(path)

    @test companies2 == companies
    @test meta2["test"] == true
    @test model2 isa SwingPredictor{DualCNN}
    @test model2.arch.name == DUAL_CNN_V1.name

    # Weights must be identical after roundtrip
    B      = 2
    market = rand(Float32, N_MARKET_DAYS, N_MARKET_CHANNELS, N_TEST, B)
    hourly = rand(Float32, N_HOURLY_BARS, B)
    llm    = rand(Float32, N_LLM_FEATURES, B)

    Flux.testmode!(model);  Flux.testmode!(model2)
    @test model(market, hourly, llm) ≈ model2(market, hourly, llm)

end
