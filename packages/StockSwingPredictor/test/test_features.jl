@testset "features — LLM serialisation" begin

    @testset "llm_to_vec length" begin
        v = llm_to_vec(MISSING_LLM)
        @test length(v) == N_LLM_FEATURES
        @test eltype(v) == Float32
    end

    @testset "llm_to_vec values match struct fields" begin
        f = LLMFeatures(0.5f0, -1f0, 0.3f0, 0.7f0, -0.5f0,
                        0.2f0, 1f0, 0f0, 1f0,
                        0.4f0, 0f0, 1f0, 0f0,
                        0.9f0, 30f0)
        v = llm_to_vec(f)
        @test v[1]  == f.management_tone
        @test v[2]  == f.guidance_direction
        @test v[14] == f.extraction_confidence
        @test v[15] == f.doc_age_days
    end

    @testset "llm_feature_names length matches N_LLM_FEATURES" begin
        @test length(llm_feature_names()) == N_LLM_FEATURES
    end

end

@testset "features — latest_before" begin

    cache = Dict(
        Date(2024, 1, 1) => 1.0,
        Date(2024, 3, 1) => 3.0,
        Date(2024, 6, 1) => 6.0,
    )

    @test latest_before(cache, Date(2024, 4, 1), 0.0) == 3.0
    @test latest_before(cache, Date(2024, 1, 1), 0.0) == 1.0   # exact match
    @test latest_before(cache, Date(2023, 12, 31), 0.0) == 0.0  # before all keys → default
    @test latest_before(cache, Date(2025, 1, 1), 0.0) == 6.0   # after all keys → latest

end

@testset "features — find_date_index" begin

    dates = [Date(2024, 1, 1), Date(2024, 1, 2), Date(2024, 1, 5), Date(2024, 1, 8)]

    @test find_date_index(dates, Date(2024, 1, 5))  == 3   # exact match
    @test find_date_index(dates, Date(2024, 1, 6))  == 3   # between dates → floor
    @test find_date_index(dates, Date(2024, 1, 8))  == 4   # last
    @test find_date_index(dates, Date(2023, 12, 31)) == 0  # before first → 0

end
