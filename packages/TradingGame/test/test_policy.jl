# ── Tests: observation assembly + ActorCriticPolicy forward pass shapes ──────────
#
# No training here — just shape/differentiability checks, per Stage 2 of the
# build plan. Reuses `make_test_cache`/`make_test_env` from test_rules.jl.

@testset "Observation assembly" begin

    @testset "Shapes match the declared constants" begin
        env = make_test_env(n_days=30)
        step!(env, RawAction[])   # give the hourly lookback some history
        N = length(env.candidate_order)

        obs = assemble_observation(env)
        @test size(obs.hourly)    == (N_HOURLY_BARS_SHORT, N_PRICE_CHANNELS, N)
        @test size(obs.macro_ctx) == (N_MACRO_DAYS, N_MACRO_SERIES)
        @test size(obs.news)      == (N_NEWS_FEATURES, N)
        @test size(obs.holding)   == (N_HOLDING_FEATURES, N)
        @test size(obs.portfolio) == (N_PORTFOLIO_SCALARS,)
        @test obs.candidates == env.candidate_order
    end

    @testset "Volatility channel reads the PREVIOUS day's range (no intraday look-ahead)" begin
        env = make_test_env(n_days=30)
        env.cache.vols .= Float32.(1:size(env.cache.vols, 1))   # day i's (H-L)/C := i, unmistakable
        @test N_PRICE_CHANNELS == 2

        obs0 = assemble_observation(env)
        d0 = env.cache.date_index[env.current_date]
        @test all(obs0.hourly[:, 2, :] .== (d0 > 1 ? d0 - 1 : 0))

        for _ in 1:40   # cross at least one day boundary
            step!(env, RawAction[])
        end
        d = env.cache.date_index[env.current_date]
        @test d > d0
        obs = assemble_observation(env)
        @test all(obs.hourly[:, 2, :] .== d - 1)
        @test !any(obs.hourly[:, 2, :] .== d)
    end

    @testset "News feature defaults to zero; a custom news_fn is honoured" begin
        env = make_test_env(n_days=30)
        step!(env, RawAction[])

        obs = assemble_observation(env)
        @test all(iszero, obs.news)

        marker(env, sym_idx, hour_idx) = fill(Float32(sym_idx), N_NEWS_FEATURES)
        obs2 = assemble_observation(env; news_fn=marker)
        for (col, sym_idx) in enumerate(env.candidate_order)
            @test all(==(Float32(sym_idx)), obs2.news[:, col])
        end
    end

    @testset "Holding features reflect an open position, neutral for unheld stocks" begin
        env = make_test_env(n_days=30)
        buy_idx = first(env.candidate_sym_idx)
        step!(env, [RawAction(buy_idx, BUY, 1.0)])

        obs = assemble_observation(env)
        col = findfirst(==(buy_idx), env.candidate_order)
        @test obs.holding[1, col] == 1f0          # held flag
        @test obs.holding[3, col] <= 1f0            # remaining-hold-budget fraction

        for (c, sym_idx) in enumerate(env.candidate_order)
            sym_idx == buy_idx && continue
            @test obs.holding[:, c] == Float32[0, 0, 1]   # neutral default for unheld
        end
    end

    @testset "build_macro_cache: real repo data loads; a directory with no CSVs is all-neutral" begin
        real_dir = joinpath(@__DIR__, "..", "..", "..", "website", "data", "ohlcv", "macro")
        if isdir(real_dir)
            mc = build_macro_cache(real_dir)
            @test size(mc.closes, 2) == N_MACRO_SERIES
            @test length(mc.dates) > 0

            env = make_test_env(n_days=30)
            obs = assemble_observation(env; macro_cache=mc)
            @test size(obs.macro_ctx) == (N_MACRO_DAYS, N_MACRO_SERIES)
        else
            @info "Skipping real-macro-data test — $real_dir not present in this environment"
        end

        empty_dir = mktempdir()
        mc_empty = build_macro_cache(empty_dir)
        @test isempty(mc_empty.dates)
        env = make_test_env(n_days=30)
        obs = assemble_observation(env; macro_cache=mc_empty)
        @test all(iszero, obs.macro_ctx)   # no cached series → documented neutral default
    end

    @testset "stack_observations batches correctly and rejects mismatched candidate sets" begin
        env = make_test_env(n_days=30)
        step!(env, RawAction[])
        obs = [assemble_observation(env) for _ in 1:4]
        batch = stack_observations(obs)

        N = length(env.candidate_order)
        @test size(batch.hourly)    == (N_HOURLY_BARS_SHORT, N_PRICE_CHANNELS, N, 4)
        @test size(batch.news)      == (N_NEWS_FEATURES, N, 4)
        @test size(batch.holding)   == (N_HOLDING_FEATURES, N, 4)
        @test size(batch.macro_ctx) == (N_MACRO_DAYS, N_MACRO_SERIES, 4)
        @test size(batch.portfolio) == (N_PORTFOLIO_SCALARS, 4)

        @test_throws ErrorException stack_observations(Observation[])
    end

end

@testset "ActorCriticPolicy forward pass" begin

    function _batch(env; B::Int=4)
        obs = [assemble_observation(env) for _ in 1:B]
        return stack_observations(obs)
    end

    @testset "CPU forward pass shapes" begin
        env = make_test_env(n_days=30)
        step!(env, RawAction[])
        N = length(env.candidate_order)
        batch = _batch(env)

        policy = ActorCriticPolicy(embed_dim=16, macro_embed_dim=8, attn_heads=2, critic_hidden=[16])
        action_logits, buy_weight_logit, value =
            policy(batch.hourly, batch.news, batch.holding, batch.macro_ctx, batch.portfolio)

        @test size(action_logits)    == (3, N, 4)
        @test size(buy_weight_logit) == (N, 4)
        @test size(value)            == (4,)
        @test all(isfinite, action_logits)
        @test all(isfinite, buy_weight_logit)
        @test all(isfinite, value)
    end

    @testset "seed makes initial weights reproducible" begin
        p1 = ActorCriticPolicy(embed_dim=16, macro_embed_dim=8, attn_heads=2, critic_hidden=[16], seed=42)
        p2 = ActorCriticPolicy(embed_dim=16, macro_embed_dim=8, attn_heads=2, critic_hidden=[16], seed=42)
        p3 = ActorCriticPolicy(embed_dim=16, macro_embed_dim=8, attn_heads=2, critic_hidden=[16], seed=7)

        @test p1.actor_head.weight        == p2.actor_head.weight
        @test p1.hourly_encoder.cell.Wi   == p2.hourly_encoder.cell.Wi
        @test p1.attn.q_proj.weight       == p2.attn.q_proj.weight

        @test p1.actor_head.weight != p3.actor_head.weight
    end

    @testset "Gradients flow through every sub-layer" begin
        env = make_test_env(n_days=30)
        step!(env, RawAction[])
        batch = _batch(env)
        policy = ActorCriticPolicy(embed_dim=16, macro_embed_dim=8, attn_heads=2, critic_hidden=[16])

        loss, grads = Flux.withgradient(policy) do m
            al, bw, v = m(batch.hourly, batch.news, batch.holding, batch.macro_ctx, batch.portfolio)
            sum(abs2, al) + sum(abs2, bw) + sum(abs2, v)
        end
        @test isfinite(loss)
        g = grads[1]
        @test !isnothing(g.hourly_encoder)
        @test !isnothing(g.macro_encoder)
        @test !isnothing(g.attn)
        @test sum(abs, g.actor_head.weight) > 0
        @test sum(abs, g.critic_head.layers[end].weight) > 0
    end

    @testset "GPU forward pass matches CPU shapes (skipped if no functional GPU)" begin
        if CUDA.functional()
            env = make_test_env(n_days=30)
            step!(env, RawAction[])
            N = length(env.candidate_order)
            batch = _batch(env)

            policy = ActorCriticPolicy(embed_dim=16, macro_embed_dim=8, attn_heads=2, critic_hidden=[16]) |> gpu
            hourly, news, holding, macro_ctx, portfolio =
                gpu(batch.hourly), gpu(batch.news), gpu(batch.holding), gpu(batch.macro_ctx), gpu(batch.portfolio)

            action_logits, buy_weight_logit, value = policy(hourly, news, holding, macro_ctx, portfolio)
            @test size(action_logits)    == (3, N, 4)
            @test size(buy_weight_logit) == (N, 4)
            @test size(value)            == (4,)
        else
            @info "Skipping GPU forward-pass test — CUDA.functional() is false in this environment"
        end
    end

end
