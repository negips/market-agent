# ── Tests: PPO mechanics + train_policy! conventions ──────────────────────────────
#
# Mechanical/shape/plumbing tests only — these run a handful of iterations on
# the tiny synthetic fixture, fast enough for CI. The actual "does the policy
# learn" demonstration (episode return trending upward over ~80 iterations on
# a strongly-trending synthetic stock) was run manually as the Stage 3 sanity
# check; see the TradingGame module docstring's quick-start for how to repeat it.

@testset "PPO rollout + update mechanics" begin

    @testset "collect_rollout: shapes and a genuine trade occurs under a random policy" begin
        env = make_test_env(n_days=30)
        train_config = EpisodeConfig(initial_cash=100_000.0, start_date=env.cache.dates[1],
                                      end_date=env.cache.dates[end], candidate_universe=env.cache.companies)
        policy = ActorCriticPolicy(embed_dim=8, macro_embed_dim=4, attn_heads=2, critic_hidden=[8])

        buffer = collect_rollout(env, policy, train_config; rng=MersenneTwister(3))
        @test length(buffer) > 0
        @test buffer[end].done
        @test all(s -> length(s.action_idx) == length(env.candidate_order), buffer)
        @test all(s -> all(∈((1, 2, 3)), s.action_idx), buffer)
        @test all(s -> isfinite(s.logprob) && isfinite(s.value), buffer)

        # With an untrained (near-uniform) policy over ~200 decision steps × 3
        # candidates, the all-neutral-portfolio-scalar bug (fixed during Stage 3)
        # would make every reward exactly 0 — assert at least one real trade fired.
        @test any(!iszero, [s.reward for s in buffer])
    end

    @testset "compute_gae: terminal step has no bootstrap term" begin
        rewards = Float32[0.1, -0.05, 0.2]
        values  = Float32[1.0, 1.1, 0.9]
        dones   = [false, false, true]
        adv, ret = compute_gae(rewards, values, dones; gamma=0.99, gae_lambda=0.95)
        @test length(adv) == 3
        @test ret ≈ adv .+ values
        # At the terminal step, δ_T = r_T - V_T (no next-value term) — verify directly.
        @test adv[3] ≈ rewards[3] - values[3]
    end

    @testset "ppo_update! runs and produces a finite loss" begin
        env = make_test_env(n_days=30)
        train_config = EpisodeConfig(initial_cash=100_000.0, start_date=env.cache.dates[1],
                                      end_date=env.cache.dates[end], candidate_universe=env.cache.companies)
        policy = ActorCriticPolicy(embed_dim=8, macro_embed_dim=4, attn_heads=2, critic_hidden=[8])
        opt_state = Flux.setup(Flux.Adam(1f-3), policy)

        buffer = collect_rollout(env, policy, train_config; rng=MersenneTwister(4))
        stats = ppo_update!(policy, opt_state, buffer; k_epochs=2, minibatch_size=16, rng=MersenneTwister(5))
        @test isfinite(stats["loss"])
    end

end

@testset "train_policy! conventions" begin

    function _train_setup(; n_days=30)
        env = make_test_env(n_days=n_days)
        train_config = EpisodeConfig(initial_cash=100_000.0, start_date=env.cache.dates[1],
                                      end_date=env.cache.dates[end], candidate_universe=env.cache.companies)
        policy = ActorCriticPolicy(embed_dim=8, macro_embed_dim=4, attn_heads=2, critic_hidden=[8])
        return env, train_config, policy
    end

    @testset "log structure, checkpoint save, and reload round-trips the policy" begin
        env, train_config, policy = _train_setup()
        ckpt = tempname() * ".bson"
        elog = tempname() * ".jsonl"

        policy, log = train_policy!(policy, env, train_config;
                                     iterations=5, k_epochs=2, minibatch_size=16,
                                     checkpoint_path=ckpt, episode_log_path=elog,
                                     embed_dim=8, macro_embed_dim=4, attn_heads=2, critic_hidden=[8],
                                     rng=MersenneTwister(6))

        @test log["iterations_run"] == 5
        @test length(log["train_return"]) == 5
        @test length(log["loss"]) == 5
        @test log["stop_now"] == false
        @test isfinite(log["best_return"])
        @test 1 <= log["best_iteration"] <= 5

        @test isfile(ckpt)
        loaded, meta = load_policy(ckpt)
        @test meta["checkpoint"] == true
        # train_policy! always reloads best-checkpointed weights before returning,
        # so the returned `policy` and the on-disk checkpoint must match exactly.
        obs = assemble_observation(env)
        batch = stack_observations([obs])
        al1, bw1, v1 = policy(batch.hourly, batch.news, batch.holding, batch.macro_ctx, batch.portfolio)
        al2, bw2, v2 = loaded(batch.hourly, batch.news, batch.holding, batch.macro_ctx, batch.portfolio)
        @test al2 ≈ al1
        @test bw2 ≈ bw1
        @test v2  ≈ v1

        @test isfile(elog)
        lines = readlines(elog)
        @test length(lines) == 5
        rec = JSON3.read(lines[1])
        @test rec.iteration == 1
        @test haskey(rec, :train_return) && haskey(rec, :loss) && haskey(rec, :best_return)
    end

    @testset "STOP file: clean interrupt after the current iteration, file removed" begin
        env, train_config, policy = _train_setup()
        stop_file = tempname() * "_STOP"   # must literally contain "STOP" — train_policy! derives
        touch(stop_file)                    # the STOP_NOW path via string replacement on this name


        policy, log = train_policy!(policy, env, train_config;
                                     iterations=50, k_epochs=1, minibatch_size=16,
                                     stop_file=stop_file, rng=MersenneTwister(7))

        @test log["iterations_run"] == 1
        @test log["stop_now"] == false
        @test !isfile(stop_file)
    end

    @testset "STOP_NOW file: hard interrupt, distinct from STOP" begin
        env, train_config, policy = _train_setup()
        stop_file     = tempname() * "_STOP"
        stop_now_file = replace(stop_file, "STOP" => "STOP_NOW")
        touch(stop_now_file)

        policy, log = train_policy!(policy, env, train_config;
                                     iterations=50, k_epochs=1, minibatch_size=16,
                                     stop_file=stop_file, rng=MersenneTwister(8))

        @test log["iterations_run"] == 1
        @test log["stop_now"] == true
        @test !isfile(stop_now_file)
    end

    @testset "val_config drives checkpointing when provided" begin
        env, train_config, policy = _train_setup(n_days=40)
        val_config = EpisodeConfig(initial_cash=100_000.0, start_date=env.cache.dates[21],
                                    end_date=env.cache.dates[end], candidate_universe=env.cache.companies)

        policy, log = train_policy!(policy, env, train_config; val_config=val_config,
                                     iterations=6, k_epochs=1, minibatch_size=16, eval_every=2,
                                     rng=MersenneTwister(9))
        @test length(log["val_return"]) == 3   # iterations 2, 4, 6
        @test log["best_return"] in log["val_return"]   # checkpoint metric was the held-out return
    end

end
