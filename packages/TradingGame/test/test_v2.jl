# ── Tests: game v2 (same-bar execution, no forced exit, soft penalties, cash token) ──
#
# Reuses `make_test_cache` from test_rules.jl. Every v1 behaviour is covered by
# the existing files and must be unchanged — these only pin what v2 adds.

function make_rules_env(rules::GameRules; n_days::Int=40, initial_cash::Float64=1_000_000.0)
    cache = make_test_cache(n_days=n_days)
    env   = TradingGameEnv(cache)
    reset!(env, EpisodeConfig(initial_cash=initial_cash, start_date=cache.dates[1],
                               end_date=cache.dates[end], candidate_universe=cache.companies,
                               rules=rules))
    return env
end

buy_a(env) = [RawAction(env.cache.sym_index["AAA"], BUY, 1.0)]

@testset "Game v2" begin

    @testset "Rule sets: v1 is the default, v2 differs exactly where specified" begin
        r1, r2 = rules_v1(), rules_v2()
        @test EpisodeConfig(initial_cash=1.0, start_date=Date(2024,1,1), end_date=Date(2024,1,2),
                             candidate_universe=["A"]).rules == r1
        @test (r1.same_bar_execution, r1.forced_exit, r1.max_hold_days, r1.cash_token) == (false, true, MAX_HOLD_DAYS, false)
        @test (r2.same_bar_execution, r2.forced_exit, r2.max_hold_days, r2.cash_token) == (true, false, MAX_HOLD_DAYS_V2, true)
        @test r2.cash_penalty_coef == 0.0 && r2.hold_penalty_coef == 0.0       # both penalties default off
        @test rules_v2(cash_penalty=0.05, hold_penalty=0.1).cash_penalty_coef == 0.05
        @test rules_v1().cash_penalty_coef == CASH_CEILING_PENALTY_COEF
        @test rules_v1(cash_penalty=0.2).cash_penalty_coef == 0.2
        @test n_portfolio_scalars(r1) == N_PORTFOLIO_SCALARS && n_portfolio_scalars(r2) == N_PORTFOLIO_SCALARS_V2
    end

    @testset "A decision fills at the SAME bar's close in v2, the NEXT bar's in v1" begin
        j = 1   # "AAA"
        env2 = make_rules_env(rules_v2())
        h0   = env2.current_hour_idx
        res2 = step!(env2, buy_a(env2))
        ev2  = only(res2.info["trades"])
        @test ev2.price == Float64(env2.cache.hourly_closes[h0, j])
        @test ev2.t == string(env2.cache.hourly_datetimes[h0])
        @test only(env2.portfolio.holdings).entry_hour_idx == h0
        @test env2.current_hour_idx == h0 + 1                                  # clock moved after filling

        env1 = make_rules_env(rules_v1())
        res1 = step!(env1, buy_a(env1))
        @test only(res1.info["trades"]).price == Float64(env1.cache.hourly_closes[h0 + 1, j])
    end

    @testset "v2 never force-exits; v1 still does" begin
        env2 = make_rules_env(rules_v2(); n_days=40)
        step!(env2, buy_a(env2))
        kinds = String[]
        for _ in 1:(20 * 7)    # 20 trading days >> MAX_HOLD_DAYS_V2 and MAX_HOLD_DAYS
            res = step!(env2, RawAction[]); append!(kinds, [e.kind for e in res.info["trades"]])
            res.done && break
        end
        @test !("forced_exit" in kinds)
        @test length(env2.portfolio.holdings) == 1

        env1 = make_rules_env(rules_v1(); n_days=40)
        step!(env1, buy_a(env1))
        kinds1 = String[]
        for _ in 1:(20 * 7)
            res = step!(env1, RawAction[]); append!(kinds1, [e.kind for e in res.info["trades"]])
            res.done && break
        end
        @test "forced_exit" in kinds1
    end

    @testset "Hold penalty: zero before MAX_HOLD_DAYS_V2, coef x overdue share after, off by default" begin
        coef = 0.5
        a = make_rules_env(rules_v2(hold_penalty=0.0))
        b = make_rules_env(rules_v2(hold_penalty=coef))
        step!(a, buy_a(a)); step!(b, buy_a(b))
        seen_overdue = false
        for _ in 1:(15 * 7)
            ra = step!(a, RawAction[]); rb = step!(b, RawAction[])
            od = rb.info["overdue_fraction"]
            @test ra.info["overdue_fraction"] == od
            @test rb.reward ≈ ra.reward - coef * od atol=1e-12
            held_days = b.cache.date_index[b.current_date] - only(b.portfolio.holdings).entry_date_idx
            @test (od > 0) == (held_days >= MAX_HOLD_DAYS_V2)
            seen_overdue |= od > 0
            (ra.done || rb.done) && break
        end
        @test seen_overdue
    end

    @testset "Cash penalty flag: off by default, coef x excess over MAX_CASH_FRACTION when set" begin
        off = make_rules_env(rules_v2())
        on  = make_rules_env(rules_v2(cash_penalty=0.1))
        r_off = step!(off, RawAction[]); r_on = step!(on, RawAction[])     # all-cash episode
        @test r_off.info["cash_fraction"] ≈ 1.0
        @test r_off.reward == 0.0                                           # no window close, no penalty
        @test r_on.reward ≈ -0.1 * (1.0 - MAX_CASH_FRACTION)
    end

    @testset "Observation + policy: cash token features, shapes, gradients, checkpoint round-trip" begin
        env = make_rules_env(rules_v2())
        obs = assemble_observation(env)
        @test length(obs.portfolio) == N_PORTFOLIO_SCALARS_V2
        @test obs.portfolio[5] ≈ min(3.0, 1.0 / MAX_CASH_FRACTION)          # all cash: capped utilisation
        @test obs.portfolio[6] == 0f0                                        # just started: no days over yet
        for _ in 1:(3 * 7); step!(env, RawAction[]); end                     # stay all-cash 3 days
        @test assemble_observation(env).portfolio[6] ≈ 3 / MAX_HOLD_DAYS_V2 atol=0.2

        env1 = make_rules_env(rules_v1())
        @test length(assemble_observation(env1).portfolio) == N_PORTFOLIO_SCALARS
        @test_throws ErrorException TradingGame.assemble_observation!(                  # wrong-width vector fails loudly
            zeros(Float32, N_HOURLY_BARS_SHORT, N_PRICE_CHANNELS, 3), zeros(Float32, N_MACRO_DAYS, N_MACRO_SERIES),
            zeros(Float32, N_NEWS_FEATURES, 3), zeros(Float32, N_HOLDING_FEATURES, 3),
            zeros(Float32, N_PORTFOLIO_SCALARS), env)

        policy = ActorCriticPolicy(embed_dim=8, macro_embed_dim=4, attn_heads=2, critic_hidden=[8], cash_token=true)
        @test policy.cash_encoder !== nothing
        batch = stack_observations([obs, assemble_observation(env)])
        logits, bw, value = policy(batch.hourly, batch.news, batch.holding, batch.macro_ctx, batch.portfolio)
        N = length(obs.candidates)
        @test size(logits) == (3, N, 2) && size(bw) == (N, 2) && size(value) == (2,)
        grads = Flux.gradient(m -> sum(first(m(batch.hourly, batch.news, batch.holding, batch.macro_ctx, batch.portfolio))), policy)[1]
        @test any(!iszero, grads.cash_encoder.weight)                        # the cash token really feeds the output

        mktempdir() do dir
            path = joinpath(dir, "p.bson")
            save_policy(policy, path; embed_dim=8, macro_embed_dim=4, attn_heads=2, critic_hidden=[8])
            loaded, hp, _ = load_policy(path)
            @test hp.cash_token && loaded.cash_encoder !== nothing
            @test first(loaded(batch.hourly, batch.news, batch.holding, batch.macro_ctx, batch.portfolio)) ≈ logits
        end
    end

    @testset "Random and heuristic baselines keep every invariant under v2" begin
        for policy_fn in (env -> random_policy(env; rng=MersenneTwister(7)), heuristic_policy)
            env = make_rules_env(rules_v2(cash_penalty=0.02, hold_penalty=0.02); n_days=40)
            done, steps = false, 0
            while !done
                r = step!(env, policy_fn(env))
                done = r.done; steps += 1
                b = portfolio_breakdown(env)
                @test env.portfolio.cash >= -1e-6
                @test b.value ≈ b.stocks_value + b.cash_value atol=1e-6
                @test length(Set(h.sym_idx for h in env.portfolio.holdings)) <= n_max_holdings(length(env.candidate_sym_idx))
                @test !any(e -> e.kind == "forced_exit", r.info["trades"])
                @test steps <= 10_000
            end
            @test steps > 0
        end
    end

    @testset "A full v2 training iteration runs end to end" begin
        cache = make_test_cache(n_days=30)
        env = TradingGameEnv(cache)
        cfg = EpisodeConfig(initial_cash=100_000.0, start_date=cache.dates[1], end_date=cache.dates[end],
                             candidate_universe=cache.companies, rules=rules_v2(cash_penalty=0.05, hold_penalty=0.05))
        policy = ActorCriticPolicy(embed_dim=8, macro_embed_dim=4, attn_heads=2, critic_hidden=[8], cash_token=true)
        buffer = collect_rollout(env, policy, cfg; rng=MersenneTwister(1))
        @test buffer[end].done && all(s -> isfinite(s.reward), buffer)
        @test length(buffer[1].obs.portfolio) == N_PORTFOLIO_SCALARS_V2
        stats = ppo_update!(policy, Flux.setup(Flux.Adam(1f-3), policy), buffer; k_epochs=1, minibatch_size=16,
                             rng=MersenneTwister(2))
        @test isfinite(stats["loss"])
    end
end
