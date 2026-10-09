# ── Tests: 15-minute / BSE cache and cadence ──────────────────────────────────────
#
# Synthetic CSVs in the per-granularity layout (`{exchange}/daily`, `{exchange}/15min`),
# built into a real cache with `build_inference_cache` — offline, no Kite data.

function write_intraday_fixture(root::String; n_days::Int=12)
    mkpath(joinpath(root, "daily")); mkpath(joinpath(root, "15min"))
    days = [Date(2024, 3, 4) + Day(i - 1) for i in 1:n_days]
    for (k, sym) in enumerate(["AAA", "BBB"])
        open(joinpath(root, "daily", "$sym.csv"), "w") do io
            println(io, "date,open,high,low,close,volume")
            for (i, d) in enumerate(days)
                p = 100.0 * k + i
                println(io, "$d,$p,$(p + 1),$(p - 1),$p,1000")
            end
        end
        open(joinpath(root, "15min", "$sym.csv"), "w") do io
            println(io, "datetime,open,high,low,close,volume")
            n = 0
            for d in days, b in 0:24
                n += 1
                dt = DateTime(d) + Hour(9) + Minute(15) + Minute(15 * b)
                p = 100.0 * k + 0.01 * n
                println(io, "$(Dates.format(dt, "yyyy-mm-ddTHH:MM:SS")).0,$p,$p,$p,$p,10")
            end
            # Muhurat-style evening bar: must be dropped
            println(io, "$(days[1])T18:15:00.0,1.0,1.0,1.0,1.0,0")
        end
    end
    return days
end

@testset "15-minute BSE cache" begin
    root = mktempdir()
    days = write_intraday_fixture(root)
    path = joinpath(mktempdir(), "cache.bson")

    cache = build_inference_cache(root, ["AAA", "BBB"], path; granularity="15min", exchange="bse", history="resample")

    @testset "build records bar length, exchange and filters out-of-session bars" begin
        @test cache.bar_minutes == 15
        @test cache.exchange == "bse"
        @test length(cache.hourly_datetimes) == 25 * length(days)
        @test all(dt -> Time(9, 15) <= Time(dt) <= Time(15, 15), cache.hourly_datetimes)
        @test cache.hourly_datetimes[2] - cache.hourly_datetimes[1] == Minute(15)
    end

    @testset "hourly history axis is resampled from the 15-minute bars and never runs ahead" begin
        @test has_history(cache)
        @test length(cache.history_datetimes) == 7 * length(days)                 # slots 9:15 … 15:15
        @test cache.history_datetimes[1:7] == [DateTime(days[1]) + Hour(9) + Minute(15) + Hour(k) for k in 0:6]
        j = cache.sym_index["AAA"]
        # a slot's close is the close of its LAST 15-minute bar (fixture price rises 0.01 per bar)
        for slot in (1, 2, 7)
            last_clock = findfirst(==(cache.history_last_bar[slot]), cache.hourly_datetimes)
            @test cache.history_closes[slot, j] == cache.hourly_closes[last_clock, j]
        end
        # at 15-minute step s the newest usable history bar is already complete at s
        for i in eachindex(cache.hourly_datetimes)
            h = cache.history_end_idx[i]
            h == 0 && continue
            @test cache.history_last_bar[h] <= cache.hourly_datetimes[i]
            h < length(cache.history_datetimes) && @test cache.history_last_bar[h + 1] > cache.hourly_datetimes[i]
        end
        # 10:00 is when the 9:15 slot completes; one step earlier it is not yet usable
        i1000 = findfirst(==(DateTime(days[1]) + Hour(10)), cache.hourly_datetimes)
        @test cache.history_end_idx[i1000] == 1 && cache.history_end_idx[i1000 - 1] == 0
        # an hourly-only cache has no history axis
        @test !has_history(build_inference_cache(root, ["AAA"], joinpath(mktempdir(), "n.bson"); granularity="15min", exchange="bse"))
        @test_throws ErrorException build_inference_cache(root, ["AAA"], path; granularity="hourly", history="resample")
    end

    @testset "BSON round-trip keeps bar_minutes/exchange" begin
        loaded = load_inference_cache(path)
        @test loaded.bar_minutes == 15 && loaded.exchange == "bse"
        @test loaded.hourly_closes == cache.hourly_closes
        @test loaded.history_closes == cache.history_closes && loaded.history_end_idx == cache.history_end_idx
    end

    @testset "legacy 9-argument constructor defaults to hourly NSE" begin
        c = make_test_cache()
        @test c.bar_minutes == 60 && c.exchange == "nse"
        @test decision_granularity(c) == HOURLY
        @test decision_granularity(cache) == MINUTE_15
    end

    @testset "unknown granularity is rejected" begin
        @test_throws ErrorException build_inference_cache(root, ["AAA"], path; granularity="5min")
    end

    @testset "env decides at every 15-minute bar and fills at that bar's close" begin
        env = TradingGameEnv(cache)
        reset!(env, EpisodeConfig(initial_cash=100_000.0, start_date=days[1], end_date=days[end],
                                   candidate_universe=["AAA", "BBB"], rules=rules_v2()))
        @test is_decision_bar(env)
        n = 0
        t0 = env.cache.hourly_datetimes[env.current_hour_idx]
        done = false
        while !done
            n += 1
            @test is_decision_bar(env)
            done = step!(env, RawAction[RawAction(1, BUY, 1.0)]).done
        end
        @test n == 25 * length(days) - 1
        @test portfolio_value(env) > 0
        @test t0 == DateTime(days[1]) + Hour(9) + Minute(15)
    end

    @testset "game v3 = v2 rules, but only on a BSE 15-minute cache" begin
        r2, r3 = rules_v2(cash_penalty=0.1, hold_penalty=0.2), rules_v3(cash_penalty=0.1, hold_penalty=0.2)
        @test r3.version == 3
        @test all(getfield(r3, f) == getfield(r2, f) for f in
                  (:same_bar_execution, :forced_exit, :max_hold_days, :hold_penalty_coef, :cash_penalty_coef))
        @test r2.cash_token && !r3.cash_token                    # v3 has no cash token...
        @test r3.cap_features && r3.portfolio_in_fusion          # ...the cash state rides in the portfolio scalars into fusion
        @test r3.portfolio_to_critic && !rules_v2().portfolio_to_critic   # ...and the critic reads them directly too
        @test n_portfolio_scalars(r3) == N_PORTFOLIO_SCALARS_V2
        @test (r2.use_macro, r2.use_news) == (true, true)
        @test (r3.use_macro, r3.use_news) == (false, false)
        cfg(rules, cache) = EpisodeConfig(initial_cash=1e5, start_date=cache.dates[1], end_date=cache.dates[end],
                                           candidate_universe=cache.companies, rules=rules)
        reset!(TradingGameEnv(cache), cfg(r3, cache))                       # BSE / 15 min: accepted
        nse_hourly = make_test_cache()
        @test_throws ErrorException reset!(TradingGameEnv(nse_hourly), cfg(r3, nse_hourly))
        reset!(TradingGameEnv(nse_hourly), cfg(r2, nse_hourly))             # v2 has no such requirement
    end

    @testset "v3 policy has no macro or news inputs and ignores those tensors" begin
        small = (embed_dim=8, macro_embed_dim=4, attn_heads=2, critic_hidden=[8])
        p2 = ActorCriticPolicy(; small..., cash_token=true, seed=1)
        p3 = ActorCriticPolicy(; small..., cash_token=true, use_macro=false, use_news=false, seed=1)
        @test p3.macro_encoder === nothing
        @test size(p3.fusion.weight, 2) == 8 + N_HOLDING_FEATURES
        @test size(p3.portfolio_encoder.weight, 2) == N_PORTFOLIO_SCALARS_V2
        n2 = sum(length, Flux.trainables(p2)); n3 = sum(length, Flux.trainables(p3))
        @test n3 < n2

        N, B = 5, 2
        hourly = randn(Float32, N_HOURLY_BARS_SHORT, N_PRICE_CHANNELS, N, B)
        holding = rand(Float32, N_HOLDING_FEATURES, N, B)
        portfolio = rand(Float32, N_PORTFOLIO_SCALARS_V2, B)
        zeros_macro = zeros(Float32, N_MACRO_DAYS, N_MACRO_SERIES, B)
        a = p3(hourly, zeros(Float32, N_NEWS_FEATURES, N, B), holding, zeros_macro, portfolio)
        b = p3(hourly, randn(Float32, N_NEWS_FEATURES, N, B), holding, randn(Float32, N_MACRO_DAYS, N_MACRO_SERIES, B), portfolio)
        @test all(a .≈ b)   # news/macro tensors have no effect

        path = tempname() * ".bson"
        save_policy(p3, path; small...)
        loaded, hp, _ = load_policy(path)
        @test !hp.use_macro && !hp.use_news && loaded.macro_encoder === nothing
        @test all(loaded(hourly, zeros(Float32, N_NEWS_FEATURES, N, B), holding, zeros_macro, portfolio) .≈ a)
    end

    @testset "v3 price window: close only, two weeks of bars" begin
        r3 = rules_v3()
        @test r3.history_encoder == :direct && rules_v3(history_encoder=:gru).history_encoder == :gru
        @test_throws ErrorException rules_v3(history_encoder=:lstm)
        @test !r3.use_volatility && r3.obs_window_days == OBS_WINDOW_DAYS_V3 == 10
        @test n_price_channels(r3) == 1 && n_price_channels(rules_v2()) == 2
        @test r3.use_history && n_stock_features(r3) == 4 && n_stock_features(rules_v2()) == 3
        @test obs_window_bars(r3, cache) == 70                           # 10 trading days x 7 hourly history bars
        @test obs_window_bars(r3, make_test_cache()) == 70
        @test obs_window_bars(rules_v2(), cache) == N_HOURLY_BARS_SHORT   # v1/v2 keep the fixed 120 bars
    end

    @testset "discount factors are rescaled to the same wall-clock horizon" begin
        @test bar_scaled(GAMMA, 60) == GAMMA
        @test bar_scaled(GAMMA, 15)^4 ≈ GAMMA
        @test bar_scaled(GAE_LAMBDA, 15)^4 ≈ GAE_LAMBDA
    end

    @testset "min_market_cap_cr filters the eligible pool" begin
        cpath = tempname() * ".json"
        write_companies_fixture(cpath, [
            (symbol="AAA", name="Alpha", market_cap_cr=900, confidence=(score=60, pass=true)),
            (symbol="BBB", name="Beta",  market_cap_cr=50,  confidence=(score=60, pass=true)),
        ])
        @test [e.symbol for e in eligible_candidates(cache, cpath)] == ["AAA", "BBB"]
        @test [e.symbol for e in eligible_candidates(cache, cpath; min_market_cap_cr=100.0)] == ["AAA"]
    end
end

@testset "Impossible moves are masked; other illegal moves are charged at once" begin
    syms  = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]   # rule 13 allows 5 holdings of 10
    mkenv(rules) = begin
        cache = make_test_cache(symbols=syms)
        env = TradingGameEnv(cache)
        reset!(env, EpisodeConfig(initial_cash=100_000.0, start_date=cache.dates[1], end_date=cache.dates[end],
                                   candidate_universe=syms, rules=rules))
        env
    end
    one(kind, i=1, w=1.0) = RawAction[RawAction(i, kind, w)]
    # v3's rules minus its BSE/15-minute data requirement, so the synthetic NSE fixture can run them
    v3_like() = GameRules(version=3, same_bar_execution=true, forced_exit=false, max_hold_days=MAX_HOLD_DAYS_V2,
                          cash_token=true, use_macro=false, use_news=false, premask=true,
                          illegal_penalty_coef=ILLEGAL_PENALTY_COEF_V3)

    @testset "pre-sampling mask: SELL is removed where nothing is sellable" begin
        env = mkenv(v3_like())
        date_idx = env.cache.date_index[env.current_date]
        @test !any(sellable_mask(env, date_idx))
        logits = zeros(Float32, 3, 10, 1)
        m = mask_action_logits(logits, reshape(sellable_mask(env, date_idx), 10, 1))
        p = Flux.softmax(m; dims=1)
        @test all(p[2, :, 1] .== 0)
        @test all(p[1, :, 1] .≈ 0.5) && all(p[3, :, 1] .≈ 0.5)          # conditional: only HOLD and BUY remain
        sell_ok = trues(10, 1)
        @test mask_action_logits(logits, sell_ok) == logits              # nothing masked when sells are possible
        @test rules_v3().premask && !rules_v2().premask && !rules_v1().premask
    end

    @testset "a sell with nothing sellable (non-masked caller) is charged the coefficient" begin
        e0 = mkenv(rules_v2()); e1 = mkenv(rules_v2(illegal_penalty=0.01))
        r0 = step!(e0, one(SELL)); r1 = step!(e1, one(SELL))
        @test r1.info["illegal_penalty"] ≈ 0.01 && r0.info["illegal_penalty"] == 0.0
        @test r0.reward - r1.reward ≈ 0.01
        @test portfolio_value(e0) == portfolio_value(e1) == 100_000.0     # the charge is reward-only
    end

    @testset "an affordable, in-cap buy is not charged" begin
        env = mkenv(rules_v2(illegal_penalty=0.01))
        r = step!(env, RawAction[RawAction(i, BUY, 1.0) for i in 1:5])   # 20% of cash each, 5 slots, under the 30% cap
        @test r.info["illegal_penalty"] == 0.0
        @test length(r.info["trades"]) == 5
    end

    @testset "a buy over the position cap is charged once, and the in-cap part fills" begin
        env = mkenv(rules_v2(illegal_penalty=0.01))
        r = step!(env, one(BUY))
        @test r.info["illegal_penalty"] ≈ 0.01
        @test length(r.info["trades"]) == 1
    end

    @testset "a buy with no cash is charged, then rejected" begin
        env = mkenv(rules_v2(illegal_penalty=0.01))
        env.portfolio.cash = 0.0
        r = step!(env, RawAction[RawAction(1, BUY, 1.0), RawAction(2, BUY, 1.0)])
        @test r.info["illegal_penalty"] ≈ 0.02
        @test isempty(r.info["trades"])
    end

    @testset "a buy the cash cannot fund even one share of is charged, then rejected" begin
        env = mkenv(rules_v2(illegal_penalty=0.01))
        env.portfolio.cash = 50.0                      # every stock in the fixture costs >= 100
        r = step!(env, one(BUY))
        @test r.info["illegal_penalty"] ≈ 0.01
        @test isempty(r.info["trades"])
    end

    @testset "buys beyond the holdings cap: each candidate without a slot is charged" begin
        env = mkenv(rules_v2(illegal_penalty=0.01))
        r = step!(env, RawAction[RawAction(i, BUY, 1.0) for i in 1:8])    # 5 slots, 8 candidates
        @test r.info["illegal_penalty"] ≈ 0.03
    end

    @testset "defaults: v1/v2 charge nothing, v3 charges 1%" begin
        @test rules_v1().illegal_penalty_coef == 0.0
        @test rules_v2().illegal_penalty_coef == 0.0
        @test rules_v3().illegal_penalty_coef == ILLEGAL_PENALTY_COEF_V3 == 0.01
    end

    @testset "PPO round-trip with the mask stored per step" begin
        cache = make_test_cache(symbols=syms)
        env = TradingGameEnv(cache)
        cfg = EpisodeConfig(initial_cash=100_000.0, start_date=cache.dates[1], end_date=cache.dates[end],
                             candidate_universe=syms, rules=v3_like())
        policy = ActorCriticPolicy(embed_dim=8, macro_embed_dim=4, attn_heads=2, critic_hidden=[8], cash_token=true,
                                    use_macro=false, use_news=false)
        buf = collect_rollout(env, policy, cfg; rng=MersenneTwister(3))
        @test !all(all, (s.sell_ok for s in buf))                         # some steps had nothing sellable
        sold = [(s.action_idx[c] == 2 && !s.sell_ok[c]) for s in buf for c in 1:10]
        @test !any(sold)                                                  # a masked SELL is never sampled
        opt = Flux.setup(Flux.Adam(1e-3), policy)
        stats = ppo_update!(policy, opt, buf; k_epochs=1, minibatch_size=32)
        @test isfinite(stats["loss"])
    end
end

@testset "v3 observation: 14-day hourly history + instantaneous 15-minute snapshot" begin
    root = mktempdir(); days = write_intraday_fixture(root; n_days=30)
    cache = build_inference_cache(root, ["AAA", "BBB"], joinpath(mktempdir(), "c.bson");
                                  granularity="15min", exchange="bse", history="resample")
    rules = rules_v3()
    start = days[14]
    cfg = EpisodeConfig(initial_cash=1e5, start_date=start, end_date=days[end], candidate_universe=["AAA", "BBB"], rules=rules)
    env = TradingGameEnv(cache); reset!(env, cfg)

    obs = assemble_observation(env)
    @test size(obs.hourly) == (70, 1, 2)                       # 70 hourly history bars, close only
    @test size(obs.holding) == (4, 2)                          # held, P&L, hold left, snapshot
    @test all(isfinite, obs.hourly) && all(obs.hourly .> 0)

    # the window is 70 log-returns between the newest 71 COMPLETED hourly closes
    t = env.current_hour_idx; h = cache.history_end_idx[t]; j = cache.sym_index["AAA"]
    raw = cache.history_closes[h-70:h, j]
    @test obs.hourly[:, 1, 1] ≈ Float32.(LOG_RETURN_SCALE .* log.(raw[2:end] ./ raw[1:end-1]))
    @test obs.hourly[1, 1, 1] ≈ LOG_RETURN_SCALE * log(raw[2] / raw[1])
    # the snapshot is the next return in the series: last completed hourly close -> this bar's 15-minute close
    @test obs.holding[4, 1] ≈ LOG_RETURN_SCALE * log(cache.hourly_closes[t, j] / raw[end])
    # no look-ahead: the newest history bar finished no later than the current 15-minute bar started
    @test cache.history_last_bar[h] <= cache.hourly_datetimes[t]

    # stepping the 15-minute clock changes the snapshot every bar, the history only when an hour completes
    snaps = Float32[]; hists = Vector{Float32}[]
    for _ in 1:8
        o = assemble_observation(env); push!(snaps, o.holding[4, 1]); push!(hists, copy(o.hourly[:, 1, 1]))
        step!(env, RawAction[])
    end
    @test length(unique(hists)) < 8
    @test length(unique(snaps)) > 1

    small = (embed_dim=8, macro_embed_dim=4, attn_heads=2, critic_hidden=[8])
    for (name, hb) in (("direct", 70), ("gru", 0))
        policy = ActorCriticPolicy(; small..., use_macro=false, use_news=false, portfolio_in_fusion=true,
                                   portfolio_scalars=6, portfolio_to_critic=true,
                                   price_channels=1, stock_features=4, history_bars=hb)
        @test policy.portfolio_encoder === nothing && policy.cash_encoder === nothing && policy.global_in_fusion == 6
        @test (policy.hourly_encoder === nothing) == (hb > 0)
        @test policy_stock_inputs(policy) == (false, 4)
        reset!(env, cfg)
        buf = collect_rollout(env, policy, cfg; rng=MersenneTwister(2))
        @test size(buf[1].obs.hourly) == (70, 1, 2) && size(buf[1].obs.holding) == (4, 2)
        stats = ppo_update!(policy, Flux.setup(Flux.Adam(1e-3), policy), buf; k_epochs=1, minibatch_size=32)
        @test isfinite(stats["loss"])

        path = tempname() * ".bson"
        save_policy(policy, path; small...)
        loaded, hp, _ = load_policy(path)
        @test hp.price_channels == 1 && hp.stock_features == 4 && hp.history_bars == hb && hp.global_in_fusion == 6 && hp.critic_global == 6
        @test loaded.portfolio_encoder === nothing && loaded.global_in_fusion == 6
        @test (loaded.hourly_encoder === nothing) == (hb > 0)
    end

    # the direct policy is one fusion layer: Dense(70 + 4 -> embed), no GRU
    direct = ActorCriticPolicy(; small..., use_macro=false, use_news=false, portfolio_in_fusion=true,
                               portfolio_scalars=6, price_channels=1, stock_features=4, history_bars=70)
    @test size(direct.fusion.weight) == (8, 80)                   # 70 returns + 4 state + 6 portfolio scalars
    @test_throws ErrorException direct(zeros(Float32, 60, 1, 2, 1), zeros(Float32, N_NEWS_FEATURES, 2, 1),
        zeros(Float32, 4, 2, 1), zeros(Float32, N_MACRO_DAYS, N_MACRO_SERIES, 1), zeros(Float32, N_PORTFOLIO_SCALARS_V2, 1))

    # the critic also reads the portfolio scalars directly: its first layer is (embed + 6) wide, and with the
    # portfolio zeroed out of the actor's route (same fusion weights) the value still reacts to it
    with_c = ActorCriticPolicy(; small..., use_macro=false, use_news=false, portfolio_in_fusion=true,
                               portfolio_scalars=6, portfolio_to_critic=true, price_channels=1, stock_features=4, history_bars=70)
    @test size(with_c.critic_head.layers[1].weight, 2) == 8 + 6
    @test size(direct.critic_head.layers[1].weight, 2) == 8
    @test sum(length, Flux.trainables(with_c)) == sum(length, Flux.trainables(direct)) + 6 * 8
    xx = randn(Float32, 70, 1, 3, 1); ss = rand(Float32, 4, 3, 1); mm = zeros(Float32, N_MACRO_DAYS, N_MACRO_SERIES, 1)
    nn = zeros(Float32, N_NEWS_FEATURES, 3, 1)
    _, _, va = with_c(xx, nn, ss, mm, fill(0.2f0, 6, 1)); _, _, vb = with_c(xx, nn, ss, mm, fill(0.9f0, 6, 1))
    @test !(va ≈ vb)
    g = Flux.gradient(m -> sum(m(xx, nn, ss, mm, fill(0.5f0, 6, 1))[3]), with_c)[1]
    @test any(!iszero, g.critic_head.layers[1].weight[:, 9:14])           # gradient reaches the direct-portfolio weights

    # no portfolio/cash token: attention sees N tokens, and the book state reaches every stock through fusion
    N, B = 5, 2
    x = randn(Float32, 70, 1, N, B); st = rand(Float32, 4, N, B); mc = zeros(Float32, N_MACRO_DAYS, N_MACRO_SERIES, B)
    nw = zeros(Float32, N_NEWS_FEATURES, N, B)
    pf1 = rand(Float32, 6, B); pf2 = pf1 .+ 0.5f0
    l1, w1, v1 = direct(x, nw, st, mc, pf1); l2, w2, v2 = direct(x, nw, st, mc, pf2)
    @test size(l1) == (3, N, B) && size(w1) == (N, B) && size(v1) == (B,)
    @test !(l1 ≈ l2) && !(v1 ≈ v2)                                  # portfolio state changes both the actor and the critic
    perm = [3, 1, 5, 2, 4]                                          # stocks are interchangeable tokens: permuting them permutes the outputs
    lp, _, vp = direct(x[:, :, perm, :], nw, st[:, perm, :], mc, pf1)
    @test isapprox(lp, l1[:, perm, :]; atol=1e-5)
    @test isapprox(vp, v1; atol=1e-5)
    @test_throws ErrorException ActorCriticPolicy(; small..., portfolio_in_fusion=true, cash_token=true)
    @test_throws ErrorException ActorCriticPolicy(; small..., portfolio_in_fusion=true, use_macro=true)

    # v3 refuses a cache without a history axis, and a wrongly shaped buffer
    nohist = build_inference_cache(root, ["AAA", "BBB"], joinpath(mktempdir(), "n.bson"); granularity="15min", exchange="bse")
    @test_throws ErrorException reset!(TradingGameEnv(nohist), cfg)
    @test_throws ErrorException TradingGame.assemble_observation!(zeros(Float32, 70, 2, 2), zeros(Float32, N_MACRO_DAYS, N_MACRO_SERIES),
        zeros(Float32, N_NEWS_FEATURES, 2), zeros(Float32, 4, 2), zeros(Float32, N_PORTFOLIO_SCALARS_V2), env)
end

@testset "Per-step validation log: every decision bar, trades or not" begin
    syms = ["A", "B", "C", "D", "E", "F"]
    cache = make_test_cache(symbols=syms, n_days=12)
    rules = GameRules(version=3, same_bar_execution=true, forced_exit=false, max_hold_days=MAX_HOLD_DAYS_V2,
                      use_macro=false, use_news=false, premask=true, cap_features=true, portfolio_in_fusion=true,
                      portfolio_to_critic=true, illegal_penalty_coef=0.01)
    cfg = EpisodeConfig(initial_cash=1e5, start_date=cache.dates[1], end_date=cache.dates[end], candidate_universe=syms, rules=rules)
    env = TradingGameEnv(cache)
    policy = ActorCriticPolicy(embed_dim=8, macro_embed_dim=4, attn_heads=2, critic_hidden=[8], use_macro=false, use_news=false,
                               portfolio_in_fusion=true, portfolio_scalars=6, portfolio_to_critic=true, seed=11)

    ref = Ref{Any}(nothing)
    buf = collect_rollout(env, policy, cfg; greedy=true, rng=MersenneTwister(1), step_log=ref)
    log = ref[]
    T = length(buf)
    @test log.filled == T
    @test size(log.probs) == (3, 6, T) && size(log.price) == (6, T) && length(log.value) == T
    @test all(!isempty, log.datetimes) && issorted(log.datetimes)
    @test log.symbols == syms
    # probabilities are a proper distribution on every bar for every stock, masked sells are exactly zero
    @test all(abs.(dropdims(sum(Float32.(log.probs); dims=1); dims=1) .- 1) .< 5e-3)
    sell_p = log.probs[2, :, :]
    @test all(sell_p[.!log.sell_ok] .== 0)
    @test all(log.action[.!log.sell_ok] .!= 2)                         # a masked sell is never chosen
    @test all(log.p_sell_raw[.!log.sell_ok] .>= 0) && any(log.p_sell_raw[.!log.sell_ok] .> 0)   # what it wanted before masking
    # the step log agrees with the rollout buffer, bar by bar
    @test log.value ≈ Float32[s.value for s in buf]
    @test log.reward ≈ Float32[s.reward for s in buf]
    @test Int.(log.action) == reduce(hcat, s.action_idx for s in buf)
    @test log.portfolio_value[end] ≈ portfolio_value(env)
    @test sum(log.n_trades) > 0 && any(log.n_trades .== 0)              # steps with no trade are logged too
    @test log.n_holdings[end] == length(unique(h.sym_idx for h in env.portfolio.holdings))
    @test sum(log.illegal_penalty) > 0                                   # random-ish policy makes illegal moves

    # round trip through the BSON file
    path = joinpath(mktempdir(), "iter_00001.bson")
    save_step_log(path, log; meta=Dict("iteration" => 1))
    s = load_val_steps(path)
    @test s.meta["iteration"] == 1 && s.symbols == syms && length(s.datetimes) == T
    @test s.probs == log.probs && s.action == log.action && s.value == log.value
    @test s.probs isa Array{Float16, 3}

    # the website's per-bar summary: one entry per bar, trade bars and no-trade bars alike
    sm = step_summary(log)
    @test length(sm.t) == T && length(sm.nt) == T && length(sm.ph) == T
    @test all(abs.(sm.ph .+ sm.ps .+ sm.pb .- 1) .< 5e-3)
    @test sm.nt == Int.(log.n_trades)
    @test all(sm.mb .>= sm.pb .- 1e-3) && all(1 .<= sm.mbs .<= 6)
    @test sm.n_pick_buy == vec(sum(log.action .== 3; dims=1))
    js = joinpath(mktempdir(), "iter_00001.json"); save_step_summary(js, log; meta=Dict("iteration" => 1))
    parsed = JSON3.read(read(js, String))
    @test length(parsed.t) == T && parsed.meta.iteration == 1 && collect(parsed.symbols) == syms

    # train_policy! writes one file per logged validation run (every `val_steps_every`-th)
    dir = mktempdir()
    train_policy!(policy, env, cfg; val_config=cfg, iterations=3, k_epochs=1, minibatch_size=64, eval_every=1,
                  val_steps_dir=dir, val_steps_every=2, embed_dim=8, macro_embed_dim=4, attn_heads=2, critic_hidden=[8])
    @test sort(readdir(dir)) == ["iter_00001.bson", "iter_00001.json", "iter_00003.bson", "iter_00003.json"]
    @test load_val_steps(joinpath(dir, "iter_00003.bson")).meta["iteration"] == 3
    dir0 = mktempdir()
    train_policy!(policy, env, cfg; val_config=cfg, iterations=1, k_epochs=1, minibatch_size=64, eval_every=1,
                  val_steps_dir=dir0, val_steps_every=0, embed_dim=8, macro_embed_dim=4, attn_heads=2, critic_hidden=[8])
    @test isempty(readdir(dir0))
end

@testset "Terminal reward: window gain minus the window's penalties, paid at the window's end" begin
    syms = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    cache = make_test_cache(symbols=syms, n_days=10)
    mk(; window=0, terminal=true, ill=0.01, cash=0.0, hold=0.0) = GameRules(version=3, same_bar_execution=true, forced_exit=false,
        max_hold_days=MAX_HOLD_DAYS_V2, use_macro=false, use_news=false, premask=true, cap_features=true,
        portfolio_in_fusion=true, portfolio_to_critic=true, illegal_penalty_coef=ill, cash_penalty_coef=cash,
        hold_penalty_coef=hold, terminal_reward=terminal, reward_window_days=window)
    cfg(r) = EpisodeConfig(initial_cash=100_000.0, start_date=cache.dates[1], end_date=cache.dates[end], candidate_universe=syms, rules=r)
    v0 = 100_000.0
    # per step: reward, value at the decision, the illegal-move fraction, value after the step, trading-day index after the step
    function play(rules; actions)
        env = TradingGameEnv(cache); reset!(env, cfg(rules))
        out = NamedTuple[]; done = false; t = 0
        while !done
            t += 1
            vdec = portfolio_value(env)
            res = step!(env, actions(t))
            push!(out, (reward=res.reward, vdec=vdec, frac=res.info["illegal_penalty"], value=res.info["portfolio_value"],
                        day=env.cache.date_index[env.current_date]))
            done = res.done
        end
        return env, out
    end
    # one legal buy-in, then repeated illegal "buy everything into stock A" (over the 30% cap) every 5th bar
    acts(t) = t == 1 ? RawAction[RawAction(i, BUY, 1.0) for i in 1:5] : (t % 5 == 0 ? RawAction[RawAction(1, BUY, 1.0)] : RawAction[])

    @testset "one window = the whole episode (window 0)" begin
        env, o = play(mk(); actions=acts)
        r = [x.reward for x in o]
        @test all(r[1:end-1] .== 0.0) && r[end] != 0.0
        pen = sum(x.vdec * x.frac for x in o)
        @test pen > 0
        @test r[end] ≈ (o[end].value - v0) / v0 - pen / v0
        @test env.penalty_accum == 0.0                                  # the ledger is cleared when it is paid
        env0, o0 = play(mk(ill=0.0); actions=acts)
        @test o0[end].reward ≈ (o0[end].value - v0) / v0 && all(x.reward == 0.0 for x in o0[1:end-1])
    end

    @testset "windows of N trading days: each normalised by the value at its own start" begin
        N = 3
        env, o = play(mk(window=N); actions=acts)
        r = [x.reward for x in o]
        # replay the window logic from the recorded per-step data
        expect = zeros(length(o)); vstart = v0; dstart = cache.date_index[cache.dates[1]]; ledger = 0.0
        for (k, x) in enumerate(o)
            ledger += x.vdec * x.frac
            if k == length(o) || x.day - dstart >= N
                expect[k] = (x.value - vstart) / vstart - ledger / vstart
                vstart = x.value; dstart = x.day; ledger = 0.0
            end
        end
        @test r ≈ expect
        @test count(!=(0.0), r) >= 3                                     # several payouts in a 10-day episode
        @test r[end] != 0.0
        @test env.penalty_accum == 0.0
        # the windows tile the episode: compounded window gains (before penalties) give the total gain
        _, o2 = play(mk(window=N, ill=0.0); actions=acts)
        gains = Float64[]; vs = v0
        for (k, x) in enumerate(o2)
            x.reward != 0.0 && (push!(gains, x.reward); vs = x.value)
        end
        @test prod(1 .+ gains) ≈ o2[end].value / v0
    end

    @testset "cash/hold penalties join the window's ledger" begin
        envc, oc = play(mk(ill=0.0, cash=0.5); actions=t -> RawAction[])
        exp_pen = sum(x.value * 0.5 * max(0.0, 1.0 - MAX_CASH_FRACTION) for x in oc)   # all cash, every bar
        @test oc[end].reward ≈ (oc[end].value - v0) / v0 - exp_pen / v0 rtol=1e-6
    end

    @testset "stepwise rules, resets and the PPO discount" begin
        _, os = play(mk(terminal=false); actions=acts)
        @test any(x.reward < 0 for x in os)
        env2 = TradingGameEnv(cache); reset!(env2, cfg(mk())); step!(env2, acts(5)); @test env2.penalty_accum > 0
        reset!(env2, cfg(mk())); @test env2.penalty_accum == 0.0 && env2.reward_window_start_value == v0
        @test discount_factors(mk(), cache) == (1.0, 1.0)                                   # one reward at the very end
        @test discount_factors(mk(window=7), cache) == (bar_scaled(GAMMA, 60), 1.0)          # a reward every week
        @test discount_factors(mk(terminal=false), cache) == (bar_scaled(GAMMA, 60), bar_scaled(GAE_LAMBDA, 60))
        @test rules_v3().terminal_reward && rules_v3().reward_window_days == 0 && rules_v3(reward_window_days=5).reward_window_days == 5
        @test !rules_v3(terminal_reward=false).terminal_reward && !rules_v2().terminal_reward
    end
end
