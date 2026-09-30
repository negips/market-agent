# ── Synthetic fixture ─────────────────────────────────────────────────────────────
#
# A small, deterministic InferenceCache — built directly (no CSVs/BSON, no
# sidecar) so these tests run instantly and never depend on real market data.

"""Build a synthetic `InferenceCache`: `n_days` trading days, `bars_per_day`
hourly bars each, one symbol per entry of `symbols` with its own price drift
(so momentum-based policies have a real signal to act on)."""
function make_test_cache(; n_days::Int=30, bars_per_day::Int=7,
                          symbols::Vector{String}=["AAA", "BBB", "CCC"])
    dates = [Date(2024, 1, 1) + Day(i - 1) for i in 1:n_days]
    n_comp = length(symbols)

    hourly_datetimes = DateTime[]
    for d in dates, h in 0:bars_per_day-1
        push!(hourly_datetimes, DateTime(d) + Hour(9) + Minute(15) + Hour(h))
    end
    n_hourly = length(hourly_datetimes)

    hourly_closes = zeros(Float32, n_hourly, n_comp)
    for (j, _) in enumerate(symbols)
        base  = 100f0 * j
        drift = 0.002f0 * j - 0.003f0   # each symbol trends differently
        for i in 1:n_hourly
            hourly_closes[i, j] = base * (1 + drift)^(i - 1)
        end
    end

    closes = zeros(Float32, n_days, n_comp)
    for i in 1:n_days, j in 1:n_comp
        closes[i, j] = hourly_closes[(i - 1) * bars_per_day + 1, j]
    end

    vols     = zeros(Float32, n_days, n_comp)
    rel_vols = ones(Float32, n_days, n_comp)

    date_index = Dict(d => i for (i, d) in enumerate(dates))
    sym_index  = Dict(s => i for (i, s) in enumerate(symbols))

    return InferenceCache(closes, vols, rel_vols, hourly_closes, dates,
                           hourly_datetimes, symbols, date_index, sym_index)
end

function make_test_env(; n_days::Int=30, initial_cash::Float64=1_000_000.0)
    cache = make_test_cache(n_days=n_days)
    env   = TradingGameEnv(cache)
    reset!(env, EpisodeConfig(
        initial_cash       = initial_cash,
        start_date         = cache.dates[1],
        end_date           = cache.dates[end],
        candidate_universe = cache.companies,
    ))
    return env
end

# ── Tests ─────────────────────────────────────────────────────────────────────────

@testset "TradingGame rule compliance" begin

    @testset "Starting portfolio value equals initial cash exactly" begin
        env = make_test_env(initial_cash=54_321.0)
        @test portfolio_value(env) == 54_321.0
    end

    @testset "All hourly bars fall within NSE market hours (rule 8 precondition)" begin
        cache = make_test_cache()
        for dt in cache.hourly_datetimes
            t = Time(dt)
            @test t >= Time(9, 15) && t <= Time(15, 30)
        end
    end

    @testset "Buy debits exactly floor(notional/price)*price*(1+FEE_RATE)" begin
        env = make_test_env()
        sym_idx  = first(env.candidate_sym_idx)
        price    = env.cache.hourly_closes[env.current_hour_idx, sym_idx]
        notional = 1000.0   # chosen to divide evenly (price == 100.0) — the flooring test below covers the remainder case
        date_idx = env.cache.date_index[env.current_date]
        cash_before = env.portfolio.cash

        TradingGame._apply_actions!(env, [ResolvedTrade(sym_idx, BUY, notional)], date_idx)

        expected_qty      = floor(notional / price)
        expected_notional = expected_qty * price
        expected_fee      = FEE_RATE * expected_notional
        expected_debit    = expected_notional + expected_fee
        @test env.portfolio.cash ≈ cash_before - expected_debit
        @test length(env.portfolio.holdings) == 1
        h = env.portfolio.holdings[1]
        @test h.quantity  == expected_qty
        @test h.entry_fee ≈ expected_fee
    end

    @testset "Buy quantity is always a whole number of shares" begin
        env = make_test_env()
        sym_idx  = first(env.candidate_sym_idx)
        price    = env.cache.hourly_closes[env.current_hour_idx, sym_idx]
        # A notional that does NOT divide evenly by price — half a share's worth left over.
        notional = 10.5 * price
        date_idx = env.cache.date_index[env.current_date]
        cash_before = env.portfolio.cash

        TradingGame._apply_actions!(env, [ResolvedTrade(sym_idx, BUY, notional)], date_idx)

        h = env.portfolio.holdings[1]
        @test isinteger(h.quantity)
        @test h.quantity == 10.0
        # The unspent half-share's worth of cash stays as cash, not lost or rounded away.
        spent = cash_before - env.portfolio.cash
        @test spent < notional * (1 + FEE_RATE)
        @test spent ≈ (10.0 * price) * (1 + FEE_RATE)
    end

    @testset "A buy request too small to afford one share is a no-op" begin
        env = make_test_env()
        sym_idx  = first(env.candidate_sym_idx)
        price    = env.cache.hourly_closes[env.current_hour_idx, sym_idx]
        date_idx = env.cache.date_index[env.current_date]
        cash_before = env.portfolio.cash

        TradingGame._apply_actions!(env, [ResolvedTrade(sym_idx, BUY, price * 0.5)], date_idx)

        @test env.portfolio.cash == cash_before
        @test isempty(env.portfolio.holdings)
    end

    @testset "Sell credits proceeds-minus-fee to reserved cash, not cash, immediately" begin
        env = make_test_env()
        sym_idx  = first(env.candidate_sym_idx)
        date_idx = env.cache.date_index[env.current_date]
        TradingGame._apply_actions!(env, [ResolvedTrade(sym_idx, BUY, 1000.0)], date_idx)
        cash_after_buy = env.portfolio.cash
        h = env.portfolio.holdings[1]

        # cross the 1-day lock-up so this is a genuine voluntary sell
        entry_date_idx = h.entry_date_idx
        while env.cache.date_index[env.current_date] == entry_date_idx
            step!(env, RawAction[])
        end
        date_idx2     = env.cache.date_index[env.current_date]
        price_at_sell = env.cache.hourly_closes[env.current_hour_idx, sym_idx]
        expected_proceeds = h.quantity * price_at_sell
        expected_fee      = FEE_RATE * expected_proceeds

        TradingGame._apply_actions!(env, [ResolvedTrade(sym_idx, SELL, 0.0)], date_idx2)

        @test env.portfolio.cash ≈ cash_after_buy   # sell proceeds never touch cash directly
        @test isempty(env.portfolio.holdings)
        @test length(env.portfolio.reserved) == 1
        @test env.portfolio.reserved[1].amount ≈ expected_proceeds - expected_fee
    end

    @testset "Insufficient-cash buy: rejected defensively, and never produced by resolve_actions" begin
        env = make_test_env(initial_cash=1000.0)
        date_idx = env.cache.date_index[env.current_date]

        bogus = ResolvedTrade(first(env.candidate_sym_idx), BUY, 1_000_000.0)
        @test_throws CashConstraintViolation TradingGame._apply_actions!(env, [bogus], date_idx)

        raw = [RawAction(idx, BUY, 1.0) for idx in env.candidate_sym_idx]
        resolved = resolve_actions(env, raw, date_idx)
        total_debit = isempty(resolved) ? 0.0 :
            sum(t.notional * (1 + FEE_RATE) for t in resolved if t.kind == BUY; init=0.0)
        @test total_debit <= env.portfolio.cash + 1e-6
    end

    @testset "Reserved cash settles at exactly +SETTLEMENT_DAYS trading days" begin
        env = make_test_env()
        date_idx0   = env.cache.date_index[env.current_date]
        cash_before = env.portfolio.cash
        push!(env.portfolio.reserved, ReservedCashLot(500.0, date_idx0 + SETTLEMENT_DAYS, "TEST"))

        settled_at = nothing
        for _ in 1:(3 * 7)
            step!(env, RawAction[])
            if settled_at === nothing && env.portfolio.cash > cash_before + 1e-6
                settled_at = env.cache.date_index[env.current_date]
            end
        end
        @test settled_at == date_idx0 + SETTLEMENT_DAYS
        @test env.portfolio.cash ≈ cash_before + 500.0
    end

    @testset "1-day lock-up blocks a voluntary sell until MIN_HOLD_DAYS (rule 10)" begin
        env = make_test_env()
        sym_idx = first(env.candidate_sym_idx)

        step!(env, [RawAction(sym_idx, BUY, 1.0)])
        @test length(env.portfolio.holdings) == 1
        entry_date_idx = env.portfolio.holdings[1].entry_date_idx

        # same day, later bar: sell must be masked — lot survives
        step!(env, [RawAction(sym_idx, SELL, 0.0)])
        @test length(env.portfolio.holdings) == 1
        @test env.portfolio.holdings[1].entry_date_idx == entry_date_idx

        # cross into the next trading day, then sell must succeed
        while env.cache.date_index[env.current_date] == entry_date_idx
            step!(env, RawAction[])
        end
        step!(env, [RawAction(sym_idx, SELL, 0.0)])
        @test isempty(env.portfolio.holdings)
    end

    @testset "Forced exit fires at exactly entry+MAX_HOLD_DAYS, never before (rule 9)" begin
        env = make_test_env(n_days=30)
        sym_idx = first(env.candidate_sym_idx)
        step!(env, [RawAction(sym_idx, BUY, 1.0)])
        entry_date_idx = env.portfolio.holdings[1].entry_date_idx

        forced_exit_date_idx = nothing
        for _ in 1:(15 * 7)
            step!(env, RawAction[])   # never voluntarily sell
            if isempty(env.portfolio.holdings) && forced_exit_date_idx === nothing
                forced_exit_date_idx = env.cache.date_index[env.current_date]
            end
            forced_exit_date_idx !== nothing && break
        end
        @test forced_exit_date_idx == entry_date_idx + MAX_HOLD_DAYS
    end

    @testset "Position size capped at MAX_POSITION_FRACTION of portfolio value (rule 12)" begin
        env = make_test_env(initial_cash=100_000.0)
        sym_idx  = first(env.candidate_sym_idx)
        price    = env.cache.hourly_closes[env.current_hour_idx, sym_idx]
        date_idx = env.cache.date_index[env.current_date]

        # weight=1.0 requests ~100% of available cash on a single symbol — the cap must bind
        resolved = resolve_actions(env, [RawAction(sym_idx, BUY, 1.0)], date_idx)
        @test length(resolved) == 1
        total_value = portfolio_value(env)
        @test resolved[1].notional <= MAX_POSITION_FRACTION * total_value + 1e-6
        @test resolved[1].notional > 0.10 * total_value   # confirms it actually bound, not coincidentally under

        TradingGame._apply_actions!(env, resolved, date_idx)
        h = env.portfolio.holdings[1]
        @test h.quantity * price <= MAX_POSITION_FRACTION * total_value + price   # within one share's rounding

        # A further buy into the SAME symbol is capped down to ~0 (already at the ceiling)...
        resolved2 = resolve_actions(env, [RawAction(sym_idx, BUY, 1.0)], date_idx)
        @test isempty(resolved2)

        # ...but a different symbol still has its own 15% headroom.
        other_idx = first(setdiff(env.candidate_sym_idx, Set([sym_idx])))
        resolved3 = resolve_actions(env, [RawAction(other_idx, BUY, 1.0)], date_idx)
        @test length(resolved3) == 1
    end

    @testset "At most N_MAX_HOLDINGS distinct symbols held at once (rule 13)" begin
        symbols = ["S$i" for i in 1:25]
        cache   = make_test_cache(symbols=symbols)
        env     = TradingGameEnv(cache)
        reset!(env, EpisodeConfig(initial_cash=10_000_000.0, start_date=cache.dates[1],
                                   end_date=cache.dates[end], candidate_universe=symbols))
        date_idx = env.cache.date_index[env.current_date]

        # 21 simultaneous new-symbol buys, equal weight — cash and the 15% cap are
        # both slack here (each gets ~1/21 of a large cash pile), isolating rule 13.
        first21 = [env.cache.sym_index[s] for s in symbols[1:21]]
        resolved = resolve_actions(env, [RawAction(idx, BUY, 1.0) for idx in first21], date_idx)
        bought = Set(t.sym_idx for t in resolved if t.kind == BUY)
        @test length(bought) == N_MAX_HOLDINGS   # the 21st is masked out

        TradingGame._apply_actions!(env, resolved, date_idx)
        @test length(Set(h.sym_idx for h in env.portfolio.holdings)) == N_MAX_HOLDINGS

        # Adding to an already-held symbol is still allowed at the cap...
        resolved2 = resolve_actions(env, [RawAction(first(bought), BUY, 1.0)], date_idx)
        @test any(t.kind == BUY for t in resolved2)

        # ...but opening a brand-new (21st) position is not.
        unheld_idx = env.cache.sym_index[symbols[22]]
        resolved3 = resolve_actions(env, [RawAction(unheld_idx, BUY, 1.0)], date_idx)
        @test isempty(resolved3)
    end

    @testset "Under flat prices, portfolio value is fee-drag-only: non-increasing, never negative" begin
        cache = make_test_cache(n_days=20)
        cache.hourly_closes .= 100f0
        cache.closes        .= 100f0
        env = TradingGameEnv(cache)
        reset!(env, EpisodeConfig(initial_cash=100_000.0, start_date=cache.dates[1],
                                   end_date=cache.dates[end], candidate_universe=cache.companies))

        rng = MersenneTwister(7)
        prev_value = portfolio_value(env)
        for _ in 1:(19 * 7)
            step!(env, random_policy(env; rng=rng))
            @test env.portfolio.cash   >= -1e-6
            @test portfolio_value(env) >= -1e-6
            @test portfolio_value(env) <= prev_value + 1e-6
            prev_value = portfolio_value(env)
        end
    end

    @testset "Full historical episode: random and heuristic baselines obey every rule" begin
        policies = [
            env -> random_policy(env; rng=MersenneTwister(1)),
            heuristic_policy,
        ]
        for policy_fn in policies
            env = make_test_env(n_days=40)
            done, steps = false, 0
            while !done
                r = step!(env, policy_fn(env))
                done = r.done
                steps += 1

                @test env.portfolio.cash   >= -1e-6
                @test portfolio_value(env) >= -1e-6
                date_idx = env.cache.date_index[env.current_date]
                for h in env.portfolio.holdings
                    @test date_idx - h.entry_date_idx < MAX_HOLD_DAYS
                end

                # N_MAX_HOLDINGS is a true invariant (rule 13 masks every buy that would
                # exceed it). MAX_POSITION_FRACTION (rule 12) is purchase-time-only by
                # design — price drift after a buy can legitimately carry a position
                # above 15%, so it's NOT asserted here; see the dedicated rule-12 testset
                # below for what IS guaranteed (the cap binds at the moment of purchase).
                @test length(Set(h.sym_idx for h in env.portfolio.holdings)) <= N_MAX_HOLDINGS

                @test steps <= 10_000   # runaway guard — episode must terminate via `done`
            end
            @test steps > 0
        end
    end

end
