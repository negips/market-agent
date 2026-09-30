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

function make_test_env(; n_days::Int=30, initial_cash::Float64=100_000.0)
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

                @test steps <= 10_000   # runaway guard — episode must terminate via `done`
            end
            @test steps > 0
        end
    end

end
