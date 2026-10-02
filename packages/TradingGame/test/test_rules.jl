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
        # 8 symbols (not make_test_env's default 3) so n_max_holdings(8)=2 —
        # this test needs room for 2 simultaneously-held distinct symbols,
        # which the default 3-symbol fixture's n_max_holdings(3)=1 no longer allows.
        symbols = ["S$i" for i in 1:8]
        cache   = make_test_cache(symbols=symbols)
        env     = TradingGameEnv(cache)
        reset!(env, EpisodeConfig(initial_cash=100_000.0, start_date=cache.dates[1],
                                   end_date=cache.dates[end], candidate_universe=symbols))
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

    @testset "At most n_max_holdings(N) distinct symbols held at once (rule 13)" begin
        # n_max_holdings(40) = round(0.25*40) = 10, exact — no rounding
        # ambiguity, and large enough that each accepted buy's equal-weight
        # share (1/10 = 10%) stays under MAX_POSITION_FRACTION's 15% cap, so
        # rule 12 doesn't also clip these positions and eat the headroom the
        # later "adding to an already-held symbol" check needs.
        symbols = ["S$i" for i in 1:40]
        n_max   = n_max_holdings(length(symbols))
        @test n_max == 10
        cache   = make_test_cache(symbols=symbols)
        env     = TradingGameEnv(cache)
        reset!(env, EpisodeConfig(initial_cash=10_000_000.0, start_date=cache.dates[1],
                                   end_date=cache.dates[end], candidate_universe=symbols))
        date_idx = env.cache.date_index[env.current_date]

        # n_max+1 simultaneous new-symbol buys, equal weight. Only n_max of
        # them pass rule 13's mask, so cash splits n_max ways (not n_max+1)
        # among the accepted ones — see the comment above for why that still
        # stays under the 15% cap here.
        requested = [env.cache.sym_index[s] for s in symbols[1:n_max+1]]
        resolved = resolve_actions(env, [RawAction(idx, BUY, 1.0) for idx in requested], date_idx)
        bought = Set(t.sym_idx for t in resolved if t.kind == BUY)
        @test length(bought) == n_max   # the (n_max+1)th is masked out

        TradingGame._apply_actions!(env, resolved, date_idx)
        @test length(Set(h.sym_idx for h in env.portfolio.holdings)) == n_max

        # Adding to an already-held symbol is still allowed at the cap...
        resolved2 = resolve_actions(env, [RawAction(first(bought), BUY, 1.0)], date_idx)
        @test any(t.kind == BUY for t in resolved2)

        # ...but opening a brand-new ((n_max+1)th) position is not.
        unheld_idx = env.cache.sym_index[symbols[n_max+2]]
        resolved3 = resolve_actions(env, [RawAction(unheld_idx, BUY, 1.0)], date_idx)
        @test isempty(resolved3)
    end

    @testset "n_max_holdings scales with universe size and floors at 1 (rule 13)" begin
        @test n_max_holdings(100) == 25    # the live run's actual universe size
        @test n_max_holdings(60)  == 15    # N_CANDIDATE_STOCKS default
        @test n_max_holdings(20)  == 5
        @test n_max_holdings(3)   == 1     # would round to 0.75→1 anyway, but the floor guarantees it
        @test n_max_holdings(1)   == 1
    end

    @testset "A sold symbol can't be newly bought again for REBUY_COOLDOWN_DAYS (rule 15)" begin
        env = make_test_env(initial_cash=1_000_000.0)
        sym_idx  = first(env.candidate_sym_idx)
        date_idx = env.cache.date_index[env.current_date]

        # Buy, wait out the 1-day lock-up, then voluntarily sell.
        TradingGame._apply_actions!(env, resolve_actions(env, [RawAction(sym_idx, BUY, 1.0)], date_idx), date_idx)
        sell_date_idx = date_idx + MIN_HOLD_DAYS
        TradingGame._apply_actions!(env, resolve_actions(env, [RawAction(sym_idx, SELL, 0.0)], sell_date_idx), sell_date_idx)
        @test isempty(env.portfolio.holdings)
        @test env.portfolio.rebuy_cooldown[sym_idx] == sell_date_idx + REBUY_COOLDOWN_DAYS

        # Still inside the cooldown: masked to HOLD.
        resolved_early = resolve_actions(env, [RawAction(sym_idx, BUY, 1.0)], sell_date_idx + REBUY_COOLDOWN_DAYS - 1)
        @test isempty(resolved_early)

        # Exactly at the cooldown's end: eligible again.
        resolved_ok = resolve_actions(env, [RawAction(sym_idx, BUY, 1.0)], sell_date_idx + REBUY_COOLDOWN_DAYS)
        @test length(resolved_ok) == 1 && resolved_ok[1].kind == BUY

        # A forced exit starts the same cooldown — shared code path, no special-casing.
        env2 = make_test_env(initial_cash=1_000_000.0)
        sym_idx2  = first(env2.candidate_sym_idx)
        date_idx2 = env2.cache.date_index[env2.current_date]
        TradingGame._apply_actions!(env2, resolve_actions(env2, [RawAction(sym_idx2, BUY, 1.0)], date_idx2), date_idx2)
        forced_date_idx = date_idx2 + MAX_HOLD_DAYS
        TradingGame._force_exit_stale_holdings!(env2, forced_date_idx)
        @test env2.portfolio.rebuy_cooldown[sym_idx2] == forced_date_idx + REBUY_COOLDOWN_DAYS

        # Cooldown only blocks opening a NEW position — adding to a symbol
        # that still has a separate, currently-held (not-yet-sold) lot is
        # unaffected. Needs an 8-symbol universe (n_max_holdings(8)=2) and a
        # competing decoy buy: a *lone* buy request always normalises to 100%
        # share (see resolve_actions — weight only matters relative to other
        # simultaneous buys), which would immediately hit MAX_POSITION_FRACTION
        # and leave no headroom to test a follow-up add.
        symbols3 = ["S$i" for i in 1:8]
        cache3   = make_test_cache(symbols=symbols3)
        env3     = TradingGameEnv(cache3)
        reset!(env3, EpisodeConfig(initial_cash=1_000_000.0, start_date=cache3.dates[1],
                                    end_date=cache3.dates[end], candidate_universe=symbols3))
        date_idx3 = env3.cache.date_index[env3.current_date]
        sym_idx3  = env3.cache.sym_index["S1"]
        decoy_idx = env3.cache.sym_index["S2"]
        # sym_idx3 gets only 10% of cash share (well under the 15% cap),
        # leaving headroom for a follow-up add.
        TradingGame._apply_actions!(env3,
            resolve_actions(env3, [RawAction(sym_idx3, BUY, 0.1), RawAction(decoy_idx, BUY, 0.9)], date_idx3),
            date_idx3)
        @test sym_idx3 in Set(h.sym_idx for h in env3.portfolio.holdings)
        # Manually seed a (hypothetical) cooldown on this still-held symbol —
        # adding to the existing lot must not be blocked by it.
        env3.portfolio.rebuy_cooldown[sym_idx3] = date_idx3 + 999
        resolved_add = resolve_actions(env3, [RawAction(sym_idx3, BUY, 1.0)], date_idx3)
        @test any(t.kind == BUY && t.sym_idx == sym_idx3 for t in resolved_add)
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

    @testset "Cash above MAX_CASH_FRACTION costs a reward penalty, not a blocked trade (rule 14)" begin
        # Never trade: cash stays pinned at 100% of portfolio value all episode,
        # the extreme case of a rule-14 violation.
        env = make_test_env(initial_cash=100_000.0)
        r = step!(env, RawAction[])
        @test r.info["cash_fraction"] ≈ 1.0
        @test r.info["cash_ceiling_violated"] == true

        # The penalty is exactly what the formula says: reward is price drift
        # (here zero, flat first bar's log-return) minus the coefficient times
        # the excess over the ceiling.
        expected_excess = 1.0 - MAX_CASH_FRACTION
        @test r.reward ≈ -CASH_CEILING_PENALTY_COEF * expected_excess atol=1e-6

        # A trade that brings cash to/under the ceiling clears the flag and the
        # penalty. MAX_POSITION_FRACTION (rule 12) caps each symbol at 15%, so
        # getting cash under 30% needs several symbols bought at once (>=5 to
        # cover the >=70% that must be deployed) — a single buy can't do it.
        # The universe needs n_max_holdings(N) >= 6 too (rule 13), or fewer
        # than 6 of these buys would even be accepted; 24 symbols gives
        # exactly 6, just enough for all of them.
        symbols = ["S$i" for i in 1:24]
        buy_symbols = symbols[1:6]
        cache   = make_test_cache(symbols=symbols)
        env2    = TradingGameEnv(cache)
        reset!(env2, EpisodeConfig(initial_cash=100_000.0, start_date=cache.dates[1],
                                    end_date=cache.dates[end], candidate_universe=symbols))
        date_idx = env2.cache.date_index[env2.current_date]
        sym_idxs = [env2.cache.sym_index[s] for s in buy_symbols]
        r2 = step!(env2, [RawAction(idx, BUY, 1.0) for idx in sym_idxs])
        @test r2.info["cash_fraction"] < MAX_CASH_FRACTION
        @test r2.info["cash_ceiling_violated"] == false

        # Rule 14 is explicitly NOT structural: resolve_actions never blocks or
        # shrinks a trade purely because post-trade cash would still be high —
        # a HOLD-only episode (tested above) is a legal, if penalized, trajectory.
        @test length(r2.info["trades"]) >= 1   # the buy above executed unmasked
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

                # n_max_holdings(N) is a true invariant (rule 13 masks every buy that
                # would exceed it). MAX_POSITION_FRACTION (rule 12) is purchase-time-only
                # by design — price drift after a buy can legitimately carry a position
                # above 15%, so it's NOT asserted here; see the dedicated rule-12 testset
                # below for what IS guaranteed (the cap binds at the moment of purchase).
                @test length(Set(h.sym_idx for h in env.portfolio.holdings)) <=
                      n_max_holdings(length(env.candidate_sym_idx))

                @test steps <= 10_000   # runaway guard — episode must terminate via `done`
            end
            @test steps > 0
        end
    end

end
