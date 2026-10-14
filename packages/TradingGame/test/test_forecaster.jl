# ── Tests: v4 minute forecaster (cache, model, loss, training) ────────────────────
#
# All on small synthetic data — no CSVs from website/data and no GPU needed.

const _M = TradingGame.MINUTES_PER_DAY

"""A synthetic cache: `ndays` weekday-ish dates × 375 minutes, `nsym` symbols, random returns, ~60% traded."""
function make_minute_cache(; ndays=8, nsym=5, seed=1)
    rng = MersenneTwister(seed)
    dates = [Date(2024, 1, 1) + Day(i - 1 + 2 * ((i - 1) ÷ 5)) for i in 1:ndays]   # skips weekends
    T = ndays * _M
    logret = 0.3f0 .* randn(rng, Float32, T, nsym)
    traded = rand(rng, T, nsym) .< 0.6
    logret[.!traded] .= 0f0
    return MinuteCache(["S$i" for i in 1:nsym], dates, logret, traded)
end

@testset "v4 minute forecaster" begin

    @testset "MinuteCache: axis, time features, scale" begin
        c = make_minute_cache()
        @test size(c.logret) == (8 * _M, 5)
        @test size(c.tau) == (8 * _M, TradingGame.N_TIME_FEATURES) && size(c.tdesc) == (8 * _M, 3)
        @test c.tau[1, 5] == 1f0 && c.tau[2, 5] == 0f0 && c.tau[_M + 1, 5] == 1f0      # first-bar flag
        @test c.tau[2, 1] ≈ 1 / _M && c.tdesc[_M, 1] ≈ (_M - 1) / _M                      # minute-of-day fraction
        @test c.tau[5 * _M + 1, 4] ≈ log1p(3)                                              # Friday → Monday gap of 3 days
        @test c.scale > 0
        @test_throws DimensionMismatch MinuteCache(c.symbols, c.dates, c.logret[1:10, :], c.traded[1:10, :])
    end

    @testset "make_batch: alignment, masking, and no look-ahead in the context features" begin
        c = make_minute_cache()
        M, cols = 12, [1, 3, 4]
        s = _M + 20
        b = make_batch(c, cols, [s], M)
        @test size(b.X) == (8, 3, M, 1) && size(b.Y) == (M, 3, 1) && size(b.W) == (M, 3, 1) && size(b.T) == (3, M, 1)
        @test b.X[1, 2, 1, 1] ≈ c.logret[s, 3] / c.scale                  # input at position 1 is bar s
        @test b.Y[1, 2, 1] ≈ c.logret[s + 1, 3] / c.scale                 # its target is bar s+1
        @test b.T[1, 5, 1] ≈ c.tdesc[s + 5, 1]
        @test b.W[1, 2, 1] == (c.traded[s + 1, 3] ? 1f0 : 0f0)
        # a target on the first bar of a day is masked even if traded
        s2 = 2 * _M - 5
        b2 = make_batch(c, [1, 2, 3, 4, 5], [s2], M)
        j = 2 * _M + 1 - s2                                               # position whose target is the first bar of day 3
        @test all(b2.W[j, :, 1] .== 0)
        # the context features come only from bars before the window
        c2 = MinuteCache(c.symbols, c.dates, copy(c.logret), copy(c.traded))
        c2.logret[s:end, :] .= 5f0
        b3 = make_batch(c2, cols, [s], M)
        @test b3.X[7:8, :, :, 1] == b.X[7:8, :, :, 1]
        @test b3.X[1, :, 1, 1] != b.X[1, :, 1, 1]
    end

    @testset "window_starts: inside the date range with a full context behind it" begin
        c = make_minute_cache()
        r = window_starts(c, c.dates[3], c.dates[6], 50)
        @test first(r) == 2 * _M + 1 && last(r) == 6 * _M - 50
        @test first(window_starts(c, c.dates[1], c.dates[6], 50)) == TradingGame.CONTEXT_BARS + 1
        @test isempty(window_starts(c, c.dates[1], c.dates[1], 50))
    end

    @testset "cache save/load round trip" begin
        c = make_minute_cache()
        path = tempname() * ".bson"
        save_minute_cache(c, path)
        c2 = load_minute_cache(path)
        @test c2.symbols == c.symbols && c2.dates == c.dates && c2.logret == c.logret && c2.traded == c.traded && c2.scale == c.scale
        @test_throws MinuteCacheError load_minute_cache(tempname())
    end

    @testset "build_minute_cache: alignment, forward fill, traded flag, day filter" begin
        dir = mktempdir()
        function write_csv(sym, rows)
            open(joinpath(dir, "$sym.csv"), "w") do io
                println(io, "datetime,open,high,low,close,volume")
                for (dt, close, vol) in rows
                    println(io, "$dt,$close,$close,$close,$close,$vol")
                end
            end
        end
        days = ["2024-03-04", "2024-03-05", "2024-03-06"]
        rows(extra=[]) = vcat([("$(d)T$(h):$(m):00.0", 100.0 + i, 10.0) for (i, d) in enumerate(days) for (h, m) in (("09", "15"), ("09", "16"), ("15", "29"))], extra)
        write_csv("AAA", rows())
        write_csv("BBB", rows([("2024-03-05T09:20:00.0", 105.0, 0.0), ("2024-03-05T20:00:00.0", 999.0, 5.0)]))
        write_csv("CCC", [("2024-03-04T09:15:00.0", 50.0, 1.0)])      # active on one day only, with too few bars
        c = build_minute_cache(dir, ["AAA", "BBB", "CCC"]; from=Date(2024, 3, 4), to=Date(2024, 3, 6), min_day_fraction=0.6, min_day_bars=3, verbose=false)
        @test c.dates == [Date(2024, 3, 4), Date(2024, 3, 5), Date(2024, 3, 6)]
        @test size(c.logret) == (3 * _M, 3)
        @test c.traded[1, 1] && c.traded[2, 1] && !c.traded[3, 1]            # rows at minute 0 and 1, none at minute 2
        @test c.logret[3, 1] == 0f0                                            # forward-filled minute
        @test c.logret[2, 1] ≈ 100 * log(101.0 / 101.0)                        # same close
        @test c.logret[_M + 1, 1] ≈ Float32(100 * log(102.0 / 101.0))          # next day's first bar vs previous close
        @test !c.traded[_M + 6, 2] && c.logret[_M + 6, 2] ≈ Float32(100 * log(105.0 / 102.0))   # zero-volume row moves the price but is not "traded"
        @test_throws MinuteCacheError build_minute_cache(dir, ["AAA", "NOPE"]; from=Date(2024, 3, 4), to=Date(2024, 3, 6), verbose=false)
    end

    @testset "Forecaster: parameter counts match the architecture page" begin
        @test n_parameters(Forecaster(ForecasterConfig(window=96, d=64))) == 228_037
        @test n_parameters(Forecaster(ForecasterConfig(window=96, d=128))) == 890_245
        @test_throws ForecasterConfigError Forecaster(ForecasterConfig(d=30, heads=4))
        cfg = ForecasterConfig(window=10, d=16, heads=2, blocks=3, dropout=0.2f0)
        @test forecaster_config(Forecaster(cfg)) == cfg
    end

    @testset "Forecaster: shapes, causality, stock symmetry, any N" begin
        cfg = ForecasterConfig(window=12, d=16, heads=2, blocks=2, dropout=0f0)
        m = Forecaster(cfg; seed=3)
        X = randn(Float32, 8, 5, 12, 2)
        r, ls, t = m(X)
        @test size(r) == (12, 5, 2) && size(ls) == (12, 5, 2) && size(t) == (3, 12, 2)
        @test all(LOGSIGMA_CENTER - LOGSIGMA_HALFRANGE .<= ls .<= LOGSIGMA_CENTER + LOGSIGMA_HALFRANGE)
        X2 = copy(X); X2[:, :, 7:end, :] .= randn(Float32, 8, 5, 6, 2)
        r2, ls2, t2 = m(X2)
        @test isapprox(r[1:6, :, :], r2[1:6, :, :]; atol=1e-5) && isapprox(ls[1:6, :, :], ls2[1:6, :, :]; atol=1e-5) && isapprox(t[:, 1:6, :], t2[:, 1:6, :]; atol=1e-5)      # the past ignores the future
        @test maximum(abs.(r[7:end, :, :] .- r2[7:end, :, :])) > 1e-3
        p = [3, 1, 5, 2, 4]
        rp, lsp, tp = m(X[:, p, :, :])
        @test isapprox(rp, r[:, p, :]; atol=1e-5) && isapprox(lsp, ls[:, p, :]; atol=1e-5) && isapprox(tp, t; atol=1e-5)                       # stocks are interchangeable
        r3, _, _ = m(X[:, 1:3, :, :]); @test size(r3) == (12, 3, 2)                                      # a different N, same weights
        r4, _, _ = m(X[:, :, 1:7, :]); @test size(r4) == (7, 5, 2) && isapprox(r4, r[1:7, :, :]; atol=1e-5)       # a shorter window
        @test_throws ForecasterConfigError m(randn(Float32, 8, 5, 13, 1))
    end

    @testset "forecast_loss: only counted targets contribute; gradients are finite" begin
        cfg = ForecasterConfig(window=8, d=16, heads=2, blocks=1, dropout=0f0)
        m = Forecaster(cfg; seed=4)
        X = randn(Float32, 8, 4, 8, 2); Y = randn(Float32, 8, 4, 2); T = randn(Float32, 3, 8, 2)
        W = Float32.(rand(MersenneTwister(1), 8, 4, 2) .< 0.5)
        l1, lr1, _ = forecast_loss(m, X, Y, W, T)
        Y2 = copy(Y); Y2[W .== 0] .= 1f3
        @test forecast_loss(m, X, Y2, W, T)[2] ≈ lr1
        @test forecast_loss(m, X, Y, zeros(Float32, 8, 4, 2), T)[2] == 0f0
        @test l1 ≈ lr1 + TIME_LOSS_COEF * forecast_loss(m, X, Y, W, T)[3]
        @test forecast_loss(m, X, Y2, W, T; loss=:huber)[2] ≈ forecast_loss(m, X, Y, W, T; loss=:huber)[2]
        @test forecast_loss(m, X, Y2, W, T; loss=:gaussian)[2] ≈ forecast_loss(m, X, Y, W, T; loss=:gaussian)[2]
        g = Flux.gradient(mm -> forecast_loss(mm, X, Y, W, T)[1], m)[1]
        @test all(isfinite, vcat((vec(x) for x in Flux.trainables(g))...))
    end

    @testset "checkpoint round trip" begin
        cfg = ForecasterConfig(window=8, d=16, heads=2, blocks=2, dropout=0.1f0)
        m = Forecaster(cfg; seed=5)
        path = tempname() * ".bson"
        save_forecaster(m, path; meta=Dict{String, Any}("note" => "x"))
        m2, cfg2, meta = load_forecaster(path)
        @test cfg2 == cfg && meta["note"] == "x"
        X = randn(Float32, 8, 3, 8, 1)
        Flux.testmode!(m); Flux.testmode!(m2)
        @test m(X)[1] ≈ m2(X)[1]
        @test_throws ForecasterConfigError load_forecaster(tempname())
    end

    @testset "evaluate_forecaster: a zero predictor has skill 0; baselines and report shapes" begin
        c = make_minute_cache()
        cfg = ForecasterConfig(window=16, d=16, heads=2, blocks=1, dropout=0f0)
        m = Forecaster(cfg; seed=6)
        m.ret_head.weight .= 0f0; m.ret_head.bias .= 0f0     # location 0, scale at its centre
        es = EvalSet("t", [1, 2, 3], spread_starts(window_starts(c, c.dates[2], c.dates[8], 16), 6, 16))
        r = evaluate_forecaster(m, c, es, 16; batch=4)
        @test r.skill == 0 && r.mse ≈ r.mse_zero
        @test r.n_targets > 0 && length(r.per_stock_skill) == 3 && length(r.per_slot_skill) == TradingGame.N_SLOT_BUCKETS
        @test 0 <= r.hit_rate <= 1 && 0 <= r.minute_acc <= 1
        @test isfinite(r.skill_last)
    end

    @testset "scheduled_lr: warm-up then cosine to a tenth" begin
        @test TradingGame.scheduled_lr(1, 100, 1e-3, 10) ≈ 1e-4
        @test TradingGame.scheduled_lr(10, 100, 1e-3, 10) ≈ 1e-3
        @test TradingGame.scheduled_lr(100, 100, 1e-3, 10) ≈ 1e-4
        @test TradingGame.scheduled_lr(55, 100, 1e-3, 10) < 1e-3
    end

    @testset "train_forecaster!: learns a predictable series (overfit check), writes logs, STOP works" begin
        # next-minute return = -0.8 × this minute's return, a pattern the model must find
        rng = MersenneTwister(7)
        T = 6 * _M
        base = 0.3f0 .* randn(rng, Float32, T, 4)
        lr = similar(base)
        lr[1, :] .= 0
        for t in 2:T
            lr[t, :] .= -0.8f0 .* lr[t - 1, :] .+ 0.05f0 .* base[t, :]
        end
        dates = [Date(2024, 1, 1) + Day(i - 1) for i in 1:6]
        c = MinuteCache(["A", "B", "C", "D"], dates, lr, trues(T, 4))
        cfg = ForecasterConfig(window=16, d=16, heads=2, blocks=2, dropout=0f0)
        m = Forecaster(cfg; seed=8)
        starts = window_starts(c, dates[2], dates[5], 16)
        sets = [EvalSet("val", [1, 2], spread_starts(starts, 8, 16))]
        before = evaluate_forecaster(m, c, sets[1], 16)
        dir = mktempdir()
        log = train_forecaster!(m, c, [3, 4], starts, sets; epochs=3, steps_per_epoch=60, batch=8, lr=3e-3,
                                warmup_steps=10, out_dir=dir, rng=MersenneTwister(9))
        after = evaluate_forecaster(m, c, sets[1], 16)
        @test after.loss < before.loss - 0.5                            # the Student-t NLL falls well below its starting value
        @test after.skill > 0.3                                         # unseen stocks A, B: the rule transfers
        @test after.minute_acc > 0.9                                    # the calendar prior gives the minute exactly
        @test after.skill > after.skill_last                            # and beats just repeating the last return (which is wrong here)
        @test log["epochs_run"] == 3 && 1 <= log["best_epoch"] <= 3
        @test isfile(joinpath(dir, "forecaster.bson")) && isfile(joinpath(dir, "timing.json"))
        rows = [JSON3.read(l) for l in eachline(joinpath(dir, "epoch_log.jsonl"))]
        @test [r.epoch for r in rows] == [1, 2, 3] && haskey(rows[1].eval, :val) && haskey(rows[1].eval.val, :skill)
        # STOP ends after the current epoch and the file is consumed
        touch(joinpath(dir, "STOP"))
        log2 = train_forecaster!(m, c, [3, 4], starts, sets; epochs=5, steps_per_epoch=2, batch=4, out_dir=dir,
                                 epoch_offset=3, rng=MersenneTwister(10))
        @test log2["epochs_run"] == 1 && log2["stopped"] == "STOP" && !isfile(joinpath(dir, "STOP"))
        @test [JSON3.read(l).epoch for l in eachline(joinpath(dir, "epoch_log.jsonl"))][end] == 4
    end
end

@testset "v4 forecaster number types" begin
    @test forecast_dtype("Float16") === Float16 && forecast_dtype("float32") === Float32 && forecast_dtype(" float64 ") === Float64
    @test_throws ForecasterConfigError forecast_dtype("float8")

    cfg = ForecasterConfig(window=12, d=16, heads=2, blocks=2, dropout=0f0)
    m = Forecaster(cfg; seed=3)
    X = randn(Float32, 8, 5, 12, 2)
    X[2, :, :, :] .= reshape(Float32.((0:11) ./ 375), 1, 12, 1)       # a real minute-of-day column, the same for every stock
    for T in (Float16, Float64)
        mt = convert_forecaster(m, T)
        @test eltype(mt.embed.weight) == T && eltype(mt.pos) == T
        r, ls, t = mt(T.(X))
        @test eltype(r) == T && all(isfinite, r) && all(isfinite, ls) && all(isfinite, t)
        r32, _, t32 = m(X)
        @test isapprox(Float32.(r), r32; atol=T === Float16 ? 5e-2 : 1e-5)
        @test isapprox(Float32.(t), t32; atol=T === Float16 ? 5e-3 : 1e-5)                # the timestamp prior stays exact
    end
    Y = randn(Float32, 12, 5, 2); W = ones(Float32, 12, 5, 2); Tt = randn(Float32, 3, 12, 2)
    @test forecast_loss(convert_forecaster(m, Float16), Float16.(X), Y, W, Tt)[1] isa Float32      # the loss is computed in Float32
    @test isapprox(forecast_loss(convert_forecaster(m, Float16), Float16.(X), Y, W, Tt)[1], forecast_loss(m, X, Y, W, Tt)[1]; rtol=0.05)

    # training in Float16 and Float64 still learns the same predictable series, and saves Float32 weights
    rng = MersenneTwister(7)
    Tn = 6 * _M
    lr = zeros(Float32, Tn, 4)
    for t in 2:Tn
        lr[t, :] .= -0.8f0 .* lr[t - 1, :] .+ 0.3f0 .* randn(rng, Float32, 4)
    end
    dates = [Date(2024, 1, 1) + Day(i - 1) for i in 1:6]
    c = MinuteCache(["A", "B", "C", "D"], dates, lr, trues(Tn, 4))
    starts = window_starts(c, dates[2], dates[5], 16)
    sets = [EvalSet("val", [1, 2], spread_starts(starts, 8, 16))]
    for T in (Float16, Float64)
        mm = Forecaster(ForecasterConfig(window=16, d=16, heads=2, blocks=2, dropout=0f0); seed=8)
        before = evaluate_forecaster(mm, c, sets[1], 16; dtype=T)
        dir = mktempdir()
        log = train_forecaster!(mm, c, [3, 4], starts, sets; epochs=2, steps_per_epoch=60, batch=8, lr=3e-3,
                                warmup_steps=10, dtype=T, out_dir=dir, rng=MersenneTwister(9))
        after = evaluate_forecaster(mm, c, sets[1], 16; dtype=T)
        @test after.loss < before.loss && after.skill > 0.2
        row = JSON3.read(readlines(joinpath(dir, "epoch_log.jsonl"))[1])
        @test row.dtype == string(T) && haskey(row, :loss_scale) && row.skipped_steps >= 0
        loaded, _, _ = load_forecaster(joinpath(dir, "forecaster.bson"))
        @test eltype(loaded.embed.weight) == Float32
    end
end


@testset "v4 forecaster predictive distribution" begin
    @testset "likelihood helpers" begin
        @test TradingGame._loggamma(5) ≈ log(24) atol=1e-9
        @test TradingGame._loggamma(0.5) ≈ log(sqrt(π)) atol=1e-9
        @test student_t_const(1) ≈ log(π) atol=1e-9                     # Cauchy: density 1/(π(1+x²))
        @test student_t_const(2) ≈ log(2 * sqrt(2)) atol=1e-9           # density at 0 is 1/(2√2)
        @test forecast_nll(:gaussian, [0f0], [0f0], [0f0], 4)[1] ≈ 0.5 * log(2π) atol=1e-6
        @test forecast_nll(:student_t, [0f0], [0f0], [0f0], 1)[1] ≈ log(π) atol=1e-6
        @test isapprox(forecast_nll(:student_t, [0.7f0], [0f0], [0f0], 1e5)[1], forecast_nll(:gaussian, [0.7f0], [0f0], [0f0], 4)[1]; atol=1e-3)
        @test central_halfwidth(:gaussian, 4, 0.9545) ≈ 2.0 atol=1e-2
        @test central_halfwidth(:gaussian, 4, 0.95) ≈ 1.960 atol=1e-3
        @test central_halfwidth(:student_t, 4, 0.95) ≈ 2.776 atol=1e-3   # the t(4) 97.5% quantile
        @test unit_std(:student_t, 4) ≈ sqrt(2) && unit_std(:gaussian, 4) == 1
        @test_throws ForecastLossError unit_std(:student_t, 2)
        @test forecast_loss_kind("Student_T") === :student_t
        @test_throws ForecastLossError forecast_loss_kind("laplace")
    end

    @testset "the scale head learns a volatility that depends on the time of day, and stays calibrated" begin
        rng = MersenneTwister(21)
        T = 8 * _M
        slot = repeat(0:_M - 1, 8)
        sd = [m < _M ÷ 2 ? 0.1f0 : 1.0f0 for m in slot]                 # quiet morning, wild afternoon, zero mean
        lr = sd .* randn(rng, Float32, T, 4)
        dates = [Date(2024, 1, 1) + Day(i - 1) for i in 1:8]
        c = MinuteCache(["A", "B", "C", "D"], dates, lr, trues(T, 4))
        starts = window_starts(c, dates[2], dates[7], 16)
        sets = [EvalSet("val", [1, 2], spread_starts(starts, 40, 16))]
        for kind in (:student_t, :gaussian)
            m = Forecaster(ForecasterConfig(window=16, d=16, heads=2, blocks=1, dropout=0f0); seed=11)
            log = train_forecaster!(m, c, [3, 4], starts, sets; epochs=2, steps_per_epoch=120, batch=8, lr=3e-3,
                                    warmup_steps=10, loss=kind, rng=MersenneTwister(5))
            r = evaluate_forecaster(m, c, sets[1], 16; loss=kind)
            early = sum(r.std_pred_slot[1:8]) / 8; late = sum(r.std_pred_slot[end-8:end]) / 9
            @test late > 3 * early                                      # it found the pattern
            @test r.nll < r.nll_const - 0.3                             # and beats one scale per stock on likelihood
            @test 0.55 < r.cov68 < 0.85 && 0.88 < r.cov95 < 0.99        # near the nominal 68% / 95%
            @test abs(r.skill) < 0.05                                   # nothing to forecast in the mean
        end
    end
end

@testset "v4 AR baseline" begin
    @testset "it recovers an AR(1) location and a time-of-day scale, and is scored by the same code" begin
        rng = MersenneTwister(31)
        T = 10 * _M
        slot = repeat(0:_M - 1, 10)
        sd = [m < _M ÷ 2 ? 0.2f0 : 1.0f0 for m in slot]
        lr = zeros(Float32, T, 4)
        for s in 1:4, t in 2:T
            (t - 1) % _M == 0 && continue
            lr[t, s] = 0.4f0 * lr[t - 1, s] + sd[t] * randn(rng, Float32)
        end
        dates = [Date(2024, 1, 1) + Day(i - 1) for i in 1:10]
        c = MinuteCache(["A", "B", "C", "D"], dates, lr, trues(T, 4))
        starts = window_starts(c, dates[2], dates[9], 32)
        b = fit_ar_baseline(c, [3, 4], starts, 32; lags=3, windows=300, rng=MersenneTwister(2))
        @test b.mean_coef[1] ≈ 0.4 atol=0.06
        @test abs(b.mean_coef[2]) < 0.06
        es = EvalSet("val", [1, 2], spread_starts(starts, 40, 32))
        r = evaluate_ar_baseline(b, c, es, 32)
        @test r.skill > 0.05                                             # the AR(1) structure is found
        @test r.nll < r.nll_const - 0.2                                   # and so is the quiet morning
        @test 0.55 < r.cov68 < 0.85 && 0.88 < r.cov95 < 0.99
        @test isnan(r.minute_acc) && r.n_targets > 0
    end

    @testset "mean_abs_factor and the guard rails" begin
        @test TradingGame.mean_abs_factor(:student_t, 4) ≈ 1.0 atol=1e-9      # E|t₄| = 1
        @test TradingGame.mean_abs_factor(:gaussian, 4) ≈ sqrt(2 / π)
        @test_throws ForecastLossError TradingGame.mean_abs_factor(:huber, 4)
        @test_throws ForecastLossError TradingGame.mean_abs_factor(:student_t, 2)
    end
end

@testset "v4 warm start" begin
    cfg = ForecasterConfig(window=16, d=16, heads=2, blocks=1, dropout=0.1f0)
    src = Forecaster(cfg; seed=3)
    path = joinpath(mktempdir(), "ck.bson")
    save_forecaster(src, path)

    @testset "weights are copied; dropout and seed of the new run apply" begin
        m = warm_start_forecaster(ForecasterConfig(window=16, d=16, heads=2, blocks=1, dropout=0.3f0), path; seed=99)
        X = randn(MersenneTwister(1), Float32, TradingGame.N_FORECAST_FEATURES, 3, 8, 2)
        a = Flux.testmode!(src)(X); b = Flux.testmode!(m)(X)
        @test all(a[i] ≈ b[i] for i in 1:3)
        @test forecaster_config(m).dropout ≈ 0.3f0
    end

    @testset "an architecture that does not fit is refused, naming what differs" begin
        err = try warm_start_forecaster(ForecasterConfig(window=16, d=32, heads=2, blocks=1), path); nothing catch e; e end
        @test err isa ForecasterConfigError && occursin("d = 16", err.msg) && occursin("32 asked for", err.msg)
        @test_throws ForecasterConfigError warm_start_forecaster(ForecasterConfig(window=16, d=16, heads=2, blocks=2), path)
        @test_throws ForecasterConfigError warm_start_forecaster(cfg, joinpath(tempdir(), "no_such_checkpoint.bson"))
    end
end

@testset "v4 forecaster cache at every bar size" begin
    function write_bars(dir, sym, stamps)
        mkpath(dir)
        open(joinpath(dir, "$sym.csv"), "w") do io
            println(io, occursin("T", first(stamps)[1]) ? "datetime,open,high,low,close,volume" : "date,open,high,low,close,volume")
            for (st, close, vol) in stamps
                println(io, "$st,$close,$close,$close,$close,$vol")
            end
        end
    end
    days = ["2024-03-04", "2024-03-05", "2024-03-06"]
    from, to = Date(2024, 3, 4), Date(2024, 3, 6)

    @testset "granularity table" begin
        @test Set(keys(GRANULARITIES)) == Set(["1min", "5min", "15min", "hourly", "daily"])
        @test [cld(TradingGame.MINUTES_PER_DAY, GRANULARITIES[g].bar_minutes) for g in ("1min", "5min", "15min", "hourly", "daily")] == [375, 75, 25, 7, 1]
        @test_throws MinuteCacheError build_forecaster_cache(mktempdir(), ["A"]; granularity="3min", from=from, to=to, verbose=false)
        @test_throws MinuteCacheError MinuteCache(["A"], [Date(2024, 3, 4)], zeros(Float32, 1, 1), trues(1, 1); granularity="weekly")
    end

    @testset "5min, 15min and hourly: a bar is placed by the start of its interval" begin
        for (g, label, per_day, hits) in (("5min", "5min", 75, [("09:15", 0), ("09:20", 1), ("15:25", 74)]),
                                          ("15min", "15min", 25, [("09:15", 0), ("09:30", 1), ("15:15", 24)]),
                                          ("hourly", "hourly", 7, [("09:15", 0), ("10:15", 1), ("15:15", 6)]))
            dir = joinpath(mktempdir(), label)
            for sym in ("AAA", "BBB")
                write_bars(dir, sym, [("$(d)T$(hm):00.0", 100.0 + i + k, 10.0) for (i, d) in enumerate(days) for (k, (hm, _)) in enumerate(hits)])
            end
            c = build_forecaster_cache(dir, ["AAA", "BBB"]; granularity=g, from=from, to=to, min_day_fraction=0.5, verbose=false)
            @test c.granularity == g && c.bars_per_day == per_day && size(c.logret) == (3 * per_day, 2)
            for (k, (_, idx)) in enumerate(hits)
                @test c.traded[idx + 1, 1]
            end
            @test sum(c.traded[:, 1]) == 3 * length(hits)
            @test c.tau[2, 1] ≈ 1 / per_day                                    # bar-of-day fraction of the second bar
            @test c.context_bars == max(per_day, TradingGame.MIN_CONTEXT_BARS)
            @test n_slot_buckets(c) == cld(per_day, c.bucket_bars)
            @test maximum(Int.(make_batch(c, [1, 2], [c.context_bars + 1], 8).slot)) < n_slot_buckets(c)
        end
    end

    @testset "daily: one bar a day, every target counts, calendar prior steps weekday to weekday" begin
        dir = joinpath(mktempdir(), "daily")
        dates = [Date(2024, 3, 1) + Day(i - 1) for i in 1:200 if dayofweek(Date(2024, 3, 1) + Day(i - 1)) <= 5]
        rng = MersenneTwister(5)
        for sym in ("AAA", "BBB", "CCC")
            px = 100.0 .* exp.(cumsum(0.01 .* randn(rng, length(dates))))
            write_bars(dir, sym, [(string(d), px[i], 1000.0) for (i, d) in enumerate(dates)])
        end
        c = build_forecaster_cache(dir, ["AAA", "BBB", "CCC"]; granularity="daily", from=first(dates), to=last(dates), verbose=false)
        @test c.bars_per_day == 1 && length(c.dates) == length(dates) && size(c.logret) == (length(dates), 3)
        @test c.tau[:, 1] == zeros(Float32, length(dates)) && all(c.tau[:, 5] .== 1)   # fraction constant, every bar is a first bar
        b = make_batch(c, [1, 2, 3], [c.context_bars + 1], 16)
        @test all(b.W .== 1) && all(b.slot .== 0)                                      # nothing is excluded as an overnight gap
        i = findfirst(d -> dayofweek(d) == 5, c.dates[2:end]) + 1                      # a Friday: the next bar's gap is 3 calendar days
        @test c.tdesc[i + 1, 2] ≈ Float32(log1p(3)) && c.tdesc[i, 3] ≈ 4 / 4 && c.tdesc[i + 1, 3] ≈ 0
        m = Forecaster(ForecasterConfig(window=16, d=16, heads=2, blocks=1, bars_per_day=1); seed=2)
        _, _, tm = Flux.testmode!(m)(b.X)
        @test size(tm) == (3, 16, 1)
        prior = TradingGame.next_time_prior(b.X, 1)
        @test all(prior[1, :, :] .== 0f0)                                              # the next daily bar is bar 0 again
    end

    @testset "a model of any bar size trains and evaluates; the checkpoint remembers it" begin
        dir = joinpath(mktempdir(), "15min")
        rng = MersenneTwister(8)
        for sym in ("AAA", "BBB", "CCC", "DDD")
            px = 100.0 .* exp.(cumsum(0.002 .* randn(rng, 25 * 70)))
            rows = Tuple{String, Float64, Float64}[]
            for d in 0:69, k in 0:24
                dt = Date(2024, 1, 1) + Day(d + 2 * (d ÷ 5))
                m = 9 * 60 + 15 + 15 * k
                push!(rows, ("$(dt)T$(lpad(m ÷ 60, 2, '0')):$(lpad(m % 60, 2, '0')):00.0", px[d * 25 + k + 1], 100.0))
            end
            write_bars(dir, sym, rows)
        end
        c = build_forecaster_cache(dir, ["AAA", "BBB", "CCC", "DDD"]; granularity="15min", from=Date(2024, 1, 1), to=Date(2024, 6, 1), verbose=false)
        @test c.bars_per_day == 25 && length(c.dates) == 70
        cfg = ForecasterConfig(window=24, d=16, heads=2, blocks=1, bars_per_day=c.bars_per_day)
        model = Forecaster(cfg; seed=1)
        starts = window_starts(c, c.dates[1], c.dates[end], cfg.window)
        es = EvalSet("val", [1, 2, 3, 4], collect(starts)[1:8:end][1:6])
        r = evaluate_forecaster(model, c, es, cfg.window; batch=3)
        @test length(r.per_slot_skill) == n_slot_buckets(c) && isfinite(r.minute_acc)
        out = mktempdir()
        train_forecaster!(model, c, [1, 2, 3, 4], starts, [es]; epochs=1, steps_per_epoch=3, batch=4, out_dir=out, warmup_steps=1)
        m2, cfg2, _ = load_forecaster(joinpath(out, "forecaster.bson"))
        @test cfg2.bars_per_day == 25 && forecaster_config(m2).bars_per_day == 25
    end
end
