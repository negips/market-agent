"""
memory_profile.jl

GPU memory and step time of the v4 minute forecaster, measured on real batches from the cache: one training step
(forward, backward, AdamW update) and one evaluation forward pass, in a fresh process so nothing else sits in the pool.

The peak is the CUDA memory pool's own high-water mark (bytes in use, reset before each measurement), not the figure
`nvidia-smi` shows, which also counts freed blocks the pool keeps. "Retained by the forward pass" is what the backward
pass needs: the pool's growth between the start of the forward pass and its end, with the pullback still alive.
`--breakdown` repeats the measurement with 1 and 2 time blocks, so the cost of one block and of the rest (embedding, stock
mixing, heads) follow by subtraction.

Usage:
  julia --project=packages/TradingGame scripts/forecaster/memory_profile.jl --config configs/bse_v3.toml
  julia --project=packages/TradingGame scripts/forecaster/memory_profile.jl --config configs/bse_v3.toml --batch 8,16,32,64
  julia --project=packages/TradingGame scripts/forecaster/memory_profile.jl --config configs/bse_v3.toml --breakdown --dtype float16
  julia --project=packages/TradingGame scripts/forecaster/memory_profile.jl --config configs/bse_v3.toml --n-stocks 40 --out /tmp/profile.json

Options (the [forecaster] and [forecaster_cache] sections of --config supply the defaults; a flag overrides the file):
  --config PATH        TOML run file
  --granularity G      forecaster cache to read (1min default; 5min, 15min, hourly, daily)
  --window M, --d D, --heads H, --blocks L, --dropout P, --dtype T, --loss L
                       the model and number type to measure (as in train_minute_forecaster.jl)
  --batch B[,B...]     windows per batch; a list measures each in turn (default: the config's batch)
  --micro-batches K    split each batch into K pieces whose gradients are summed (the trainer's option); "kept for backward" is then per piece
  --n-stocks N         stocks per window: the first N train companies (default: all of them)
  --steps K            timed steps after 2 warm-up steps (default 5)
  --breakdown          also measure with 1 and 2 time blocks (first batch size only)
  --out PATH           write the result as JSON
  --force              run although another process holds most of the GPU (it can make that process run out of memory)
  -h, --help           this text

With `--micro-batches K` a piece's activations stay in the pool until Julia's collector frees them (CUDA.jl collects on demand when an allocation
fails), so "step peak" overstates what is needed. To see whether a setting fits a budget, cap the pool and look for OOM:
`JULIA_CUDA_HARD_MEMORY_LIMIT=10GiB julia ... --batch 64 --micro-batches 4` (the step is then slower, because it collects more often).

Needs a GPU with little else on it: stop a running training first (`touch website/data/STOP_PIPELINE`, then `STOP` in the run directory).
A batch size that does not fit is reported as OOM and the next one is tried.
"""

using TradingGame, Flux, CUDA, JSON3, Random, Printf, Statistics, Dates

include(joinpath(@__DIR__, "..", "lib", "run_config.jl"))
using .RunConfigs

const REPO_ROOT     = joinpath(@__DIR__, "..", "..")
const DATA_DIR      = joinpath(REPO_ROOT, "website", "data")
const UNIVERSE_FILE = joinpath(DATA_DIR, "trading_game", "universe_latest.json")
const CORE          = CUDA.CUDACore

"""Free GPU memory below which the script refuses to run without `--force`: another process is using the card."""
const MIN_FREE_FRACTION = 0.8

const WARMUP_STEPS = 2

function parse_args()
    config_path = RunConfigs.take_config_arg!(ARGS)
    cfg = config_path === nothing ? nothing : RunConfigs.load_or_exit(config_path)
    opts = Dict{String, Any}("window" => 96, "d" => 64, "heads" => 4, "blocks" => 4, "dropout" => 0.1,
        "batch" => 32, "dtype" => "float32", "loss" => "student_t", "nu" => 4.0, "nll_beta" => 0.5,
        "granularity" => "1min", "out" => nothing)
    for sec in ("forecaster_cache", "forecaster")
        merge!(opts, Dict(k => v for (k, v) in RunConfigs.section_opts(cfg, sec) if k in ("out", "granularity", keys(opts)...)))
    end
    opts["micro"] = 1; opts["n_stocks"] = 0; opts["steps"] = 5; opts["breakdown"] = false; opts["json"] = nothing; opts["force"] = false
    opts["batches"] = [Int(opts["batch"])]
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        takes_value = a in ("--granularity", "--window", "--d", "--heads", "--blocks", "--dropout", "--dtype", "--loss", "--batch", "--micro-batches", "--n-stocks", "--steps", "--out")
        takes_value && i == length(ARGS) && error("$a needs a value")
        if a in ("-h", "--help")
            println(match(r"\"\"\"\n(.*?)\"\"\""s, read(@__FILE__, String)).captures[1])
            exit(0)
        elseif a == "--micro-batches"; opts["micro"] = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--breakdown"; opts["breakdown"] = true; i += 1
        elseif a == "--force"; opts["force"] = true; i += 1
        elseif a == "--granularity"; opts["granularity"] = lowercase(ARGS[i+1]); i += 2
        elseif a == "--batch"; opts["batches"] = parse.(Int, split(ARGS[i+1], ",")); i += 2
        elseif a == "--out"; opts["json"] = abspath(ARGS[i+1]); i += 2
        elseif a in ("--window", "--d", "--heads", "--blocks", "--n-stocks", "--steps")
            opts[replace(a[3:end], "-" => "_")] = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--dropout"; opts["dropout"] = parse(Float64, ARGS[i+1]); i += 2
        elseif a in ("--dtype", "--loss"); opts[a[3:end]] = ARGS[i+1]; i += 2
        else
            error("Unknown argument: $a (see --help)")
        end
    end
    forecast_dtype(opts["dtype"])
    forecast_loss_kind(opts["loss"])
    all(>(0), opts["batches"]) || error("--batch must be positive")
    opts["steps"] >= 1 || error("--steps must be at least 1")
    return opts
end

mib(x) = x / 2^20
gib(x) = x / 2^30

pool() = CORE.pool_create(CUDA.device())
used_now() = (CUDA.synchronize(); Int(CORE.attribute(UInt64, pool(), CORE.MEMPOOL_ATTR_USED_MEM_CURRENT)))
peak_now() = (CUDA.synchronize(); Int(CORE.attribute(UInt64, pool(), CORE.MEMPOOL_ATTR_USED_MEM_HIGH)))
reset_peak!() = (CUDA.synchronize(); CORE.attribute!(pool(), CORE.MEMPOOL_ATTR_USED_MEM_HIGH, UInt64(0)); nothing)

"""Free the pool completely, so the next measurement starts from the model alone."""
function clean!()
    GC.gc(true)
    CUDA.reclaim()
    return nothing
end

"""Bytes of the arrays in a (nested) structure."""
function nbytes(x)
    total = 0
    Flux.fmap(x) do a
        a isa AbstractArray && (total += sizeof(a))
        a
    end
    return total
end

"""
One measurement: build the model (`blocks` time blocks), run `steps` timed training steps and one evaluation forward pass.
Returns a named tuple, or one with `oom = true` when the batch does not fit.
"""
function profile_once(mcfg::ForecasterConfig, cache::MinuteCache, cols::Vector{Int}, starts::UnitRange{Int},
                      batch::Int, micro::Int, dtype::Type, loss::Symbol, nu, beta, steps::Int, rng)
    clean!()
    base_used = used_now()
    try
        model = Flux.gpu(Forecaster(mcfg; seed=1))
        master = model
        net = dtype === Float32 ? master : convert_forecaster(master, dtype)
        opt = Flux.setup(Flux.OptimiserChain(Flux.ClipNorm(1f0), Flux.AdamW(1f-4)), master)
        n_param = n_parameters(master)
        params_bytes = nbytes(master)
        loss_scale = sizeof(dtype) < 4 ? 1024f0 : 1f0
        draw() = begin
            b = make_batch(cache, cols, rand(rng, starts, batch), mcfg.window)
            (Flux.gpu(dtype.(b.X)), Flux.gpu(b.Y), Flux.gpu(b.W), Flux.gpu(b.T))
        end
        chunk_size = cld(batch, micro)

        step_peak = 0
        times = Float64[]
        for k in 1:(WARMUP_STEPS + steps)
            chunks = Iterators.partition(rand(rng, starts, batch), chunk_size)
            bs = [make_batch(cache, cols, collect(c), mcfg.window) for c in chunks]
            Flux.trainmode!(net)
            GC.gc(true)
            reset_peak!()
            t0 = time()
            acc = accumulated_gradient(net, bs, Flux.gpu, dtype; loss=loss, nu=nu, beta=beta, loss_scale=loss_scale)
            Flux.update!(opt, master, acc.grads)
            net = dtype === Float32 ? master : convert_forecaster(master, dtype)
            CUDA.synchronize()
            dt = time() - t0
            if k > WARMUP_STEPS
                push!(times, dt)
                step_peak = max(step_peak, peak_now())
            end
            acc = nothing
        end

        retained = 0; fwd_peak = 0
        Flux.trainmode!(net)
        for _ in 1:2
            b = make_batch(cache, cols, rand(rng, starts, chunk_size), mcfg.window)
            X, Y, W, T = Flux.gpu(dtype.(b.X)), Flux.gpu(b.Y), Flux.gpu(b.W), Flux.gpu(b.T)
            GC.gc(true)
            reset_peak!()
            u0 = used_now()
            res, pb = Flux.Zygote.pullback(net) do m
                forecast_loss(m, X, Y, W, T; loss=loss, nu=nu, beta=beta)[1]
            end
            retained = used_now() - u0
            fwd_peak = peak_now() - u0
            pb = nothing; X = Y = W = T = nothing
        end

        Flux.testmode!(net)
        X, _, _, _ = draw()
        GC.gc(true)
        reset_peak!()
        u0 = used_now()
        mu, ls, tm = net(X)
        CUDA.synchronize()
        eval_peak = peak_now() - u0
        mu = ls = tm = X = nothing
        return (oom=false, batch=batch, micro_batches=micro, blocks=mcfg.blocks, params=n_param, params_mib=mib(params_bytes),
                base_mib=mib(base_used), retained_mib=mib(retained), forward_peak_mib=mib(fwd_peak),
                step_peak_mib=mib(step_peak), eval_peak_mib=mib(eval_peak),
                step_secs=median(times), step_secs_min=minimum(times))
    catch e
        e isa CUDA.OutOfGPUMemoryError || rethrow()
        return (oom=true, batch=batch, blocks=mcfg.blocks)
    end
end

function main()
    opts = parse_args()
    CUDA.functional() || error("memory_profile.jl needs a working CUDA GPU")
    free, total = CUDA.memory_info()
    @info "GPU: $(CUDA.name(CUDA.device())), $(@sprintf("%.1f", gib(free))) of $(@sprintf("%.1f", gib(total))) GiB free"
    if free < MIN_FREE_FRACTION * total && !opts["force"]
        error("only $(@sprintf("%.1f", gib(free))) GiB of the GPU is free: another process is using it (a running training?). " *
              "Stop it first, or pass --force (the other process may then run out of memory).")
    end

    haskey(GRANULARITIES, opts["granularity"]) || error("--granularity must be one of $(join(sort(collect(keys(GRANULARITIES))), ", "))")
    cache_path = opts["out"] === nothing ? normpath(joinpath(DATA_DIR, "forecaster_cache_$(opts["granularity"]).bson")) : abspath(joinpath(REPO_ROOT, opts["out"]))
    isfile(cache_path) || error("Not found: $cache_path\nBuild it: julia -t auto --project=packages/TradingGame scripts/forecaster/build_forecaster_cache.jl --granularity $(opts["granularity"])")
    cache = load_minute_cache(cache_path)
    universe = load_universe_snapshot(UNIVERSE_FILE)
    idx = Dict(s => i for (i, s) in enumerate(cache.symbols))
    gone = [s for s in universe.train if !haskey(idx, s)]
    isempty(gone) || error("$(length(gone)) train companies are not in the cache (e.g. $(first(gone)))")
    cols = [idx[s] for s in universe.train]
    opts["n_stocks"] > 0 && (cols = cols[1:min(opts["n_stocks"], length(cols))])
    M = opts["window"]
    starts = window_starts(cache, first(cache.dates), last(cache.dates), M)
    isempty(starts) && error("no window of $M bars fits in the cache")

    dtype = forecast_dtype(opts["dtype"])
    loss = forecast_loss_kind(opts["loss"])
    rng = MersenneTwister(1)
    cfg_for(blocks) = ForecasterConfig(window=M, d=opts["d"], heads=opts["heads"], blocks=blocks,
                                       dropout=Float32(opts["dropout"]), bars_per_day=cache.bars_per_day)
    N = length(cols)
    @info "Model: window $M, d $(opts["d"]), $(opts["heads"]) heads, $(opts["blocks"]) blocks, dropout $(opts["dropout"]), $(opts["dtype"]); $N stocks, cache $(cache.granularity)"

    runs = []
    for b in opts["batches"]
        r = profile_once(cfg_for(opts["blocks"]), cache, cols, starts, b, opts["micro"], dtype, loss, opts["nu"], opts["nll_beta"], opts["steps"], rng)
        push!(runs, r)
        r.oom ? @info("batch $b: out of GPU memory") :
                @info(@sprintf("batch %d: step peak %.0f MiB, %.3f s/step", b, r.step_peak_mib, r.step_secs))
    end

    breakdown = nothing
    if opts["breakdown"] && !runs[1].oom
        b = opts["batches"][1]
        r1 = profile_once(cfg_for(1), cache, cols, starts, b, opts["micro"], dtype, loss, opts["nu"], opts["nll_beta"], opts["steps"], rng)
        r2 = profile_once(cfg_for(2), cache, cols, starts, b, opts["micro"], dtype, loss, opts["nu"], opts["nll_beta"], opts["steps"], rng)
        if !r1.oom && !r2.oom
            blk_ret = r2.retained_mib - r1.retained_mib; blk_peak = r2.step_peak_mib - r1.step_peak_mib; blk_secs = r2.step_secs - r1.step_secs
            breakdown = (batch=b, mixing_retained_mib=r1.retained_mib - blk_ret, per_block_retained_mib=blk_ret,
                         mixing_step_peak_mib=r1.step_peak_mib - blk_peak, per_block_step_peak_mib=blk_peak,
                         mixing_step_secs=r1.step_secs - blk_secs, per_block_step_secs=blk_secs)
        end
    end

    B = cld(runs[1].batch, opts["micro"])
    seqs = N * B
    score_mib = mib(seqs * opts["heads"] * M * M * sizeof(dtype))
    mix_mib = mib(M * B * opts["heads"] * N * N * sizeof(dtype))
    token_mib = mib(seqs * M * opts["d"] * sizeof(dtype))
    println()
    println("v4 forecaster memory profile — ", Dates.format(now(), "yyyy-mm-dd HH:MM"))
    println("  $(opts["dtype"]), window $M, d $(opts["d"]), $(opts["heads"]) heads, $(opts["blocks"]) blocks, $N stocks, $(cache.granularity) cache")
    println()
    @printf("  %-6s %9s %12s %12s %12s %12s %10s\n", "batch", "params", "retained", "fwd peak", "step peak", "eval peak", "s/step")
    for r in runs
        if r.oom
            @printf("  %-6d %9s %12s\n", r.batch, "-", "OOM")
        else
            @printf("  %-6d %9d %9.0f MiB %9.0f MiB %9.0f MiB %9.0f MiB %10.3f\n", r.batch, r.params, r.retained_mib,
                    r.forward_peak_mib, r.step_peak_mib, r.eval_peak_mib, r.step_secs)
        end
    end
    println()
    println("  retained = kept by the forward pass for the backward pass (per micro-batch); fwd peak = highest during the forward pass;")
    println("  step peak = highest over forward, backward and update (model and optimiser state included); all are pool bytes in use.")
    println()
    @printf("  Shapes at micro-batch %d: one time-attention score tensor %.0f MiB (%d sequences × %d heads × %d × %d), one mixing score tensor %.0f MiB, one d-wide token tensor %.0f MiB\n",
            B, score_mib, seqs, opts["heads"], M, M, mix_mib, token_mib)
    if breakdown !== nothing
        @printf("  Breakdown at batch %d: embedding, stock mixing and heads %.0f MiB retained, %.0f MiB peak; each time block +%.0f MiB retained, +%.0f MiB peak, +%.3f s/step\n",
                breakdown.batch, breakdown.mixing_retained_mib, breakdown.mixing_step_peak_mib,
                breakdown.per_block_retained_mib, breakdown.per_block_step_peak_mib, breakdown.per_block_step_secs)
        if !runs[1].oom
            @printf("  A block retains %.1f score tensors' worth of memory\n", breakdown.per_block_retained_mib / score_mib)
        end
    end
    @printf("  Host: peak resident set %.0f MiB\n", mib(Sys.maxrss()))

    if opts["json"] !== nothing
        mkpath(dirname(opts["json"]))
        open(opts["json"], "w") do io
            JSON3.pretty(io, Dict("when" => string(now()), "gpu" => CUDA.name(CUDA.device()), "gpu_total_mib" => mib(total),
                "config" => Dict("window" => M, "d" => opts["d"], "heads" => opts["heads"], "blocks" => opts["blocks"],
                                 "dropout" => opts["dropout"], "dtype" => opts["dtype"], "loss" => opts["loss"],
                                 "stocks" => N, "granularity" => cache.granularity),
                "runs" => runs, "breakdown" => breakdown))
        end
        println("  Written: ", opts["json"])
    end
end

main()
