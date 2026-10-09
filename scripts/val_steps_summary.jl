"""
val_steps_summary.jl

Summarise a validation run's per-decision-step log (`val_steps/iter_NNNNN.bson`,
written by `train_trading_policy.jl`) to see what the policy is doing: average
action probabilities overall, split by whether the stock is held, by time of
day (15-minute slot), the critic's value against what actually followed, and how
often moves were illegal or masked.

Usage:
  julia --project=packages/TradingGame scripts/val_steps_summary.jl            # newest run
  julia --project=packages/TradingGame scripts/val_steps_summary.jl 42         # iteration 42
  julia --project=packages/TradingGame scripts/val_steps_summary.jl a.bson b.bson   # compare files
"""

using TradingGame, Statistics, Printf, Dates

const STEPS_DIR = joinpath(@__DIR__, "..", "website", "data", "trading_game", "val_steps")

function resolve_paths(args)::Vector{String}
    function in_dir(name)
        isdir(STEPS_DIR) || error("No per-step logs yet: $STEPS_DIR (train_trading_policy.jl writes them each validation run)")
        return joinpath(STEPS_DIR, name)
    end
    if isempty(args)
        isdir(STEPS_DIR) || in_dir("")
        files = sort(filter(f -> endswith(f, ".bson"), readdir(STEPS_DIR)))
        isempty(files) && error("No iter_*.bson files in $STEPS_DIR")
        return [joinpath(STEPS_DIR, files[end])]
    end
    return [isfile(a) ? a : (all(isdigit, a) ? in_dir("iter_" * lpad(a, 5, '0') * ".bson") : a) for a in args]
end

pct(x) = @sprintf("%5.1f%%", 100 * x)
avg(x) = isempty(x) ? NaN : mean(Float64.(x))

function entropy(p)   # p :: (3, N, T) probabilities → mean per-stock entropy in nats
    h = -sum(q -> q > 0 ? q * log(q) : 0.0, Float64.(p); dims=1)
    return mean(h)
end

function summarize(path::String)
    s = load_val_steps(path)
    N, T = size(s.action)
    p = s.probs
    println("═"^72)
    @printf("%s\n  iteration %s · val return %.4f · %d stocks · %d decision bars (%s … %s)\n", basename(path),
            get(s.meta, "iteration", "?"), get(s.meta, "val_return", NaN), N, T, first(s.datetimes), last(s.datetimes))

    println("\nAverage probabilities (after the sell mask)")
    @printf("  all stock-bars    HOLD %s  SELL %s  BUY %s   entropy %.3f nats (max %.3f)\n",
            pct(avg(p[1, :, :])), pct(avg(p[2, :, :])), pct(avg(p[3, :, :])), entropy(p), log(3))
    for (label, mask) in (("held stocks", s.held), ("not held", .!s.held))
        any(mask) || continue
        @printf("  %-17s HOLD %s  SELL %s  BUY %s   (%d stock-bars)\n", label,
                pct(avg(p[1, :, :][mask])), pct(avg(p[2, :, :][mask])), pct(avg(p[3, :, :][mask])), count(mask))
    end
    held_sellable = s.held .& s.sell_ok
    any(held_sellable) && @printf("  held & sellable   SELL %s   (raw SELL before masking, all stock-bars: %s)\n",
                                  pct(avg(p[2, :, :][held_sellable])), pct(avg(s.p_sell_raw)))
    @printf("  sell masked out in %s of stock-bars\n", pct(1 - avg(s.sell_ok)))

    println("\nWhat it chose (argmax)")
    for (k, name) in ((1, "HOLD"), (2, "SELL"), (3, "BUY"))
        @printf("  %-5s %s of stock-bars\n", name, pct(avg(s.action .== k)))
    end
    @printf("  bars with a trade: %d of %d · trades %d · bars charged for illegal moves: %d (total penalty %.3f)\n",
            count(s.n_trades .> 0), T, sum(s.n_trades), count(s.illegal_penalty .> 0), sum(s.illegal_penalty))

    println("\nBook")
    @printf("  portfolio value %.0f → %.0f · cash share avg %s · holdings avg %.1f (max %d)\n", first(s.portfolio_value),
            last(s.portfolio_value), pct(avg(s.cash ./ s.portfolio_value)), avg(s.n_holdings), maximum(s.n_holdings))

    println("\nCritic")
    ret = cumsum(Float64.(s.reward))
    future = [sum(@view s.reward[t:min(t + 24, T)]) for t in 1:T]      # reward over the next ~day (25 bars)
    @printf("  V(s) mean %.4f · std %.4f · corr(V, next-day reward) %.3f\n", avg(s.value), std(Float64.(s.value)),
            T > 2 ? cor(Float64.(s.value), future) : NaN)

    println("\nBy time of day (15-minute slot)")
    println("  slot    P(BUY)  P(SELL)  chose BUY  chose SELL")
    slots = [Dates.format(DateTime(d), "HH:MM") for d in s.datetimes]
    for sl in sort(unique(slots))
        idx = findall(==(sl), slots)
        @printf("  %s  %s  %s    %s     %s\n", sl, pct(avg(p[3, :, idx])), pct(avg(p[2, :, idx])),
                pct(avg(s.action[:, idx] .== 3)), pct(avg(s.action[:, idx] .== 2)))
    end

    println("\nPer stock (most bought first)")
    order = sortperm([avg(s.action[i, :] .== 3) for i in 1:N]; rev=true)
    for i in order[1:min(8, N)]
        @printf("  %-12s P(BUY) %s  chose BUY %s  held %s of bars\n", s.symbols[i], pct(avg(p[3, i, :])),
                pct(avg(s.action[i, :] .== 3)), pct(avg(s.held[i, :])))
    end
end

for path in resolve_paths(ARGS)
    isfile(path) || error("Not found: $path")
    summarize(path)
end
