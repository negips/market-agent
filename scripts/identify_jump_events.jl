"""
identify_jump_events.jl

Two-stage jump detection on hourly OHLCV data:

  Stage 1 — BNS (Barndorff-Nielsen & Shephard 2006):
    Rolling W-day window. Tests whether a jump occurred *somewhere* in the
    window. Attributes it to the day with the largest absolute daily return.

  Stage 2 — Lee-Mykland (2008):
    Applied globally to every hourly bar. For each bar i, computes a local
    standardised return L(i) = r_i / ŝ_i where ŝ_i is a BV-based spot
    volatility from the K preceding bars. Critical values come from the
    Gumbel extreme-value distribution (joint test over all n bars). Gives
    the exact bar and timestamp of the jump.

  Combined output:
    For each BNS-detected jump day, the LM bar with the highest |L| within
    that window is shown alongside the BNS statistics. If no LM bar clears
    the threshold in the window the jump is still reported (BNS only).
    The "type" field distinguishes overnight gaps (bar 1 of the day, likely
    a catalyst after market close) from intraday moves.

## Usage

  julia --project=packages/StockSwingPredictor scripts/identify_jump_events.jl RELIANCE
  julia --project=packages/StockSwingPredictor scripts/identify_jump_events.jl RELIANCE --window 15 --alpha 0.01
  julia --project=packages/StockSwingPredictor scripts/identify_jump_events.jl RELIANCE --lm-K 14 --alpha 0.05 --min-return 0.02
"""

using CSV, DataFrames, Dates, Statistics, Printf, JSON3

const REPO_ROOT = joinpath(@__DIR__, "..")
const OHLCV_DIR = joinpath(REPO_ROOT, "website", "data", "ohlcv")

# ── Shared constants ──────────────────────────────────────────────────────────

# BV scaling: 1/μ₁² = π/2  where μ₁ = E[|Z|] = √(2/π)
const BV_SCALE = Float64(π / 2)

# TPQ scaling: 1/μ_{4/3}³ ≈ 1.7429
const TPQ_SCALE = 1.7429

# BNS asymptotic variance constant: Φ = π²/4 + π − 5 ≈ 0.6090
const Φ = π^2/4 + π - 5.0

const Z_CRIT = Dict(0.10 => 1.282, 0.05 => 1.645, 0.01 => 2.326, 0.001 => 3.090)

# ── Argument parsing ──────────────────────────────────────────────────────────

function parse_args()
    opts = Dict{String,Any}(
        "symbol"     => "",
        "window"     => 10,
        "lm_K"       => 7,
        "alpha"      => 0.01,
        "min_return" => 0.01,
    )
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a in ("-h", "--help")
            println("""
Usage:
  julia --project=packages/StockSwingPredictor scripts/identify_jump_events.jl SYMBOL [options]

Arguments:
  SYMBOL           NSE symbol (e.g. RELIANCE, INFY, HDFCBANK)

Options:
  --window N       BNS: trading days per rolling window (default: 10, ~70 returns)
  --lm-K N         LM:  preceding bars for spot-vol estimate (default: 7 = 1 day)
  --alpha FLOAT    Significance level for both tests: 0.10, 0.05, 0.01, 0.001 (default: 0.01)
  --min-return F   Minimum |daily log-return| to report (default: 0.01 = 1%)

Output:
  Console table + website/data/jump_events/{SYMBOL}_jumps.json
""")
            exit(0)
        elseif a == "--window"     ; opts["window"]      = parse(Int,     ARGS[i+1]); i += 2
        elseif a == "--lm-K"       ; opts["lm_K"]        = parse(Int,     ARGS[i+1]); i += 2
        elseif a == "--alpha"      ; opts["alpha"]        = parse(Float64, ARGS[i+1]); i += 2
        elseif a == "--min-return" ; opts["min_return"]   = parse(Float64, ARGS[i+1]); i += 2
        elseif !startswith(a, "-") ; opts["symbol"] = uppercase(strip(a));             i += 1
        else i += 1
        end
    end
    isempty(opts["symbol"]) && error("Symbol required. Run with --help for usage.")
    haskey(Z_CRIT, opts["alpha"]) ||
        error("--alpha must be one of: $(join(sort(collect(keys(Z_CRIT))), ", "))")
    return opts
end

# ── BNS functions ─────────────────────────────────────────────────────────────

function realized_variance(r::AbstractVector{<:Real})::Float64
    sum(x^2 for x in r)
end

function bipower_variation(r::AbstractVector{<:Real})::Float64
    length(r) < 2 && return 0.0
    BV_SCALE * sum(abs(r[i]) * abs(r[i-1]) for i in 2:length(r))
end

function tripower_quarticity(r::AbstractVector{<:Real})::Float64
    length(r) < 3 && return 0.0
    TPQ_SCALE * sum(abs(r[i])^(4/3) * abs(r[i-1])^(4/3) * abs(r[i-2])^(4/3)
                    for i in 3:length(r))
end

"""
BNS test statistic and decomposition for a vector of log-returns.
Returns (rv, bv, j, j_frac, z).
"""
function bns_test(r::AbstractVector{<:Real})
    n  = length(r)
    rv = realized_variance(r)
    bv = bipower_variation(r)
    rv <= 0.0 && return (rv=0.0, bv=0.0, j=0.0, j_frac=0.0, z=0.0)
    tpq    = tripower_quarticity(r)
    j      = max(rv - bv, 0.0)
    j_frac = j / rv
    var_rj = Φ / n * max(1.0, tpq / max(bv^2, 1e-20))
    z      = (rv - bv) / (rv * sqrt(max(var_rj, 1e-20)))
    return (rv=rv, bv=bv, j=j, j_frac=j_frac, z=z)
end

# ── Lee-Mykland functions ─────────────────────────────────────────────────────

"""
Gumbel extreme-value critical value for the maximum of |L(i)| over n bars.

Lee & Mykland (2008): under H₀ the maximum of the standardised bar returns
converges to a Gumbel distribution with location c_n and scale s_n.
A bar whose |L(i)| exceeds this threshold is flagged as a jump at level α.
"""
function lm_critical_value(n::Int, alpha::Float64)::Float64
    log2n = 2.0 * log(n)
    c_n   = sqrt(log2n) - (log(π) + log(log(n))) / (2.0 * sqrt(log2n))
    s_n   = 1.0 / sqrt(log2n)
    return c_n - s_n * log(-log(1.0 - alpha))
end

"""
BV-based spot volatility estimate for bar i using the K returns preceding it.
Returns NaN if there are fewer than 2 adjacent pairs in the window.
"""
function spot_vol(r::AbstractVector{<:Real}, i::Int, K::Int)::Float64
    lo = max(1, i - K)
    hi = i - 1
    hi < lo + 1 && return NaN
    bv = BV_SCALE * sum(abs(r[j]) * abs(r[j-1]) for j in lo+1:hi)
    bv <= 0.0 && return NaN
    return sqrt(bv / (hi - lo))
end

"""
Apply the Lee-Mykland test globally to all bars in `r`.

Returns (hits, cv) where `hits` is a vector of NamedTuples
(idx, l_stat, sigma_hat) for every bar whose |L(i)| exceeds the Gumbel
critical value, and `cv` is that critical value.
"""
function lee_mykland(r::AbstractVector{<:Real}; K::Int=7, alpha::Float64=0.01)
    n  = length(r)
    cv = lm_critical_value(n, alpha)
    hits = NamedTuple[]
    for i in (K+2):n
        σ̂ = spot_vol(r, i, K)
        isnan(σ̂) && continue
        L  = r[i] / σ̂
        abs(L) >= cv || continue
        push!(hits, (idx=i, l_stat=L, sigma_hat=σ̂))
    end
    return hits, cv
end

# ── Main ──────────────────────────────────────────────────────────────────────

function main()
    opts   = parse_args()
    sym    = opts["symbol"]
    W      = opts["window"]
    K      = opts["lm_K"]
    alpha  = opts["alpha"]
    z_crit = Z_CRIT[alpha]

    # ── Load hourly data ──────────────────────────────────────────────────────

    path = joinpath(OHLCV_DIR, "$(sym)_hourly.csv")
    isfile(path) || error("Hourly data not found: $path")

    df = CSV.read(path, DataFrame; types=Dict(:datetime => DateTime))
    sort!(df, :datetime)
    nrow(df) >= 2 || error("Not enough data in $path")

    @info "$(sym): $(nrow(df)) hourly bars  ($(df.datetime[1]) → $(df.datetime[end]))"

    # ── Compute returns and bar metadata ─────────────────────────────────────

    n_returns      = nrow(df) - 1
    log_returns    = [log(df.close[i] / df.close[i-1]) for i in 2:nrow(df)]
    bar_datetimes  = [df.datetime[i]  for i in 2:nrow(df)]
    bar_dates      = [Date(dt)        for dt in bar_datetimes]

    # Intraday bar index (1 = first bar of that trading day)
    intraday_idx   = zeros(Int, n_returns)
    day_count      = Dict{Date, Int}()
    for i in 1:n_returns
        d = bar_dates[i]
        day_count[d] = get(day_count, d, 0) + 1
        intraday_idx[i] = day_count[d]
    end

    # Total bars per trading day (for "bar X/N" display)
    bars_per_day   = Dict{Date, Int}(d => c for (d, c) in day_count)

    # Close-to-close daily returns and last close per day
    daily_close    = Dict{Date, Float64}()
    for row in eachrow(df)
        daily_close[Date(row.datetime)] = row.close
    end
    trading_days = sort(collect(keys(daily_close)))
    n_days       = length(trading_days)

    daily_return = Dict{Date, Float64}()
    for i in 2:n_days
        daily_return[trading_days[i]] =
            log(daily_close[trading_days[i]] / daily_close[trading_days[i-1]])
    end

    # Group return indices by date
    idx_by_date = Dict{Date, Vector{Int}}()
    for i in 1:n_returns
        push!(get!(idx_by_date, bar_dates[i], Int[]), i)
    end

    avg_bars = round(Int, n_returns / n_days)
    @info "BNS window: $W days (~$(W * avg_bars) returns) | LM K=$K | α=$alpha"

    # ── Stage 1: BNS rolling window ───────────────────────────────────────────

    raw_bns = NamedTuple[]

    for t in W:n_days
        window_days = trading_days[t-W+1:t]
        window_idx  = reduce(vcat, [get(idx_by_date, d, Int[]) for d in window_days])
        length(window_idx) < 10 && continue

        window_returns = log_returns[window_idx]
        stat = bns_test(window_returns)
        stat.z <= z_crit && continue

        # Attribute to the day with the largest |daily return| in the window
        jump_day = window_days[1]; jump_ret = 0.0
        for d in window_days
            r = get(daily_return, d, 0.0)
            abs(r) > abs(jump_ret) && (jump_day = d; jump_ret = r)
        end

        push!(raw_bns, (
            window_start = window_days[1],
            window_end   = window_days[end],
            jump_day     = jump_day,
            daily_return = jump_ret,
            z            = stat.z,
            j_frac       = stat.j_frac,
            rv           = stat.rv,
            bv           = stat.bv,
            n_obs        = length(window_returns),
        ))
    end

    # Deduplicate: keep highest-z detection per jump day
    best_bns = Dict{Date, eltype(raw_bns)}()
    for d in raw_bns
        if !haskey(best_bns, d.jump_day) || d.z > best_bns[d.jump_day].z
            best_bns[d.jump_day] = d
        end
    end

    bns_events = sort(collect(values(best_bns)), by = e -> e.jump_day)
    bns_events = filter(e -> abs(e.daily_return) >= opts["min_return"], bns_events)

    # ── Stage 2: Lee-Mykland global bar-level test ────────────────────────────

    lm_hits, lm_cv = lee_mykland(log_returns; K=K, alpha=alpha)

    # Index LM hits by date for fast lookup
    lm_by_date = Dict{Date, Vector{eltype(lm_hits)}}()
    for h in lm_hits
        d = bar_dates[h.idx]
        push!(get!(lm_by_date, d, eltype(lm_hits)[]), h)
    end

    @info "LM: $(length(lm_hits)) bar-level hits  (cv=$(round(lm_cv, digits=3)))"

    # ── Merge: for each BNS event, find best LM bar within its window ─────────

    events = map(bns_events) do e
        # Collect all LM hits whose date falls in the BNS window
        window_lm = NamedTuple[]
        for d in e.window_start:Day(1):e.window_end
            append!(window_lm, get(lm_by_date, d, eltype(lm_hits)[]))
        end

        best_lm = isempty(window_lm) ? nothing :
                  window_lm[argmax(abs(h.l_stat) for h in window_lm)]

        lm_datetime = isnothing(best_lm) ? missing : bar_datetimes[best_lm.idx]
        lm_bar_idx  = isnothing(best_lm) ? missing : intraday_idx[best_lm.idx]
        lm_bar_tot  = isnothing(best_lm) ? missing : bars_per_day[bar_dates[best_lm.idx]]
        lm_l_stat   = isnothing(best_lm) ? missing : best_lm.l_stat
        lm_return   = isnothing(best_lm) ? missing : log_returns[best_lm.idx]
        lm_type     = isnothing(best_lm) ? missing :
                      (lm_bar_idx == 1 ? "overnight gap" : "intraday")

        merge(e, (
            lm_datetime = lm_datetime,
            lm_bar_idx  = lm_bar_idx,
            lm_bar_tot  = lm_bar_tot,
            lm_l_stat   = lm_l_stat,
            lm_return   = lm_return,
            lm_type     = lm_type,
        ))
    end

    # ── Print results ─────────────────────────────────────────────────────────

    println()
    println("═" ^ 88)
    @printf("  BNS + Lee-Mykland Jump Events: %s  |  α=%.3f  |  BNS W=%d  |  LM K=%d\n",
            sym, alpha, W, K)
    println("═" ^ 88)
    println()
    @printf("  %-12s  %-9s  %-17s  %-5s  %-9s  %-8s  %-8s  %-15s\n",
            "Jump Day", "Daily Ret", "LM Bar Datetime", "Bar", "LM stat",
            "BNS z", "J/RV", "Type")
    println("  " * "─" ^ 84)

    for e in events
        lm_dt_str  = ismissing(e.lm_datetime) ? "—" :
                     string(Date(e.lm_datetime)) * " " *
                     string(Time(e.lm_datetime))[1:5]
        lm_bar_str = ismissing(e.lm_bar_idx)  ? "—" :
                     "$(e.lm_bar_idx)/$(e.lm_bar_tot)"
        lm_L_str   = ismissing(e.lm_l_stat)   ? "—" :
                     @sprintf("%+.2f", e.lm_l_stat)
        type_str   = ismissing(e.lm_type)      ? "BNS only" : e.lm_type

        @printf("  %-12s  %+8.2f%%  %-17s  %-5s  %-9s  %6.2f  %6.1f%%  %s\n",
                e.jump_day, e.daily_return * 100,
                lm_dt_str, lm_bar_str, lm_L_str,
                e.z, e.j_frac * 100, type_str)
    end

    println()
    n_both   = count(e -> !ismissing(e.lm_l_stat), events)
    n_bns    = length(events) - n_both
    pct_days = length(events) / n_days * 100
    @printf("  %d jump events (%.1f%% of trading days) — %d confirmed by both tests, %d BNS-only\n",
            length(events), pct_days, n_both, n_bns)

    if !isempty(events)
        zs   = [e.z           for e in events]
        rets = [e.daily_return for e in events]
        jfs  = [e.j_frac       for e in events]
        lm_confirmed = filter(e -> !ismissing(e.lm_type), events)
        n_gap      = count(e -> e.lm_type == "overnight gap", lm_confirmed)
        n_intraday = count(e -> e.lm_type == "intraday",      lm_confirmed)
        println()
        @printf("  BNS z:     mean=%.2f  max=%.2f\n",   mean(zs),        maximum(zs))
        @printf("  |return|:  mean=%.2f%%  max=%.2f%%\n", mean(abs.(rets))*100, maximum(abs.(rets))*100)
        @printf("  J/RV:      mean=%.1f%%  max=%.1f%%\n", mean(jfs)*100,   maximum(jfs)*100)
        @printf("  Jump type: %d overnight gaps  |  %d intraday\n", n_gap, n_intraday)
    end
    println()

    # ── Save JSON ─────────────────────────────────────────────────────────────

    out_dir  = joinpath(REPO_ROOT, "website", "data", "jump_events")
    mkpath(out_dir)
    out_path = joinpath(out_dir, "$(sym)_jumps.json")

    json_out = Dict(
        "symbol"          => sym,
        "alpha"           => alpha,
        "bns_window_days" => W,
        "lm_K"            => K,
        "bns_z_critical"  => z_crit,
        "lm_cv"           => round(lm_cv, digits=4),
        "n_events"        => length(events),
        "n_trading_days"  => n_days,
        "events"          => [Dict(
            "date"           => string(e.jump_day),
            "daily_return"   => round(e.daily_return, digits=6),
            "bns_z"          => round(e.z,     digits=4),
            "bns_j_fraction" => round(e.j_frac, digits=4),
            "lm_datetime"    => ismissing(e.lm_datetime) ? nothing : string(e.lm_datetime),
            "lm_bar"         => ismissing(e.lm_bar_idx)  ? nothing :
                                "$(e.lm_bar_idx)/$(e.lm_bar_tot)",
            "lm_stat"        => ismissing(e.lm_l_stat)   ? nothing :
                                round(e.lm_l_stat, digits=4),
            "lm_bar_return"  => ismissing(e.lm_return)   ? nothing :
                                round(e.lm_return, digits=6),
            "type"           => ismissing(e.lm_type) ? "bns_only" : e.lm_type,
        ) for e in events],
    )
    open(out_path, "w") do io; JSON3.pretty(io, json_out); end
    @info "Saved → $out_path"
end

main()
