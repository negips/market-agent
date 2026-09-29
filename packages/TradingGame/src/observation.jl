"""
Observation tensor assembly from `InferenceCache` + macro OHLCV + portfolio
state. All slicing is O(1) against already-loaded matrices — no CSV/network
I/O happens per call, matching the `inference_cache.jl`/`dataset.jl` discipline
in `StockSwingPredictor`.

News features are a neutral zero until `news_features.jl` (Stage 0/2 backfill)
lands — see `_zero_news`.
"""

using Dates, DataFrames
using StockSwingPredictor: load_macro_ohlcv

# ── Macro cache: aligned daily macro matrix, built once ───────────────────────────

"""Aligned daily macro-series matrix, mirroring `InferenceCache`'s pattern for
equities: build once from the OHLCV CSVs `StockSwingPredictor.macro_data.jl`
already produces, then slice in O(1) per observation."""
struct MacroCache
    closes     :: Matrix{Float32}   # (n_dates, N_MACRO_SERIES), column order == MACRO_SERIES_NAMES
    dates      :: Vector{Date}
    date_index :: Dict{Date, Int}
end

"""
Build a `MacroCache` from the daily OHLCV CSVs in `out_dir` (default:
`website/data/ohlcv/macro`, via `StockSwingPredictor.load_macro_ohlcv`).
Missing instruments/dates are left `NaN` then forward-filled per series;
a series with no cached CSV at all stays `NaN` throughout (read as "unavailable"
by `_macro_context`, which then contributes zeros for that series).
"""
function build_macro_cache(out_dir::String)::MacroCache
    per_series = Dict{String, DataFrame}()
    all_dates  = Set{Date}()
    for name in MACRO_SERIES_NAMES
        df = load_macro_ohlcv(name; out_dir=out_dir)
        per_series[name] = df
        isempty(df) || union!(all_dates, df.date)
    end

    dates      = sort(collect(all_dates))
    n_dates    = length(dates)
    date_index = Dict(d => i for (i, d) in enumerate(dates))
    closes     = fill(NaN32, n_dates, N_MACRO_SERIES)

    for (j, name) in enumerate(MACRO_SERIES_NAMES)
        df = per_series[name]
        isempty(df) && continue
        for row in eachrow(df)
            i = get(date_index, row.date, 0)
            i == 0 && continue
            closes[i, j] = Float32(row.close)
        end
        last_c = NaN32
        for i in 1:n_dates
            if !isnan(closes[i, j])
                last_c = closes[i, j]
            elseif !isnan(last_c)
                closes[i, j] = last_c
            end
        end
    end

    return MacroCache(closes, dates, date_index)
end

"""Right-aligned, anchor-normalised `(N_MACRO_DAYS, N_MACRO_SERIES)` window
ending at or before `current_date`. Zero-padded at the front when less than
`N_MACRO_DAYS` of history exists; an all-`NaN` series (never cached) or a
missing cache (`nothing`) contributes zeros — the documented neutral default."""
function _macro_context(macro_cache::Union{Nothing, MacroCache}, current_date::Date)::Matrix{Float32}
    out = zeros(Float32, N_MACRO_DAYS, N_MACRO_SERIES)
    macro_cache === nothing && return out

    idx = searchsortedlast(macro_cache.dates, current_date)
    idx == 0 && return out

    lo = max(1, idx - N_MACRO_DAYS + 1)
    n  = idx - lo + 1
    for j in 1:N_MACRO_SERIES
        col = @view macro_cache.closes[lo:idx, j]
        anchor_i = findfirst(!isnan, col)
        anchor_i === nothing && continue
        anchor = col[anchor_i]
        anchor <= 0 && continue
        for i in 1:n
            v = col[i]
            out[N_MACRO_DAYS - n + i, j] = isnan(v) ? 1f0 : Float32(v / anchor)
        end
    end
    return out
end

# ── News features (neutral default until Stage 0/2 backfill) ─────────────────────

"""Neutral placeholder: no news signal for any candidate. Real recency-decayed
features (own-symbol + market-wide) land with `news_features.jl`; `assemble_observation`
accepts any `(env, sym_idx, hour_idx) -> Vector{Float32}` of length `N_NEWS_FEATURES`
via its `news_fn` keyword, so swapping this in later touches no other file."""
_zero_news(::TradingGameEnv, ::Int, ::Int) = zeros(Float32, N_NEWS_FEATURES)

# ── Observation ────────────────────────────────────────────────────────────────────

"""
One decision step's full observation, for the `N = length(candidates)`
candidates in `env.candidate_order`.
"""
struct Observation
    hourly     :: Array{Float32, 3}   # (N_HOURLY_BARS_SHORT, N_PRICE_CHANNELS, N)
    macro_ctx  :: Matrix{Float32}     # (N_MACRO_DAYS, N_MACRO_SERIES)
    news       :: Matrix{Float32}     # (N_NEWS_FEATURES, N)
    holding    :: Matrix{Float32}     # (N_HOLDING_FEATURES, N)
    portfolio  :: Vector{Float32}     # (N_PORTFOLIO_SCALARS,)
    candidates :: Vector{Int}         # sym_idx per column, == env.candidate_order
end

"""
Assemble the current observation for `env`. `macro_cache`/`news_fn` are
optional — omitting them yields the documented neutral defaults (all-zero
macro context / news features), which is exactly what Stage 1/2 rule and shape
tests use; real training runs pass a `build_macro_cache(...)` result and a
`news_features.jl` lookup once those exist.
"""
function assemble_observation(env::TradingGameEnv;
                               macro_cache::Union{Nothing, MacroCache}=nothing,
                               news_fn::Function=_zero_news)::Observation
    env.config === nothing && error("assemble_observation: call reset! before observing")

    N = length(env.candidate_order)
    t = env.current_hour_idx
    date_idx = env.cache.date_index[env.current_date]

    # ── Per-stock hourly sequence ────────────────────────────────────────────
    hourly = zeros(Float32, N_HOURLY_BARS_SHORT, N_PRICE_CHANNELS, N)
    lo = t - N_HOURLY_BARS_SHORT + 1
    for (col, sym_idx) in enumerate(env.candidate_order)
        if lo >= 1
            raw = @view env.cache.hourly_closes[lo:t, sym_idx]
            anchor_i = findfirst(!isnan, raw)
            anchor = anchor_i === nothing ? 1f0 : max(raw[anchor_i], 1f-6)
            for i in 1:N_HOURLY_BARS_SHORT
                c = raw[i]
                hourly[i, 1, col] = isnan(c) ? 1f0 : c / anchor
            end
        end
        # channels 2/3 are daily-granularity in InferenceCache (no intraday vol
        # series exists) — broadcast today's value across the hourly window.
        v  = env.cache.vols[date_idx, sym_idx]
        rv = env.cache.rel_vols[date_idx, sym_idx]
        hourly[:, 2, col] .= isnan(v)  ? 0f0 : v
        hourly[:, 3, col] .= isnan(rv) ? 1f0 : rv
    end

    # ── News (neutral zero by default) ───────────────────────────────────────
    news = zeros(Float32, N_NEWS_FEATURES, N)
    for (col, sym_idx) in enumerate(env.candidate_order)
        news[:, col] .= news_fn(env, sym_idx, t)
    end

    # ── Per-holding state — aggregated across concurrent lots of the same stock:
    #    held flag, quantity-weighted unrealised P&L, and the MOST URGENT lot's
    #    remaining-hold-budget fraction (the one that force-exits soonest). ────
    sym_col = Dict(sym_idx => col for (col, sym_idx) in enumerate(env.candidate_order))
    qty_sum  = zeros(Float64, N)
    val_sum  = zeros(Float64, N)
    cost_sum = zeros(Float64, N)
    min_remaining = fill(1f0, N)
    held = falses(N)
    for h in env.portfolio.holdings
        col = get(sym_col, h.sym_idx, 0)
        col == 0 && continue
        price = env.cache.hourly_closes[t, h.sym_idx]
        isnan(price) && continue
        held[col]      = true
        qty_sum[col]  += h.quantity
        val_sum[col]  += h.quantity * price
        cost_sum[col] += h.quantity * h.entry_price
        days_held = date_idx - h.entry_date_idx
        remaining = Float32(clamp((MAX_HOLD_DAYS - days_held) / MAX_HOLD_DAYS, 0, 1))
        min_remaining[col] = min(min_remaining[col], remaining)
    end
    holding = zeros(Float32, N_HOLDING_FEATURES, N)
    for col in 1:N
        holding[1, col] = held[col] ? 1f0 : 0f0
        holding[2, col] = (held[col] && cost_sum[col] > 0) ?
                           Float32((val_sum[col] - cost_sum[col]) / cost_sum[col]) : 0f0
        holding[3, col] = held[col] ? min_remaining[col] : 1f0
    end

    # ── Macro context ─────────────────────────────────────────────────────────
    macro_ctx = _macro_context(macro_cache, env.current_date)

    # ── Global portfolio scalars — all O(1)-scale ratios, never raw rupees.
    #    Feeding cash/value directly (routinely 1e5–1e7 ₹) into a Dense layer
    #    alongside O(1) price/news/holding features saturates it — every
    #    scalar here is normalised against `value` or `initial_cash` instead. ─
    value    = portfolio_value(env)
    reserved = isempty(env.portfolio.reserved) ? 0.0 : sum(l.amount for l in env.portfolio.reserved)
    cash_frac     = value > 0 ? Float32(env.portfolio.cash / value) : 0f0
    reserved_frac = value > 0 ? Float32(reserved / value) : 0f0
    stocks_frac   = clamp(1f0 - cash_frac - reserved_frac, 0f0, 1f0)
    value_ratio   = Float32(value / env.config.initial_cash)   # 1.0 = breakeven
    portfolio = Float32[cash_frac, reserved_frac, value_ratio, stocks_frac]

    return Observation(hourly, macro_ctx, news, holding, portfolio, copy(env.candidate_order))
end

"""
Stack several single-step `Observation`s (all from the same episode, so they
share `candidates`) into batched tensors ready for `ActorCriticPolicy`.

# Returns
Named tuple `(hourly, news, holding, macro_ctx, portfolio)` with a trailing
batch dimension `B = length(obs)` on every field.
"""
function stack_observations(obs::Vector{Observation})
    B = length(obs)
    B == 0 && error("stack_observations: empty batch")
    N = length(obs[1].candidates)
    for o in obs
        length(o.candidates) == N || error("stack_observations: candidate-set size mismatch across batch")
    end

    hourly    = cat((o.hourly for o in obs)...; dims=4)
    news      = cat((reshape(o.news, size(o.news)..., 1) for o in obs)...; dims=3)
    holding   = cat((reshape(o.holding, size(o.holding)..., 1) for o in obs)...; dims=3)
    macro_ctx = cat((reshape(o.macro_ctx, size(o.macro_ctx)..., 1) for o in obs)...; dims=3)
    portfolio = hcat((o.portfolio for o in obs)...)

    return (hourly=hourly, news=news, holding=holding, macro_ctx=macro_ctx, portfolio=portfolio)
end
