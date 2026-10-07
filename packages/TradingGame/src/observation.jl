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
candidates in `env.candidate_order`. Parametric over its field array types so
it can hold either plain `Array`s (the single-step, allocating
`assemble_observation` path — tests and simple callers) or `SubArray` views
into a preallocated per-episode tensor (`collect_rollout`'s path — see its
docstring) without sacrificing concrete, inferrable field types either way.
"""
struct Observation{H<:AbstractArray{Float32,3}, M<:AbstractMatrix{Float32},
                    NW<:AbstractMatrix{Float32}, HO<:AbstractMatrix{Float32},
                    P<:AbstractVector{Float32}}
    hourly     :: H    # (N_HOURLY_BARS_SHORT, N_PRICE_CHANNELS, N)
    macro_ctx  :: M    # (N_MACRO_DAYS, N_MACRO_SERIES)
    news       :: NW   # (N_NEWS_FEATURES, N)
    holding    :: HO   # (N_HOLDING_FEATURES, N)
    portfolio  :: P    # (N_PORTFOLIO_SCALARS,)
    candidates :: Vector{Int}   # sym_idx per column, == env.candidate_order
end

"""
Write the current observation for `env` into the given output arrays/views
in place — no allocation beyond whatever `macro_cache`/`news_fn` themselves
allocate internally (small, `N_MACRO_SERIES`/`N_NEWS_FEATURES`-sized, not
per-episode-retained). `assemble_observation` (below) is a thin allocating
wrapper around this for callers that just want one fresh `Observation`;
`collect_rollout` calls this directly against views into a preallocated
per-episode tensor instead — see its docstring for why.

Unlike the old single-array `assemble_observation`, callers must zero-init
`hourly`/`news`/`holding` themselves if reusing a buffer across calls (this
function only writes the entries it computes — channel 1 of `hourly` is left
untouched, not zeroed, when `lo < 1`, matching the old implicit-zero
behaviour only when the buffer started as `zeros`).
"""
function assemble_observation!(hourly::AbstractArray{Float32,3}, macro_ctx::AbstractMatrix{Float32},
                                news::AbstractMatrix{Float32}, holding::AbstractMatrix{Float32},
                                portfolio::AbstractVector{Float32}, env::TradingGameEnv;
                                macro_cache::Union{Nothing, MacroCache}=nothing,
                                news_fn::Function=_zero_news)::Nothing
    env.config === nothing && error("assemble_observation!: call reset! before observing")

    N = length(env.candidate_order)
    t = env.current_hour_idx
    date_idx = env.cache.date_index[env.current_date]
    rules    = env.config.rules
    max_hold = rules.max_hold_days
    length(portfolio) == n_portfolio_scalars(rules) ||
        error("assemble_observation!: portfolio vector has $(length(portfolio)) entries, " *
              "game v$(rules.version) needs $(n_portfolio_scalars(rules))")

    # ── Per-stock hourly sequence ────────────────────────────────────────────
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
            # The window's last bar (= the current bar, time `t`) reflects a
            # news-instant override when one exists, same anchor as the rest
            # of the window — so the policy sees the price it's about to
            # trade at, not the enclosing hour's close.
            ov = _price_override(env, sym_idx)
            ov !== nothing && (hourly[N_HOURLY_BARS_SHORT, 1, col] = ov / anchor)
        else
            hourly[:, 1, col] .= 0f0
        end
        # channel 2 is daily-granularity in InferenceCache (no intraday vol
        # series exists) — broadcast across the hourly window. Uses the
        # PREVIOUS trading day's (H-L)/C: today's daily bar isn't complete at
        # an intraday decision bar, so its range would leak the rest of the day.
        v = date_idx > 1 ? env.cache.vols[date_idx - 1, sym_idx] : NaN32
        hourly[:, 2, col] .= isnan(v) ? 0f0 : v
    end

    # ── News (neutral zero by default) ───────────────────────────────────────
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
        price = current_price(env, h.sym_idx)
        isnan(price) && continue
        held[col]      = true
        qty_sum[col]  += h.quantity
        val_sum[col]  += h.quantity * price
        cost_sum[col] += h.quantity * h.entry_price
        days_held = date_idx - h.entry_date_idx
        remaining = Float32(clamp((max_hold - days_held) / max_hold, 0, 1))
        min_remaining[col] = min(min_remaining[col], remaining)
    end
    for col in 1:N
        holding[1, col] = held[col] ? 1f0 : 0f0
        holding[2, col] = (held[col] && cost_sum[col] > 0) ?
                           Float32((val_sum[col] - cost_sum[col]) / cost_sum[col]) : 0f0
        holding[3, col] = held[col] ? min_remaining[col] : 1f0
    end

    # ── Macro context ─────────────────────────────────────────────────────────
    macro_ctx .= _macro_context(macro_cache, env.current_date)

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
    portfolio[1] = cash_frac
    portfolio[2] = reserved_frac
    portfolio[3] = value_ratio
    portfolio[4] = stocks_frac
    if rules.cash_token
        days_over = env.cash_over_since_date_idx > 0 ? date_idx - env.cash_over_since_date_idx : 0
        portfolio[5] = Float32(clamp(cash_frac / MAX_CASH_FRACTION, 0, 3))
        portfolio[6] = Float32(clamp(days_over / max_hold, 0, 1))
    end

    return nothing
end

"""
Assemble the current observation for `env`. `macro_cache`/`news_fn` are
optional — omitting them yields the documented neutral defaults (all-zero
macro context / news features), which is exactly what Stage 1/2 rule and shape
tests use; real training runs pass a `build_macro_cache(...)` result and a
`news_features.jl` lookup once those exist.

A thin allocating wrapper around `assemble_observation!` — fine for one-off
calls (tests, REPL exploration), but `collect_rollout` calls `assemble_observation!`
directly against a preallocated per-episode tensor instead, to avoid one
fresh heap allocation per field per decision bar over a multi-thousand-bar
episode (see `collect_rollout`'s docstring)."""
function assemble_observation(env::TradingGameEnv;
                               macro_cache::Union{Nothing, MacroCache}=nothing,
                               news_fn::Function=_zero_news)::Observation
    env.config === nothing && error("assemble_observation: call reset! before observing")
    N = length(env.candidate_order)
    hourly    = zeros(Float32, N_HOURLY_BARS_SHORT, N_PRICE_CHANNELS, N)
    macro_ctx = zeros(Float32, N_MACRO_DAYS, N_MACRO_SERIES)
    news      = zeros(Float32, N_NEWS_FEATURES, N)
    holding   = zeros(Float32, N_HOLDING_FEATURES, N)
    portfolio = zeros(Float32, n_portfolio_scalars(env.config.rules))
    assemble_observation!(hourly, macro_ctx, news, holding, portfolio, env;
                           macro_cache=macro_cache, news_fn=news_fn)
    return Observation(hourly, macro_ctx, news, holding, portfolio, copy(env.candidate_order))
end

"""
Stack several single-step `Observation`s (all from the same episode, so they
share `candidates`) into batched tensors ready for `ActorCriticPolicy`.

# Returns
Named tuple `(hourly, news, holding, macro_ctx, portfolio)` with a trailing
batch dimension `B = length(obs)` on every field.
"""
function stack_observations(obs::Vector{<:Observation})
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
