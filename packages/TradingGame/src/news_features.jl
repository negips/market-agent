"""
Historical news features: loads `scripts/backfill_news_signals.jl`'s output
(`website/data/news_signals.db`) into a `NewsFeatureCache` and exposes it
three ways, matching rule 8's "decisions happen at most every 15 minutes, or
immediately on a news release":

1. **The observation's news channel** (`N_NEWS_FEATURES`, `observation.jl`'s
   `news_fn` hook) — for a given `(sym_idx, hour_idx)`, an exponentially
   decayed sum of `sentiment * severity` for that symbol, plus a second
   market-wide sum pooled across every classified signal. `news_feature_fn`
   builds the closure `assemble_observation!`/`collect_rollout` expect.
2. **Forced decision bars** (`TradingGameEnv.news_hour_indices`) — any
   classified signal at or above `NEWS_DECISION_SEVERITY_THRESHOLD` maps its
   real `published_at` timestamp onto the hourly bar it falls in
   (`find_hourly_end`, the same mapping `reset!` uses for episode starts).
   Every bar of the cache (hourly or 15-minute) is already a
   decision bar (see `env.jl`'s module docstring), so this alone doesn't
   change decision *cadence* — the hour was already going to be a decision
   bar regardless.
3. **Instant market snapshot at that same hour** (`TradingGameEnv.
   price_overrides`, consumed via `current_price` in `env.jl`) — this is
   what actually matters at `HOURLY` granularity: instead of reacting to news
   using the hourly bar's close (which could be up to an hour stale relative
   to the news), every candidate's price at that bar is replaced with its
   **1-minute open** at-or-just-after the news's exact timestamp, read
   straight from `StockSwingPredictor.load_cached_ohlcv_1min`. That's "look
   at the current state of the whole market universe the instant the news
   lands, trade off that" rather than off a once-an-hour-stale close — a
   sparse, on-demand lookup at only the handful of news-bearing bars an
   episode has, not a continuous 1-minute `InferenceCache` (which would be
   both unnecessary here and, at full history for hundreds of symbols,
   expensive to hold in memory). A symbol with no 1-minute CSV yet, or whose
   nearest 1-minute bar is too stale (`NEWS_SNAPSHOT_MAX_LAG_MINUTES`), simply
   gets no override for that bar and falls back to the hourly close —
   `current_price` making that fallback automatic (see its docstring) is
   exactly why this degrades gracefully symbol-by-symbol rather than needing
   all-or-nothing 1-minute coverage before any of this works at all.

No dependency on the `NewsMonitor` package — this reads `news_signals.db`
directly via `SQLite.jl` (the exact table `backfill_news_signals.jl`
writes), keeping `TradingGame` independently usable without pulling in
`NewsMonitor`'s HTTP/classification stack just to read a few columns.
"""

using Dates, SQLite, DataFrames
using StockSwingPredictor: InferenceCache, find_hourly_end, load_cached_ohlcv_1min

"""How stale the nearest 1-minute bar at-or-after a news timestamp is allowed
to be before `build_news_snapshots` gives up and leaves that `(hour_idx,
sym_idx)` with no override (falls back to the hourly close). Guards against
a symbol whose 1-minute CSV has a large gap (or doesn't cover that date at
all) silently snapping to some much-later, unrelated price."""
const NEWS_SNAPSHOT_MAX_LAG_MINUTES = 30.0

"""One classified signal's decay-relevant fields: when it happened and its
signed impact (`sentiment * severity`) — event_type/summary/etc. aren't
needed past classification time, so they're not retained here."""
const NewsEvent = Tuple{DateTime, Float32}

const _NO_EVENTS = NewsEvent[]

"""
Loaded, decay-ready view of `news_signals.db`: per-symbol event lists plus a
pooled market-wide list, all sorted ascending by `published_at`; the set of
hourly-bar indices any at-or-above-threshold signal falls in; the earliest
qualifying timestamp within each of those bars (`decision_instants`); and,
when 1-minute OHLCV is available, a per-bar instant-price snapshot for every
candidate symbol at those same bars (see this file's module docstring,
point 3).

`decision_instants` is exposed (not just folded into `snapshots`) so
`scripts/fetch_news_snapshot_ohlcv.jl` can call `build_news_feature_cache`
with no `snapshot_symbols` (skipping snapshot building, which needs 1-minute
data that doesn't exist yet) purely to get the exact same hour-bucketed,
earliest-timestamp-per-bar targets training will later consume — guaranteeing
the fetch script fetches precisely what `build_news_snapshots` will look up,
nothing more.

Built once per training run (like `MacroCache`/`InferenceCache`), not
per-episode or per-step — `news_feature_fn`'s closure and `current_price`'s
`price_overrides` lookup are both O(log n) / O(1), no I/O.
"""
struct NewsFeatureCache
    own_events        :: Dict{Int, Vector{NewsEvent}}         # sym_idx => sorted events
    market_events     :: Vector{NewsEvent}                     # sorted, pooled across every classified symbol
    decision_hours    :: Set{Int}                               # hourly-bar indices with a >= threshold-severity signal
    decision_instants :: Dict{Int, DateTime}                   # hour_idx => earliest qualifying timestamp in that bar
    snapshots         :: Dict{Int, Dict{Int, Float32}}         # hour_idx => sym_idx => 1-minute-open instant price
end

"""
Build a `NewsFeatureCache` from `db_path` (default: `website/data/news_signals.db`,
`backfill_news_signals.jl`'s output) against `cache`'s symbol/hourly-bar
indexing. Rows for a symbol not present in `cache.sym_index` (news for a
company outside the cache's universe) are skipped for the own-symbol
channel but still counted in the market-wide pool.

# Arguments
- `cache`: the `InferenceCache` whose `sym_index`/`hourly_datetimes` this
  cache's lookups will be queried against.
- `db_path`: path to the classified-signals SQLite DB.
- `severity_threshold`: minimum `severity` for a signal to enter
  `decision_hours`/trigger a snapshot (default `NEWS_DECISION_SEVERITY_THRESHOLD`).
- `snapshot_symbols`: candidate symbols to build 1-minute instant-price
  snapshots for (typically the training+val universe — the same scope
  `backfill_news_signals.jl` classifies). Empty (default) skips snapshot
  building entirely — `price_overrides` then stays empty and every price
  read falls back to the hourly close, same as before this mechanism existed.
- `ohlcv_1min_dir`: directory of `{SYMBOL}.csv` 1-minute OHLCV files (e.g.
  `website/data/ohlcv/nse/1min`, from `collect_nse_ohlcv.jl --1min-only` /
  `backfill_ohlcv.jl`). Required if `snapshot_symbols` is non-empty.
"""
function build_news_feature_cache(cache::InferenceCache, db_path::String;
                                   severity_threshold::Float64=NEWS_DECISION_SEVERITY_THRESHOLD,
                                   snapshot_symbols::Vector{String}=String[],
                                   ohlcv_1min_dir::Union{Nothing, String}=nothing)::NewsFeatureCache
    isfile(db_path) || error(
        "TradingGame.build_news_feature_cache: not found: $db_path\n" *
        "Run: julia --project=packages/NewsMonitor scripts/backfill_news_signals.jl")

    db = SQLite.DB(db_path)
    own_events        = Dict{Int, Vector{NewsEvent}}()
    market_events     = NewsEvent[]
    decision_hours    = Set{Int}()
    decision_instants = Dict{Int, DateTime}()   # hour_idx => earliest qualifying timestamp in that bar

    for row in DBInterface.execute(db, "SELECT symbol, sentiment, severity, published_at FROM news_signals")
        sentiment = Float32(row.sentiment)
        severity  = Float32(row.severity)
        impact    = sentiment * severity
        dt        = DateTime(row.published_at)

        push!(market_events, (dt, impact))

        sym_idx = get(cache.sym_index, String(row.symbol), 0)
        if sym_idx != 0
            push!(get!(own_events, sym_idx, NewsEvent[]), (dt, impact))
            if severity >= severity_threshold
                h = find_hourly_end(cache, dt)
                if h > 0
                    push!(decision_hours, h)
                    (!haskey(decision_instants, h) || dt < decision_instants[h]) && (decision_instants[h] = dt)
                end
            end
        end
    end

    for v in values(own_events)
        sort!(v, by = first)
    end
    sort!(market_events, by = first)

    snapshots = if isempty(snapshot_symbols) || isempty(decision_instants)
        Dict{Int, Dict{Int, Float32}}()
    else
        ohlcv_1min_dir === nothing && error(
            "TradingGame.build_news_feature_cache: snapshot_symbols given without ohlcv_1min_dir")
        build_news_snapshots(cache, decision_instants, snapshot_symbols, ohlcv_1min_dir)
    end

    return NewsFeatureCache(own_events, market_events, decision_hours, decision_instants, snapshots)
end

"""First 1-minute bar's `open` at-or-after `t` in `df` (sorted ascending by
`datetime`), or `nothing` if `df` is empty, `t` is past the end of `df`, or
the nearest bar is more than `max_lag_minutes` later than `t`."""
function _snapshot_price(df::DataFrame, t::DateTime, max_lag_minutes::Float64)::Union{Nothing, Float32}
    isempty(df) && return nothing
    i = searchsortedfirst(df.datetime, t)
    i > nrow(df) && return nothing
    lag_minutes = Dates.value(df.datetime[i] - t) / 60_000
    lag_minutes > max_lag_minutes && return nothing
    return Float32(df.open[i])
end

"""
Build the instant-price snapshot table: for each `(hour_idx, timestamp)` in
`decision_instants` and each symbol in `symbols` with a 1-minute CSV in
`ohlcv_1min_dir`, look up that symbol's first 1-minute bar at-or-after
`timestamp` and record its `open` as `snapshots[hour_idx][sym_idx]`.

One `load_cached_ohlcv_1min` read per symbol (not per event) — each symbol's
full 1-minute history is read once, the handful of relevant prices are
extracted via binary search, then the (potentially large, multi-year)
DataFrame is dropped rather than retained, since `NewsFeatureCache` only
ever needs the sparse snapshot values afterward, not the full series.
"""
function build_news_snapshots(cache::InferenceCache, decision_instants::Dict{Int, DateTime},
                               symbols::Vector{String}, ohlcv_1min_dir::String;
                               max_lag_minutes::Float64=NEWS_SNAPSHOT_MAX_LAG_MINUTES)::Dict{Int, Dict{Int, Float32}}
    snapshots = Dict{Int, Dict{Int, Float32}}()
    for sym in symbols
        sym_idx = get(cache.sym_index, sym, 0)
        sym_idx == 0 && continue

        df = load_cached_ohlcv_1min(sym, ohlcv_1min_dir)
        isempty(df) && continue
        issorted(df.datetime) || sort!(df, :datetime)

        for (hour_idx, t) in decision_instants
            price = _snapshot_price(df, t, max_lag_minutes)
            price === nothing && continue
            get!(snapshots, hour_idx, Dict{Int, Float32}())[sym_idx] = price
        end
    end
    return snapshots
end

"""Exponentially decayed sum of `events`' impacts as of `t`, half-life
`halflife_hours`. `events` must be sorted ascending by timestamp. Only
events within `window_halflives` half-lives of `t` are summed — beyond that
a half-life's contribution is negligible (`2^-window_halflives`), and
bounding the window keeps this O(log n + window size) rather than O(n) per
call regardless of how much history `events` spans."""
function _decayed_sum(events::Vector{NewsEvent}, t::DateTime, halflife_hours::Float64;
                       window_halflives::Float64=10.0)::Float32
    isempty(events) && return 0f0
    lookback = Millisecond(round(Int, window_halflives * halflife_hours * 3_600_000))
    lo = searchsortedfirst(events, (t - lookback, -Inf32); by = first)
    hi = searchsortedlast(events, (t, Inf32); by = first)
    hi < lo && return 0f0

    s = 0f0
    for i in lo:hi
        dt, impact = events[i]
        dt > t && continue
        hours_ago = Dates.value(t - dt) / 3_600_000
        s += impact * Float32(exp(-hours_ago / halflife_hours))
    end
    return s
end

"""
Build the `news_fn` closure `assemble_observation!`/`collect_rollout` expect
— `(env, sym_idx, hour_idx) -> Vector{Float32}` of length `N_NEWS_FEATURES`
(`[own-symbol decayed signal, market-wide decayed signal]`). A plain closure
rather than making `NewsFeatureCache` itself callable, since the `news_fn::
Function` annotations in `observation.jl`/`ppo.jl`/`train.jl` require an
actual `Function` subtype — a callable struct doesn't satisfy that without
also declaring `NewsFeatureCache <: Function`, which would be a strange
thing for a data cache to inherit from.
"""
function news_feature_fn(fc::NewsFeatureCache; halflife_hours::Float64=DECAY_HALFLIFE_HOURS)::Function
    return (env::TradingGameEnv, sym_idx::Int, hour_idx::Int) -> begin
        t = env.cache.hourly_datetimes[hour_idx]
        own = _decayed_sum(get(fc.own_events, sym_idx, _NO_EVENTS), t, halflife_hours)
        mkt = _decayed_sum(fc.market_events, t, halflife_hours)
        Float32[own, mkt]
    end
end
