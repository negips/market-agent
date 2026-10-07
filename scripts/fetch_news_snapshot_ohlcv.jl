"""
fetch_news_snapshot_ohlcv.jl

Fetches 1-minute OHLCV from Kite for just the few minutes around EVERY
classified news event — not a continuous historical backfill, and not a
whole trading day per event either. `TradingGame.news_features.jl`'s
instant-price snapshot mechanism only ever looks up the first 1-minute bar
at-or-after a news timestamp, within `TradingGame.NEWS_SNAPSHOT_MAX_LAG_MINUTES`
(30 min), so fetching a whole day (~375 bars) per (symbol, event) pair would
be ~12x more than ever gets read.

Deliberately NOT filtered by severity: whether a news event is severe
enough to actually act on (override a price, force a decision) is a
training-time judgement call — `TradingGame.build_news_feature_cache`'s
`severity_threshold` argument, applied when training reads
`news_signals.db` — not something baked into which 1-minute data exists on
disk. Fetching every classified event's window regardless of severity means
changing that threshold later (tuning it, or just trying a lower one) never
requires re-fetching; the data's already there either way. Nothing on disk
satisfies even this narrow need today — `website/data/ohlcv/nse/1min/` has
zero files.

Scope:
  - Timestamps: every distinct `published_at` in `website/data/news_signals.db`
    (`backfill_news_signals.jl`'s output), any severity, optionally narrowed
    by `--from`/`--to`.
  - Symbols: every symbol in `universe_latest.json`'s train+val candidates
    (or `--symbol`), because the snapshot genuinely needs ALL candidates'
    prices at that instant, not just whichever symbol the news was actually
    about — this mimics what live operation will actually do: a severe-
    enough news event triggers a check of the market universe's full
    instantaneous state (every tracked symbol), combined with hourly/daily
    history, to decide buy/sell/hold across the whole book. Narrowing to
    just the newsy symbol would be cheaper but wouldn't be simulating that.

`--role {train,val,both}` (default `both`) picks which half of
`universe_latest.json` to use — train and val are genuinely independent
(usually different symbols, often different date windows), so there's no
need to fetch one combined (train ∪ val symbols) × (every event anywhere)
job. Run `--role train --from <train window>` and `--role val --from <val
window>` as two separate, independently-sized invocations instead — this
is the real lever for cutting the job down, not narrowing which symbols get
snapshots within a given role (see above for why that narrowing is the
wrong cut). `--role both` (the default) keeps the original combined
behavior for when train/val share the same date window (`--val-window
same`) or you just want one pass.

`--from`/`--to` exist because Kite's 1-minute coverage, unlike daily/hourly/
5-min/15-min, is NOT reliably available arbitrarily far back — verified live
against the real API (RELIANCE/TCS/INFY, weekday, mid-market-hours): 2015-01
returns nothing, 2016-01 returns real bars, 2016-07 returns nothing again,
2017-01 onward returns bars consistently. It's patchy, not a single clean
cutoff date, so no default filter is applied here — an unfiltered run will
keep re-attempting pre-2017-ish events on every future run too, since a
missing bar is indistinguishable from "not yet fetched" and never a reason
to stop retrying. `--from 2017-01-01` is a reasonable starting point to
avoid that wasted effort, not a guarantee every date after it succeeds.

For each (symbol, timestamp) pair not already covered by a bar within the
same `NEWS_SNAPSHOT_MAX_LAG_MINUTES` window in that symbol's existing
`website/data/ohlcv/nse/1min/{SYMBOL}.csv`, fetches a short window
(`timestamp - 1 minute` … `timestamp + NEWS_SNAPSHOT_MAX_LAG_MINUTES`) via
`StockSwingPredictor.fetch_ohlcv_1min_window` (one Kite historical-data call
per pair, paced the same 0.35s as `fetch_ohlcv_1min`'s internal chunking)
and merges the handful of returned bars in (de-duplicated by `datetime`,
re-sorted) — never overwrites the file, so this composes cleanly with a
later full `--1min-only` backfill or `update_ohlcv.jl`'s daily runs.

Usage:
  julia --project=packages/TradingGame scripts/fetch_news_snapshot_ohlcv.jl
  julia --project=packages/TradingGame scripts/fetch_news_snapshot_ohlcv.jl --dry-run
  julia --project=packages/TradingGame scripts/fetch_news_snapshot_ohlcv.jl --symbol RELIANCE --symbol TCS
  julia --project=packages/TradingGame scripts/fetch_news_snapshot_ohlcv.jl --news-db other_signals.db
  julia --project=packages/TradingGame scripts/fetch_news_snapshot_ohlcv.jl --from 2017-01-01
  julia --project=packages/TradingGame scripts/fetch_news_snapshot_ohlcv.jl --role train --from 2021-09-09 --to 2026-05-11
  julia --project=packages/TradingGame scripts/fetch_news_snapshot_ohlcv.jl --role val   --from 2026-05-12

Output: website/data/ohlcv/nse/1min/{SYMBOL}.csv (merged in-place, same
schema as every other OHLCV CSV: datetime,open,high,low,close,volume).
"""

using StockSwingPredictor, TradingGame, SQLite, JSON3, CSV, DataFrames, Dates

const REPO_ROOT      = joinpath(@__DIR__, "..")
const OHLCV_1MIN_DIR = normpath(joinpath(REPO_ROOT, "website", "data", "ohlcv", "nse", "1min"))
const UNIVERSE_FILE  = normpath(joinpath(REPO_ROOT, "website", "data", "trading_game", "universe_latest.json"))
const NEWS_DB_FILE   = normpath(joinpath(REPO_ROOT, "website", "data", "news_signals.db"))

# Same pacing fetch_ohlcv_1min uses internally between its own chunks —
# this script calls fetch_ohlcv_1min_window once per (symbol, event) pair
# instead, so the pacing has to live in this loop rather than inside the
# fetch function itself.
const KITE_CALL_DELAY = 0.35

function parse_args()
    args = Dict{String,Any}(
        "symbols"   => String[],
        "universe"  => UNIVERSE_FILE,
        "news_db"   => NEWS_DB_FILE,
        "out_dir"   => OHLCV_1MIN_DIR,
        "role"      => "both",
        "from"      => nothing,
        "to"        => nothing,
        "dry_run"   => false,
    )
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a in ("-h", "--help")
            println("""
fetch_news_snapshot_ohlcv.jl — fetch 1-minute OHLCV around every news event

Fetches regardless of severity — severity only matters later, when training
decides whether to act on a given signal (`--severity-threshold` on the
training/consumption side, not here).

Flags:
  --symbol SYM     Add one symbol to fetch for (repeatable). Overrides
                    --role entirely — default: --role's resolved symbols.
  --role ROLE      train | val | both (default: both). Picks which half of
                    --universe's candidates to use — train and val are
                    independent, usually different symbols and date
                    windows, so run each separately with its own --from/--to
                    instead of one combined (train+val) x (every event) pass.
  --universe PATH  Universe snapshot JSON (default: $(args["universe"]))
  --news-db PATH   Classified-signals DB (default: $(args["news_db"]))
  --out-dir PATH   1-minute OHLCV output dir (default: $(args["out_dir"]))
  --from DATE      Only fetch events on/after this date (yyyy-mm-dd). Kite's
                    1-minute coverage is patchy before ~2017 — see the module
                    docstring; --from 2017-01-01 is a reasonable start.
  --to DATE        Only fetch events on/before this date (yyyy-mm-dd)
  --account N      Kite API-key slot (1 = KITE_HISTORICAL_*, 2 = KITE_HISTORICAL2_*, …)
  --dry-run        Print symbol/event/call counts, no API calls or writes
  -h, --help       Show this message
""")
            exit(0)
        elseif a == "--account" && i + 1 <= length(ARGS)
            i += 2   # read by kite_account_from_args() when the session loads
        elseif a == "--symbol" && i + 1 <= length(ARGS)
            push!(args["symbols"], ARGS[i+1]); i += 2
        elseif a == "--role" && i + 1 <= length(ARGS)
            args["role"] = ARGS[i+1]; i += 2
        elseif a == "--universe" && i + 1 <= length(ARGS)
            args["universe"] = ARGS[i+1]; i += 2
        elseif a == "--news-db" && i + 1 <= length(ARGS)
            args["news_db"] = ARGS[i+1]; i += 2
        elseif a == "--out-dir" && i + 1 <= length(ARGS)
            args["out_dir"] = ARGS[i+1]; i += 2
        elseif a == "--from" && i + 1 <= length(ARGS)
            args["from"] = Date(ARGS[i+1]); i += 2
        elseif a == "--to" && i + 1 <= length(ARGS)
            args["to"] = Date(ARGS[i+1]); i += 2
        elseif a == "--dry-run"
            args["dry_run"] = true; i += 1
        else
            @warn "Unknown argument: $a"; i += 1
        end
    end
    args["role"] in ("train", "val", "both") ||
        error("fetch_news_snapshot_ohlcv.jl: --role must be train, val, or both (got '$(args["role"])')")
    return args
end

# ── Universe symbol resolution (same dual-format handling as backfill_news_signals.jl) ──

"""Symbols to fetch for: `--symbol` if given (overrides `role` entirely),
else `role`'s slice of `path`'s candidates (`"train"`/`"val"`/`"both"` ==
`train_candidates`/`val_candidates`/their union). The old flat format
(`candidates`, pre train/val split) has no role distinction — `role in
("train", "val")` against it is an error asking the caller to regenerate
`universe_latest.json` or use `--role both`/`--symbol` instead, rather than
silently treating "train" and "val" as identical to "both"."""
function resolve_symbols(path::String, explicit::Vector{String}, role::String)::Vector{String}
    isempty(explicit) || return sort(unique(explicit))

    isfile(path) || error(
        "fetch_news_snapshot_ohlcv.jl: no --symbol given and universe snapshot not found: $path\n" *
        "Run build_market_universe_snapshot.jl first, or pass --symbol explicitly.")

    raw = JSON3.read(read(path, String))
    syms = Set{String}()
    if haskey(raw, :train_candidates)
        role in ("train", "both") && for c in raw.train_candidates; push!(syms, String(c.symbol)); end
        role in ("val", "both")   && for c in get(raw, :val_candidates, []); push!(syms, String(c.symbol)); end
    elseif haskey(raw, :candidates)
        role == "both" || error(
            "fetch_news_snapshot_ohlcv.jl: --role $role requested but $path is the old flat format " *
            "(no train/val split) — regenerate it with build_market_universe_snapshot.jl, or use " *
            "--role both / --symbol instead.")
        for c in raw.candidates; push!(syms, String(c.symbol)); end
    else
        error("fetch_news_snapshot_ohlcv.jl: $path has neither train_candidates/val_candidates nor candidates")
    end
    return sort(collect(syms))
end

# ── Event timestamps: every classified signal, any severity ────────────────────

"""Every distinct `published_at` in `news_db` — no severity filter. Whether
a given timestamp ends up mattering is a training-time decision
(`build_news_feature_cache`'s `severity_threshold`), not a fetch-time one;
see this file's module docstring for why."""
function all_event_timestamps(news_db::String)::Vector{DateTime}
    db = SQLite.DB(news_db)
    times = Set{DateTime}()
    for row in DBInterface.execute(db, "SELECT DISTINCT published_at FROM news_signals")
        push!(times, DateTime(row.published_at))
    end
    return sort(collect(times))
end

# ── Per-symbol coverage check + merge ───────────────────────────────────────────

"""Sorted `datetime`s already present in `path` (empty if the file doesn't
exist yet) — read once per symbol, not once per event."""
function covered_datetimes(path::String)::Vector{DateTime}
    isfile(path) || return DateTime[]
    df = CSV.read(path, DataFrame; select=[:datetime], types=Dict(:datetime => DateTime))
    return sort(df.datetime)
end

"""Whether `times` (sorted) already has a bar within `[t, t + max_lag_minutes]`
— the same at-or-after-within-lag rule `_snapshot_price` (`news_features.jl`)
uses to consume this data, so "already covered" here means exactly "building
the snapshot later would already succeed without fetching anything more"."""
function has_covering_bar(times::Vector{DateTime}, t::DateTime, max_lag_minutes::Float64)::Bool
    isempty(times) && return false
    i = searchsortedfirst(times, t)
    i > length(times) && return false
    return Dates.value(times[i] - t) / 60_000 <= max_lag_minutes
end

"""Merge `new_bars` into `path`, de-duplicated by `datetime` and re-sorted —
never overwrites existing bars, so this composes with whatever else writes
to the same file (a later full backfill, `update_ohlcv.jl`'s daily runs)."""
function merge_bars!(path::String, new_bars::DataFrame)
    isempty(new_bars) && return nothing
    mkpath(dirname(path))
    merged = isfile(path) ?
        vcat(CSV.read(path, DataFrame; types=Dict(:datetime => DateTime)), new_bars) :
        new_bars
    unique!(merged, :datetime)
    sort!(merged, :datetime)
    CSV.write(path, merged)
    return nothing
end

# ── Main ──────────────────────────────────────────────────────────────────────

function main()
    args = parse_args()
    symbols = resolve_symbols(args["universe"], args["symbols"], args["role"])
    isempty(symbols) && error("fetch_news_snapshot_ohlcv.jl: no symbols to fetch for")

    isfile(args["news_db"]) || error(
        "Not found: $(args["news_db"])\n" *
        "Run: julia --project=packages/NewsMonitor scripts/backfill_news_signals.jl")

    times = all_event_timestamps(args["news_db"])
    isempty(times) && error("fetch_news_snapshot_ohlcv.jl: no news signals in $(args["news_db"])")

    from_d, to_d = args["from"], args["to"]
    if from_d !== nothing || to_d !== nothing
        times = filter(times) do t
            d = Date(t)
            (from_d === nothing || d >= from_d) && (to_d === nothing || d <= to_d)
        end
        isempty(times) && error("fetch_news_snapshot_ohlcv.jl: no news events in --from/--to range")
    end

    @info "Role: $(args["role"])  Symbols: $(length(symbols))  News events (any severity" *
          (from_d === nothing && to_d === nothing ? ""  : ", filtered to $(from_d)…$(to_d)") *
          "): $(length(times))"

    # ── Figure out exactly which (symbol, event) pairs are missing ─────────────
    needed = Dict{String, Vector{DateTime}}()
    n_needed = 0
    for sym in symbols
        have = covered_datetimes(joinpath(args["out_dir"], "$sym.csv"))
        missing_events = [t for t in times if !has_covering_bar(have, t, NEWS_SNAPSHOT_MAX_LAG_MINUTES)]
        isempty(missing_events) && continue
        needed[sym] = missing_events
        n_needed += length(missing_events)
    end

    @info "$(n_needed) (symbol, event) pairs need fetching " *
          "($(length(symbols) * length(times) - n_needed) already covered)"

    if args["dry_run"]
        @info "--dry-run: no API calls or writes performed."
        return
    end

    n_needed == 0 && (@info "Nothing to do."; return)

    session   = load_kite_session(REPO_ROOT; account=kite_account_from_args())
    @info "Loading NSE instrument list from Kite…"
    instr     = load_instruments(session; exchange="NSE", refresh=true)
    token_map = build_token_map(instr; exchange="NSE")

    n_done = 0
    n_failed = 0
    t0 = time()
    for (sym, sym_times) in needed
        token = get(token_map, sym, nothing)
        if isnothing(token)
            @warn "No NSE token for $sym — skipping ($(length(sym_times)) events)"
            n_failed += length(sym_times)
            continue
        end
        path = joinpath(args["out_dir"], "$sym.csv")
        for t in sym_times
            from_dt = t - Minute(1)
            to_dt   = t + Minute(ceil(Int, NEWS_SNAPSHOT_MAX_LAG_MINUTES))
            df = fetch_ohlcv_1min_window(token, from_dt, to_dt, session)
            if isempty(df)
                n_failed += 1
            else
                merge_bars!(path, df)
                n_done += 1
            end
            sleep(KITE_CALL_DELAY)

            if (n_done + n_failed) % 50 == 0
                elapsed = round(Int, time() - t0)
                @info "  [$(n_done + n_failed)/$n_needed] fetched=$n_done failed=$n_failed — $(elapsed)s elapsed"
            end
        end
    end

    @info "Done. fetched=$n_done failed=$n_failed → $(args["out_dir"])"
end

main()
