"""
backfill_ohlcv.jl

Extends existing OHLCV CSVs in website/data/ohlcv/{nse,bse}/{daily,hourly,
5min,15min,1min}/ *backward* to an earlier `--from` date (both exchanges by
default — pass --nse-only or --bse-only to restrict to one). The mirror
image of update_ohlcv.jl, which only ever extends forward to yesterday.

For each existing CSV, reads the EARLIEST date/datetime in the file and, if
that's later than `--from`, fetches only the older gap — `--from` through
the day before the earliest bar already on disk — then merges it into the
file (de-duplicated, re-sorted), not a full re-fetch-and-overwrite the way
collect_nse_ohlcv.jl's `--refresh` would be. A symbol already extending back
to `--from` or earlier is left untouched. A symbol with no existing CSV at
all is skipped (there's nothing to extend) — run collect_nse_ohlcv.jl /
collect_bse_ohlcv.jl first for a brand-new granularity.

This exists because Kite's per-interval day limits (60/100/200/400/2000
days for 1min/5min/15min/60min/day) are a single-request span cap, not a
total retention cliff — verified live against the real API, every
granularity here can still be fetched arbitrarily far back. Every existing
archive was collected under the old (incorrect) assumption that those
numbers were hard retention windows, so most symbols' files start much
later than Kite can actually provide.

After every daily pass, `{nse,bse}/earliest_trading_day.json` is rebuilt
from the current daily CSVs (`build_earliest_trading_day`) — one JSON file
per exchange mapping symbol → earliest date daily has ever found for it.
Every finer granularity (hourly/5min/15min/1min) then fetches from
`max(--from, earliest_trading_day[symbol])`, never `--from` alone. Without
this, a symbol daily already proved has nothing before, say, a 2021 IPO
still gets re-probed from `--from` at every finer granularity on every
future run — and the finer the granularity, the more chunks that empty
re-probe costs (hourly's 59-day chunks vs. daily's 2000-day span cap is a
~30x difference per symbol). Daily is the cheapest granularity to establish
this floor, so it's the one source of truth for all the others.

That daily-derived floor is necessary but not sufficient: each finer
granularity ALSO has its own Kite retention floor, independent of the
symbol's real IPO date and independent of the other granularities —
observed live, hourly simply has no data before ~2015-02-02 for the vast
majority of NSE symbols, regardless of how far back daily goes for the same
symbol. Without tracking this separately, every symbol whose daily history
predates that floor gets its full multi-chunk gap re-probed and
re-reconfirmed empty on EVERY run, forever, even though the previous run
already learned the answer. Each of hourly/5min/15min/1min therefore keeps
its own `{granularity}/confirmed_floor.json` (inside that granularity's own
subfolder, not the exchange root, since it doesn't apply to the others) —
`{symbol => earliest date confirmed to have no older bars}` — written the
first time a "no older bars" result is seen for that symbol, and consulted
as a second floor term alongside `earliest_trading_day.json` from then on.

Prerequisites:
  - sidecar/kite_session.json present         (node sidecar/kite_login.js)
  - website/data/ohlcv/nse/{daily,hourly,5min,15min,1min}/ populated
    (run collect_nse_ohlcv.jl first, unless --bse-only)
  - website/data/ohlcv/bse/{daily,hourly,5min,15min,1min}/ populated
    (run collect_bse_ohlcv.jl first, unless --nse-only)

Usage:
  julia --project=packages/StockSwingPredictor scripts/backfill_ohlcv.jl              # --from defaults to 2010-01-04
  julia --project=packages/StockSwingPredictor scripts/backfill_ohlcv.jl --from 2015-01-01 --nse-only
  julia --project=packages/StockSwingPredictor scripts/backfill_ohlcv.jl --hourly-only
  julia --project=packages/StockSwingPredictor scripts/backfill_ohlcv.jl --skip-1min
  julia --project=packages/StockSwingPredictor scripts/backfill_ohlcv.jl --dry-run
  julia --project=packages/StockSwingPredictor scripts/backfill_ohlcv.jl --symbol RELIANCE

NOTE: going back to 2010 for the finer granularities, especially 1-minute,
across the full symbol universe is a genuinely large job — many chunked API
calls per symbol. Use --dry-run first to see the call-count estimate, and
--symbol / --hourly-only / --15min-only etc. to scope a trial run.
"""

using StockSwingPredictor, CSV, DataFrames, Dates, JSON3

const REPO_ROOT     = joinpath(@__DIR__, "..")
const OHLCV_ROOT    = joinpath(REPO_ROOT, "website", "data", "ohlcv")
const NSE_OHLCV_DIR = joinpath(OHLCV_ROOT, "nse")
const BSE_OHLCV_DIR = joinpath(OHLCV_ROOT, "bse")

# Deliberately NOT 2010-01-01 like collect_nse_ohlcv.jl/collect_bse_ohlcv.jl's
# --from default (where it's harmless — those are one-shot full collections).
# Here it's compared every run via `first_date <= from_target` to decide
# whether a symbol needs anything at all, and 2010-01-01 was a Friday NSE
# holiday — every long-listed symbol's real earliest bar is 2010-01-04 (the
# first trading Monday that year), which is never <= 2010-01-01. That made
# the skip check permanently unsatisfiable for virtually every symbol: every
# run, forever, re-requested the same already-exhausted few-day gap.
# 2010-01-04 is the actual earliest date any symbol's data can start at, so
# it's the correct default for this check to ever succeed.
const DEFAULT_FROM = Date(2010, 1, 4)

# One subfolder per granularity under each exchange root — see
# kite_data.jl's module docstring. Macro is deliberately out of scope here
# (same as update_ohlcv.jl's reasoning): too few files to be worth it.
nse_gran_dir(granularity::String) = joinpath(NSE_OHLCV_DIR, granularity)
bse_gran_dir(granularity::String) = joinpath(BSE_OHLCV_DIR, granularity)

# One reference file per exchange, at the exchange root (not inside any one
# granularity's subfolder, since it's used by all of them) — see
# `build_earliest_trading_day`'s docstring for what it's for.
nse_earliest_file() = joinpath(NSE_OHLCV_DIR, "earliest_trading_day.json")
bse_earliest_file() = joinpath(BSE_OHLCV_DIR, "earliest_trading_day.json")

# One confirmed-empty-floor file PER GRANULARITY, inside that granularity's
# own subfolder — unlike earliest_trading_day.json, this doesn't apply to
# the other granularities, so it has no business living at the exchange
# root. See the module docstring for why each finer granularity needs its
# own copy of this mechanism.
confirmed_floor_file(gran_dir::String) = joinpath(gran_dir, "confirmed_floor.json")

# See ScriptLog's docstring (StockSwingPredictor/src/script_log.jl) — every
# per-symbol outcome goes here; the terminal only gets stage headers,
# periodic heartbeats, and warnings. Override with --log-file.
const DEFAULT_LOG_FILE = joinpath(OHLCV_ROOT, "logs", "backfill_ohlcv.log")

"""`readdir`, but `String[]` instead of an error when `dir` doesn't exist yet."""
_readdir_safe(dir::String) = isdir(dir) ? readdir(dir) : String[]

# ── Helpers ───────────────────────────────────────────────────────────────────

"""
Full-file fallback for `_first_value`: reads the whole CSV via CSV.jl and
takes the minimum of the (possibly partially corrupted) column. This is the
ORIGINAL implementation, kept as the accurate-but-slow path for files the
fast path below can't trust (missing header column, unparseable first data
row — i.e. the file might be corrupted or not sorted as expected). Returns
`nothing` if the file is missing, empty, or every row fails to parse.
"""
function _first_value_full_scan(path::String, col::Symbol, T::Type)
    df = try
        CSV.read(path, DataFrame; select=[col], types=Dict(col => T))
    catch e
        @warn "Could not read $path: $(sprint(showerror, e))"
        return nothing
    end
    isempty(df) && return nothing

    n_bad = count(ismissing, df[!, col])
    n_bad > 0 && @warn "$path: $n_bad row(s) failed to parse as $T — file may be " *
                        "corrupted. Ignoring them for the earliest-known-good " *
                        "timestamp; inspect the file directly if this recurs."

    good = skipmissing(df[!, col])
    isempty(good) && return nothing
    return minimum(good)
end

"""
Read only the date/datetime column from a CSV and return the MINIMUM value
(the mirror image of update_ohlcv.jl's `_last_value`, which returns the
maximum). Every writer in this codebase (`fetch_ohlcv*`'s `sort!`,
`_prepend_merge!`'s `sort!`) guarantees these CSVs are sorted ascending by
`col`, so the minimum is always just the FIRST data row — reading that one
line is enough, no need to parse the whole file. This matters because
`backfill_ohlcv.jl` calls this once per symbol per granularity just to
decide whether a symbol needs any work at all: with ~1900 symbols per
granularity, a full `CSV.read` per file (the original implementation, now
`_first_value_full_scan`) turned "nothing to do" into roughly an hour of
pure I/O before a single real API call — observed live, re-running after an
earlier pass had already finished most symbols. Falls back to the full
accurate scan — same corrupted-row handling as before — if the header
doesn't have `col`, the file has no data rows, or the first row fails to
parse as `T` (any of which could mean a corrupted or unexpectedly unsorted
file, where trusting just the first line would be wrong).
"""
function _first_value(path::String, col::Symbol, T::Type)
    isfile(path) || return nothing
    header_line, first_data_line = open(path, "r") do io
        eof(io) && return "", ""
        h = readline(io)
        d = eof(io) ? "" : readline(io)
        return h, d
    end
    isempty(header_line) && return nothing

    col_idx = findfirst(==(string(col)), split(header_line, ','))
    isnothing(col_idx) && return _first_value_full_scan(path, col, T)
    isempty(first_data_line) && return _first_value_full_scan(path, col, T)

    row_fields = split(first_data_line, ',')
    col_idx > length(row_fields) && return _first_value_full_scan(path, col, T)

    return try
        T(row_fields[col_idx])
    catch
        _first_value_full_scan(path, col, T)
    end
end

_first_daily_date(sym::String, dir::String) =
    _first_value(joinpath(dir, "$sym.csv"), :date,     Date)
_first_hourly_datetime(sym::String, dir::String) =
    _first_value(joinpath(dir, "$sym.csv"), :datetime, DateTime)
_first_5min_datetime(sym::String, dir::String) =
    _first_value(joinpath(dir, "$sym.csv"), :datetime, DateTime)
_first_15min_datetime(sym::String, dir::String) =
    _first_value(joinpath(dir, "$sym.csv"), :datetime, DateTime)
_first_1min_datetime(sym::String, dir::String) =
    _first_value(joinpath(dir, "$sym.csv"), :datetime, DateTime)

"""
Build `{symbol => earliest known trading date}` from every daily CSV in
`daily_dir`. Daily is the cheapest granularity to have already found each
symbol's true earliest bar (or confirmed there's nothing earlier within
whatever `--from` has been tried so far) — its per-request span cap is 2000
days, vs. hourly's 59, so it reaches the same conclusion in far fewer calls.

This is the reference the finer granularities (`backfill_hourly!`/`_5min!`/
`_15min!`/`_1min!`) consult before requesting anything: without it, a symbol
daily already proved has zero data before, say, 2021-11-10 (a late IPO)
still gets re-probed from `--from` (e.g. 2010-01-04) at EVERY finer
granularity, and the finer the granularity the more chunks that empty
re-probe costs — observed live, hourly's 59-day chunking meant re-confirming
"nothing here" for ~1250 such symbols was on track to cost tens of thousands
of silently-empty API calls and many hours, despite daily having already
settled the question cheaply.
"""
function build_earliest_trading_day(daily_dir::String)::Dict{String, Date}
    out = Dict{String, Date}()
    for f in _readdir_safe(daily_dir)
        sym = replace(f, ".csv" => "")
        d = _first_daily_date(sym, daily_dir)
        isnothing(d) || (out[sym] = d)
    end
    return out
end

"""Persist a `{symbol => date}` reference as JSON: `{"SYMBOL": "yyyy-mm-dd",
…}`. Generic over what the dates MEAN — used both for the daily-derived
earliest-trading-day reference and for each finer granularity's own
confirmed-empty-floor reference (same shape, same file format, different
path and different source of truth for the dates)."""
function save_earliest_trading_day(ref::Dict{String, Date}, path::String)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON3.write(io, Dict(s => string(d) for (s, d) in ref))
    end
end

"""Load a previously-saved `{symbol => date}` reference (see
`save_earliest_trading_day`). Returns an empty `Dict` (not an error) if the
file doesn't exist yet — e.g. the daily pass hasn't completed yet, or a
finer granularity's `confirmed_floor.json` hasn't been written yet — in
which case that floor term simply doesn't constrain anything."""
function load_earliest_trading_day(path::String)::Dict{String, Date}
    isfile(path) || return Dict{String, Date}()
    raw = JSON3.read(read(path, String), Dict{String, String})
    return Dict(s => Date(d) for (s, d) in raw)
end

"""
The latest (most restrictive) of `from_target` and every known floor for
`sym` across however many `floors` dicts are given — the actual date a
fetch should start from. Never narrower than an explicit `--from` (a
shallower explicit request still wins, e.g. deliberately backfilling
hourly only to 2015 even though daily goes back to 2010), and never wastes
calls re-probing a gap some floor already proved is empty. Each `floors`
argument is consulted independently — a symbol absent from a given dict
just doesn't constrain that term — so finer granularities can pass BOTH
daily's earliest-trading-day reference AND their own confirmed-empty-floor
reference, while daily itself only ever has the one.

Caveat shared by every floor this feeds: a transient error on a single
chunk during the gap probe (`fetch_ohlcv*`'s per-chunk `catch` skips to the
next chunk rather than retrying) looks identical to a genuine "no data
here" and would get persisted as a floor just the same. This was already
true of `earliest_trading_day.json` before confirmed-empty floors existed;
accepted here for the same reason — a very rare transient failure
incorrectly pinning a floor a little too shallow is a far smaller cost than
re-probing every symbol's full gap on every run forever.
"""
function _effective_from(sym::String, from_target::Date, floors::Dict{String, Date}...)
    result = from_target
    for d in floors
        haskey(d, sym) && (result = max(result, d[sym]))
    end
    return result
end

"""
Merge `older` (freshly fetched, strictly before the file's current earliest
bar) into the existing CSV at `path`, de-duplicated and re-sorted ascending
by `col`. Unlike update_ohlcv.jl's forward `append=true`, a backward
extension can't just append — the new rows belong at the *start* of the
file — so this reads the whole existing file back in and rewrites it. Fine
at these file sizes (single-symbol OHLCV CSVs, at most a few hundred
thousand rows even for 1-minute across 15+ years).
"""
function _prepend_merge!(path::String, older::DataFrame, col::Symbol, T::Type)
    existing = CSV.read(path, DataFrame; types=Dict(col => T))
    merged = vcat(older, existing)
    unique!(merged, col)
    sort!(merged, col)
    CSV.write(path, merged)
end

# Last bar times for "day complete" checks (IST) — same convention as
# update_ohlcv.jl, reused here only for the dry-run day-count estimate.
const HOURLY_CHUNK_DAYS     = 59
const FIVEMIN_CHUNK_DAYS    = 90
const FIFTEENMIN_CHUNK_DAYS = 175
const ONEMIN_CHUNK_DAYS     = 55

# How often (in symbols) a granularity loop prints a console progress
# heartbeat — see `ScriptLog`'s docstring for why routine per-symbol detail
# otherwise goes to the log file only.
const HEARTBEAT_INTERVAL = 100

# ── Core backfill loops ──────────────────────────────────────────────────────

function backfill_daily!(symbols, token_map, session, from_target::Date, earliest::Dict{String, Date}, slog::ScriptLog;
                         dry_run::Bool, out_dir::String)
    current = updated = failed = 0
    total   = length(symbols)

    for (i, sym) in enumerate(symbols)
        if !dry_run && i % HEARTBEAT_INTERVAL == 0
            logboth(slog, "[$i/$total] daily progress: $updated backfilled, $current at floor, $failed failed so far")
        end

        path = joinpath(out_dir, "$sym.csv")
        first_date = _first_daily_date(sym, out_dir)
        if isnothing(first_date)
            @warn "[$i/$total] $sym daily — no existing CSV, skipping (run collect_nse_ohlcv.jl first)"
            failed += 1; continue
        end

        floor_date = _effective_from(sym, from_target, earliest)
        if first_date <= floor_date
            current += 1; continue
        end

        to  = first_date - Day(1)
        gap = (to - floor_date).value + 1

        if dry_run
            @info "[$i/$total] $sym daily: would backfill $floor_date → $to ($gap calendar days)"
            updated += 1; continue
        end

        token = get(token_map, sym, nothing)
        if isnothing(token)
            logboth(slog, "[$i/$total] $sym — no instrument token"; warn=true); failed += 1; continue
        end

        older_df = fetch_ohlcv(token, floor_date, to, session)
        if isempty(older_df)
            logf(slog, "[$i/$total] $sym daily — no older bars in $floor_date…$to")
            failed += 1; sleep(0.35); continue
        end

        _prepend_merge!(path, older_df, :date, Date)
        updated += 1
        logf(slog, "[$i/$total] $sym daily +$(nrow(older_df)) older bars ($floor_date → $to)")
        sleep(0.35)
    end

    summary = dry_run ?
        "Daily: $updated would be backfilled, $current already at their floor" :
        "Daily: $updated backfilled, $current already at their floor, $failed failed"
    dry_run ? (@info summary) : logboth(slog, summary)
end

function backfill_hourly!(symbols, token_map, session, from_target::Date, earliest::Dict{String, Date}, slog::ScriptLog;
                          dry_run::Bool, out_dir::String)
    current = updated = failed = 0
    total   = length(symbols)

    confirmed_path  = confirmed_floor_file(out_dir)
    confirmed_floor = load_earliest_trading_day(confirmed_path)
    confirmed_added = 0

    for (i, sym) in enumerate(symbols)
        if !dry_run && i % HEARTBEAT_INTERVAL == 0
            logboth(slog, "[$i/$total] hourly progress: $updated backfilled, $current at floor, $failed failed so far")
        end

        path = joinpath(out_dir, "$sym.csv")
        first_dt = _first_hourly_datetime(sym, out_dir)
        if isnothing(first_dt)
            @warn "[$i/$total] $sym hourly — no existing CSV, skipping (run collect_nse_ohlcv.jl first)"
            failed += 1; continue
        end
        first_date = Date(first_dt)

        floor_date = _effective_from(sym, from_target, earliest, confirmed_floor)
        if first_date <= floor_date
            current += 1; continue
        end

        # Fetch through the whole earliest day (not day-1) and de-dup via
        # the datetime filter below — same partial-day defensiveness as
        # update_ohlcv.jl's forward version, just mirrored.
        to     = first_date
        gap    = (to - floor_date).value + 1
        n_chks = ceil(Int, gap / HOURLY_CHUNK_DAYS)

        if dry_run
            @info "[$i/$total] $sym hourly: would backfill $floor_date → $to ($gap days, ~$n_chks API call$(n_chks==1 ? "" : "s"))"
            updated += 1; continue
        end

        token = get(token_map, sym, nothing)
        if isnothing(token)
            logboth(slog, "[$i/$total] $sym — no instrument token"; warn=true); failed += 1; continue
        end

        older_df = fetch_ohlcv_hourly(token, floor_date, to, session)
        filter!(row -> row.datetime < first_dt, older_df)

        if isempty(older_df)
            logf(slog, "[$i/$total] $sym hourly — no older bars before $first_dt")
            confirmed_floor[sym] = first_date
            save_earliest_trading_day(confirmed_floor, confirmed_path)
            confirmed_added += 1
            failed += 1; continue
        end

        _prepend_merge!(path, older_df, :datetime, DateTime)
        updated += 1
        logf(slog, "[$i/$total] $sym hourly +$(nrow(older_df)) older bars ($(older_df.datetime[1]) → $(older_df.datetime[end]))")
    end

    confirmed_added > 0 &&
        logboth(slog, "  Confirmed-empty hourly floor: +$confirmed_added new, $(length(confirmed_floor)) total → $confirmed_path")

    summary = dry_run ?
        "Hourly: $updated would be backfilled, $current already at their floor" :
        "Hourly: $updated backfilled, $current already at their floor, $failed failed"
    dry_run ? (@info summary) : logboth(slog, summary)
end

function backfill_5min!(symbols, token_map, session, from_target::Date, earliest::Dict{String, Date}, slog::ScriptLog;
                        dry_run::Bool, out_dir::String)
    current = updated = failed = 0
    total   = length(symbols)

    confirmed_path  = confirmed_floor_file(out_dir)
    confirmed_floor = load_earliest_trading_day(confirmed_path)
    confirmed_added = 0

    for (i, sym) in enumerate(symbols)
        if !dry_run && i % HEARTBEAT_INTERVAL == 0
            logboth(slog, "[$i/$total] 5min progress: $updated backfilled, $current at floor, $failed failed so far")
        end

        path = joinpath(out_dir, "$sym.csv")
        first_dt = _first_5min_datetime(sym, out_dir)
        if isnothing(first_dt)
            @warn "[$i/$total] $sym 5min — no existing CSV, skipping (run collect_nse_ohlcv.jl first)"
            failed += 1; continue
        end
        first_date = Date(first_dt)

        floor_date = _effective_from(sym, from_target, earliest, confirmed_floor)
        if first_date <= floor_date
            current += 1; continue
        end

        to     = first_date
        gap    = (to - floor_date).value + 1
        n_chks = ceil(Int, gap / FIVEMIN_CHUNK_DAYS)

        if dry_run
            @info "[$i/$total] $sym 5min: would backfill $floor_date → $to ($gap days, ~$n_chks call$(n_chks==1 ? "" : "s"))"
            updated += 1; continue
        end

        token = get(token_map, sym, nothing)
        if isnothing(token)
            logboth(slog, "[$i/$total] $sym — no instrument token"; warn=true); failed += 1; continue
        end

        older_df = fetch_ohlcv_5min(token, floor_date, to, session)
        filter!(row -> row.datetime < first_dt, older_df)

        if isempty(older_df)
            logf(slog, "[$i/$total] $sym 5min — no older bars before $first_dt")
            confirmed_floor[sym] = first_date
            save_earliest_trading_day(confirmed_floor, confirmed_path)
            confirmed_added += 1
            failed += 1; continue
        end

        _prepend_merge!(path, older_df, :datetime, DateTime)
        updated += 1
        logf(slog, "[$i/$total] $sym 5min +$(nrow(older_df)) older bars ($(older_df.datetime[1]) → $(older_df.datetime[end]))")
    end

    confirmed_added > 0 &&
        logboth(slog, "  Confirmed-empty 5min floor: +$confirmed_added new, $(length(confirmed_floor)) total → $confirmed_path")

    summary = dry_run ?
        "5min: $updated would be backfilled, $current already at their floor" :
        "5min: $updated backfilled, $current already at their floor, $failed failed"
    dry_run ? (@info summary) : logboth(slog, summary)
end

function backfill_15min!(symbols, token_map, session, from_target::Date, earliest::Dict{String, Date}, slog::ScriptLog;
                         dry_run::Bool, out_dir::String)
    current = updated = failed = 0
    total   = length(symbols)

    confirmed_path  = confirmed_floor_file(out_dir)
    confirmed_floor = load_earliest_trading_day(confirmed_path)
    confirmed_added = 0

    for (i, sym) in enumerate(symbols)
        if !dry_run && i % HEARTBEAT_INTERVAL == 0
            logboth(slog, "[$i/$total] 15min progress: $updated backfilled, $current at floor, $failed failed so far")
        end

        path = joinpath(out_dir, "$sym.csv")
        first_dt = _first_15min_datetime(sym, out_dir)
        if isnothing(first_dt)
            @warn "[$i/$total] $sym 15min — no existing CSV, skipping (run collect_nse_ohlcv.jl first)"
            failed += 1; continue
        end
        first_date = Date(first_dt)

        floor_date = _effective_from(sym, from_target, earliest, confirmed_floor)
        if first_date <= floor_date
            current += 1; continue
        end

        to     = first_date
        gap    = (to - floor_date).value + 1
        n_chks = ceil(Int, gap / FIFTEENMIN_CHUNK_DAYS)

        if dry_run
            @info "[$i/$total] $sym 15min: would backfill $floor_date → $to ($gap days, ~$n_chks call$(n_chks==1 ? "" : "s"))"
            updated += 1; continue
        end

        token = get(token_map, sym, nothing)
        if isnothing(token)
            logboth(slog, "[$i/$total] $sym — no instrument token"; warn=true); failed += 1; continue
        end

        older_df = fetch_ohlcv_15min(token, floor_date, to, session)
        filter!(row -> row.datetime < first_dt, older_df)

        if isempty(older_df)
            logf(slog, "[$i/$total] $sym 15min — no older bars before $first_dt")
            confirmed_floor[sym] = first_date
            save_earliest_trading_day(confirmed_floor, confirmed_path)
            confirmed_added += 1
            failed += 1; continue
        end

        _prepend_merge!(path, older_df, :datetime, DateTime)
        updated += 1
        logf(slog, "[$i/$total] $sym 15min +$(nrow(older_df)) older bars ($(older_df.datetime[1]) → $(older_df.datetime[end]))")
    end

    confirmed_added > 0 &&
        logboth(slog, "  Confirmed-empty 15min floor: +$confirmed_added new, $(length(confirmed_floor)) total → $confirmed_path")

    summary = dry_run ?
        "15min: $updated would be backfilled, $current already at their floor" :
        "15min: $updated backfilled, $current already at their floor, $failed failed"
    dry_run ? (@info summary) : logboth(slog, summary)
end

function backfill_1min!(symbols, token_map, session, from_target::Date, earliest::Dict{String, Date}, slog::ScriptLog;
                        dry_run::Bool, out_dir::String)
    current = updated = failed = 0
    total   = length(symbols)

    confirmed_path  = confirmed_floor_file(out_dir)
    confirmed_floor = load_earliest_trading_day(confirmed_path)
    confirmed_added = 0

    for (i, sym) in enumerate(symbols)
        if !dry_run && i % HEARTBEAT_INTERVAL == 0
            logboth(slog, "[$i/$total] 1min progress: $updated backfilled, $current at floor, $failed failed so far")
        end

        path = joinpath(out_dir, "$sym.csv")
        first_dt = _first_1min_datetime(sym, out_dir)
        if isnothing(first_dt)
            @warn "[$i/$total] $sym 1min — no existing CSV, skipping (run collect_nse_ohlcv.jl first)"
            failed += 1; continue
        end
        first_date = Date(first_dt)

        floor_date = _effective_from(sym, from_target, earliest, confirmed_floor)
        if first_date <= floor_date
            current += 1; continue
        end

        to     = first_date
        gap    = (to - floor_date).value + 1
        n_chks = ceil(Int, gap / ONEMIN_CHUNK_DAYS)

        if dry_run
            @info "[$i/$total] $sym 1min: would backfill $floor_date → $to ($gap days, ~$n_chks call$(n_chks==1 ? "" : "s"))"
            updated += 1; continue
        end

        token = get(token_map, sym, nothing)
        if isnothing(token)
            logboth(slog, "[$i/$total] $sym — no instrument token"; warn=true); failed += 1; continue
        end

        older_df = fetch_ohlcv_1min(token, floor_date, to, session)
        filter!(row -> row.datetime < first_dt, older_df)

        if isempty(older_df)
            logf(slog, "[$i/$total] $sym 1min — no older bars before $first_dt")
            confirmed_floor[sym] = first_date
            save_earliest_trading_day(confirmed_floor, confirmed_path)
            confirmed_added += 1
            failed += 1; continue
        end

        _prepend_merge!(path, older_df, :datetime, DateTime)
        updated += 1
        logf(slog, "[$i/$total] $sym 1min +$(nrow(older_df)) older bars ($(older_df.datetime[1]) → $(older_df.datetime[end]))")
    end

    confirmed_added > 0 &&
        logboth(slog, "  Confirmed-empty 1min floor: +$confirmed_added new, $(length(confirmed_floor)) total → $confirmed_path")

    summary = dry_run ?
        "1min: $updated would be backfilled, $current already at their floor" :
        "1min: $updated backfilled, $current already at their floor, $failed failed"
    dry_run ? (@info summary) : logboth(slog, summary)
end

# ── Entry point ───────────────────────────────────────────────────────────────

function main()
    if "--help" in ARGS || "-h" in ARGS
        println("""
Usage:
  julia --project=packages/StockSwingPredictor scripts/backfill_ohlcv.jl [FLAGS]

Flags:
  --from DATE     Extend every selected CSV back to this date (default: 2010-01-04,
                  the real earliest NSE trading day of 2010 — see DEFAULT_FROM).
  --symbol SYM    Backfill only this one symbol (applied to whichever exchange(s) run).
  --nse-only      Backfill NSE only (default: both NSE and BSE). Mutually exclusive with --bse-only.
  --bse-only      Backfill BSE only (default: both NSE and BSE). Mutually exclusive with --nse-only.
  --daily-only    Backfill only daily bars.
  --hourly-only   Backfill only hourly (60-min) bars.
  --5min-only     Backfill only 5-minute bars.
  --15min-only    Backfill only 15-minute bars.
  --1min-only     Backfill only 1-minute bars.
  --skip-daily    Skip the daily pass (overrides --daily-only if both given).
  --skip-hourly   Skip the hourly pass (overrides --hourly-only if both given).
  --skip-5min     Skip the 5-minute pass (overrides --5min-only if both given).
  --skip-15min    Skip the 15-minute pass (overrides --15min-only if both given).
  --skip-1min     Skip the 1-minute pass (overrides --1min-only if both given).
  --dry-run       Report what would be fetched without making any API calls.
  --log-file PATH Full per-symbol detail (default: $DEFAULT_LOG_FILE;
                  with --account N>=2: <name>.accountN.log).
  --account N     Kite API-key slot (1 = KITE_HISTORICAL_*, 2 = KITE_HISTORICAL2_*, …);
                  rate limits are per key. Needs: node sidecar/kite_login.js --account N.
                  The terminal only shows stage headers, a progress
                  heartbeat every $HEARTBEAT_INTERVAL symbols, and warnings —
                  tail -f the log file for live per-symbol status instead.

For each existing {SYMBOL}.csv in website/data/ohlcv/{nse,bse}/{daily,
hourly,5min,15min,1min}/, reads the earliest date/datetime already on disk
and fetches only the older gap down to --from, merging it into the file. A
symbol with no existing CSV is skipped — this tool only EXTENDS, it doesn't
do the initial collection (see collect_nse_ohlcv.jl/collect_bse_ohlcv.jl for
that, including brand-new granularities like NSE 15-min that have zero
files today).

After each daily pass, {nse,bse}/earliest_trading_day.json is rebuilt from
the current daily CSVs. Hourly/5min/15min/1min then use
max(--from, earliest_trading_day[symbol]) as their actual floor, so a
symbol daily already proved has no data before some date (e.g. a 2021 IPO)
is never re-probed from --from at the finer granularities — see the module
docstring above for why this matters.

Each of hourly/5min/15min/1min ALSO keeps its own
{granularity}/confirmed_floor.json (inside that granularity's own
subfolder), recording the first time a "no older bars" result is seen for
a symbol at that specific granularity. This is a separate floor from
earliest_trading_day.json because Kite's own retention floor for each
intraday granularity is shallower than daily's and shared across symbols
regardless of IPO date (hourly, for example, has nothing before
~2015-02-02 for virtually every NSE symbol) — without this, every such
symbol's full multi-chunk gap gets silently re-probed and re-confirmed
empty on every single run, forever.

NOTE: Kite's per-interval day limits (60/100/200/400/2000 days for
1min/5min/15min/60min/day) are a single-request span cap, not a retention
cliff (verified live) — --from 2010-01-04 is achievable for every
granularity, it just costs more chunked API calls (and disk) the further
back and the finer the granularity. Use --dry-run to see the call-count
estimate before committing to a full run.
""")
        return
    end

    from_idx = findfirst(==("--from"), ARGS)
    from_target = if isnothing(from_idx)
        DEFAULT_FROM
    else
        from_idx == length(ARGS) && error("--from given with no DATE after it — see --help")
        Date(ARGS[from_idx + 1])
    end

    daily_only      = "--daily-only"   in ARGS
    hourly_only     = "--hourly-only"  in ARGS
    fivemin_only    = "--5min-only"    in ARGS
    fifteenmin_only = "--15min-only"   in ARGS
    onemin_only     = "--1min-only"    in ARGS
    skip_daily      = "--skip-daily"   in ARGS
    skip_hourly     = "--skip-hourly"  in ARGS
    skip_5min       = "--skip-5min"    in ARGS
    skip_15min      = "--skip-15min"   in ARGS
    skip_1min       = "--skip-1min"    in ARGS
    nse_only        = "--nse-only"     in ARGS
    bse_only        = "--bse-only"     in ARGS
    dry_run         = "--dry-run"      in ARGS

    nse_only && bse_only && error("--nse-only and --bse-only are mutually exclusive.")
    run_nse = !bse_only
    run_bse = !nse_only

    any_only   = daily_only || hourly_only || fivemin_only || fifteenmin_only || onemin_only
    run_daily  = (!any_only || daily_only)      && !skip_daily
    run_hourly = (!any_only || hourly_only)     && !skip_hourly
    run_5min   = (!any_only || fivemin_only)    && !skip_5min
    run_15min  = (!any_only || fifteenmin_only) && !skip_15min
    run_1min   = (!any_only || onemin_only)     && !skip_1min

    sym_idx    = findfirst(==("--symbol"), ARGS)
    sym_filter = (!isnothing(sym_idx) && sym_idx < length(ARGS)) ? ARGS[sym_idx + 1] : nothing

    log_idx  = findfirst(==("--log-file"), ARGS)
    log_path = (!isnothing(log_idx) && log_idx < length(ARGS)) ? ARGS[log_idx + 1] :
               account_log_path(DEFAULT_LOG_FILE, kite_account_from_args())
    slog     = open_script_log(log_path)
    @info "Logging full per-symbol detail to: $log_path"

    dry_run && @info "[DRY RUN] No API calls will be made."
    logboth(slog, "Backfilling to: $from_target")

    session = dry_run ? (api_key="", access_token="") :
                        load_kite_session(REPO_ROOT; account=kite_account_from_args())

    nse_token_map = if !run_nse || dry_run
        Dict{String, Int}()
    else
        @info "Loading NSE instrument list from Kite…"
        instr = load_instruments(session; exchange="NSE")
        t = build_token_map(instr; exchange="NSE")
        logboth(slog, "  $(length(t)) NSE instruments loaded")
        t
    end

    bse_token_map = if !run_bse || dry_run
        Dict{String, Int}()
    else
        @info "Loading BSE instrument list from Kite…"
        instr = load_instruments(session; exchange="BSE", refresh=true)
        t = build_token_map(instr; exchange="BSE")
        logboth(slog, "  $(length(t)) BSE instruments loaded")
        t
    end

    # ── NSE backfill ──────────────────────────────────────────────────────────
    if run_nse
        isdir(NSE_OHLCV_DIR) || error("NSE OHLCV directory not found: $NSE_OHLCV_DIR\n" *
                                       "Run collect_nse_ohlcv.jl first (or pass --bse-only).")

        daily_syms      = [replace(f, ".csv" => "") for f in _readdir_safe(nse_gran_dir("daily"))]
        hourly_syms     = [replace(f, ".csv" => "") for f in _readdir_safe(nse_gran_dir("hourly"))]
        fivemin_syms    = [replace(f, ".csv" => "") for f in _readdir_safe(nse_gran_dir("5min"))]
        fifteenmin_syms = [replace(f, ".csv" => "") for f in _readdir_safe(nse_gran_dir("15min"))]
        onemin_syms     = [replace(f, ".csv" => "") for f in _readdir_safe(nse_gran_dir("1min"))]

        if !isnothing(sym_filter)
            daily_syms      = filter(==(sym_filter), daily_syms)
            hourly_syms     = filter(==(sym_filter), hourly_syms)
            fivemin_syms    = filter(==(sym_filter), fivemin_syms)
            fifteenmin_syms = filter(==(sym_filter), fifteenmin_syms)
            onemin_syms     = filter(==(sym_filter), onemin_syms)
            isempty(daily_syms) && isempty(hourly_syms) &&
            isempty(fivemin_syms) && isempty(fifteenmin_syms) && isempty(onemin_syms) &&
                error("No existing CSV found for symbol '$sym_filter' in $NSE_OHLCV_DIR")
            @info "Filtering to symbol: $sym_filter"
        end

        logboth(slog, "Found $(length(daily_syms)) NSE daily, $(length(hourly_syms)) hourly, $(length(fivemin_syms)) 5-min, $(length(fifteenmin_syms)) 15-min, $(length(onemin_syms)) 1-min CSVs")
        logboth(slog, "═══ NSE ═══")

        nse_earliest = load_earliest_trading_day(nse_earliest_file())

        if run_daily
            logboth(slog, "── Daily bars ──")
            backfill_daily!(daily_syms, nse_token_map, session, from_target, nse_earliest, slog;
                            dry_run, out_dir=nse_gran_dir("daily"))
            if !dry_run
                nse_earliest = build_earliest_trading_day(nse_gran_dir("daily"))
                save_earliest_trading_day(nse_earliest, nse_earliest_file())
                logboth(slog, "  Updated earliest-trading-day reference: $(length(nse_earliest)) symbols → $(nse_earliest_file())")
            end
        end

        if run_hourly
            logboth(slog, "── Hourly bars ──")
            backfill_hourly!(hourly_syms, nse_token_map, session, from_target, nse_earliest, slog;
                             dry_run, out_dir=nse_gran_dir("hourly"))
        end

        if run_5min
            logboth(slog, "── 5-min bars ──")
            backfill_5min!(fivemin_syms, nse_token_map, session, from_target, nse_earliest, slog;
                           dry_run, out_dir=nse_gran_dir("5min"))
        end

        if run_15min
            logboth(slog, "── 15-min bars ──")
            backfill_15min!(fifteenmin_syms, nse_token_map, session, from_target, nse_earliest, slog;
                            dry_run, out_dir=nse_gran_dir("15min"))
        end

        if run_1min
            logboth(slog, "── 1-min bars ──")
            backfill_1min!(onemin_syms, nse_token_map, session, from_target, nse_earliest, slog;
                           dry_run, out_dir=nse_gran_dir("1min"))
        end
    end

    # ── BSE backfill ──────────────────────────────────────────────────────────
    if run_bse
        if isdir(BSE_OHLCV_DIR)
            bse_daily_syms      = [replace(f, ".csv" => "") for f in _readdir_safe(bse_gran_dir("daily"))]
            bse_hourly_syms     = [replace(f, ".csv" => "") for f in _readdir_safe(bse_gran_dir("hourly"))]
            bse_5min_syms       = [replace(f, ".csv" => "") for f in _readdir_safe(bse_gran_dir("5min"))]
            bse_15min_syms      = [replace(f, ".csv" => "") for f in _readdir_safe(bse_gran_dir("15min"))]
            bse_1min_syms       = [replace(f, ".csv" => "") for f in _readdir_safe(bse_gran_dir("1min"))]

            if !isnothing(sym_filter)
                bse_daily_syms  = filter(==(sym_filter), bse_daily_syms)
                bse_hourly_syms = filter(==(sym_filter), bse_hourly_syms)
                bse_5min_syms   = filter(==(sym_filter), bse_5min_syms)
                bse_15min_syms  = filter(==(sym_filter), bse_15min_syms)
                bse_1min_syms   = filter(==(sym_filter), bse_1min_syms)
            end

            logboth(slog, "Found $(length(bse_daily_syms)) BSE daily, $(length(bse_hourly_syms)) hourly, $(length(bse_5min_syms)) 5-min, $(length(bse_15min_syms)) 15-min, $(length(bse_1min_syms)) 1-min CSVs")
            logboth(slog, "═══ BSE ═══")

            bse_earliest = load_earliest_trading_day(bse_earliest_file())

            if run_daily
                logboth(slog, "── Daily bars ──")
                backfill_daily!(bse_daily_syms, bse_token_map, session, from_target, bse_earliest, slog;
                                dry_run, out_dir=bse_gran_dir("daily"))
                if !dry_run
                    bse_earliest = build_earliest_trading_day(bse_gran_dir("daily"))
                    save_earliest_trading_day(bse_earliest, bse_earliest_file())
                    logboth(slog, "  Updated earliest-trading-day reference: $(length(bse_earliest)) symbols → $(bse_earliest_file())")
                end
            end

            if run_hourly
                logboth(slog, "── Hourly bars ──")
                backfill_hourly!(bse_hourly_syms, bse_token_map, session, from_target, bse_earliest, slog;
                                 dry_run, out_dir=bse_gran_dir("hourly"))
            end

            if run_5min
                logboth(slog, "── 5-min bars ──")
                backfill_5min!(bse_5min_syms, bse_token_map, session, from_target, bse_earliest, slog;
                               dry_run, out_dir=bse_gran_dir("5min"))
            end

            if run_15min
                logboth(slog, "── 15-min bars ──")
                backfill_15min!(bse_15min_syms, bse_token_map, session, from_target, bse_earliest, slog;
                                dry_run, out_dir=bse_gran_dir("15min"))
            end

            if run_1min
                logboth(slog, "── 1-min bars ──")
                backfill_1min!(bse_1min_syms, bse_token_map, session, from_target, bse_earliest, slog;
                               dry_run, out_dir=bse_gran_dir("1min"))
            end
        else
            @warn "BSE directory not found ($BSE_OHLCV_DIR) — run collect_bse_ohlcv.jl first (or pass --nse-only)."
        end
    end

    close_script_log(slog, "exit normally")
end

main()
