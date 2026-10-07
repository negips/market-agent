"""
update_ohlcv.jl

Incrementally update all OHLCV CSVs in website/data/ohlcv/{nse,bse}/{daily,
hourly,5min,15min,1min}/ (both exchanges by default — pass --nse-only or
--bse-only to restrict to one; macro CSVs under website/data/ohlcv/macro/
keep their own flat, suffixed layout — too few files to need subfolders)
with bars added since the last collection run.

For each existing CSV, reads the last date/datetime in the file and fetches
only the gap since then. Symbols already current are skipped. Appends new
rows in-place — no full-file rewrite needed.

Run daily after kite_login.js. Missing a day never loses data — every
granularity here, including 1-minute, can still be fetched arbitrarily far
back (verified live against the real API; Kite's per-interval day limits are
a single-request span cap, not a retention cliff — see kite_data.jl). The
reason to run this daily anyway is purely practical: incremental is one
small chunked request per symbol, where catching up a long-neglected gap
means re-chunking the whole gap, which is slower and heavier per run the
longer it's left.

Prerequisites:
  - sidecar/kite_session.json present         (node sidecar/kite_login.js)
  - website/data/ohlcv/nse/{daily,hourly,5min,15min,1min}/ populated
    (run collect_nse_ohlcv.jl first, unless --bse-only)
  - website/data/ohlcv/bse/{daily,hourly,5min,15min,1min}/ populated
    (run collect_bse_ohlcv.jl first, unless --nse-only)

Usage:
  julia --project=packages/StockSwingPredictor scripts/update_ohlcv.jl
  julia --project=packages/StockSwingPredictor scripts/update_ohlcv.jl --nse-only
  julia --project=packages/StockSwingPredictor scripts/update_ohlcv.jl --bse-only
  julia --project=packages/StockSwingPredictor scripts/update_ohlcv.jl --daily-only
  julia --project=packages/StockSwingPredictor scripts/update_ohlcv.jl --hourly-only
  julia --project=packages/StockSwingPredictor scripts/update_ohlcv.jl --5min-only
  julia --project=packages/StockSwingPredictor scripts/update_ohlcv.jl --15min-only
  julia --project=packages/StockSwingPredictor scripts/update_ohlcv.jl --1min-only
  julia --project=packages/StockSwingPredictor scripts/update_ohlcv.jl --skip-5min
  julia --project=packages/StockSwingPredictor scripts/update_ohlcv.jl --skip-15min
  julia --project=packages/StockSwingPredictor scripts/update_ohlcv.jl --skip-1min
  julia --project=packages/StockSwingPredictor scripts/update_ohlcv.jl --dry-run
  julia --project=packages/StockSwingPredictor scripts/update_ohlcv.jl --symbol RELIANCE
"""

using StockSwingPredictor, CSV, DataFrames, Dates

const REPO_ROOT     = joinpath(@__DIR__, "..")
const OHLCV_ROOT    = joinpath(REPO_ROOT, "website", "data", "ohlcv")
const NSE_OHLCV_DIR = joinpath(OHLCV_ROOT, "nse")
const BSE_OHLCV_DIR = joinpath(OHLCV_ROOT, "bse")
const MACRO_DIR     = joinpath(OHLCV_ROOT, "macro")   # macro stays flat/suffixed — not part of this layout

# One subfolder per granularity under each exchange root (e.g.
# website/data/ohlcv/nse/hourly/) — see kite_data.jl's module docstring.
# Macro is intentionally excluded: it's a handful of files, not thousands,
# so it keeps its original flat {NAME}_5min.csv/{NAME}_15min.csv layout.
nse_gran_dir(granularity::String) = joinpath(NSE_OHLCV_DIR, granularity)
bse_gran_dir(granularity::String) = joinpath(BSE_OHLCV_DIR, granularity)

"""`readdir`, but `String[]` instead of an error when `dir` doesn't exist yet
— e.g. a freshly-added granularity's subfolder before its first collection
run."""
_readdir_safe(dir::String) = isdir(dir) ? readdir(dir) : String[]

# ── Helpers ───────────────────────────────────────────────────────────────────

"""
Read only the date/datetime column from a CSV and return the maximum value.
Uses `select` so it never loads OHLCV columns for large files.
Returns `nothing` if the file is missing, empty, or every row fails to parse.

A row that fails to parse as `T` (e.g. two bars glued together with no
newline between them, from an interrupted or concurrent append — the CSV
appends in this script are not safe to run two-at-once against the same
file) becomes `missing` rather than raising here; `maximum` alone would then
silently propagate that single `missing` to the *whole file's* result,
turning one corrupted row into "I can't tell you anything about this file"
instead of "one row was bad, here's the latest good timestamp." We skip
missing rows and warn instead, so the caller still gets a usable answer and
the corruption is visible rather than surfacing later as an unrelated
`MethodError` deep in `Dates`.
"""
function _last_value(path::String, col::Symbol, T::Type)
    isfile(path) || return nothing
    df = try
        CSV.read(path, DataFrame; select=[col], types=Dict(col => T))
    catch e
        @warn "Could not read $path: $(sprint(showerror, e))"
        return nothing
    end
    isempty(df) && return nothing

    n_bad = count(ismissing, df[!, col])
    n_bad > 0 && @warn "$path: $n_bad row(s) failed to parse as $T — file may be " *
                        "corrupted (e.g. two bars glued together from an interrupted " *
                        "or concurrent write). Ignoring them for the last-known-good " *
                        "timestamp; inspect the file directly if this recurs."

    good = skipmissing(df[!, col])
    isempty(good) && return nothing
    return maximum(good)
end

_last_daily_date(sym::String, dir::String=nse_gran_dir("daily")) =
    _last_value(joinpath(dir, "$sym.csv"), :date,     Date)
_last_hourly_datetime(sym::String, dir::String=nse_gran_dir("hourly")) =
    _last_value(joinpath(dir, "$sym.csv"), :datetime, DateTime)
_last_5min_datetime(sym::String, dir::String=nse_gran_dir("5min")) =
    _last_value(joinpath(dir, "$sym.csv"), :datetime, DateTime)
_last_15min_datetime(sym::String, dir::String=nse_gran_dir("15min")) =
    _last_value(joinpath(dir, "$sym.csv"), :datetime, DateTime)
_last_1min_datetime(sym::String, dir::String=nse_gran_dir("1min")) =
    _last_value(joinpath(dir, "$sym.csv"), :datetime, DateTime)
_last_macro_5min_datetime(name::String) =
    _last_value(joinpath(MACRO_DIR, "$(name)_5min.csv"),  :datetime, DateTime)
_last_macro_15min_datetime(name::String) =
    _last_value(joinpath(MACRO_DIR, "$(name)_15min.csv"), :datetime, DateTime)

# Last bar times for "day complete" checks (IST)
const HOURLY_LAST_BAR      = Time(15,  0, 0)   # last 60-min bar opens at 15:00
const FIVEMIN_LAST_BAR     = Time(15, 25, 0)   # last 5-min bar opens at 15:25 (NSE)
const FIFTEENMIN_LAST_BAR  = Time(15, 15, 0)   # last 15-min bar opens at 15:15 (NSE)
const ONEMIN_LAST_BAR      = Time(15, 29, 0)   # last 1-min bar opens at 15:29 (NSE)
const MCX_FIVEMIN_LAST_BAR    = Time(23, 25, 0)  # last 5-min bar opens at 23:25 (MCX)
const MCX_FIFTEENMIN_LAST_BAR = Time(23, 15, 0)  # last 15-min bar opens at 23:15 (MCX)

# How often (in symbols) a loop prints a console progress heartbeat — see
# ScriptLog's docstring (StockSwingPredictor/src/script_log.jl) for why
# routine per-symbol detail otherwise goes to the log file only.
const HEARTBEAT_INTERVAL = 100

# Full per-symbol detail goes here; the terminal only gets stage headers,
# the progress heartbeat, and warnings. Override with --log-file.
const DEFAULT_LOG_FILE = joinpath(OHLCV_ROOT, "logs", "update_ohlcv.log")

# ── Core update loops ─────────────────────────────────────────────────────────

function update_daily!(symbols, token_map, session, yest::Date, slog::ScriptLog;
                       dry_run::Bool, out_dir::String=nse_gran_dir("daily"))
    current = updated = failed = 0
    total   = length(symbols)

    for (i, sym) in enumerate(symbols)
        if !dry_run && i % HEARTBEAT_INTERVAL == 0
            logboth(slog, "[$i/$total] daily progress: $updated updated, $current current, $failed failed so far")
        end

        last = _last_daily_date(sym, out_dir)
        if isnothing(last)
            @warn "[$i/$total] $sym daily — could not determine last date, skipping"
            failed += 1
            continue
        end

        from = last + Day(1)
        if from > yest
            current += 1
            continue
        end

        gap = (yest - from).value + 1
        if dry_run
            @info "[$i/$total] $sym daily: would fetch $from → $yest ($gap calendar days)"
            updated += 1
            continue
        end

        token = get(token_map, sym, nothing)
        if isnothing(token)
            logboth(slog, "[$i/$total] $sym — no instrument token"; warn=true)
            failed += 1
            continue
        end

        new_df = fetch_ohlcv(token, from, yest, session)
        if isempty(new_df)
            # Normal for a holiday gap with no trading days in the range.
            logf(slog, "[$i/$total] $sym daily — no new bars in $from…$yest (holiday gap?)")
            failed += 1
            sleep(0.35)
            continue
        end

        path = joinpath(out_dir, "$sym.csv")
        CSV.write(path, new_df; append=true)
        updated += 1
        logf(slog, "[$i/$total] $sym daily +$(nrow(new_df)) bars ($from → $yest)")
        sleep(0.35)
    end

    summary = dry_run ?
        "Daily: $updated would be updated, $current already current" :
        "Daily: $updated updated, $current already current, $failed failed"
    dry_run ? (@info summary) : logboth(slog, summary)
end

function update_hourly!(symbols, token_map, session, yest::Date, slog::ScriptLog;
                        dry_run::Bool, out_dir::String=nse_gran_dir("hourly"))
    current = updated = failed = 0
    total   = length(symbols)

    for (i, sym) in enumerate(symbols)
        if !dry_run && i % HEARTBEAT_INTERVAL == 0
            logboth(slog, "[$i/$total] hourly progress: $updated updated, $current current, $failed failed so far")
        end

        last_dt = _last_hourly_datetime(sym, out_dir)
        if isnothing(last_dt)
            @warn "[$i/$total] $sym hourly — could not determine last datetime, skipping"
            failed += 1
            continue
        end

        last_date = Date(last_dt)
        last_time = Time(last_dt)

        # Skip only when the last day is fully collected (last bar at or after
        # HOURLY_LAST_BAR) and that day is already yesterday or later.
        if last_date > yest || (last_date == yest && last_time >= HOURLY_LAST_BAR)
            current += 1
            continue
        end

        # Always start from last_date so a partial last day gets completed.
        # Rows already in the file are stripped after the fetch via datetime filter.
        from = last_date
        gap  = (yest - from).value + 1
        n_chunks = ceil(Int, gap / 59)

        if dry_run
            partial_note = last_time < HOURLY_LAST_BAR ? " (last bar $(last_time), day incomplete)" : ""
            @info "[$i/$total] $sym hourly: would fetch $from → $yest ($gap days, ~$n_chunks API call$(n_chunks == 1 ? "" : "s"))$partial_note"
            updated += 1
            continue
        end

        token = get(token_map, sym, nothing)
        if isnothing(token)
            logboth(slog, "[$i/$total] $sym — no instrument token"; warn=true)
            failed += 1
            continue
        end

        new_df = fetch_ohlcv_hourly(token, from, yest, session)

        # Drop bars already present in the file (covers the partial last day).
        filter!(row -> row.datetime > last_dt, new_df)

        if isempty(new_df)
            logf(slog, "[$i/$total] $sym hourly — no new bars after $last_dt")
            failed += 1
            continue
        end

        path = joinpath(out_dir, "$sym.csv")
        CSV.write(path, new_df; append=true)
        updated += 1
        logf(slog, "[$i/$total] $sym hourly +$(nrow(new_df)) bars (from $(new_df.datetime[1]) → $(new_df.datetime[end]))")
    end

    summary = dry_run ?
        "Hourly: $updated would be updated, $current already current" :
        "Hourly: $updated updated, $current already current, $failed failed"
    dry_run ? (@info summary) : logboth(slog, summary)
end

function update_5min!(symbols, token_map, session, yest::Date, slog::ScriptLog;
                      dry_run::Bool, out_dir::String=nse_gran_dir("5min"))
    current = updated = failed = 0
    total   = length(symbols)

    for (i, sym) in enumerate(symbols)
        if !dry_run && i % HEARTBEAT_INTERVAL == 0
            logboth(slog, "[$i/$total] 5min progress: $updated updated, $current current, $failed failed so far")
        end

        last_dt = _last_5min_datetime(sym, out_dir)
        if isnothing(last_dt)
            @warn "[$i/$total] $sym 5min — no existing CSV, skipping (run collect_nse_ohlcv.jl first)"
            failed += 1; continue
        end

        last_date = Date(last_dt)
        last_time = Time(last_dt)

        if last_date > yest || (last_date == yest && last_time >= FIVEMIN_LAST_BAR)
            current += 1; continue
        end

        from   = last_date
        gap    = (yest - from).value + 1
        n_chks = ceil(Int, gap / 90)

        if dry_run
            partial = last_time < FIVEMIN_LAST_BAR ? " (partial last day at $last_time)" : ""
            @info "[$i/$total] $sym 5min: would fetch $from → $yest ($gap days, ~$n_chks call$(n_chks==1 ? "" : "s"))$partial"
            updated += 1; continue
        end

        token = get(token_map, sym, nothing)
        if isnothing(token)
            logboth(slog, "[$i/$total] $sym — no instrument token"; warn=true); failed += 1; continue
        end

        new_df = fetch_ohlcv_5min(token, from, yest, session)
        filter!(row -> row.datetime > last_dt, new_df)

        if isempty(new_df)
            logf(slog, "[$i/$total] $sym 5min — no new bars after $last_dt")
            failed += 1; continue
        end

        path = joinpath(out_dir, "$sym.csv")
        CSV.write(path, new_df; append=true)
        updated += 1
        logf(slog, "[$i/$total] $sym 5min +$(nrow(new_df)) bars ($(new_df.datetime[1]) → $(new_df.datetime[end]))")
    end

    summary = dry_run ?
        "5min: $updated would be updated, $current already current" :
        "5min: $updated updated, $current already current, $failed failed"
    dry_run ? (@info summary) : logboth(slog, summary)
end

function update_macro_5min!(session, yest::Date; dry_run::Bool)
    macro_dir = MACRO_DIR
    isdir(macro_dir) || return

    token_map = dry_run ? Dict{String, Tuple{Int,Bool}}() :
                          build_macro_kite_tokens(session)

    current = updated = failed = 0

    for inst in KITE_MACRO_INSTRUMENTS
        last_dt = _last_macro_5min_datetime(inst.name)
        if isnothing(last_dt)
            @warn "  $(inst.name) 5min — no existing CSV, skipping (run collect_nse_ohlcv.jl first)"
            failed += 1; continue
        end

        last_date = Date(last_dt)
        last_time = Time(last_dt)
        # Use MCX threshold for MCX instruments, NSE threshold for INDIA_VIX
        last_bar = inst.exchange == "NSE" ? FIVEMIN_LAST_BAR : MCX_FIVEMIN_LAST_BAR

        if last_date > yest || (last_date == yest && last_time >= last_bar)
            current += 1; continue
        end

        from = last_date
        gap  = (yest - from).value + 1

        if dry_run
            @info "  $(inst.name) 5min: would fetch $from → $yest ($gap days)"
            updated += 1; continue
        end

        entry = get(token_map, inst.name, nothing)
        if isnothing(entry)
            @warn "  $(inst.name) — token not found"; failed += 1; continue
        end
        token, continuous = entry

        new_df = fetch_kite_macro_5min(token, from, yest, session)
        filter!(row -> row.datetime > last_dt, new_df)

        if isempty(new_df)
            @debug "  $(inst.name) 5min — no new bars after $last_dt"
            failed += 1; continue
        end

        path = joinpath(macro_dir, "$(inst.name)_5min.csv")
        CSV.write(path, new_df; append=true)
        updated += 1
        @info "  $(inst.name) 5min +$(nrow(new_df)) bars ($(new_df.datetime[1]) → $(new_df.datetime[end]))"
    end

    if dry_run
        @info "Macro 5min: $updated would be updated, $current already current"
    else
        @info "Macro 5min: $updated updated, $current already current, $failed failed"
    end
end

function update_15min!(symbols, token_map, session, yest::Date, slog::ScriptLog;
                       dry_run::Bool, out_dir::String=nse_gran_dir("15min"))
    current = updated = failed = 0
    total   = length(symbols)

    for (i, sym) in enumerate(symbols)
        if !dry_run && i % HEARTBEAT_INTERVAL == 0
            logboth(slog, "[$i/$total] 15min progress: $updated updated, $current current, $failed failed so far")
        end

        last_dt = _last_15min_datetime(sym, out_dir)
        if isnothing(last_dt)
            @warn "[$i/$total] $sym 15min — no existing CSV, skipping (run collect_nse_ohlcv.jl first)"
            failed += 1; continue
        end

        last_date = Date(last_dt)
        last_time = Time(last_dt)

        if last_date > yest || (last_date == yest && last_time >= FIFTEENMIN_LAST_BAR)
            current += 1; continue
        end

        from   = last_date
        gap    = (yest - from).value + 1
        n_chks = ceil(Int, gap / 175)

        if dry_run
            partial = last_time < FIFTEENMIN_LAST_BAR ? " (partial last day at $last_time)" : ""
            @info "[$i/$total] $sym 15min: would fetch $from → $yest ($gap days, ~$n_chks call$(n_chks==1 ? "" : "s"))$partial"
            updated += 1; continue
        end

        token = get(token_map, sym, nothing)
        if isnothing(token)
            logboth(slog, "[$i/$total] $sym — no instrument token"; warn=true); failed += 1; continue
        end

        new_df = fetch_ohlcv_15min(token, from, yest, session)
        filter!(row -> row.datetime > last_dt, new_df)

        if isempty(new_df)
            logf(slog, "[$i/$total] $sym 15min — no new bars after $last_dt")
            failed += 1; continue
        end

        path = joinpath(out_dir, "$sym.csv")
        CSV.write(path, new_df; append=true)
        updated += 1
        logf(slog, "[$i/$total] $sym 15min +$(nrow(new_df)) bars ($(new_df.datetime[1]) → $(new_df.datetime[end]))")
    end

    summary = dry_run ?
        "15min: $updated would be updated, $current already current" :
        "15min: $updated updated, $current already current, $failed failed"
    dry_run ? (@info summary) : logboth(slog, summary)
end

function update_1min!(symbols, token_map, session, yest::Date, slog::ScriptLog;
                      dry_run::Bool, out_dir::String=nse_gran_dir("1min"))
    current = updated = failed = 0
    total   = length(symbols)

    for (i, sym) in enumerate(symbols)
        if !dry_run && i % HEARTBEAT_INTERVAL == 0
            logboth(slog, "[$i/$total] 1min progress: $updated updated, $current current, $failed failed so far")
        end

        last_dt = _last_1min_datetime(sym, out_dir)
        if isnothing(last_dt)
            @warn "[$i/$total] $sym 1min — no existing CSV, skipping (run collect_nse_ohlcv.jl first)"
            failed += 1; continue
        end

        last_date = Date(last_dt)
        last_time = Time(last_dt)

        if last_date > yest || (last_date == yest && last_time >= ONEMIN_LAST_BAR)
            current += 1; continue
        end

        # 60 days is Kite's per-request span cap for this interval, not a
        # retention cliff (verified live) — fetch_ohlcv_1min already chunks
        # under it, so a gap wider than that just costs more chunked calls
        # here, nothing is unrecoverable.
        from   = last_date
        gap    = (yest - from).value + 1
        n_chks = ceil(Int, gap / 55)

        if dry_run
            partial = last_time < ONEMIN_LAST_BAR ? " (partial last day at $last_time)" : ""
            @info "[$i/$total] $sym 1min: would fetch $from → $yest ($gap days, ~$n_chks call$(n_chks==1 ? "" : "s"))$partial"
            updated += 1; continue
        end

        token = get(token_map, sym, nothing)
        if isnothing(token)
            logboth(slog, "[$i/$total] $sym — no instrument token"; warn=true); failed += 1; continue
        end

        new_df = fetch_ohlcv_1min(token, from, yest, session)
        filter!(row -> row.datetime > last_dt, new_df)

        if isempty(new_df)
            logf(slog, "[$i/$total] $sym 1min — no new bars after $last_dt")
            failed += 1; continue
        end

        path = joinpath(out_dir, "$sym.csv")
        CSV.write(path, new_df; append=true)
        updated += 1
        logf(slog, "[$i/$total] $sym 1min +$(nrow(new_df)) bars ($(new_df.datetime[1]) → $(new_df.datetime[end]))")
    end

    summary = dry_run ?
        "1min: $updated would be updated, $current already current" :
        "1min: $updated updated, $current already current, $failed failed"
    dry_run ? (@info summary) : logboth(slog, summary)
end

function update_macro_15min!(session, yest::Date; dry_run::Bool)
    macro_dir = MACRO_DIR
    isdir(macro_dir) || return

    token_map = dry_run ? Dict{String, Tuple{Int,Bool}}() :
                          build_macro_kite_tokens(session)

    current = updated = failed = 0

    for inst in KITE_MACRO_INSTRUMENTS
        last_dt = _last_macro_15min_datetime(inst.name)
        if isnothing(last_dt)
            @warn "  $(inst.name) 15min — no existing CSV, skipping (run collect_nse_ohlcv.jl first)"
            failed += 1; continue
        end

        last_date = Date(last_dt)
        last_time = Time(last_dt)
        last_bar  = inst.exchange == "NSE" ? FIFTEENMIN_LAST_BAR : MCX_FIFTEENMIN_LAST_BAR

        if last_date > yest || (last_date == yest && last_time >= last_bar)
            current += 1; continue
        end

        from = last_date
        gap  = (yest - from).value + 1

        if dry_run
            @info "  $(inst.name) 15min: would fetch $from → $yest ($gap days)"
            updated += 1; continue
        end

        entry = get(token_map, inst.name, nothing)
        if isnothing(entry)
            @warn "  $(inst.name) — token not found"; failed += 1; continue
        end
        token, continuous = entry

        new_df = fetch_kite_macro_15min(token, from, yest, session)
        filter!(row -> row.datetime > last_dt, new_df)

        if isempty(new_df)
            @debug "  $(inst.name) 15min — no new bars after $last_dt"
            failed += 1; continue
        end

        path = joinpath(macro_dir, "$(inst.name)_15min.csv")
        CSV.write(path, new_df; append=true)
        updated += 1
        @info "  $(inst.name) 15min +$(nrow(new_df)) bars ($(new_df.datetime[1]) → $(new_df.datetime[end]))"
    end

    if dry_run
        @info "Macro 15min: $updated would be updated, $current already current"
    else
        @info "Macro 15min: $updated updated, $current already current, $failed failed"
    end
end

# ── Entry point ───────────────────────────────────────────────────────────────

function main()
    if "--help" in ARGS || "-h" in ARGS
        println("""
Usage:
  julia --project=packages/StockSwingPredictor scripts/update_ohlcv.jl [FLAGS]

Flags:
  --symbol SYM    Update only this one symbol (applied to whichever exchange(s) run).
  --nse-only      Update NSE only (default: both NSE and BSE). Mutually exclusive with --bse-only.
  --bse-only      Update BSE only (default: both NSE and BSE). Mutually exclusive with --nse-only.
  --daily-only    Update only daily bars.
  --hourly-only   Update only hourly (60-min) bars.
  --5min-only     Update only 5-minute bars.
  --15min-only    Update only 15-minute bars.
  --1min-only     Update only 1-minute bars.
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

Reads each existing {SYMBOL}.csv in website/data/ohlcv/{nse,bse}/{daily,
hourly,5min,15min,1min}/ (both exchanges by default — pass --nse-only or
--bse-only to update a single exchange), finds the last date, and fetches
only the gap to yesterday. New rows are appended in-place. Macro instrument
CSVs (website/data/ohlcv/macro/, flat and suffixed — not part of this
subfolder layout) are updated once regardless of exchange selection, during
the 5-min/15-min passes — there is no macro 1-min collection today, so
--1min-only/--skip-1min never affect it.

NOTE: no granularity here actually loses data if you skip a day — Kite's
per-interval day limits are a single-request span cap, not a retention
cliff (verified live). Running this daily is just cheaper than catching up
a big gap in one go, not a race against data disappearing.
""")
        return
    end

    daily_only      = "--daily-only"   in ARGS
    hourly_only     = "--hourly-only"  in ARGS
    fivemin_only    = "--5min-only"    in ARGS
    fifteenmin_only = "--15min-only"   in ARGS
    onemin_only     = "--1min-only"    in ARGS
    skip_5min       = "--skip-5min"    in ARGS
    skip_15min      = "--skip-15min"   in ARGS
    skip_1min       = "--skip-1min"    in ARGS
    nse_only        = "--nse-only"     in ARGS
    bse_only        = "--bse-only"     in ARGS
    dry_run         = "--dry-run"      in ARGS

    nse_only && bse_only && error("--nse-only and --bse-only are mutually exclusive.")
    run_nse = !bse_only   # both exchanges run by default
    run_bse = !nse_only

    any_only  = daily_only || hourly_only || fivemin_only || fifteenmin_only || onemin_only
    run_daily  = !any_only || daily_only
    run_hourly = !any_only || hourly_only
    run_5min   = (!any_only || fivemin_only)    && !skip_5min
    run_15min  = (!any_only || fifteenmin_only) && !skip_15min
    run_1min   = (!any_only || onemin_only)     && !skip_1min

    # Optional single-symbol filter (--symbol INFY)
    sym_idx   = findfirst(==("--symbol"), ARGS)
    sym_filter = (!isnothing(sym_idx) && sym_idx < length(ARGS)) ? ARGS[sym_idx + 1] : nothing

    log_idx  = findfirst(==("--log-file"), ARGS)
    log_path = (!isnothing(log_idx) && log_idx < length(ARGS)) ? ARGS[log_idx + 1] :
               account_log_path(DEFAULT_LOG_FILE, kite_account_from_args())
    slog     = open_script_log(log_path)
    @info "Logging full per-symbol detail to: $log_path"

    yest = today() - Day(1)

    dry_run && @info "[DRY RUN] No API calls will be made."

    logboth(slog, "Updating to: $yest")

    # ── Load session (skipped in dry-run) ─────────────────────────────────────
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

    # ── NSE update ────────────────────────────────────────────────────────────
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

        if run_daily
            logboth(slog, "── Daily bars ──")
            update_daily!(daily_syms, nse_token_map, session, yest, slog;
                          dry_run, out_dir=nse_gran_dir("daily"))
        end

        if run_hourly
            logboth(slog, "── Hourly bars ──")
            update_hourly!(hourly_syms, nse_token_map, session, yest, slog;
                           dry_run, out_dir=nse_gran_dir("hourly"))
        end

        if run_5min
            logboth(slog, "── 5-min bars ──")
            update_5min!(fivemin_syms, nse_token_map, session, yest, slog;
                        dry_run, out_dir=nse_gran_dir("5min"))
        end

        if run_15min
            logboth(slog, "── 15-min bars ──")
            update_15min!(fifteenmin_syms, nse_token_map, session, yest, slog;
                         dry_run, out_dir=nse_gran_dir("15min"))
        end

        if run_1min
            logboth(slog, "── 1-min bars ──")
            update_1min!(onemin_syms, nse_token_map, session, yest, slog;
                        dry_run, out_dir=nse_gran_dir("1min"))
        end
    end

    # ── BSE update ────────────────────────────────────────────────────────────
    if run_bse
        if isdir(BSE_OHLCV_DIR)
            bse_daily_syms   = [replace(f, ".csv" => "") for f in _readdir_safe(bse_gran_dir("daily"))]
            bse_hourly_syms  = [replace(f, ".csv" => "") for f in _readdir_safe(bse_gran_dir("hourly"))]
            bse_5min_syms    = [replace(f, ".csv" => "") for f in _readdir_safe(bse_gran_dir("5min"))]
            bse_15min_syms   = [replace(f, ".csv" => "") for f in _readdir_safe(bse_gran_dir("15min"))]
            bse_1min_syms    = [replace(f, ".csv" => "") for f in _readdir_safe(bse_gran_dir("1min"))]

            if !isnothing(sym_filter)
                bse_daily_syms  = filter(==(sym_filter), bse_daily_syms)
                bse_hourly_syms = filter(==(sym_filter), bse_hourly_syms)
                bse_5min_syms   = filter(==(sym_filter), bse_5min_syms)
                bse_15min_syms  = filter(==(sym_filter), bse_15min_syms)
                bse_1min_syms   = filter(==(sym_filter), bse_1min_syms)
            end

            logboth(slog, "Found $(length(bse_daily_syms)) BSE daily, $(length(bse_hourly_syms)) hourly, $(length(bse_5min_syms)) 5-min, $(length(bse_15min_syms)) 15-min, $(length(bse_1min_syms)) 1-min CSVs")
            logboth(slog, "═══ BSE ═══")

            if run_daily
                logboth(slog, "── Daily bars ──")
                update_daily!(bse_daily_syms, bse_token_map, session, yest, slog;
                              dry_run, out_dir=bse_gran_dir("daily"))
            end

            if run_hourly
                logboth(slog, "── Hourly bars ──")
                update_hourly!(bse_hourly_syms, bse_token_map, session, yest, slog;
                               dry_run, out_dir=bse_gran_dir("hourly"))
            end

            if run_5min
                logboth(slog, "── 5-min bars ──")
                update_5min!(bse_5min_syms, bse_token_map, session, yest, slog;
                             dry_run, out_dir=bse_gran_dir("5min"))
            end

            if run_15min
                logboth(slog, "── 15-min bars ──")
                update_15min!(bse_15min_syms, bse_token_map, session, yest, slog;
                              dry_run, out_dir=bse_gran_dir("15min"))
            end

            if run_1min
                logboth(slog, "── 1-min bars ──")
                update_1min!(bse_1min_syms, bse_token_map, session, yest, slog;
                             dry_run, out_dir=bse_gran_dir("1min"))
            end
        else
            @warn "BSE directory not found ($BSE_OHLCV_DIR) — run collect_bse_ohlcv.jl first (or pass --nse-only)."
        end
    end

    # ── Macro update (exchange-independent — runs regardless of --nse-only/--bse-only) ──
    if run_5min || run_15min
        @info "═══ Macro ═══"
        run_5min  && (@info "── 5-min bars ──";  update_macro_5min!(session, yest; dry_run))
        run_15min && (@info "── 15-min bars ──"; update_macro_15min!(session, yest; dry_run))
    end

    close_script_log(slog, "exit normally")
end

main()
