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

Prerequisites:
  - sidecar/kite_session.json present         (node sidecar/kite_login.js)
  - website/data/ohlcv/nse/{daily,hourly,5min,15min,1min}/ populated
    (run collect_nse_ohlcv.jl first, unless --bse-only)
  - website/data/ohlcv/bse/{daily,hourly,5min,15min,1min}/ populated
    (run collect_bse_ohlcv.jl first, unless --nse-only)

Usage:
  julia --project=packages/StockSwingPredictor scripts/backfill_ohlcv.jl              # --from defaults to 2010-01-01
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

using StockSwingPredictor, CSV, DataFrames, Dates

const REPO_ROOT     = joinpath(@__DIR__, "..")
const OHLCV_ROOT    = joinpath(REPO_ROOT, "website", "data", "ohlcv")
const NSE_OHLCV_DIR = joinpath(OHLCV_ROOT, "nse")
const BSE_OHLCV_DIR = joinpath(OHLCV_ROOT, "bse")

# Same default as collect_nse_ohlcv.jl/collect_bse_ohlcv.jl's --from.
const DEFAULT_FROM = Date(2010, 1, 1)

# One subfolder per granularity under each exchange root — see
# kite_data.jl's module docstring. Macro is deliberately out of scope here
# (same as update_ohlcv.jl's reasoning): too few files to be worth it.
nse_gran_dir(granularity::String) = joinpath(NSE_OHLCV_DIR, granularity)
bse_gran_dir(granularity::String) = joinpath(BSE_OHLCV_DIR, granularity)

"""`readdir`, but `String[]` instead of an error when `dir` doesn't exist yet."""
_readdir_safe(dir::String) = isdir(dir) ? readdir(dir) : String[]

# ── Helpers ───────────────────────────────────────────────────────────────────

"""
Read only the date/datetime column from a CSV and return the MINIMUM value
(the mirror image of update_ohlcv.jl's `_last_value`, which returns the
maximum). Returns `nothing` if the file is missing, empty, or every row
fails to parse — same corrupted-row handling as `_last_value`.
"""
function _first_value(path::String, col::Symbol, T::Type)
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
                        "corrupted. Ignoring them for the earliest-known-good " *
                        "timestamp; inspect the file directly if this recurs."

    good = skipmissing(df[!, col])
    isempty(good) && return nothing
    return minimum(good)
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

# ── Core backfill loops ──────────────────────────────────────────────────────

function backfill_daily!(symbols, token_map, session, from_target::Date;
                         dry_run::Bool, out_dir::String)
    current = updated = failed = 0
    total   = length(symbols)

    for (i, sym) in enumerate(symbols)
        path = joinpath(out_dir, "$sym.csv")
        first_date = _first_daily_date(sym, out_dir)
        if isnothing(first_date)
            @warn "[$i/$total] $sym daily — no existing CSV, skipping (run collect_nse_ohlcv.jl first)"
            failed += 1; continue
        end

        if first_date <= from_target
            current += 1; continue
        end

        to  = first_date - Day(1)
        gap = (to - from_target).value + 1

        if dry_run
            @info "[$i/$total] $sym daily: would backfill $from_target → $to ($gap calendar days)"
            updated += 1; continue
        end

        token = get(token_map, sym, nothing)
        if isnothing(token)
            @warn "[$i/$total] $sym — no instrument token"; failed += 1; continue
        end

        older_df = fetch_ohlcv(token, from_target, to, session)
        if isempty(older_df)
            @debug "[$i/$total] $sym daily — no older bars in $from_target…$to"
            failed += 1; sleep(0.35); continue
        end

        _prepend_merge!(path, older_df, :date, Date)
        updated += 1
        @info "[$i/$total] $sym daily +$(nrow(older_df)) older bars ($from_target → $to)"
        sleep(0.35)
    end

    if dry_run
        @info "Daily: $updated would be backfilled, $current already back to $from_target"
    else
        @info "Daily: $updated backfilled, $current already back to $from_target, $failed failed"
    end
end

function backfill_hourly!(symbols, token_map, session, from_target::Date;
                          dry_run::Bool, out_dir::String)
    current = updated = failed = 0
    total   = length(symbols)

    for (i, sym) in enumerate(symbols)
        path = joinpath(out_dir, "$sym.csv")
        first_dt = _first_hourly_datetime(sym, out_dir)
        if isnothing(first_dt)
            @warn "[$i/$total] $sym hourly — no existing CSV, skipping (run collect_nse_ohlcv.jl first)"
            failed += 1; continue
        end
        first_date = Date(first_dt)

        if first_date <= from_target
            current += 1; continue
        end

        # Fetch through the whole earliest day (not day-1) and de-dup via
        # the datetime filter below — same partial-day defensiveness as
        # update_ohlcv.jl's forward version, just mirrored.
        to     = first_date
        gap    = (to - from_target).value + 1
        n_chks = ceil(Int, gap / HOURLY_CHUNK_DAYS)

        if dry_run
            @info "[$i/$total] $sym hourly: would backfill $from_target → $to ($gap days, ~$n_chks API call$(n_chks==1 ? "" : "s"))"
            updated += 1; continue
        end

        token = get(token_map, sym, nothing)
        if isnothing(token)
            @warn "[$i/$total] $sym — no instrument token"; failed += 1; continue
        end

        older_df = fetch_ohlcv_hourly(token, from_target, to, session)
        filter!(row -> row.datetime < first_dt, older_df)

        if isempty(older_df)
            @debug "[$i/$total] $sym hourly — no older bars before $first_dt"
            failed += 1; continue
        end

        _prepend_merge!(path, older_df, :datetime, DateTime)
        updated += 1
        @info "[$i/$total] $sym hourly +$(nrow(older_df)) older bars ($(older_df.datetime[1]) → $(older_df.datetime[end]))"
    end

    if dry_run
        @info "Hourly: $updated would be backfilled, $current already back to $from_target"
    else
        @info "Hourly: $updated backfilled, $current already back to $from_target, $failed failed"
    end
end

function backfill_5min!(symbols, token_map, session, from_target::Date;
                        dry_run::Bool, out_dir::String)
    current = updated = failed = 0
    total   = length(symbols)

    for (i, sym) in enumerate(symbols)
        path = joinpath(out_dir, "$sym.csv")
        first_dt = _first_5min_datetime(sym, out_dir)
        if isnothing(first_dt)
            @warn "[$i/$total] $sym 5min — no existing CSV, skipping (run collect_nse_ohlcv.jl first)"
            failed += 1; continue
        end
        first_date = Date(first_dt)

        if first_date <= from_target
            current += 1; continue
        end

        to     = first_date
        gap    = (to - from_target).value + 1
        n_chks = ceil(Int, gap / FIVEMIN_CHUNK_DAYS)

        if dry_run
            @info "[$i/$total] $sym 5min: would backfill $from_target → $to ($gap days, ~$n_chks call$(n_chks==1 ? "" : "s"))"
            updated += 1; continue
        end

        token = get(token_map, sym, nothing)
        if isnothing(token)
            @warn "[$i/$total] $sym — no instrument token"; failed += 1; continue
        end

        older_df = fetch_ohlcv_5min(token, from_target, to, session)
        filter!(row -> row.datetime < first_dt, older_df)

        if isempty(older_df)
            @debug "[$i/$total] $sym 5min — no older bars before $first_dt"
            failed += 1; continue
        end

        _prepend_merge!(path, older_df, :datetime, DateTime)
        updated += 1
        @info "[$i/$total] $sym 5min +$(nrow(older_df)) older bars ($(older_df.datetime[1]) → $(older_df.datetime[end]))"
    end

    if dry_run
        @info "5min: $updated would be backfilled, $current already back to $from_target"
    else
        @info "5min: $updated backfilled, $current already back to $from_target, $failed failed"
    end
end

function backfill_15min!(symbols, token_map, session, from_target::Date;
                         dry_run::Bool, out_dir::String)
    current = updated = failed = 0
    total   = length(symbols)

    for (i, sym) in enumerate(symbols)
        path = joinpath(out_dir, "$sym.csv")
        first_dt = _first_15min_datetime(sym, out_dir)
        if isnothing(first_dt)
            @warn "[$i/$total] $sym 15min — no existing CSV, skipping (run collect_nse_ohlcv.jl first)"
            failed += 1; continue
        end
        first_date = Date(first_dt)

        if first_date <= from_target
            current += 1; continue
        end

        to     = first_date
        gap    = (to - from_target).value + 1
        n_chks = ceil(Int, gap / FIFTEENMIN_CHUNK_DAYS)

        if dry_run
            @info "[$i/$total] $sym 15min: would backfill $from_target → $to ($gap days, ~$n_chks call$(n_chks==1 ? "" : "s"))"
            updated += 1; continue
        end

        token = get(token_map, sym, nothing)
        if isnothing(token)
            @warn "[$i/$total] $sym — no instrument token"; failed += 1; continue
        end

        older_df = fetch_ohlcv_15min(token, from_target, to, session)
        filter!(row -> row.datetime < first_dt, older_df)

        if isempty(older_df)
            @debug "[$i/$total] $sym 15min — no older bars before $first_dt"
            failed += 1; continue
        end

        _prepend_merge!(path, older_df, :datetime, DateTime)
        updated += 1
        @info "[$i/$total] $sym 15min +$(nrow(older_df)) older bars ($(older_df.datetime[1]) → $(older_df.datetime[end]))"
    end

    if dry_run
        @info "15min: $updated would be backfilled, $current already back to $from_target"
    else
        @info "15min: $updated backfilled, $current already back to $from_target, $failed failed"
    end
end

function backfill_1min!(symbols, token_map, session, from_target::Date;
                        dry_run::Bool, out_dir::String)
    current = updated = failed = 0
    total   = length(symbols)

    for (i, sym) in enumerate(symbols)
        path = joinpath(out_dir, "$sym.csv")
        first_dt = _first_1min_datetime(sym, out_dir)
        if isnothing(first_dt)
            @warn "[$i/$total] $sym 1min — no existing CSV, skipping (run collect_nse_ohlcv.jl first)"
            failed += 1; continue
        end
        first_date = Date(first_dt)

        if first_date <= from_target
            current += 1; continue
        end

        to     = first_date
        gap    = (to - from_target).value + 1
        n_chks = ceil(Int, gap / ONEMIN_CHUNK_DAYS)

        if dry_run
            @info "[$i/$total] $sym 1min: would backfill $from_target → $to ($gap days, ~$n_chks call$(n_chks==1 ? "" : "s"))"
            updated += 1; continue
        end

        token = get(token_map, sym, nothing)
        if isnothing(token)
            @warn "[$i/$total] $sym — no instrument token"; failed += 1; continue
        end

        older_df = fetch_ohlcv_1min(token, from_target, to, session)
        filter!(row -> row.datetime < first_dt, older_df)

        if isempty(older_df)
            @debug "[$i/$total] $sym 1min — no older bars before $first_dt"
            failed += 1; continue
        end

        _prepend_merge!(path, older_df, :datetime, DateTime)
        updated += 1
        @info "[$i/$total] $sym 1min +$(nrow(older_df)) older bars ($(older_df.datetime[1]) → $(older_df.datetime[end]))"
    end

    if dry_run
        @info "1min: $updated would be backfilled, $current already back to $from_target"
    else
        @info "1min: $updated backfilled, $current already back to $from_target, $failed failed"
    end
end

# ── Entry point ───────────────────────────────────────────────────────────────

function main()
    if "--help" in ARGS || "-h" in ARGS
        println("""
Usage:
  julia --project=packages/StockSwingPredictor scripts/backfill_ohlcv.jl [FLAGS]

Flags:
  --from DATE     Extend every selected CSV back to this date (default: 2010-01-01).
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

For each existing {SYMBOL}.csv in website/data/ohlcv/{nse,bse}/{daily,
hourly,5min,15min,1min}/, reads the earliest date/datetime already on disk
and fetches only the older gap down to --from, merging it into the file. A
symbol with no existing CSV is skipped — this tool only EXTENDS, it doesn't
do the initial collection (see collect_nse_ohlcv.jl/collect_bse_ohlcv.jl for
that, including brand-new granularities like NSE 15-min that have zero
files today).

NOTE: Kite's per-interval day limits (60/100/200/400/2000 days for
1min/5min/15min/60min/day) are a single-request span cap, not a retention
cliff (verified live) — --from 2010-01-01 is achievable for every
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

    dry_run && @info "[DRY RUN] No API calls will be made."
    @info "Backfilling to: $from_target"

    session = dry_run ? (api_key="", access_token="") :
                        load_kite_session(REPO_ROOT)

    nse_token_map = if !run_nse || dry_run
        Dict{String, Int}()
    else
        @info "Loading NSE instrument list from Kite…"
        instr = load_instruments(session; exchange="NSE")
        t = build_token_map(instr; exchange="NSE")
        @info "  $(length(t)) NSE instruments loaded"
        t
    end

    bse_token_map = if !run_bse || dry_run
        Dict{String, Int}()
    else
        @info "Loading BSE instrument list from Kite…"
        instr = load_instruments(session; exchange="BSE", refresh=true)
        t = build_token_map(instr; exchange="BSE")
        @info "  $(length(t)) BSE instruments loaded"
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

        @info "Found $(length(daily_syms)) NSE daily, $(length(hourly_syms)) hourly, $(length(fivemin_syms)) 5-min, $(length(fifteenmin_syms)) 15-min, $(length(onemin_syms)) 1-min CSVs"
        @info "═══ NSE ═══"

        if run_daily
            @info "── Daily bars ──"
            backfill_daily!(daily_syms, nse_token_map, session, from_target;
                            dry_run, out_dir=nse_gran_dir("daily"))
        end

        if run_hourly
            @info "── Hourly bars ──"
            backfill_hourly!(hourly_syms, nse_token_map, session, from_target;
                             dry_run, out_dir=nse_gran_dir("hourly"))
        end

        if run_5min
            @info "── 5-min bars ──"
            backfill_5min!(fivemin_syms, nse_token_map, session, from_target;
                           dry_run, out_dir=nse_gran_dir("5min"))
        end

        if run_15min
            @info "── 15-min bars ──"
            backfill_15min!(fifteenmin_syms, nse_token_map, session, from_target;
                            dry_run, out_dir=nse_gran_dir("15min"))
        end

        if run_1min
            @info "── 1-min bars ──"
            backfill_1min!(onemin_syms, nse_token_map, session, from_target;
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

            @info "Found $(length(bse_daily_syms)) BSE daily, $(length(bse_hourly_syms)) hourly, $(length(bse_5min_syms)) 5-min, $(length(bse_15min_syms)) 15-min, $(length(bse_1min_syms)) 1-min CSVs"
            @info "═══ BSE ═══"

            if run_daily
                @info "── Daily bars ──"
                backfill_daily!(bse_daily_syms, bse_token_map, session, from_target;
                                dry_run, out_dir=bse_gran_dir("daily"))
            end

            if run_hourly
                @info "── Hourly bars ──"
                backfill_hourly!(bse_hourly_syms, bse_token_map, session, from_target;
                                 dry_run, out_dir=bse_gran_dir("hourly"))
            end

            if run_5min
                @info "── 5-min bars ──"
                backfill_5min!(bse_5min_syms, bse_token_map, session, from_target;
                               dry_run, out_dir=bse_gran_dir("5min"))
            end

            if run_15min
                @info "── 15-min bars ──"
                backfill_15min!(bse_15min_syms, bse_token_map, session, from_target;
                                dry_run, out_dir=bse_gran_dir("15min"))
            end

            if run_1min
                @info "── 1-min bars ──"
                backfill_1min!(bse_1min_syms, bse_token_map, session, from_target;
                               dry_run, out_dir=bse_gran_dir("1min"))
            end
        else
            @warn "BSE directory not found ($BSE_OHLCV_DIR) — run collect_bse_ohlcv.jl first (or pass --nse-only)."
        end
    end
end

main()
