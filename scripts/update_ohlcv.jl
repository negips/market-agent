"""
update_ohlcv.jl

Incrementally update all OHLCV CSVs in website/data/ohlcv/nse/ AND
website/data/ohlcv/bse/ (both by default — pass --nse-only or --bse-only to
restrict to one exchange) with bars added since the last collection run.

For each existing CSV, reads the last date/datetime in the file and fetches
only the gap since then. Symbols already current are skipped. Appends new
rows in-place — no full-file rewrite needed.

Run daily after kite_login.js. For 5-min bars, missing a day means that
data is permanently lost after Kite's 100-day retention window. 15-min bars
have a 200-day retention window.

Prerequisites:
  - sidecar/kite_session.json present         (node sidecar/kite_login.js)
  - website/data/ohlcv/nse/ populated         (run collect_nse_ohlcv.jl first, unless --bse-only)
  - website/data/ohlcv/bse/ populated         (run collect_bse_ohlcv.jl first, unless --nse-only)

Usage:
  julia --project=packages/StockSwingPredictor scripts/update_ohlcv.jl
  julia --project=packages/StockSwingPredictor scripts/update_ohlcv.jl --nse-only
  julia --project=packages/StockSwingPredictor scripts/update_ohlcv.jl --bse-only
  julia --project=packages/StockSwingPredictor scripts/update_ohlcv.jl --daily-only
  julia --project=packages/StockSwingPredictor scripts/update_ohlcv.jl --hourly-only
  julia --project=packages/StockSwingPredictor scripts/update_ohlcv.jl --5min-only
  julia --project=packages/StockSwingPredictor scripts/update_ohlcv.jl --15min-only
  julia --project=packages/StockSwingPredictor scripts/update_ohlcv.jl --skip-5min
  julia --project=packages/StockSwingPredictor scripts/update_ohlcv.jl --skip-15min
  julia --project=packages/StockSwingPredictor scripts/update_ohlcv.jl --dry-run
  julia --project=packages/StockSwingPredictor scripts/update_ohlcv.jl --symbol RELIANCE
"""

using StockSwingPredictor, CSV, DataFrames, Dates

const REPO_ROOT     = joinpath(@__DIR__, "..")
const OHLCV_ROOT    = joinpath(REPO_ROOT, "website", "data", "ohlcv")
const NSE_OHLCV_DIR = joinpath(OHLCV_ROOT, "nse")
const BSE_OHLCV_DIR = joinpath(OHLCV_ROOT, "bse")
const MACRO_DIR     = joinpath(OHLCV_ROOT, "macro")

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

_last_daily_date(sym::String, dir::String=NSE_OHLCV_DIR) =
    _last_value(joinpath(dir, "$(sym)_daily.csv"),  :date,     Date)
_last_hourly_datetime(sym::String, dir::String=NSE_OHLCV_DIR) =
    _last_value(joinpath(dir, "$(sym)_hourly.csv"), :datetime, DateTime)
_last_5min_datetime(sym::String, dir::String=NSE_OHLCV_DIR) =
    _last_value(joinpath(dir, "$(sym)_5min.csv"),   :datetime, DateTime)
_last_15min_datetime(sym::String, dir::String=NSE_OHLCV_DIR) =
    _last_value(joinpath(dir, "$(sym)_15min.csv"),  :datetime, DateTime)
_last_macro_5min_datetime(name::String) =
    _last_value(joinpath(MACRO_DIR, "$(name)_5min.csv"),  :datetime, DateTime)
_last_macro_15min_datetime(name::String) =
    _last_value(joinpath(MACRO_DIR, "$(name)_15min.csv"), :datetime, DateTime)

# Last bar times for "day complete" checks (IST)
const HOURLY_LAST_BAR      = Time(15,  0, 0)   # last 60-min bar opens at 15:00
const FIVEMIN_LAST_BAR     = Time(15, 25, 0)   # last 5-min bar opens at 15:25 (NSE)
const FIFTEENMIN_LAST_BAR  = Time(15, 15, 0)   # last 15-min bar opens at 15:15 (NSE)
const MCX_FIVEMIN_LAST_BAR    = Time(23, 25, 0)  # last 5-min bar opens at 23:25 (MCX)
const MCX_FIFTEENMIN_LAST_BAR = Time(23, 15, 0)  # last 15-min bar opens at 23:15 (MCX)

# ── Core update loops ─────────────────────────────────────────────────────────

function update_daily!(symbols, token_map, session, yest::Date;
                       dry_run::Bool, out_dir::String=NSE_OHLCV_DIR)
    current = updated = failed = 0
    total   = length(symbols)

    for (i, sym) in enumerate(symbols)
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
            @warn "[$i/$total] $sym — no instrument token"
            failed += 1
            continue
        end

        new_df = fetch_ohlcv(token, from, yest, session)
        if isempty(new_df)
            # Normal for a holiday gap with no trading days in the range.
            @debug "[$i/$total] $sym daily — no new bars in $from…$yest (holiday gap?)"
            failed += 1
            sleep(0.35)
            continue
        end

        path = joinpath(out_dir, "$(sym)_daily.csv")
        CSV.write(path, new_df; append=true)
        updated += 1
        @info "[$i/$total] $sym daily +$(nrow(new_df)) bars ($from → $yest)"
        sleep(0.35)
    end

    if dry_run
        @info "Daily: $updated would be updated, $current already current"
    else
        @info "Daily: $updated updated, $current already current, $failed failed"
    end
end

function update_hourly!(symbols, token_map, session, yest::Date;
                        dry_run::Bool, out_dir::String=NSE_OHLCV_DIR)
    current = updated = failed = 0
    total   = length(symbols)

    for (i, sym) in enumerate(symbols)
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
            @warn "[$i/$total] $sym — no instrument token"
            failed += 1
            continue
        end

        new_df = fetch_ohlcv_hourly(token, from, yest, session)

        # Drop bars already present in the file (covers the partial last day).
        filter!(row -> row.datetime > last_dt, new_df)

        if isempty(new_df)
            @debug "[$i/$total] $sym hourly — no new bars after $last_dt"
            failed += 1
            continue
        end

        path = joinpath(out_dir, "$(sym)_hourly.csv")
        CSV.write(path, new_df; append=true)
        updated += 1
        @info "[$i/$total] $sym hourly +$(nrow(new_df)) bars (from $(new_df.datetime[1]) → $(new_df.datetime[end]))"
    end

    if dry_run
        @info "Hourly: $updated would be updated, $current already current"
    else
        @info "Hourly: $updated updated, $current already current, $failed failed"
    end
end

function update_5min!(symbols, token_map, session, yest::Date;
                      dry_run::Bool, out_dir::String=NSE_OHLCV_DIR)
    current = updated = failed = 0
    total   = length(symbols)

    for (i, sym) in enumerate(symbols)
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
            @warn "[$i/$total] $sym — no instrument token"; failed += 1; continue
        end

        new_df = fetch_ohlcv_5min(token, from, yest, session)
        filter!(row -> row.datetime > last_dt, new_df)

        if isempty(new_df)
            @debug "[$i/$total] $sym 5min — no new bars after $last_dt"
            failed += 1; continue
        end

        path = joinpath(out_dir, "$(sym)_5min.csv")
        CSV.write(path, new_df; append=true)
        updated += 1
        @info "[$i/$total] $sym 5min +$(nrow(new_df)) bars ($(new_df.datetime[1]) → $(new_df.datetime[end]))"
    end

    if dry_run
        @info "5min: $updated would be updated, $current already current"
    else
        @info "5min: $updated updated, $current already current, $failed failed"
    end
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

function update_15min!(symbols, token_map, session, yest::Date;
                       dry_run::Bool, out_dir::String=NSE_OHLCV_DIR)
    current = updated = failed = 0
    total   = length(symbols)

    for (i, sym) in enumerate(symbols)
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
            @warn "[$i/$total] $sym — no instrument token"; failed += 1; continue
        end

        new_df = fetch_ohlcv_15min(token, from, yest, session)
        filter!(row -> row.datetime > last_dt, new_df)

        if isempty(new_df)
            @debug "[$i/$total] $sym 15min — no new bars after $last_dt"
            failed += 1; continue
        end

        path = joinpath(out_dir, "$(sym)_15min.csv")
        CSV.write(path, new_df; append=true)
        updated += 1
        @info "[$i/$total] $sym 15min +$(nrow(new_df)) bars ($(new_df.datetime[1]) → $(new_df.datetime[end]))"
    end

    if dry_run
        @info "15min: $updated would be updated, $current already current"
    else
        @info "15min: $updated updated, $current already current, $failed failed"
    end
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
  --skip-5min     Skip the 5-minute pass (overrides --5min-only if both given).
  --skip-15min    Skip the 15-minute pass (overrides --15min-only if both given).
  --dry-run       Report what would be fetched without making any API calls.

Reads each existing *_daily.csv / *_hourly.csv / *_5min.csv / *_15min.csv in
website/data/ohlcv/nse/ AND website/data/ohlcv/bse/ (both by default — pass
--nse-only or --bse-only to update a single exchange), finds the last date,
and fetches only the gap to yesterday. New rows are appended in-place. Macro
instrument CSVs (website/data/ohlcv/macro/) are updated once regardless of
exchange selection, during the 5-min/15-min passes.

NOTE: 5-min bars have a 100-day retention window; 15-min bars have a
200-day retention window — run this daily or data will be permanently lost.
""")
        return
    end

    daily_only      = "--daily-only"   in ARGS
    hourly_only     = "--hourly-only"  in ARGS
    fivemin_only    = "--5min-only"    in ARGS
    fifteenmin_only = "--15min-only"   in ARGS
    skip_5min       = "--skip-5min"    in ARGS
    skip_15min      = "--skip-15min"   in ARGS
    nse_only        = "--nse-only"     in ARGS
    bse_only        = "--bse-only"     in ARGS
    dry_run         = "--dry-run"      in ARGS

    nse_only && bse_only && error("--nse-only and --bse-only are mutually exclusive.")
    run_nse = !bse_only   # both exchanges run by default
    run_bse = !nse_only

    any_only  = daily_only || hourly_only || fivemin_only || fifteenmin_only
    run_daily  = !any_only || daily_only
    run_hourly = !any_only || hourly_only
    run_5min   = (!any_only || fivemin_only)    && !skip_5min
    run_15min  = (!any_only || fifteenmin_only) && !skip_15min

    # Optional single-symbol filter (--symbol INFY)
    sym_idx   = findfirst(==("--symbol"), ARGS)
    sym_filter = (!isnothing(sym_idx) && sym_idx < length(ARGS)) ? ARGS[sym_idx + 1] : nothing

    yest = today() - Day(1)

    dry_run && @info "[DRY RUN] No API calls will be made."

    @info "Updating to: $yest"

    # ── Load session (skipped in dry-run) ─────────────────────────────────────
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

    # ── NSE update ────────────────────────────────────────────────────────────
    if run_nse
        isdir(NSE_OHLCV_DIR) || error("NSE OHLCV directory not found: $NSE_OHLCV_DIR\n" *
                                       "Run collect_nse_ohlcv.jl first (or pass --bse-only).")

        daily_syms    = [replace(f, "_daily.csv"  => "")
                         for f in readdir(NSE_OHLCV_DIR) if endswith(f, "_daily.csv")]
        hourly_syms   = [replace(f, "_hourly.csv" => "")
                         for f in readdir(NSE_OHLCV_DIR) if endswith(f, "_hourly.csv")]
        fivemin_syms  = [replace(f, "_5min.csv"   => "")
                         for f in readdir(NSE_OHLCV_DIR) if endswith(f, "_5min.csv")]
        fifteenmin_syms = [replace(f, "_15min.csv" => "")
                           for f in readdir(NSE_OHLCV_DIR) if endswith(f, "_15min.csv")]

        if !isnothing(sym_filter)
            daily_syms      = filter(==(sym_filter), daily_syms)
            hourly_syms     = filter(==(sym_filter), hourly_syms)
            fivemin_syms    = filter(==(sym_filter), fivemin_syms)
            fifteenmin_syms = filter(==(sym_filter), fifteenmin_syms)
            isempty(daily_syms) && isempty(hourly_syms) &&
            isempty(fivemin_syms) && isempty(fifteenmin_syms) &&
                error("No existing CSV found for symbol '$sym_filter' in $NSE_OHLCV_DIR")
            @info "Filtering to symbol: $sym_filter"
        end

        @info "Found $(length(daily_syms)) NSE daily, $(length(hourly_syms)) hourly, $(length(fivemin_syms)) 5-min, $(length(fifteenmin_syms)) 15-min CSVs"
        @info "═══ NSE ═══"

        if run_daily
            @info "── Daily bars ──"
            update_daily!(daily_syms, nse_token_map, session, yest; dry_run)
        end

        if run_hourly
            @info "── Hourly bars ──"
            update_hourly!(hourly_syms, nse_token_map, session, yest; dry_run)
        end

        if run_5min
            @info "── 5-min bars ──"
            update_5min!(fivemin_syms, nse_token_map, session, yest; dry_run)
        end

        if run_15min
            @info "── 15-min bars ──"
            update_15min!(fifteenmin_syms, nse_token_map, session, yest; dry_run)
        end
    end

    # ── BSE update ────────────────────────────────────────────────────────────
    if run_bse
        if isdir(BSE_OHLCV_DIR)
            bse_daily_syms = [replace(f, "_daily.csv"  => "")
                              for f in readdir(BSE_OHLCV_DIR) if endswith(f, "_daily.csv")]
            bse_hourly_syms = [replace(f, "_hourly.csv" => "")
                               for f in readdir(BSE_OHLCV_DIR) if endswith(f, "_hourly.csv")]
            bse_5min_syms  = [replace(f, "_5min.csv"   => "")
                              for f in readdir(BSE_OHLCV_DIR) if endswith(f, "_5min.csv")]
            bse_15min_syms = [replace(f, "_15min.csv"  => "")
                              for f in readdir(BSE_OHLCV_DIR) if endswith(f, "_15min.csv")]

            if !isnothing(sym_filter)
                bse_daily_syms  = filter(==(sym_filter), bse_daily_syms)
                bse_hourly_syms = filter(==(sym_filter), bse_hourly_syms)
                bse_5min_syms   = filter(==(sym_filter), bse_5min_syms)
                bse_15min_syms  = filter(==(sym_filter), bse_15min_syms)
            end

            @info "Found $(length(bse_daily_syms)) BSE daily, $(length(bse_hourly_syms)) hourly, $(length(bse_5min_syms)) 5-min, $(length(bse_15min_syms)) 15-min CSVs"
            @info "═══ BSE ═══"

            if run_daily
                @info "── Daily bars ──"
                update_daily!(bse_daily_syms, bse_token_map, session, yest;
                              dry_run, out_dir=BSE_OHLCV_DIR)
            end

            if run_hourly
                @info "── Hourly bars ──"
                update_hourly!(bse_hourly_syms, bse_token_map, session, yest;
                               dry_run, out_dir=BSE_OHLCV_DIR)
            end

            if run_5min
                @info "── 5-min bars ──"
                update_5min!(bse_5min_syms, bse_token_map, session, yest;
                             dry_run, out_dir=BSE_OHLCV_DIR)
            end

            if run_15min
                @info "── 15-min bars ──"
                update_15min!(bse_15min_syms, bse_token_map, session, yest;
                              dry_run, out_dir=BSE_OHLCV_DIR)
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
end

main()
