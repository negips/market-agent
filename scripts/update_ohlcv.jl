"""
update_ohlcv.jl

Incrementally update all OHLCV CSVs in website/data/ohlcv/ with bars
added since the last collection run.

For each existing CSV, reads the last date in the file and fetches only
the gap from (last date + 1 day) to yesterday. Symbols already current
are skipped in a single column-read check. Appends new rows in-place —
no full-file rewrite needed.

Prerequisites:
  - sidecar/kite_session.json present  (node sidecar/kite_login.js)
  - website/data/ohlcv/ populated       (run collect_ohlcv.jl first)

Usage:
  julia --project=packages/StockSwingPredictor scripts/update_ohlcv.jl
  julia --project=packages/StockSwingPredictor scripts/update_ohlcv.jl --daily-only
  julia --project=packages/StockSwingPredictor scripts/update_ohlcv.jl --dry-run
  julia --project=packages/StockSwingPredictor scripts/update_ohlcv.jl --symbol RELIANCE
"""

using StockSwingPredictor, CSV, DataFrames, Dates

const REPO_ROOT = joinpath(@__DIR__, "..")
const OHLCV_DIR = joinpath(REPO_ROOT, "website", "data", "ohlcv")

# ── Helpers ───────────────────────────────────────────────────────────────────

"""
Read only the date/datetime column from a CSV and return the maximum value.
Uses `select` so it never loads OHLCV columns for large files.
Returns `nothing` if the file is missing or empty.
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
    return maximum(df[!, col])
end

_last_daily_date(sym::String)     = _last_value(joinpath(OHLCV_DIR, "$(sym)_daily.csv"),
                                               :date, Date)
_last_hourly_datetime(sym::String) = _last_value(joinpath(OHLCV_DIR, "$(sym)_hourly.csv"),
                                                  :datetime, DateTime)

# Last 60-min bar on NSE starts at 15:00 IST — if the file's last bar is at or
# after this time, the day is complete and we don't need to re-fetch it.
const HOURLY_LAST_BAR = Time(15, 0, 0)

# ── Core update loops ─────────────────────────────────────────────────────────

function update_daily!(symbols, token_map, session, yest::Date; dry_run::Bool)
    current = updated = failed = 0
    total   = length(symbols)

    for (i, sym) in enumerate(symbols)
        last = _last_daily_date(sym)
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
            @warn "[$i/$total] $sym daily — no new bars in $from…$yest (holiday gap?)"
            failed += 1
            sleep(0.35)
            continue
        end

        path = joinpath(OHLCV_DIR, "$(sym)_daily.csv")
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

function update_hourly!(symbols, token_map, session, yest::Date; dry_run::Bool)
    current = updated = failed = 0
    total   = length(symbols)

    for (i, sym) in enumerate(symbols)
        last_dt = _last_hourly_datetime(sym)
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
            @warn "[$i/$total] $sym hourly — no new bars after $last_dt"
            failed += 1
            continue
        end

        path = joinpath(OHLCV_DIR, "$(sym)_hourly.csv")
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

# ── Entry point ───────────────────────────────────────────────────────────────

function main()
    if "--help" in ARGS || "-h" in ARGS
        println("""
Usage:
  julia --project=packages/StockSwingPredictor scripts/update_ohlcv.jl [FLAGS]

Flags:
  --symbol SYM  Update only this one NSE symbol (e.g. --symbol RELIANCE).
  --daily-only  Update only daily bars; skip hourly.
  --dry-run     Report what would be fetched without making any API calls.

Reads each existing *_daily.csv / *_hourly.csv in website/data/ohlcv/,
finds the last date, and fetches only the gap to yesterday. New rows are
appended in-place (no full-file rewrite).
""")
        return
    end

    daily_only = "--daily-only" in ARGS
    dry_run    = "--dry-run"    in ARGS

    # Optional single-symbol filter (--symbol INFY)
    sym_idx   = findfirst(==("--symbol"), ARGS)
    sym_filter = (!isnothing(sym_idx) && sym_idx < length(ARGS)) ? ARGS[sym_idx + 1] : nothing

    yest = today() - Day(1)

    dry_run && @info "[DRY RUN] No API calls will be made."

    # ── Discover existing symbols ─────────────────────────────────────────────
    isdir(OHLCV_DIR) || error("OHLCV directory not found: $OHLCV_DIR\n" *
                               "Run collect_ohlcv.jl first.")

    daily_syms  = [replace(f, "_daily.csv"  => "")
                   for f in readdir(OHLCV_DIR) if endswith(f, "_daily.csv")]
    hourly_syms = [replace(f, "_hourly.csv" => "")
                   for f in readdir(OHLCV_DIR) if endswith(f, "_hourly.csv")]

    if !isnothing(sym_filter)
        daily_syms  = filter(==(sym_filter), daily_syms)
        hourly_syms = filter(==(sym_filter), hourly_syms)
        isempty(daily_syms) && isempty(hourly_syms) &&
            error("No existing CSV found for symbol '$sym_filter' in $OHLCV_DIR")
        @info "Filtering to symbol: $sym_filter"
    end

    @info "Found $(length(daily_syms)) daily CSVs, $(length(hourly_syms)) hourly CSVs"
    @info "Updating to: $yest"

    # ── Load session + instruments (skipped in dry-run) ───────────────────────
    session   = dry_run ? (api_key="", access_token="") :
                          load_kite_session(REPO_ROOT)
    token_map = if dry_run
        Dict{String, Int}()
    else
        @info "Loading NSE instrument list from Kite…"
        instr = load_instruments(session)
        t = build_token_map(instr)
        @info "  $(length(t)) instruments loaded"
        t
    end

    # ── Update daily ──────────────────────────────────────────────────────────
    @info "── Daily bars ──"
    update_daily!(daily_syms, token_map, session, yest; dry_run)

    # ── Update hourly ─────────────────────────────────────────────────────────
    if daily_only
        @info "Skipping hourly update (--daily-only)."
    else
        @info "── Hourly bars ──"
        update_hourly!(hourly_syms, token_map, session, yest; dry_run)
    end
end

main()
