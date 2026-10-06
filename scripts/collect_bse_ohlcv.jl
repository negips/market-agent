"""
collect_bse_ohlcv.jl

Initial collection of BSE OHLCV bars for all BSE-listed EQ instruments.

Fetches daily, hourly, 5-minute, 15-minute, and 1-minute OHLCV from Kite's
BSE instrument list, all from the same `--from` start date. The BSE symbol
universe is derived directly from Kite's BSE instrument download (not from
nse_companies_latest.json).

Kite's per-interval day limits (60/100/200/400/2000 days for
1min/5min/15min/60min/day) are a single-request span cap, not a total
retention cliff — verified live against the real API, every intraday
interval here still returns genuine multi-year-old data. `fetch_ohlcv*` in
`kite_data.jl` already chunks each request to stay under that cap, so a
`--from` as old as 2010 works for every granularity, not just daily; it's
just more chunked API calls (and more disk) the further back `--from` goes,
especially for 1-minute.

Output (one subfolder per granularity):
  website/data/ohlcv/bse/daily/{SYMBOL}.csv
  website/data/ohlcv/bse/hourly/{SYMBOL}.csv
  website/data/ohlcv/bse/5min/{SYMBOL}.csv
  website/data/ohlcv/bse/15min/{SYMBOL}.csv
  website/data/ohlcv/bse/1min/{SYMBOL}.csv

Usage:
  julia --project=packages/StockSwingPredictor scripts/collect_bse_ohlcv.jl
  julia --project=packages/StockSwingPredictor scripts/collect_bse_ohlcv.jl --daily-only
  julia --project=packages/StockSwingPredictor scripts/collect_bse_ohlcv.jl --hourly-only
  julia --project=packages/StockSwingPredictor scripts/collect_bse_ohlcv.jl --5min-only
  julia --project=packages/StockSwingPredictor scripts/collect_bse_ohlcv.jl --15min-only
  julia --project=packages/StockSwingPredictor scripts/collect_bse_ohlcv.jl --1min-only
  julia --project=packages/StockSwingPredictor scripts/collect_bse_ohlcv.jl --skip-1min
  julia --project=packages/StockSwingPredictor scripts/collect_bse_ohlcv.jl --symbol RELIANCE
  julia --project=packages/StockSwingPredictor scripts/collect_bse_ohlcv.jl --refresh
  julia --project=packages/StockSwingPredictor scripts/collect_bse_ohlcv.jl --from 2015-01-01
"""

using StockSwingPredictor, Dates

const REPO_ROOT  = joinpath(@__DIR__, "..")
const OHLCV_ROOT = joinpath(REPO_ROOT, "website", "data", "ohlcv")
const BSE_DIR    = joinpath(OHLCV_ROOT, "bse")

# One subfolder per granularity under BSE_DIR — see kite_data.jl's module
# docstring for why.
bse_gran_dir(granularity::String) = joinpath(BSE_DIR, granularity)

# Kite daily OHLCV history depth for a single request (2000-bar limit).
# For daily bars that's ~8 years per chunk; we default to 2010-01-01.
const DEFAULT_FROM = Date(2010, 1, 1)

# Full per-symbol detail goes here (see ScriptLog's docstring in
# StockSwingPredictor/src/script_log.jl); the terminal only gets stage
# headers, the existing every-100-symbols heartbeat, and warnings. Override
# with --log-file.
const DEFAULT_LOG_FILE = joinpath(OHLCV_ROOT, "logs", "collect_bse_ohlcv.log")

function parse_args()
    args = Dict{String,Any}(
        "daily_only"      => false,
        "hourly_only"     => false,
        "fivemin_only"    => false,
        "fifteenmin_only" => false,
        "onemin_only"     => false,
        "skip_daily"      => false,
        "skip_hourly"     => false,
        "skip_5min"       => false,
        "skip_15min"      => false,
        "skip_1min"       => false,
        "refresh"         => false,
        "symbol"          => nothing,
        "from"            => DEFAULT_FROM,
        "log_file"        => DEFAULT_LOG_FILE,
    )
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a in ("-h", "--help")
            println("""
collect_bse_ohlcv.jl — initial BSE OHLCV collection

Fetches daily, hourly, 5-min, 15-min, and 1-min bars — all from --from — for
every BSE-listed EQ instrument in Kite's instrument list. Each interval is
chunked under Kite's per-request span cap (60/100/200/400/2000 days
respectively), which is NOT a retention limit — --from 2010-01-01 works for
every granularity, not just daily. Going back that far for the finer
granularities (especially --1min-only) means many more chunked API calls and
much more disk than the daily default; narrow --from or use --symbol to
scope a trial run first.

Flags:
  --daily-only        Only fetch daily bars
  --hourly-only       Only fetch hourly (60-min) bars
  --5min-only         Only fetch 5-minute bars
  --15min-only        Only fetch 15-minute bars
  --1min-only         Only fetch 1-minute bars
  --skip-daily        Skip the daily pass (overrides --daily-only if both given)
  --skip-hourly       Skip the hourly pass (overrides --hourly-only if both given)
  --skip-5min         Skip the 5-minute pass (overrides --5min-only if both given)
  --skip-15min        Skip the 15-minute pass (overrides --15min-only if both given)
  --skip-1min         Skip the 1-minute pass (overrides --1min-only if both given)
  --symbol SYM        Fetch only this BSE tradingsymbol (e.g. --symbol RELIANCE)
  --refresh           Re-fetch all even if CSV already exists
  --from DATE         History start date, all granularities (default: 2010-01-01)
  --log-file PATH     Full per-symbol detail (default: $DEFAULT_LOG_FILE)
  -h, --help          Show this message
""")
            exit(0)
        elseif a == "--daily-only";   args["daily_only"]      = true; i += 1
        elseif a == "--hourly-only";  args["hourly_only"]     = true; i += 1
        elseif a == "--5min-only";    args["fivemin_only"]    = true; i += 1
        elseif a == "--15min-only";   args["fifteenmin_only"] = true; i += 1
        elseif a == "--1min-only";    args["onemin_only"]     = true; i += 1
        elseif a == "--skip-daily";   args["skip_daily"]      = true; i += 1
        elseif a == "--skip-hourly";  args["skip_hourly"]     = true; i += 1
        elseif a == "--skip-5min";    args["skip_5min"]       = true; i += 1
        elseif a == "--skip-15min";   args["skip_15min"]      = true; i += 1
        elseif a == "--skip-1min";    args["skip_1min"]       = true; i += 1
        elseif a == "--refresh";      args["refresh"]         = true; i += 1
        elseif a == "--symbol" && i + 1 <= length(ARGS)
            args["symbol"] = ARGS[i+1]; i += 2
        elseif a == "--from" && i + 1 <= length(ARGS)
            args["from"] = Date(ARGS[i+1]); i += 2
        elseif a == "--log-file" && i + 1 <= length(ARGS)
            args["log_file"] = ARGS[i+1]; i += 2
        else
            @warn "Unknown argument: $a"; i += 1
        end
    end
    return args
end

function main()
    args    = parse_args()
    slog    = open_script_log(args["log_file"])
    @info "Logging full per-symbol detail to: $(args["log_file"])"
    session = load_kite_session(REPO_ROOT)
    refresh = args["refresh"]

    mkpath(BSE_DIR)

    # ── Determine which intervals to run ──────────────────────────────────────
    any_flag = args["daily_only"] || args["hourly_only"] ||
               args["fivemin_only"] || args["fifteenmin_only"] || args["onemin_only"]
    run_daily    = (!any_flag || args["daily_only"])      && !args["skip_daily"]
    run_hourly   = (!any_flag || args["hourly_only"])     && !args["skip_hourly"]
    run_5min     = (!any_flag || args["fivemin_only"])    && !args["skip_5min"]
    run_15min    = (!any_flag || args["fifteenmin_only"]) && !args["skip_15min"]
    run_1min     = (!any_flag || args["onemin_only"])     && !args["skip_1min"]

    # ── Load BSE instrument list ───────────────────────────────────────────────
    @info "Loading BSE instrument list from Kite…"
    instr     = load_instruments(session; exchange="BSE", refresh=true)
    token_map = build_token_map(instr; exchange="BSE")
    logboth(slog, "  $(length(token_map)) BSE EQ instruments found")

    symbols = collect(keys(token_map))
    if !isnothing(args["symbol"])
        symbols = filter(==(args["symbol"]), symbols)
        isempty(symbols) && error("Symbol '$(args["symbol"])' not found in BSE instrument list")
    end
    sort!(symbols)

    to_date = today() - Day(1)

    # ── Daily ─────────────────────────────────────────────────────────────────
    if run_daily
        logboth(slog, "── BSE Daily: $(length(symbols)) symbols ($(args["from"]) → $to_date) ──")
        collect_ohlcv(symbols, token_map, session, bse_gran_dir("daily"),
                      args["from"], to_date; refresh=refresh, slog)
    end

    # ── Hourly ────────────────────────────────────────────────────────────────
    if run_hourly
        logboth(slog, "── BSE Hourly: $(length(symbols)) symbols ($(args["from"]) → $to_date) ──")
        collect_ohlcv_hourly(symbols, token_map, session, bse_gran_dir("hourly"),
                             args["from"], to_date; refresh=refresh, slog)
    end

    # ── 5-minute ──────────────────────────────────────────────────────────────
    if run_5min
        logboth(slog, "── BSE 5-min: $(length(symbols)) symbols ($(args["from"]) → $to_date) ──")
        collect_ohlcv_5min(symbols, token_map, session, bse_gran_dir("5min"),
                           args["from"], to_date; refresh=refresh, slog)
    end

    # ── 15-minute ─────────────────────────────────────────────────────────────
    if run_15min
        logboth(slog, "── BSE 15-min: $(length(symbols)) symbols ($(args["from"]) → $to_date) ──")
        collect_ohlcv_15min(symbols, token_map, session, bse_gran_dir("15min"),
                            args["from"], to_date; refresh=refresh, slog)
    end

    # ── 1-minute ──────────────────────────────────────────────────────────────
    if run_1min
        logboth(slog, "── BSE 1-min: $(length(symbols)) symbols ($(args["from"]) → $to_date) ──")
        collect_ohlcv_1min(symbols, token_map, session, bse_gran_dir("1min"),
                           args["from"], to_date; refresh=refresh, slog)
    end

    close_script_log(slog, "exit normally")
end

main()
