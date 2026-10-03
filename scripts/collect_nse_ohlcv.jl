"""
collect_nse_ohlcv.jl

Initial collection of NSE OHLCV bars for all NSE-listed EQ and INDICES instruments.

Fetches daily, hourly, 5-minute, 15-minute, and 1-minute OHLCV from Kite's
NSE instrument list. The NSE symbol universe is derived directly from Kite's
NSE instrument download (all EQ and INDICES).

Output (one subfolder per granularity):
  website/data/ohlcv/nse/daily/{SYMBOL}.csv
  website/data/ohlcv/nse/hourly/{SYMBOL}.csv
  website/data/ohlcv/nse/5min/{SYMBOL}.csv
  website/data/ohlcv/nse/15min/{SYMBOL}.csv
  website/data/ohlcv/nse/1min/{SYMBOL}.csv

Usage:
  julia --project=packages/StockSwingPredictor scripts/collect_nse_ohlcv.jl
  julia --project=packages/StockSwingPredictor scripts/collect_nse_ohlcv.jl --daily-only
  julia --project=packages/StockSwingPredictor scripts/collect_nse_ohlcv.jl --hourly-only
  julia --project=packages/StockSwingPredictor scripts/collect_nse_ohlcv.jl --5min-only
  julia --project=packages/StockSwingPredictor scripts/collect_nse_ohlcv.jl --15min-only
  julia --project=packages/StockSwingPredictor scripts/collect_nse_ohlcv.jl --1min-only
  julia --project=packages/StockSwingPredictor scripts/collect_nse_ohlcv.jl --symbol RELIANCE
  julia --project=packages/StockSwingPredictor scripts/collect_nse_ohlcv.jl --refresh
  julia --project=packages/StockSwingPredictor scripts/collect_nse_ohlcv.jl --from 2015-01-01
"""

using StockSwingPredictor, Dates

const REPO_ROOT  = joinpath(@__DIR__, "..")
const OHLCV_ROOT = joinpath(REPO_ROOT, "website", "data", "ohlcv")
const NSE_DIR    = joinpath(OHLCV_ROOT, "nse")

# One subfolder per granularity under NSE_DIR — see kite_data.jl's module
# docstring for why (each collect_ohlcv_* function just writes {SYMBOL}.csv
# into whichever directory it's given).
nse_gran_dir(granularity::String) = joinpath(NSE_DIR, granularity)

const DEFAULT_FROM = Date(2010, 1, 1)

function parse_args()
    args = Dict{String,Any}(
        "daily_only"      => false,
        "hourly_only"     => false,
        "fivemin_only"    => false,
        "fifteenmin_only" => false,
        "onemin_only"     => false,
        "refresh"         => false,
        "symbol"          => nothing,
        "from"            => DEFAULT_FROM,
    )
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a in ("-h", "--help")
            println("""
collect_nse_ohlcv.jl — initial NSE OHLCV collection

Fetches daily (full history from --from), hourly (400-day retention),
5-min (100-day retention), 15-min (200-day retention), and 1-min (60-day
retention) bars for every NSE-listed EQ and INDICES instrument in Kite's
instrument list.

Flags:
  --daily-only        Only fetch daily bars
  --hourly-only       Only fetch hourly (60-min) bars
  --5min-only         Only fetch 5-minute bars
  --15min-only        Only fetch 15-minute bars
  --1min-only         Only fetch 1-minute bars
  --symbol SYM        Fetch only this NSE tradingsymbol (e.g. --symbol RELIANCE)
  --refresh           Re-fetch all even if CSV already exists
  --from DATE         Daily history start date (default: 2010-01-01)
  -h, --help          Show this message
""")
            exit(0)
        elseif a == "--daily-only";   args["daily_only"]      = true; i += 1
        elseif a == "--hourly-only";  args["hourly_only"]     = true; i += 1
        elseif a == "--5min-only";    args["fivemin_only"]    = true; i += 1
        elseif a == "--15min-only";   args["fifteenmin_only"] = true; i += 1
        elseif a == "--1min-only";    args["onemin_only"]     = true; i += 1
        elseif a == "--refresh";      args["refresh"]         = true; i += 1
        elseif a == "--symbol" && i + 1 <= length(ARGS)
            args["symbol"] = ARGS[i+1]; i += 2
        elseif a == "--from" && i + 1 <= length(ARGS)
            args["from"] = Date(ARGS[i+1]); i += 2
        else
            @warn "Unknown argument: $a"; i += 1
        end
    end
    return args
end

function main()
    args    = parse_args()
    session = load_kite_session(REPO_ROOT)
    refresh = args["refresh"]

    mkpath(NSE_DIR)

    # ── Determine which intervals to run ──────────────────────────────────────
    any_flag = args["daily_only"] || args["hourly_only"] ||
               args["fivemin_only"] || args["fifteenmin_only"] || args["onemin_only"]
    run_daily    = !any_flag || args["daily_only"]
    run_hourly   = !any_flag || args["hourly_only"]
    run_5min     = !any_flag || args["fivemin_only"]
    run_15min    = !any_flag || args["fifteenmin_only"]
    run_1min     = !any_flag || args["onemin_only"]

    # ── Load NSE instrument list ───────────────────────────────────────────────
    @info "Loading NSE instrument list from Kite…"
    instr     = load_instruments(session; exchange="NSE", refresh=true)
    token_map = build_token_map(instr; exchange="NSE")
    @info "  $(length(token_map)) NSE EQ/INDICES instruments found"

    symbols = collect(keys(token_map))
    if !isnothing(args["symbol"])
        symbols = filter(==(args["symbol"]), symbols)
        isempty(symbols) && error("Symbol '$(args["symbol"])' not found in NSE instrument list")
    end
    sort!(symbols)

    to_date = today() - Day(1)

    # ── Daily ─────────────────────────────────────────────────────────────────
    if run_daily
        @info "── NSE Daily: $(length(symbols)) symbols ($(args["from"]) → $to_date) ──"
        collect_ohlcv(symbols, token_map, session, nse_gran_dir("daily"),
                      args["from"], to_date; refresh=refresh)
    end

    # ── Hourly ────────────────────────────────────────────────────────────────
    if run_hourly
        hourly_from = today() - Day(399)
        @info "── NSE Hourly: $(length(symbols)) symbols ($hourly_from → $to_date) ──"
        collect_ohlcv_hourly(symbols, token_map, session, nse_gran_dir("hourly"),
                             hourly_from, to_date; refresh=refresh)
    end

    # ── 5-minute ──────────────────────────────────────────────────────────────
    if run_5min
        fivemin_from = today() - Day(99)
        @info "── NSE 5-min: $(length(symbols)) symbols ($fivemin_from → $to_date) ──"
        collect_ohlcv_5min(symbols, token_map, session, nse_gran_dir("5min"),
                           fivemin_from, to_date; refresh=refresh)
    end

    # ── 15-minute ─────────────────────────────────────────────────────────────
    if run_15min
        fifteenmin_from = today() - Day(199)
        @info "── NSE 15-min: $(length(symbols)) symbols ($fifteenmin_from → $to_date) ──"
        collect_ohlcv_15min(symbols, token_map, session, nse_gran_dir("15min"),
                            fifteenmin_from, to_date; refresh=refresh)
    end

    # ── 1-minute ──────────────────────────────────────────────────────────────
    if run_1min
        onemin_from = today() - Day(59)
        @info "── NSE 1-min: $(length(symbols)) symbols ($onemin_from → $to_date) ──"
        collect_ohlcv_1min(symbols, token_map, session, nse_gran_dir("1min"),
                           onemin_from, to_date; refresh=refresh)
    end
end

main()
