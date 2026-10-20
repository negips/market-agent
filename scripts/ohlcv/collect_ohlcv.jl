"""
collect_ohlcv.jl

Initial collection of OHLCV bars for every instrument Kite lists on NSE (EQ and INDICES) and BSE (EQ).
Both exchanges by default, one after the other; `--nse-only` or `--bse-only` restricts it to one (mutually
exclusive), the same convention as update_ohlcv.jl and backfill_ohlcv.jl.

Fetches daily, hourly, 15-minute, 5-minute, and 1-minute OHLCV, all from the same `--from` start date. The
symbol universe of each exchange is derived directly from Kite's instrument download (not from
nse_companies_latest.json / bse_companies_latest.json).

Kite's per-interval day limits (60/100/200/400/2000 days for
1min/5min/15min/60min/day) are a single-request span cap, not a total
retention cliff — verified live against the real API, every intraday
interval here still returns genuine multi-year-old data. `fetch_ohlcv*` in
`kite_data.jl` already chunks each request to stay under that cap, so a
`--from` as old as 2010 works for every granularity, not just daily; it's
just more chunked API calls (and more disk) the further back `--from` goes,
especially for 1-minute.

Output (one subfolder per granularity, per exchange):
  website/data/ohlcv/{nse,bse}/daily/{SYMBOL}.csv
  website/data/ohlcv/{nse,bse}/hourly/{SYMBOL}.csv
  website/data/ohlcv/{nse,bse}/5min/{SYMBOL}.csv
  website/data/ohlcv/{nse,bse}/15min/{SYMBOL}.csv
  website/data/ohlcv/{nse,bse}/1min/{SYMBOL}.csv

Usage:
  julia --project=packages/StockSwingPredictor scripts/ohlcv/collect_ohlcv.jl                   # NSE, then BSE
  julia --project=packages/StockSwingPredictor scripts/ohlcv/collect_ohlcv.jl --nse-only
  julia --project=packages/StockSwingPredictor scripts/ohlcv/collect_ohlcv.jl --bse-only
  julia --project=packages/StockSwingPredictor scripts/ohlcv/collect_ohlcv.jl --daily-only
  julia --project=packages/StockSwingPredictor scripts/ohlcv/collect_ohlcv.jl --hourly-only
  julia --project=packages/StockSwingPredictor scripts/ohlcv/collect_ohlcv.jl --15min-only
  julia --project=packages/StockSwingPredictor scripts/ohlcv/collect_ohlcv.jl --5min-only
  julia --project=packages/StockSwingPredictor scripts/ohlcv/collect_ohlcv.jl --1min-only
  julia --project=packages/StockSwingPredictor scripts/ohlcv/collect_ohlcv.jl --skip-1min
  julia --project=packages/StockSwingPredictor scripts/ohlcv/collect_ohlcv.jl --symbol RELIANCE
  julia --project=packages/StockSwingPredictor scripts/ohlcv/collect_ohlcv.jl --refresh
  julia --project=packages/StockSwingPredictor scripts/ohlcv/collect_ohlcv.jl --from 2015-01-01
  julia --project=packages/StockSwingPredictor scripts/ohlcv/collect_ohlcv.jl --nse-only --1min-only --account 1   # second API key: --bse-only --account 2
"""

using StockSwingPredictor, Dates

const REPO_ROOT  = joinpath(@__DIR__, "..", "..")
const OHLCV_ROOT = joinpath(REPO_ROOT, "website", "data", "ohlcv")

"""One exchange the script can collect: Kite's name for it, the folder under `ohlcv/`, and what its instrument list holds."""
struct Exchange
    kite::String
    dir::String
    instruments::String
end

const NSE = Exchange("NSE", "nse", "EQ/INDICES")
const BSE = Exchange("BSE", "bse", "EQ")

# One subfolder per granularity under the exchange's folder — see kite_data.jl's module
# docstring for why (each collect_ohlcv_* function just writes {SYMBOL}.csv
# into whichever directory it's given).
gran_dir(ex::Exchange, granularity::String) = joinpath(OHLCV_ROOT, ex.dir, granularity)

# Not -01-01, a Friday NSE holiday that would make backfill_ohlcv.jl's skip-check
# permanently unsatisfiable — kept consistent here too.
const DEFAULT_FROM = Date(2010, 1, 4)

# Full per-symbol detail goes here (see ScriptLog's docstring in
# StockSwingPredictor/src/script_log.jl); the terminal only gets stage
# headers, the existing every-100-symbols heartbeat, and warnings. Override
# with --log-file. Jobs started together run on different accounts, and an account N >= 2 gets
# <name>.accountN.log, so they never share a file.
const DEFAULT_LOG_FILE = joinpath(OHLCV_ROOT, "logs", "collect_ohlcv.log")

function parse_args()
    args = Dict{String,Any}(
        "nse_only"        => false,
        "bse_only"        => false,
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
        "log_file"        => nothing,
        "account"         => 1,
    )
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a in ("-h", "--help")
            println("""
collect_ohlcv.jl — initial OHLCV collection, NSE and BSE

Fetches daily, hourly, 15-min, 5-min, and 1-min bars — all from --from — for
every NSE-listed EQ and INDICES instrument and every BSE-listed EQ instrument
in Kite's instrument list. Each interval is chunked under Kite's per-request
span cap (60/100/200/400/2000 days respectively), which is NOT a retention
limit — --from 2010-01-04 works for every granularity, not just daily. Going
back that far for the finer granularities (especially --1min-only) means many
more chunked API calls and much more disk than the daily default; narrow
--from or use --symbol to scope a trial run first.

Flags:
  --nse-only          Collect NSE only (default: NSE, then BSE). Mutually exclusive with --bse-only.
  --bse-only          Collect BSE only (default: NSE, then BSE). Mutually exclusive with --nse-only.
  --daily-only        Only fetch daily bars
  --hourly-only       Only fetch hourly (60-min) bars
  --15min-only        Only fetch 15-minute bars
  --5min-only         Only fetch 5-minute bars
  --1min-only         Only fetch 1-minute bars
  --skip-daily        Skip the daily pass (overrides --daily-only if both given)
  --skip-hourly       Skip the hourly pass (overrides --hourly-only if both given)
  --skip-15min        Skip the 15-minute pass (overrides --15min-only if both given)
  --skip-5min         Skip the 5-minute pass (overrides --5min-only if both given)
  --skip-1min         Skip the 1-minute pass (overrides --1min-only if both given)
  --symbol SYM        Fetch only this tradingsymbol (e.g. --symbol RELIANCE) on the exchange(s) that list it
  --refresh           Re-fetch all even if CSV already exists
  --from DATE         History start date, all granularities (default: 2010-01-04)
  --log-file PATH     Full per-symbol detail (default: website/data/ohlcv/logs/collect_ohlcv.log;
                      with --account N>=2: collect_ohlcv.accountN.log, so parallel jobs don't share a file)
  --account N         Kite API-key slot: 1 = KITE_HISTORICAL_*, 2 = KITE_HISTORICAL2_*, …
                      Rate limits are per key, so two jobs on different accounts run in parallel
                      (e.g. --nse-only --account 1 and --bse-only --account 2).
                      Each account needs its own login: node sidecar/kite_login.js --account N
  -h, --help          Show this message
""")
            exit(0)
        elseif a == "--nse-only";     args["nse_only"]        = true; i += 1
        elseif a == "--bse-only";     args["bse_only"]        = true; i += 1
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
        elseif a == "--account" && i + 1 <= length(ARGS)
            args["account"] = parse(Int, ARGS[i+1]); i += 2
        else
            error("Unknown argument: $a (see --help)")
        end
    end
    args["nse_only"] && args["bse_only"] && error("--nse-only and --bse-only are mutually exclusive.")
    return args
end

"""
    collect_exchange(ex, args, session, slog, to_date) -> Bool

Collect the selected granularities of one exchange. Returns `false` (and collects nothing) when `--symbol` names a
symbol the exchange does not list, so a run over both exchanges can still serve a symbol that only one of them has.
"""
function collect_exchange(ex::Exchange, args, session, slog, to_date::Date)::Bool
    refresh = args["refresh"]
    any_flag = args["daily_only"] || args["hourly_only"] ||
               args["fivemin_only"] || args["fifteenmin_only"] || args["onemin_only"]
    run_daily    = (!any_flag || args["daily_only"])      && !args["skip_daily"]
    run_hourly   = (!any_flag || args["hourly_only"])     && !args["skip_hourly"]
    run_5min     = (!any_flag || args["fivemin_only"])    && !args["skip_5min"]
    run_15min    = (!any_flag || args["fifteenmin_only"]) && !args["skip_15min"]
    run_1min     = (!any_flag || args["onemin_only"])     && !args["skip_1min"]

    mkpath(joinpath(OHLCV_ROOT, ex.dir))

    @info "Loading $(ex.kite) instrument list from Kite…"
    instr     = load_instruments(session; exchange=ex.kite, refresh=true)
    token_map = build_token_map(instr; exchange=ex.kite)
    logboth(slog, "  $(length(token_map)) $(ex.kite) $(ex.instruments) instruments found")

    symbols = collect(keys(token_map))
    if !isnothing(args["symbol"])
        symbols = filter(==(args["symbol"]), symbols)
        if isempty(symbols)
            logboth(slog, "  '$(args["symbol"])' is not in the $(ex.kite) instrument list — skipping $(ex.kite)")
            return false
        end
    end
    sort!(symbols)

    from = args["from"]
    if run_daily
        logboth(slog, "── $(ex.kite) Daily: $(length(symbols)) symbols ($from → $to_date) ──")
        collect_ohlcv(symbols, token_map, session, gran_dir(ex, "daily"), from, to_date; refresh=refresh, slog)
    end
    if run_hourly
        logboth(slog, "── $(ex.kite) Hourly: $(length(symbols)) symbols ($from → $to_date) ──")
        collect_ohlcv_hourly(symbols, token_map, session, gran_dir(ex, "hourly"), from, to_date; refresh=refresh, slog)
    end
    if run_15min
        logboth(slog, "── $(ex.kite) 15-min: $(length(symbols)) symbols ($from → $to_date) ──")
        collect_ohlcv_15min(symbols, token_map, session, gran_dir(ex, "15min"), from, to_date; refresh=refresh, slog)
    end
    if run_5min
        logboth(slog, "── $(ex.kite) 5-min: $(length(symbols)) symbols ($from → $to_date) ──")
        collect_ohlcv_5min(symbols, token_map, session, gran_dir(ex, "5min"), from, to_date; refresh=refresh, slog)
    end
    if run_1min
        logboth(slog, "── $(ex.kite) 1-min: $(length(symbols)) symbols ($from → $to_date) ──")
        collect_ohlcv_1min(symbols, token_map, session, gran_dir(ex, "1min"), from, to_date; refresh=refresh, slog)
    end
    return true
end

function main()
    args      = parse_args()
    exchanges = args["nse_only"] ? [NSE] : args["bse_only"] ? [BSE] : [NSE, BSE]
    log_file  = something(args["log_file"], account_log_path(DEFAULT_LOG_FILE, args["account"]))
    slog      = open_script_log(log_file; truncate=true)
    @info "Logging full per-symbol detail to: $log_file"
    session   = load_kite_session(REPO_ROOT; account=args["account"])
    to_date   = today() - Day(1)

    collected = [collect_exchange(ex, args, session, slog, to_date) for ex in exchanges]
    if !any(collected)
        close_script_log(slog, "symbol not found")
        error("Symbol '$(args["symbol"])' not found in the instrument list of $(join((ex.kite for ex in exchanges), " or "))")
    end

    close_script_log(slog, "exit normally")
end

main()
