"""
Initial collection of 5-minute OHLCV bars for all NSE equities and macro instruments.

Kite retains 5-minute bars for 100 days. Run this once to populate the 5-min
CSVs, then use update_ohlcv.jl daily to keep them current.

Fetches for all symbols that already have a _daily.csv in website/data/ohlcv/nse/
(i.e. the equity universe from collect_ohlcv.jl). Macro instruments are fetched
separately from MCX/NSE via Kite.

Output:
  website/data/ohlcv/nse/{SYMBOL}_5min.csv   — equities
  website/data/ohlcv/macro/{NAME}_5min.csv   — macro instruments

Usage:
  julia --project=packages/StockSwingPredictor scripts/collect_5min_ohlcv.jl
  julia --project=packages/StockSwingPredictor scripts/collect_5min_ohlcv.jl --equities-only
  julia --project=packages/StockSwingPredictor scripts/collect_5min_ohlcv.jl --macro-only
  julia --project=packages/StockSwingPredictor scripts/collect_5min_ohlcv.jl --refresh
  julia --project=packages/StockSwingPredictor scripts/collect_5min_ohlcv.jl --symbol RELIANCE
"""

using StockSwingPredictor, CSV, DataFrames, Dates

const REPO_ROOT = joinpath(@__DIR__, "..")
const OHLCV_DIR = joinpath(REPO_ROOT, "website", "data", "ohlcv", "nse")

function parse_args()
    args = Dict{String,Any}(
        "equities_only" => false,
        "macro_only"    => false,
        "refresh"       => false,
        "symbol"        => nothing,
    )
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a in ("-h", "--help")
            println("""
collect_5min_ohlcv.jl — initial 5-minute OHLCV collection (last 99 days)

Flags:
  --equities-only   Only fetch equity 5-min (skip macro)
  --macro-only      Only fetch macro 5-min (skip equities)
  --refresh         Re-fetch all even if CSV already exists
  --symbol SYM      Fetch only this equity symbol (e.g. --symbol RELIANCE)
  -h, --help        Show this message
""")
            exit(0)
        elseif a == "--equities-only"; args["equities_only"] = true; i += 1
        elseif a == "--macro-only";    args["macro_only"]    = true; i += 1
        elseif a == "--refresh";       args["refresh"]       = true; i += 1
        elseif a == "--symbol" && i + 1 <= length(ARGS)
            args["symbol"] = ARGS[i+1]; i += 2
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

    # 100-day retention; fetch from 99 days ago to yesterday
    from_date = today() - Day(99)
    to_date   = today() - Day(1)

    @info "5-min collection: $from_date → $to_date"

    # ── Equities ──────────────────────────────────────────────────────────────
    if !args["macro_only"]
        isdir(OHLCV_DIR) || error("OHLCV directory not found: $OHLCV_DIR\n" *
                                   "Run collect_ohlcv.jl first.")

        symbols = [replace(f, "_daily.csv" => "")
                   for f in readdir(OHLCV_DIR) if endswith(f, "_daily.csv")]

        if !isnothing(args["symbol"])
            symbols = filter(==(args["symbol"]), symbols)
            isempty(symbols) && error("No daily CSV for symbol '$(args["symbol"])'")
        end

        @info "── Equity 5-min: $(length(symbols)) symbols ──"
        instr     = load_instruments(session; refresh=true)
        token_map = build_token_map(instr)

        collect_ohlcv_5min(symbols, token_map, session, OHLCV_DIR,
                            from_date, to_date; refresh=refresh)
    end

    # ── Macro ─────────────────────────────────────────────────────────────────
    if !args["equities_only"]
        @info "── Macro 5-min: $(length(KITE_MACRO_INSTRUMENTS)) instruments ──"
        collect_macro_5min(session; from_date=from_date, to_date=to_date,
                           refresh=refresh)
    end
end

main()
