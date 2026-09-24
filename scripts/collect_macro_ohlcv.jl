"""
Fetch and cache historical OHLCV for macro instruments.

Sources:
  Yahoo Finance — SP500, US_VIX, CRUDE_OIL, GOLD, SILVER, NATURAL_GAS,
                  COPPER, ALUMINIUM, PALM_OIL
  Kite Connect  — INDIA_VIX (NSE index), USD_INR (CDS continuous futures)

Output: website/data/ohlcv/macro/{NAME}_daily.csv

Usage:
  julia --project=packages/StockSwingPredictor scripts/collect_macro_ohlcv.jl
  julia --project=packages/StockSwingPredictor scripts/collect_macro_ohlcv.jl --from 2015-01-01
  julia --project=packages/StockSwingPredictor scripts/collect_macro_ohlcv.jl --refresh
"""

using StockSwingPredictor, Dates

function parse_args()
    args = Dict{String,Any}(
        "from"    => Date(2010, 1, 1),
        "to"      => today() - Day(1),
        "refresh" => false,
    )
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a in ("-h", "--help")
            println("""
collect_macro_ohlcv.jl — fetch historical macro instrument OHLCV

Flags:
  --from  DATE    Start date YYYY-MM-DD (default: 2010-01-01)
  --to    DATE    End date   YYYY-MM-DD (default: yesterday)
  --refresh       Re-fetch all instruments even if already cached
  -h, --help      Show this message

Output: website/data/ohlcv/macro/{NAME}_daily.csv
""")
            exit(0)
        elseif a == "--from" && i + 1 <= length(ARGS)
            args["from"] = Date(ARGS[i+1]); i += 2
        elseif a == "--to" && i + 1 <= length(ARGS)
            args["to"] = Date(ARGS[i+1]); i += 2
        elseif a == "--refresh"
            args["refresh"] = true; i += 1
        else
            @warn "Unknown argument: $a"; i += 1
        end
    end
    return args
end

args    = parse_args()
session = load_kite_session(joinpath(@__DIR__, ".."))

collect_macro_ohlcv(session;
    from_date = args["from"],
    to_date   = args["to"],
    refresh   = args["refresh"])
