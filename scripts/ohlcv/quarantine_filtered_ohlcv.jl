"""
quarantine_filtered_ohlcv.jl — move OHLCV CSVs that the current instrument filters exclude into a quarantine folder.

`StockSwingPredictor.build_token_map` drops fund vehicles (ETF / INAV / liquid funds), debt, trusts and index codes, but
files collected before a rule existed stay on disk. This script finds every CSV, in every granularity, whose symbol is
an instrument in Kite's cached instrument list but is rejected by the filters today, and moves it to
`website/data/ohlcv/_filtered_out/{exchange}/{granularity}/`. A CSV whose symbol is not in the instrument list at all (a delisted or renamed company) is left alone, because dropping it would add
survivorship bias, unless the symbol itself says it was never a company: `StockSwingPredictor.excluded_symbol` (debt and G-Sec codes,
fixed-maturity plans, rights entitlements such as `INDRA-RE`) moves those too.

Dry run by default (prints counts and sizes); nothing moves without `--move`. Nothing is deleted: remove the quarantine
folder yourself when you are satisfied. Re-running is safe, so run it again after a collect job that used the old filters finishes.

Usage:
  julia --project=packages/StockSwingPredictor scripts/ohlcv/quarantine_filtered_ohlcv.jl            # report only
  julia --project=packages/StockSwingPredictor scripts/ohlcv/quarantine_filtered_ohlcv.jl --move

Options:
  --move     move the files (default: report only)
  -h, --help this text

Reads the instrument lists cached by the collect and update scripts (`ohlcv/_instruments_NSE_cache.csv`,
`ohlcv/_instruments_bse_cache.csv`); run one of them once if those are missing.
"""

using StockSwingPredictor, CSV, DataFrames

const REPO_ROOT       = joinpath(@__DIR__, "..", "..")
const OHLCV_DIR       = joinpath(REPO_ROOT, "website", "data", "ohlcv")
const QUARANTINE_DIR  = joinpath(OHLCV_DIR, "_filtered_out")
const GRANULARITIES   = ("daily", "hourly", "5min", "15min", "1min")
const INSTRUMENT_CACHES = (("NSE", "_instruments_NSE_cache.csv"), ("BSE", "_instruments_bse_cache.csv"))

function main()
    any(a -> a in ("-h", "--help"), ARGS) && (println(match(r"\"\"\"\n(.*?)\"\"\""s, read(@__FILE__, String)).captures[1]); return)
    unknown = [a for a in ARGS if a != "--move"]
    isempty(unknown) || error("Unknown argument: $(first(unknown)) (see --help)")
    move = "--move" in ARGS
    total_files = 0; total_bytes = 0
    for (exchange, cache_file) in INSTRUMENT_CACHES
        path = joinpath(OHLCV_DIR, cache_file)
        isfile(path) || error("Instrument list not found: $path\nRun collect_ohlcv.jl / update_ohlcv.jl once to download it.")
        instruments = CSV.read(path, DataFrame)
        token_map = build_token_map(instruments; exchange=exchange)
        listed = Set(string(r.tradingsymbol) for r in eachrow(instruments) if r.exchange == exchange)
        for gran in GRANULARITIES
            dir = joinpath(OHLCV_DIR, lowercase(exchange), gran)
            isdir(dir) || continue
            files = [f for f in readdir(dir) if endswith(f, ".csv")]
            junk = [f for f in files if (s = replace(f, ".csv" => "")) in listed ? !haskey(token_map, s) : excluded_symbol(s; exchange=exchange)]
            delisted = count(f -> (s = replace(f, ".csv" => "")) ∉ listed && !excluded_symbol(s; exchange=exchange), files)
            bytes = sum(filesize(joinpath(dir, f)) for f in junk; init=0)
            println("$exchange/$gran: $(length(files)) files; excluded by the filters (or by symbol): $(length(junk)) ($(round(bytes / 1e6, digits=1)) MB); ",
                    "not in the instrument list (delisted or renamed, kept): $delisted")
            total_files += length(junk); total_bytes += bytes
            if move && !isempty(junk)
                dest = joinpath(QUARANTINE_DIR, lowercase(exchange), gran)
                mkpath(dest)
                for f in junk
                    mv(joinpath(dir, f), joinpath(dest, f); force=true)
                end
            end
        end
    end
    println("TOTAL $total_files files, $(round(total_bytes / 1e9, digits=2)) GB", move ? " moved to $(normpath(QUARANTINE_DIR))" : " (dry run — pass --move to move them)")
end

main()
