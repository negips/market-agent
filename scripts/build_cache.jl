"""
build_cache.jl

Build the inference cache from all cached OHLCV CSVs.

Reads every {SYMBOL}.csv in website/data/ohlcv/{exchange}/daily/ and
website/data/ohlcv/{exchange}/{granularity}/, aligns them onto shared time
axes, forward-fills missing bars, and writes a single binary file (~42 MB for
the hourly NSE cache; a 15-minute cache is ~3.6x more bars per company). This
file is loaded once at the start of each live session and used for both
inference and training batch assembly. The exchange and bar length are stored
in the cache, so the TradingGame scripts pick them up from it.

Run this after initial OHLCV collection, and again each morning after
update_ohlcv.jl to incorporate the latest bars.

Prerequisites:
  - website/data/ohlcv/{exchange}/ populated  (collect_nse_ohlcv.jl / collect_bse_ohlcv.jl)
  - website/data/{nse,bse}_companies_latest.json with confidence scores

Usage:
  julia --project=packages/StockSwingPredictor scripts/build_cache.jl
  julia --project=packages/StockSwingPredictor scripts/build_cache.jl 200  # top-N only
  julia --project=packages/StockSwingPredictor scripts/build_cache.jl --exchange bse --granularity 15min --min-mcap 500
"""

using StockSwingPredictor, JSON3, Dates

const REPO_ROOT      = joinpath(@__DIR__, "..")
const DATA_DIR       = joinpath(REPO_ROOT, "website", "data")
const DEFAULT_CACHE_FILE = joinpath(DATA_DIR, "inference_cache.bson")

const CONFIDENCE_THRESHOLD = 40

function parse_args()
    opts = Dict{String,Any}("top_n" => typemax(Int), "exchange" => "nse", "granularity" => "hourly",
                            "min_mcap" => 0.0, "out" => DEFAULT_CACHE_FILE, "history" => nothing)
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a == "--exchange" && i < length(ARGS);    opts["exchange"] = lowercase(ARGS[i+1]); i += 2
        elseif a == "--granularity" && i < length(ARGS); opts["granularity"] = ARGS[i+1]; i += 2
        elseif a == "--history" && i < length(ARGS); opts["history"] = ARGS[i+1]; i += 2
        elseif a == "--min-mcap" && i < length(ARGS); opts["min_mcap"] = parse(Float64, ARGS[i+1]); i += 2
        elseif a == "--out" && i < length(ARGS);     opts["out"] = abspath(ARGS[i+1]); i += 2
        elseif startswith(a, "--"); error("Unknown argument: $a")
        else opts["top_n"] = parse(Int, a); i += 1
        end
    end
    opts["exchange"] in ("nse", "bse") || error("--exchange must be nse or bse")
    opts["granularity"] in ("hourly", "15min") || error("--granularity must be hourly or 15min")
    # a 15-minute clock gets an hourly history axis by default; an hourly cache has none
    opts["history"] === nothing && (opts["history"] = opts["granularity"] == "15min" ? "resample" : "none")
    opts["history"] in ("none", "resample", "hourly") || error("--history must be none, resample or hourly")
    return opts
end

function main()
    if "--help" in ARGS || "-h" in ARGS
        println("""
Usage:
  julia --project=packages/StockSwingPredictor scripts/build_cache.jl [N] [options]

Arguments:
  N   Top N companies by market cap to include (default: all with confidence > $CONFIDENCE_THRESHOLD).

Options:
  --exchange nse|bse        Which exchange's OHLCV + company list to use (default: nse)
  --granularity hourly|15min  Intraday bar length (default: hourly). 15min needs
                            ohlcv/{exchange}/15min populated; TradingGame then decides every 15 min.
  --history resample|hourly|none
                            Second, hourly price axis for the policy's look-back window (game v3).
                            Default with --granularity 15min: resample (hourly bars built from the 15-minute
                            bars — matches Kite's own hourly closes, and covers every company that has
                            15-minute data; BSE's hourly files are incomplete). hourly: read ohlcv/{exchange}/hourly.
                            Default with --granularity hourly: none.
  --min-mcap CR             Drop companies below this market cap (₹ Cr). Strongly advised for BSE:
                            most small caps barely trade, so their 15-min bars are forward-filled.
  --out PATH                Output file (default: website/data/inference_cache.bson)

Output:
  website/data/inference_cache.bson  — ~42 MB (NSE hourly); BSE 15-min is ~0.2 MB per company-year
""")
        return
    end

    opts      = parse_args()
    exchange  = opts["exchange"]
    top_n     = opts["top_n"]
    COMPANIES_FILE = joinpath(DATA_DIR, "$(exchange)_companies_latest.json")
    OHLCV_DIR      = joinpath(DATA_DIR, "ohlcv", exchange)
    CACHE_FILE     = opts["out"]

    isfile(COMPANIES_FILE) ||
        error("Not found: $COMPANIES_FILE\nRun: julia scripts/generate_$(exchange)_list.jl")

    raw      = JSON3.read(read(COMPANIES_FILE, String))
    all_c    = collect(raw.companies)
    eligible = filter(all_c) do c
        conf = get(c, :confidence, nothing)
        !isnothing(conf) &&
        !isempty(string(get(c, :symbol, ""))) &&
        get(conf, :score, 0) > CONFIDENCE_THRESHOLD &&
        Float64(something(get(c, :market_cap_cr, nothing), 0.0)) >= opts["min_mcap"] &&
        isfile(joinpath(OHLCV_DIR, opts["granularity"], "$(c.symbol).csv"))
    end
    sort!(eligible, by = c -> Float64(something(get(c, :market_cap_cr, nothing), 0.0)), rev=true)

    universe  = first(eligible, top_n)
    companies = [string(c.symbol) for c in universe]

    @info "Universe: $(length(companies)) $(uppercase(exchange)) companies (confidence > $CONFIDENCE_THRESHOLD, " *
          "market cap >= ₹$(opts["min_mcap"]) Cr, $(opts["granularity"]) data on disk)"
    @info "Output: $CACHE_FILE"

    build_inference_cache(OHLCV_DIR, companies, CACHE_FILE;
                          granularity=opts["granularity"], exchange=exchange,
                          history=opts["history"])

    # Write the market universe JSON so the model's column ordering is auditable.
    universe_file = joinpath(DATA_DIR, "market_universe.json")
    universe_json = [(rank=i, symbol=string(c.symbol),
                      name=string(get(c, :name, "")),
                      market_cap_cr=Float64(something(get(c, :market_cap_cr, nothing), 0.0)),
                      confidence=Int(get(get(c, :confidence, Dict()), :score, 0)))
                     for (i, c) in enumerate(universe)]
    open(universe_file, "w") do io
        JSON3.pretty(io, Dict("generated_at" => string(Dates.now()),
                              "n_market_companies" => length(universe),
                              "confidence_threshold" => CONFIDENCE_THRESHOLD,
                              "companies" => universe_json))
    end
    @info "Market universe → $universe_file ($(length(universe)) companies)"
end

main()
