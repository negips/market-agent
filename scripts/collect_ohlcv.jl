"""
collect_ohlcv.jl

Download and cache daily and 60-minute OHLCV data from Kite Connect for all
NSE companies with confidence score > 40.

Daily data is used for the market context matrix; hourly data is used for
trajectory labels. Both are needed for training, but daily can be collected
independently first.

Output:
  website/data/ohlcv/{SYMBOL}_daily.csv   — daily bars
  website/data/ohlcv/{SYMBOL}_hourly.csv  — 60-minute bars

The script is resumable — already-cached symbols are skipped unless --refresh.

Prerequisites:
  - sidecar/kite_session.json present (node sidecar/kite_login.js)
  - website/data/nse_companies_latest.json present (run generate_nse_list.jl)

Usage:
  julia --project=packages/StockSwingPredictor scripts/collect_ohlcv.jl
  julia --project=packages/StockSwingPredictor scripts/collect_ohlcv.jl 3            # years of history
  julia --project=packages/StockSwingPredictor scripts/collect_ohlcv.jl 5 --daily-only
  julia --project=packages/StockSwingPredictor scripts/collect_ohlcv.jl 5 --refresh
"""

using StockSwingPredictor, JSON3, Dates, Printf

const REPO_ROOT      = joinpath(@__DIR__, "..")
const COMPANIES_FILE = joinpath(REPO_ROOT, "website", "data", "nse_companies_latest.json")
const OHLCV_DIR      = joinpath(REPO_ROOT, "website", "data", "ohlcv")

function main()
    if "--help" in ARGS || "-h" in ARGS
        println("""
Usage:
  julia --project=packages/StockSwingPredictor scripts/collect_ohlcv.jl [YEARS] [FLAGS]

Arguments:
  YEARS         Years of history to fetch (default: 5).
  --daily-only  Fetch only daily bars; skip the slow hourly collection.
  --refresh     Re-download even if a cached file exists.

Output:
  website/data/ohlcv/{SYMBOL}_daily.csv   — one row per trading day
  website/data/ohlcv/{SYMBOL}_hourly.csv  — one row per 60-minute bar

Universe: companies in nse_companies_latest.json with confidence score > 40.
""")
        return
    end

    refresh     = "--refresh"    in ARGS
    daily_only  = "--daily-only" in ARGS
    args        = filter(a -> a ∉ ("--refresh", "--daily-only"), ARGS)
    years       = length(args) >= 1 ? parse(Int, args[1]) : 5

    to_date   = today()
    from_date = to_date - Year(years)

    @info "Fetching $(years) years of OHLCV: $from_date → $to_date"
    @info "Output: $OHLCV_DIR"

    # ── Load Kite session ──────────────────────────────────────────────────────

    session = load_kite_session(REPO_ROOT)

    # ── Load instrument list ───────────────────────────────────────────────────

    @info "Loading NSE instrument list from Kite…"
    instruments = load_instruments(session; refresh=refresh)
    token_map   = build_token_map(instruments)
    @info "  $(length(token_map)) instruments loaded"

    # ── Collect equity OHLCV ──────────────────────────────────────────────────

    mkpath(OHLCV_DIR)

    isfile(COMPANIES_FILE) || error("Not found: $COMPANIES_FILE\nRun: julia scripts/generate_nse_list.jl")
    raw   = JSON3.read(read(COMPANIES_FILE, String))
    all_c = collect(raw.companies)

    # Companies with confidence score > 40 and a valid symbol, sorted by market cap.
    CONFIDENCE_THRESHOLD = 40
    eligible = filter(all_c) do c
        conf = get(c, :confidence, nothing)
        !isnothing(conf) &&
        !isempty(string(get(c, :symbol, ""))) &&
        get(conf, :score, 0) > CONFIDENCE_THRESHOLD
    end
    sort!(eligible, by = c -> Float64(get(c, :market_cap_cr, 0.0)), rev=true)
    symbols = [string(c.symbol) for c in eligible]

    @info "Universe: $(length(symbols)) companies with confidence > $CONFIDENCE_THRESHOLD"

    @info "Collecting daily OHLCV for $(length(symbols)) companies…"
    collect_ohlcv(symbols, token_map, session, OHLCV_DIR, from_date, to_date; refresh=refresh)

    # ── Collect 60-minute OHLCV (for trajectory labels) ───────────────────────

    if daily_only
        @info "Skipping hourly collection (--daily-only). Re-run without --daily-only to fetch hourly bars."
    else
        n_chunks = ceil(Int, years * 365 / 59)
        @info "Collecting 60-minute OHLCV for $(length(symbols)) companies…"
        @info "  (~$(n_chunks) API calls per company, ~$(round(Int, length(symbols) * n_chunks * 0.35 / 60)) min total)"
        collect_ohlcv_hourly(symbols, token_map, session, OHLCV_DIR, from_date, to_date; refresh=refresh)
    end
end

main()
