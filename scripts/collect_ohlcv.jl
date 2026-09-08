"""
collect_ohlcv.jl

Download and cache daily OHLCV data from Kite Connect for all confidence-scored
NSE companies and for the key index benchmarks (Nifty 50 + sector indices).

Data is saved as CSV to website/data/ohlcv/{SYMBOL}_daily.csv.
The script is resumable — already-cached symbols are skipped unless --refresh.

Prerequisites:
  - sidecar/kite_session.json present (node sidecar/kite_setup.js)
  - website/data/nse_companies_latest.json present

Usage:
  julia --project=packages/StockSwingPredictor scripts/collect_ohlcv.jl
  julia --project=packages/StockSwingPredictor scripts/collect_ohlcv.jl 3      # years of history
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
  julia --project=packages/StockSwingPredictor scripts/collect_ohlcv.jl [YEARS] [--refresh]

Arguments:
  YEARS      Years of history to fetch (default: 5).
  --refresh  Re-download even if a cached file exists.

Output:
  website/data/ohlcv/{SYMBOL}_daily.csv for each company
  website/data/ohlcv/IDX_{NAME}_daily.csv for each index
""")
        return
    end

    refresh = "--refresh" in ARGS
    args    = filter(a -> a != "--refresh", ARGS)
    years   = length(args) >= 1 ? parse(Int, args[1]) : 5

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

    # ── Collect index OHLCV ───────────────────────────────────────────────────

    @info "Fetching index data…"
    mkpath(OHLCV_DIR)
    for idx_name in NSE_INDICES
        # Kite index tradingsymbols often use spaces; the token_map key is the
        # tradingsymbol as returned by /instruments (e.g. "NIFTY 50").
        path = joinpath(OHLCV_DIR, "IDX_$(replace(idx_name, " " => "_"))_daily.csv")
        if !refresh && isfile(path)
            @info "  $idx_name — cached"
            continue
        end
        token = get(token_map, idx_name, nothing)
        if isnothing(token)
            @warn "  $idx_name — no token found, skipping"
            continue
        end
        df = fetch_ohlcv(token, from_date, to_date, session)
        if isempty(df)
            @warn "  $idx_name — empty response"
        else
            CSV.write(path, df)
            @info "  $idx_name — $(nrow(df)) days"
        end
        sleep(0.35)
    end

    # ── Collect equity OHLCV ──────────────────────────────────────────────────

    isfile(COMPANIES_FILE) || error("Not found: $COMPANIES_FILE\nRun: julia scripts/generate_nse_list.jl")
    raw  = JSON3.read(read(COMPANIES_FILE, String))
    all_c = collect(raw.companies)

    # Only companies with a confidence score and a valid symbol.
    eligible = filter(all_c) do c
        !isnothing(get(c, :confidence, nothing)) &&
        !isempty(string(get(c, :symbol, "")))
    end
    sort!(eligible, by = c -> Float64(get(c, :market_cap_cr, 0.0)), rev=true)
    symbols = [string(c.symbol) for c in eligible]

    @info "Collecting OHLCV for $(length(symbols)) confidence-scored companies…"
    collect_ohlcv(symbols, token_map, session, OHLCV_DIR, from_date, to_date; refresh=refresh)
end

main()
