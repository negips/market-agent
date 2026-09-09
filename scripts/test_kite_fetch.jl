"""
test_kite_fetch.jl

Fetch daily OHLCV for a single symbol and print the result.
Use this to verify Kite credentials and URL format before running
the full collect_ohlcv.jl script.

Usage:
  julia scripts/test_kite_fetch.jl          # defaults to RELIANCE, 30 days
  julia scripts/test_kite_fetch.jl INFY     # specific symbol
  julia scripts/test_kite_fetch.jl INFY 90  # specific symbol + days of history
"""

using StockSwingPredictor, Dates

const REPO_ROOT = joinpath(@__DIR__, "..")

symbol = length(ARGS) >= 1 ? ARGS[1] : "RELIANCE"
days   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 30

to_date   = today() - Day(30)
from_date = to_date - Day(days)

println("Testing Kite historical data fetch")
println("  Symbol:  $symbol")
println("  Range:   $from_date → $to_date")
println()

# Load session
session = load_kite_session(REPO_ROOT)
println("  Session: $(session.api_key[1:6])… (token: $(session.access_token[1:6])…)")
println()

# Load instruments
println("Loading instrument list…")
instruments = load_instruments(session)
token_map   = build_token_map(instruments)
println("  $(length(token_map)) instruments loaded")
println()

# Look up token
token = get(token_map, symbol, nothing)
if isnothing(token)
    println("ERROR: No instrument token found for '$symbol'")
    println("Check the symbol is an exact NSE tradingsymbol (e.g. RELIANCE, HDFCBANK)")
    exit(1)
end
println("  Token for $symbol: $token")

# Build and print the URL so we can inspect it
from_s = Dates.format(from_date, "yyyy-mm-dd") * "+09:15:00"
to_s   = Dates.format(to_date,   "yyyy-mm-dd") * "+15:30:00"
url = "https://api.kite.trade/instruments/historical/$token/day?from=$from_s&to=$to_s&continuous=0&oi=0"
println("  URL: $url")
println()

using HTTP, JSON3
headers = ["X-Kite-Version" => "3",
           "Authorization"  => "token $(session.api_key):$(session.access_token)"]

# ── Step 1: verify token works at all ────────────────────────────────────────
println("Checking /user/profile…")
r = HTTP.get("https://api.kite.trade/user/profile"; headers=headers,
             request_timeout=10, status_exception=false)
println("  HTTP status: $(r.status)")
println("  $(String(r.body)[1:min(300,end)])")
println()

# ── Step 2: fetch historical data ────────────────────────────────────────────
println("Fetching historical data…")
resp = HTTP.get(url; headers=headers, request_timeout=20, status_exception=false)
println("  HTTP status: $(resp.status)")
println("  Response body:")
println(String(resp.body))
println()

if resp.status == 200
    df = fetch_ohlcv(token, from_date, to_date, session)
    println("SUCCESS — $(nrow(df)) daily bars")
    println()
    println(first(df, 5))
end
