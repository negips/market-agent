"""
generate_earnings_watchlist.jl

Build website/data/earnings_watchlist_latest.json for the Watchlist page.

Earnings dates come from two sources (in priority order):
  1. NSE event-calendar  — confirmed board meeting notices (typically 2-4 weeks out)
  2. earnings_projections.json — projected dates from Tijori history

Prices and market caps are fetched live from Kite Connect (single batch call)
so the watchlist always reflects today's market data. Falls back to NSE bhavcopy
prices if Kite is unavailable or the session has expired.

Prerequisites:
  - website/data/nse_companies_latest.json    (julia scripts/generate_nse_list.jl)
  - website/data/earnings_projections.json    (julia --project=packages/TijoriData scripts/enrich_earnings_dates.jl)
  - Confidence checks run                     (julia --project=packages/CompanyConfidence scripts/run_confidence_checks.jl)
  - sidecar/kite_session.json                 (node sidecar/kite_setup.js — daily)

Usage:
  julia --project=packages/EarningsCalendar scripts/generate_earnings_watchlist.jl
  julia --project=packages/EarningsCalendar scripts/generate_earnings_watchlist.jl 365
"""

using EarningsCalendar, JSON3, Dates, Printf, HTTP

const DATA_FILE        = joinpath(@__DIR__, "..", "website", "data", "nse_companies_latest.json")
const PROJECTIONS_FILE = joinpath(@__DIR__, "..", "website", "data", "earnings_projections.json")
const SESSION_FILE     = joinpath(@__DIR__, "..", "sidecar", "kite_session.json")
const OUT_DIR          = joinpath(@__DIR__, "..", "website", "data")
const OUT_FILE         = joinpath(OUT_DIR, "earnings_watchlist_latest.json")
const KITE_BASE        = "https://api.kite.trade"

# ── Kite quote fetching ───────────────────────────────────────────────────────

struct KiteQuote
    last_price :: Float64
    prev_close :: Float64
    volume     :: Int
    change_pct :: Float64
end

function load_kite_session()::Union{NamedTuple, Nothing}
    isfile(SESSION_FILE) || return nothing
    try
        s = JSON3.read(read(SESSION_FILE, String))
        # Warn if session is from a previous day but still try — market may be closed
        if string(get(s, :date, "")) != string(today())
            @warn "Kite session is from $(get(s, :date, "?")) — token may be expired. " *
                  "Run: node sidecar/kite_setup.js"
        end
        return (api_key=string(s.api_key), access_token=string(s.access_token))
    catch
        return nothing
    end
end

function fetch_kite_quotes(symbols::Vector{String})::Dict{String, KiteQuote}
    session = load_kite_session()
    if isnothing(session)
        @warn "No Kite session found — using bhavcopy prices. Run: node sidecar/kite_setup.js"
        return Dict{String, KiteQuote}()
    end

    # Build query string: ?i=NSE:INFY&i=NSE:TCS&...
    qs = join(["i=NSE:$sym" for sym in symbols], "&")
    url = "$KITE_BASE/quote?$qs"

    headers = [
        "X-Kite-Version" => "3",
        "Authorization"  => "token $(session.api_key):$(session.access_token)",
    ]

    resp = try
        HTTP.get(url; headers=headers, request_timeout=15, status_exception=false)
    catch e
        @warn "Kite quote request failed: $(sprint(showerror, e))"
        return Dict{String, KiteQuote}()
    end

    if resp.status == 403
        @warn "Kite token rejected (403) — session may have expired. Run: node sidecar/kite_setup.js"
        return Dict{String, KiteQuote}()
    elseif resp.status != 200
        @warn "Kite quote returned HTTP $(resp.status)"
        return Dict{String, KiteQuote}()
    end

    raw = try
        JSON3.read(resp.body)
    catch
        @warn "Could not parse Kite quote response"
        return Dict{String, KiteQuote}()
    end

    quotes = Dict{String, KiteQuote}()
    for (key, q) in pairs(raw.data)
        sym = replace(string(key), "NSE:" => "")
        last  = Float64(get(q, :last_price, 0.0))
        close = Float64(get(get(q, :ohlc, (;)), :close, last))
        vol   = Int(get(q, :volume, 0))
        chg   = close > 0 ? (last - close) / close * 100 : 0.0
        quotes[sym] = KiteQuote(last, close, vol, chg)
    end

    return quotes
end

# ── Market cap adjustment ─────────────────────────────────────────────────────

# Adjusts stored market cap by the live price move: shares × new_price
function adjusted_mcap(old_mcap, old_price, new_price)
    (old_mcap === nothing || old_price === nothing ||
     old_price == 0.0 || new_price == 0.0) && return old_mcap
    return Float64(old_mcap) * (new_price / Float64(old_price))
end

# ── Main ──────────────────────────────────────────────────────────────────────

function main()
    if "--help" in ARGS || "-h" in ARGS
        println("""
Usage:
  julia --project=packages/EarningsCalendar scripts/generate_earnings_watchlist.jl [DAYS]

Arguments:
  DAYS   Optional. Number of calendar days to look ahead (default: 30).

Options:
  -h, --help   Show this message and exit.

Prerequisites:
  - website/data/nse_companies_latest.json present
  - website/data/earnings_projections.json present (optional, extends window)
  - sidecar/kite_session.json present (optional, provides live prices)

Output:
  website/data/earnings_watchlist_latest.json
""")
        return
    end

    window_days = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 30
    from_date   = today()
    to_date     = from_date + Day(window_days - 1)

    # ── Load company data ─────────────────────────────────────────────────────

    isfile(DATA_FILE) || error("Not found: $DATA_FILE\nRun: julia scripts/generate_nse_list.jl")

    raw   = JSON3.read(read(DATA_FILE, String))
    all_c = collect(raw.companies)

    company_by_symbol = Dict{String, Any}()
    for c in all_c
        isnothing(get(c, :confidence, nothing)) && continue
        sym = string(get(c, :symbol, ""))
        isempty(sym) && continue
        company_by_symbol[sym] = c
    end
    @info "$(length(company_by_symbol)) confidence-scored companies"

    # ── Load projected dates ──────────────────────────────────────────────────

    projections = Dict{String, Any}()
    if isfile(PROJECTIONS_FILE)
        try
            proj_raw = JSON3.read(read(PROJECTIONS_FILE, String))
            for (sym, val) in pairs(proj_raw.projections)
                projections[string(sym)] = val
            end
            @info "Loaded $(length(projections)) earnings projections"
        catch e
            @warn "Could not load projections file: $(sprint(showerror, e))"
        end
    else
        @warn "No projections file — using NSE calendar only\n" *
              "Run: julia --project=packages/TijoriData scripts/enrich_earnings_dates.jl"
    end

    # ── Fetch NSE confirmed events ────────────────────────────────────────────

    @info "Fetching NSE event-calendar for $from_date → $to_date…"
    nse_events = try
        fetch_earnings_calendar(from_date, to_date)
    catch e
        @warn "NSE calendar fetch failed: $(sprint(showerror, e))"
        EarningsEvent[]
    end
    @info "  $(length(nse_events)) confirmed NSE events"

    nse_by_symbol = Dict{String, EarningsEvent}()
    for ev in nse_events
        haskey(nse_by_symbol, ev.symbol) || (nse_by_symbol[ev.symbol] = ev)
    end

    # ── Build candidate list ──────────────────────────────────────────────────

    seen    = Set{String}()
    matched = Any[]

    for (sym, ev) in nse_by_symbol
        haskey(company_by_symbol, sym) || continue
        sym in seen && continue
        push!(seen, sym)
        push!(matched, _entry(sym, company_by_symbol[sym], string(ev.date), ev.purpose, "nse_confirmed"))
    end

    for (sym, c) in company_by_symbol
        sym in seen && continue
        proj = get(projections, sym, nothing)
        isnothing(proj) && continue
        date_str = string(get(proj, :projected_date, get(proj, "projected_date", nothing)))
        (isempty(date_str) || date_str == "nothing") && continue
        proj_date = tryparse(Date, date_str)
        isnothing(proj_date) && continue
        proj_date < from_date && continue
        proj_date > to_date   && continue
        push!(seen, sym)
        push!(matched, _entry(sym, c, date_str, "Projected Results", "tijori_projected",
                              string(get(proj, :projection_confidence,
                                         get(proj, "projection_confidence", "medium")))))
    end

    sort!(matched, by = e -> e["earnings_date"])

    # ── Fetch live prices from Kite ───────────────────────────────────────────

    all_symbols = [e["symbol"] for e in matched]
    quotes = Dict{String, KiteQuote}()
    if !isempty(all_symbols)
        @info "Fetching live prices from Kite for $(length(all_symbols)) symbols…"
        quotes = fetch_kite_quotes(all_symbols)
        @info "  $(length(quotes)) quotes received"
    end

    # Apply Kite prices (or fall back to bhavcopy)
    for e in matched
        sym = e["symbol"]
        c   = company_by_symbol[sym]
        if haskey(quotes, sym)
            q = quotes[sym]
            e["close_price"]   = q.last_price
            e["market_cap_cr"] = adjusted_mcap(get(c, :market_cap_cr, nothing),
                                               get(c, :close_price,   nothing),
                                               q.last_price)
            e["volume"]        = q.volume
            e["change_pct"]    = round(q.change_pct, digits=2)
            e["price_source"]  = "kite_live"
        else
            e["close_price"]   = get(c, :close_price,   nothing)
            e["market_cap_cr"] = get(c, :market_cap_cr, nothing)
            e["volume"]        = get(c, :volume,        nothing)
            e["change_pct"]    = nothing
            e["price_source"]  = "bhavcopy"
        end
    end

    n_confirmed = count(e -> e["date_source"] == "nse_confirmed",   matched)
    n_projected = count(e -> e["date_source"] == "tijori_projected", matched)
    n_live      = count(e -> e["price_source"] == "kite_live",       matched)
    @info "  $n_confirmed confirmed + $n_projected projected | $n_live live prices, $(length(matched)-n_live) from bhavcopy"

    # ── Write output ──────────────────────────────────────────────────────────

    mkpath(OUT_DIR)
    payload = Dict{String, Any}(
        "generated_at" => string(now(UTC)),
        "from_date"    => string(from_date),
        "to_date"      => string(to_date),
        "window_days"  => window_days,
        "total"        => length(matched),
        "n_confirmed"  => n_confirmed,
        "n_projected"  => n_projected,
        "events"       => matched,
    )

    open(OUT_FILE, "w") do io
        JSON3.pretty(io, payload)
    end
    @info "Written: $OUT_FILE"

    # ── Summary ───────────────────────────────────────────────────────────────

    if !isempty(matched)
        println()
        @printf("%-12s %-6s %-4s %-5s %-15s %s\n", "Date", "Score", "Src", "Price", "Symbol", "Company")
        println("─"^75)
        for e in matched
            days  = e["days_until"]
            score = e["confidence"] !== nothing ? string(round(Int, e["confidence"]["score"])) : "?"
            src   = e["date_source"] == "nse_confirmed" ? "NSE" : "est"
            psrc  = e["price_source"] == "kite_live" ? "live" : "bhav"
            day_s = days == 0 ? "today" : days == 1 ? "tmrw" : "in $days"
            @printf("%-12s %-6s %-4s %-5s %-15s %s\n",
                    day_s, score, src, psrc, e["symbol"],
                    e["company"][1:min(35, length(e["company"]))])
        end
    else
        println("\nNo scored companies have earnings in the next $window_days days.")
        println("Tip: run enrich_earnings_dates.jl to add projected dates.")
    end
end

function _entry(sym, c, date_str, purpose, source, proj_conf="")
    Dict{String, Any}(
        "symbol"                => sym,
        "company"               => string(get(c, :name, sym)),
        "earnings_date"         => date_str,
        "days_until"            => (Date(date_str) - today()).value,
        "purpose"               => purpose,
        "date_source"           => source,
        "projection_confidence" => proj_conf,
        "sector"                => string(get(c, :sector, "")),
        "confidence"            => get(c, :confidence, nothing),
        # price fields filled in after Kite fetch
        "close_price"           => nothing,
        "market_cap_cr"         => nothing,
        "volume"                => nothing,
        "change_pct"            => nothing,
        "price_source"          => "bhavcopy",
    )
end

main()
