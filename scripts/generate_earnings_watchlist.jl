"""
generate_earnings_watchlist.jl

Build website/data/earnings_watchlist_latest.json for the Watchlist page.

Combines two sources of earnings dates (in priority order):
  1. NSE event-calendar  — confirmed board meeting notices (typically 2-4 weeks out)
  2. earnings_projections.json — projected dates from Tijori history (run enrich_earnings_dates.jl)

NSE confirmed dates take priority. Projected dates fill in the rest, allowing
a much longer look-ahead window than the NSE calendar alone.

Prerequisites:
  - website/data/nse_companies_latest.json    (julia scripts/generate_nse_list.jl)
  - website/data/earnings_projections.json    (julia --project=packages/TijoriData scripts/enrich_earnings_dates.jl)
  - Confidence checks run                     (julia --project=packages/CompanyConfidence scripts/run_confidence_checks.jl)

Usage:
  julia --project=packages/EarningsCalendar scripts/generate_earnings_watchlist.jl
  julia --project=packages/EarningsCalendar scripts/generate_earnings_watchlist.jl 365
"""

using EarningsCalendar, JSON3, Dates, Printf

const DATA_FILE        = joinpath(@__DIR__, "..", "website", "data", "nse_companies_latest.json")
const PROJECTIONS_FILE = joinpath(@__DIR__, "..", "website", "data", "earnings_projections.json")
const OUT_DIR          = joinpath(@__DIR__, "..", "website", "data")
const OUT_FILE         = joinpath(OUT_DIR, "earnings_watchlist_latest.json")

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
  - website/data/earnings_projections.json present (optional but recommended for long windows)

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
            @info "Loaded $(length(projections)) earnings projections from Tijori history"
        catch e
            @warn "Could not load projections file: $(sprint(showerror, e))"
        end
    else
        @warn "No projections file found — using NSE calendar only (short window)\n" *
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

    # Symbol → confirmed NSE event (first/earliest)
    nse_by_symbol = Dict{String, EarningsEvent}()
    for ev in nse_events
        haskey(nse_by_symbol, ev.symbol) || (nse_by_symbol[ev.symbol] = ev)
    end

    # ── Build watchlist entries ───────────────────────────────────────────────

    matched = Any[]
    seen    = Set{String}()

    # Pass 1: NSE confirmed events (for scored companies)
    for (sym, ev) in nse_by_symbol
        haskey(company_by_symbol, sym) || continue
        sym in seen && continue
        push!(seen, sym)
        c = company_by_symbol[sym]
        push!(matched, _entry(sym, c, string(ev.date), ev.purpose, "nse_confirmed"))
    end

    # Pass 2: Projected dates for scored companies not already in the list
    for (sym, c) in company_by_symbol
        sym in seen && continue
        proj = get(projections, sym, nothing)
        isnothing(proj) && continue
        date_str = string(get(proj, :projected_date, nothing) === nothing ?
                          get(proj, "projected_date", nothing) :
                          get(proj, :projected_date, nothing))
        (isempty(date_str) || date_str == "nothing") && continue

        proj_date = tryparse(Date, date_str)
        isnothing(proj_date) && continue
        proj_date < from_date && continue
        proj_date > to_date   && continue

        push!(seen, sym)
        push!(matched, _entry(sym, c, date_str, "Projected Results",
                              "tijori_projected",
                              get(proj, :projection_confidence,
                                  get(proj, "projection_confidence", "medium"))))
    end

    sort!(matched, by = e -> e["earnings_date"])

    n_confirmed = count(e -> e["date_source"] == "nse_confirmed", matched)
    n_projected = count(e -> e["date_source"] == "tijori_projected", matched)
    @info "  $n_confirmed confirmed (NSE) + $n_projected projected (Tijori) = $(length(matched)) total"

    # ── Write output ──────────────────────────────────────────────────────────

    mkpath(OUT_DIR)
    payload = Dict{String, Any}(
        "generated_at"  => string(now(UTC)),
        "from_date"     => string(from_date),
        "to_date"       => string(to_date),
        "window_days"   => window_days,
        "total"         => length(matched),
        "n_confirmed"   => n_confirmed,
        "n_projected"   => n_projected,
        "events"        => matched,
    )

    open(OUT_FILE, "w") do io
        JSON3.pretty(io, payload)
    end
    @info "Written: $OUT_FILE"

    # ── Summary table ─────────────────────────────────────────────────────────

    if !isempty(matched)
        println()
        @printf("%-12s %-6s %-5s %-15s %s\n", "Date", "Score", "Src", "Symbol", "Company")
        println("─"^72)
        for e in matched
            days  = e["days_until"]
            score = e["confidence"] !== nothing ? string(round(Int, e["confidence"]["score"])) : "?"
            src   = e["date_source"] == "nse_confirmed" ? "NSE" : "est"
            day_s = days == 0 ? "today" : days == 1 ? "tmrw" : "in $days"
            @printf("%-12s %-6s %-5s %-15s %s\n",
                    day_s, score, src, e["symbol"],
                    e["company"][1:min(40,length(e["company"]))])
        end
    else
        println("\nNo scored companies have earnings in the next $window_days days.")
        println("Tip: run enrich_earnings_dates.jl to add projected dates.")
    end
end

function _entry(sym, c, date_str, purpose, source, proj_conf="")
    Dict{String, Any}(
        "symbol"               => sym,
        "company"              => string(get(c, :name, sym)),
        "earnings_date"        => date_str,
        "days_until"           => (Date(date_str) - today()).value,
        "purpose"              => purpose,
        "date_source"          => source,
        "projection_confidence"=> proj_conf,
        "sector"               => string(get(c, :sector, "")),
        "market_cap_cr"        => get(c, :market_cap_cr, nothing),
        "close_price"          => get(c, :close_price,   nothing),
        "confidence"           => get(c, :confidence,    nothing),
    )
end

main()
