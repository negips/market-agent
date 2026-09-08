"""
generate_nse_list.jl

Build a JSON file of all NSE-listed equity companies with today's closing
price and latest market cap, suitable for HTML display.

Sources:
  1. NSE equity list   (nsearchives.nseindia.com) — company name, ISIN, face value
  2. NSE sec_bhavdata  (nsearchives.nseindia.com) — today's EQ closing prices
  3. Tijori screener   (localhost:3001)           — market cap in ₹ Cr + sector

Usage (sidecar must already be running on port 3001):
  julia scripts/generate_nse_list.jl
  julia scripts/generate_nse_list.jl 2026-09-07   # explicit date
"""

using HTTP, JSON3, CSV, DataFrames, Dates

# ── Config ────────────────────────────────────────────────────────────────────

const SIDECAR  = "http://localhost:3001"
const OUT_DIR  = joinpath(@__DIR__, "..", "website", "data")
const NSE_BASE = "https://nsearchives.nseindia.com"
const HEADERS  = ["User-Agent" => "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"]

# ── Step 1: Download NSE equity list ─────────────────────────────────────────

function fetch_equity_list()::DataFrame
    @info "Fetching NSE equity list..."
    resp = HTTP.get("$NSE_BASE/content/equities/EQUITY_L.csv"; headers=HEADERS, request_timeout=30)
    # CSV has leading/trailing spaces in headers; clean them
    df = CSV.read(resp.body, DataFrame; header=1, silencewarnings=true)
    rename!(df, strip.(names(df)) .|> Symbol)

    # Keep only EQ series companies
    filter!(row -> strip(string(get(row, Symbol("SERIES"), ""))) == "EQ", df)

    select!(df, [
        Symbol("SYMBOL")         => :symbol,
        Symbol("NAME OF COMPANY") => :name,
        Symbol("ISIN NUMBER")    => :isin,
        Symbol("FACE VALUE")     => :face_value,
        Symbol("DATE OF LISTING") => :listed_since,
    ])
    df.symbol      = strip.(string.(df.symbol))
    df.name        = strip.(string.(df.name))
    df.isin        = strip.(string.(df.isin))
    df.listed_since = strip.(string.(df.listed_since))
    @info "  $(nrow(df)) EQ-series companies"
    return df
end

# ── Step 2: Download today's bhavcopy ────────────────────────────────────────

function fetch_bhavcopy(date::Date)::DataFrame
    d = Dates.format(date, "ddmmyyyy")
    url = "$NSE_BASE/products/content/sec_bhavdata_full_$(d).csv"
    @info "Fetching bhavcopy for $(date)..."
    resp = HTTP.get(url; headers=HEADERS, request_timeout=30)
    df = CSV.read(resp.body, DataFrame; header=1, silencewarnings=true)
    rename!(df, strip.(names(df)) .|> Symbol)

    # Keep EQ series only; select relevant columns
    filter!(row -> strip(string(row[:SERIES])) == "EQ", df)

    select!(df, [
        :SYMBOL          => :symbol,
        :CLOSE_PRICE     => :close_price,
        :TTL_TRD_QNTY   => :volume,
        :TURNOVER_LACS   => :turnover_lacs,
    ])
    df.symbol = strip.(string.(df.symbol))
    for col in (:close_price, :volume, :turnover_lacs)
        df[!, col] = [begin v = df[i, col]; v isa AbstractString ? parse(Float64, replace(string(v), "," => "")) : Float64(v) end for i in 1:nrow(df)]
    end
    @info "  $(nrow(df)) EQ rows in bhavcopy"
    return df
end

# ── Step 3: Paginate Tijori screener for market cap ──────────────────────────

function fetch_tijori_mktcap()::DataFrame
    @info "Fetching market cap from Tijori screener..."
    all_rows = Dict{String, Any}[]
    offset   = 0
    limit    = 50
    total    = typemax(Int)

    while offset < total
        body = JSON3.write(Dict("filters" => "Market Capitalization > 0",
                                "limit"   => limit, "offset" => offset))
        resp = HTTP.post("$SIDECAR/screen",
                         ["Content-Type" => "application/json"], body;
                         request_timeout=30)
        data = JSON3.read(resp.body)
        data.ok || error("Screener error at offset $offset: $(data.error)")

        d = data.data
        total = Int(d.total_results)
        for r in d.results
            push!(all_rows, Dict(
                "symbol_tijori" => string(get(r, Symbol("nse symbol"), "")),
                "slug"          => string(get(r, :slug, "")),
                "sector"        => string(get(r, :segment, "")),
                "market_cap_cr" => Float64(get(r, :latestMcapCr, 0.0)),
            ))
        end

        offset += limit
        offset % 500 == 0 && @info "  fetched $(min(offset, total)) / $total"
    end

    df = DataFrame(all_rows)
    # Filter to companies where NSE symbol is known
    filter!(row -> !isempty(row.symbol_tijori), df)
    # Deduplicate (keep highest market cap entry per symbol)
    sort!(df, :market_cap_cr, rev=true)
    unique!(df, :symbol_tijori)
    rename!(df, :symbol_tijori => :symbol)
    df.slug = string.(df.slug)
    @info "  $(nrow(df)) NSE companies with market cap from Tijori"
    return df
end

# ── Step 4: Join and sort ─────────────────────────────────────────────────────

function build_table(equity::DataFrame, bhav::DataFrame, mktcap::DataFrame)::DataFrame
    # Join equity list + bhavcopy on symbol
    df = leftjoin(equity, bhav,    on=:symbol)
    df = leftjoin(df,    mktcap,   on=:symbol)

    # Sort alphabetically by company name
    sort!(df, :name)

    # Replace missing with nothing-friendly values for JSON
    for col in names(df)
        if eltype(df[!, col]) >: Missing
            df[!, col] = coalesce.(df[!, col], eltype(df[!, col]) == Union{Missing, Float64} ? nothing : "")
        end
    end
    return df
end

# ── Step 5: Write JSON ────────────────────────────────────────────────────────

function write_json(df::DataFrame, date::Date, out_dir::String)
    mkpath(out_dir)
    fname = "nse_companies_" * Dates.format(date, "yyyymmdd") * ".json"
    out_path = joinpath(out_dir, fname)

    companies = []
    for row in eachrow(df)
        push!(companies, Dict(
            "symbol"        => row.symbol,
            "name"          => row.name,
            "isin"          => row.isin,
            "face_value"    => ismissing(row.face_value)    ? nothing : row.face_value,
            "listed_since"  => row.listed_since,
            "slug"          => ismissing(row.slug)          ? nothing : (isempty(row.slug) ? nothing : row.slug),
            "sector"        => ismissing(row.sector)        ? nothing : row.sector,
            "close_price"   => ismissing(row.close_price)   ? nothing : row.close_price,
            "volume"        => ismissing(row.volume)        ? nothing : row.volume,
            "turnover_lacs" => ismissing(row.turnover_lacs) ? nothing : row.turnover_lacs,
            "market_cap_cr" => ismissing(row.market_cap_cr) ? nothing : row.market_cap_cr,
        ))
    end

    payload = Dict(
        "date"          => string(date),
        "generated_at"  => string(now(UTC)),
        "total"         => length(companies),
        "sources"       => ["NSE equity list", "NSE sec_bhavdata", "Tijori Finance screener"],
        "companies"     => companies,
    )

    open(out_path, "w") do io
        JSON3.pretty(io, payload)
    end

    # Always write a stable "latest" copy for the HTML viewer
    latest_path = joinpath(out_dir, "nse_companies_latest.json")
    cp(out_path, latest_path; force=true)

    @info "Written: $out_path  ($(length(companies)) companies)"
    @info "Updated: $latest_path"
    return out_path
end

# ── Main ──────────────────────────────────────────────────────────────────────

function main()
    if "--help" in ARGS || "-h" in ARGS
        println("""
Usage:
  julia scripts/generate_nse_list.jl [DATE]

Arguments:
  DATE   Optional. Date in YYYY-MM-DD format (default: today).
         Used to fetch the NSE bhavcopy for that trading day.

Options:
  -h, --help   Show this message and exit.

Prerequisites:
  - Tijori sidecar running on port 3001 (node sidecar/server_http.js)

Output:
  website/data/nse_companies_YYYYMMDD.json   — dated snapshot
  website/data/nse_companies_latest.json     — stable copy for the HTML viewer
""")
        return
    end

    date = length(ARGS) >= 1 ? Date(ARGS[1]) : today()

    equity  = fetch_equity_list()
    bhav    = fetch_bhavcopy(date)
    mktcap  = fetch_tijori_mktcap()
    df      = build_table(equity, bhav, mktcap)
    out     = write_json(df, date, string(OUT_DIR))

    # Summary
    with_cap  = count(!ismissing, df.market_cap_cr)
    with_price = count(!ismissing, df.close_price)
    println()
    println("Summary:")
    println("  Total NSE EQ companies:  $(nrow(df))")
    println("  With today's close:      $with_price")
    println("  With market cap:         $with_cap")
    println("  Output:                  $out")
end

main()
