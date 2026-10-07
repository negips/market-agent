"""
generate_bse_list.jl

Build a JSON file of all BSE-listed company equities with their Tijori slug,
market cap and sector — the BSE counterpart of `generate_nse_list.jl`, and the
input a BSE confidence run needs (a slug is what `CompanyConfidence.analyze`
takes).

Sources:
  1. Kite BSE instrument list (cached, filtered by `build_token_map`) — the
     company-equity universe, same filter the OHLCV collectors use.
  2. Tijori screener (localhost:3001) — slug, sector, market cap in ₹ Cr, and
     the BSE `scripcode`, which equals Kite's BSE `exchange_token`; that is the
     join key. Rows also carry an `nse symbol` when the company is dual-listed.

Each company is tagged `dual_listed` so a BSE confidence run can reuse the NSE
score for those and only analyse the BSE-only remainder. Tijori's own `nse
symbol` field can't be trusted for this — for BSE-only companies it just echoes
the BSE symbol — so a company counts as dual-listed only when that symbol
appears in `nse_companies_latest.json` (NSE's EQ-series list, from
`generate_nse_list.jl`). Without that file nothing is tagged dual-listed.

Usage (sidecar must already be running on port 3001):
  julia --project=packages/StockSwingPredictor scripts/generate_bse_list.jl
  julia --project=packages/StockSwingPredictor scripts/generate_bse_list.jl --refresh-instruments

Output: website/data/bse_companies_YYYYMMDD.json and bse_companies_latest.json
"""

using StockSwingPredictor, HTTP, JSON3, DataFrames, Dates

const REPO_ROOT = joinpath(@__DIR__, "..")
const SIDECAR   = "http://localhost:3001"
const OUT_DIR   = joinpath(REPO_ROOT, "website", "data")
const SCREEN_PAGE_SIZE = 50

"""Kite BSE company equities as `(symbol, name, scripcode)` rows, after the same
non-company filtering as `build_token_map`."""
function fetch_kite_bse(; refresh::Bool)::DataFrame
    session   = load_kite_session(REPO_ROOT)
    instr     = load_instruments(session; exchange="BSE", refresh=refresh)
    token_map = build_token_map(instr; exchange="BSE")
    rows = filter(r -> haskey(token_map, string(r.tradingsymbol)) &&
                       string(r.exchange) == "BSE" && string(r.instrument_type) == "EQ", instr)
    df = DataFrame(symbol    = string.(rows.tradingsymbol),
                   name      = strip.(string.(coalesce.(rows.name, ""))),
                   scripcode = Int.(rows.exchange_token))
    unique!(df, :symbol)
    @info "  $(nrow(df)) BSE company equities from Kite"
    return df
end

"""Every company the Tijori screener knows, keyed for a BSE join."""
function fetch_tijori_screener()::DataFrame
    @info "Fetching Tijori screener (market cap, slug, scripcode)…"
    rows   = Dict{String,Any}[]
    offset = 0
    total  = typemax(Int)
    while offset < total
        body = JSON3.write(Dict("filters" => "Market Capitalization > 0",
                                "limit" => SCREEN_PAGE_SIZE, "offset" => offset))
        resp = HTTP.post("$SIDECAR/screen", ["Content-Type" => "application/json"], body;
                         request_timeout=60)
        data = JSON3.read(resp.body)
        data.ok || error("Screener error at offset $offset: $(data.error)")
        total = Int(data.data.total_results)
        for r in data.data.results
            sc = get(r, :scripcode, nothing)
            push!(rows, Dict("scripcode"  => sc === nothing ? missing : Int(sc),
                             "slug"       => string(get(r, :slug, "")),
                             "nse_symbol" => string(get(r, Symbol("nse symbol"), "")),
                             "sector"     => string(get(r, :segment, "")),
                             "market_cap_cr" => Float64(get(r, :latestMcapCr, 0.0))))
        end
        offset += SCREEN_PAGE_SIZE
        offset % 1000 == 0 && @info "  fetched $(min(offset, total)) / $total"
    end
    df = DataFrame(rows)
    dropmissing!(df, :scripcode)
    sort!(df, :market_cap_cr, rev=true)
    unique!(df, :scripcode)
    @info "  $(nrow(df)) Tijori companies with a BSE scripcode"
    return df
end

function load_nse_symbols()::Set{String}
    path = joinpath(OUT_DIR, "nse_companies_latest.json")
    if !isfile(path)
        @warn "$path not found — no company will be tagged dual-listed (run generate_nse_list.jl first)"
        return Set{String}()
    end
    return Set(string(c.symbol) for c in JSON3.read(read(path, String)).companies)
end

function build_table(kite::DataFrame, tijori::DataFrame, nse_symbols::Set{String})::DataFrame
    df = leftjoin(kite, tijori, on=:scripcode)
    df.dual_listed = [!ismissing(n) && n in nse_symbols for n in df.nse_symbol]
    df.nse_symbol  = [d ? n : missing for (d, n) in zip(df.dual_listed, df.nse_symbol)]
    sort!(df, [order(:market_cap_cr, rev=true, lt=(a, b) -> coalesce(a, -Inf) < coalesce(b, -Inf)), :name])
    return df
end

"""Confidence results already in `bse_companies_latest.json`, by scripcode, so
regenerating the list doesn't throw away hours of `run_bse_confidence_checks.jl`."""
function load_existing_confidence()
    path = joinpath(OUT_DIR, "bse_companies_latest.json")
    isfile(path) || return Dict{Int,Any}()
    out = Dict{Int,Any}()
    for c in JSON3.read(read(path, String)).companies
        conf = get(c, :confidence, nothing)
        conf === nothing || (out[Int(c.scripcode)] = conf)
    end
    return out
end

function write_json(df::DataFrame, date::Date)
    kept = load_existing_confidence()
    nz(x) = (ismissing(x) || (x isa AbstractString && isempty(x))) ? nothing : x
    companies = [merge(Dict("symbol"        => r.symbol,
                      "name"          => r.name,
                      "scripcode"     => r.scripcode,
                      "slug"          => nz(r.slug),
                      "nse_symbol"    => nz(r.nse_symbol),
                      "dual_listed"   => r.dual_listed,
                      "sector"        => nz(r.sector),
                      "market_cap_cr" => nz(r.market_cap_cr)),
                      haskey(kept, r.scripcode) ? Dict("confidence" => kept[r.scripcode]) : Dict{String,Any}())
                 for r in eachrow(df)]
    payload = Dict("date" => string(date), "generated_at" => string(now(UTC)),
                   "total" => length(companies),
                   "sources" => ["Kite BSE instruments", "Tijori Finance screener"],
                   "companies" => companies)
    mkpath(OUT_DIR)
    out = joinpath(OUT_DIR, "bse_companies_" * Dates.format(date, "yyyymmdd") * ".json")
    open(io -> JSON3.pretty(io, payload), out, "w")
    cp(out, joinpath(OUT_DIR, "bse_companies_latest.json"); force=true)
    @info "Written: $out ($(length(companies)) companies) and bse_companies_latest.json"
end

function main()
    if "--help" in ARGS || "-h" in ARGS
        println("""
Usage:
  julia --project=packages/StockSwingPredictor scripts/generate_bse_list.jl [--refresh-instruments]

Builds website/data/bse_companies_latest.json (Kite BSE company equities joined
to Tijori slug/market cap by BSE scripcode). Needs the sidecar on port 3001 and a
Kite session (only for --refresh-instruments; the cached instrument list is used
otherwise).
""")
        return
    end
    kite   = fetch_kite_bse(; refresh="--refresh-instruments" in ARGS)
    tijori = fetch_tijori_screener()
    df     = build_table(kite, tijori, load_nse_symbols())

    n         = nrow(df)
    with_slug = count(s -> !ismissing(s) && !isempty(s), df.slug)
    dual      = count(df.dual_listed)
    only_slug = count(i -> !df.dual_listed[i] && !ismissing(df.slug[i]) && !isempty(df.slug[i]), 1:n)
    @info "BSE companies: $n · with Tijori slug: $with_slug · dual-listed on NSE: $dual · " *
          "BSE-only with slug: $only_slug · no slug: $(n - with_slug)"
    write_json(df, today())
end

main()
