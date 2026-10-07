"""
run_bse_confidence_checks.jl

Attach CompanyConfidence scores to every company in `bse_companies_latest.json`
(built by `generate_bse_list.jl`), the BSE counterpart of
`run_confidence_checks.jl`.

For each BSE company with a Tijori slug:
  1. Dual-listed on NSE and already scored in `nse_companies_latest.json` →
     that score is copied (`"source": "nse"`). No Tijori calls — the signals
     (Beneish, cash flow, pledging, forensics) come from the company's
     financials, which don't depend on the exchange.
  2. Otherwise → `CompanyConfidence.analyze(slug)` through the Tijori sidecar
     (~6–15 s each), `"source": "tijori"`.

Surveillance is skipped (`check_surveillance=false`), exactly like the NSE run,
so BSE and NSE scores are directly comparable. Companies without a slug are
left unscored.

Resumable and safe to interrupt: companies that already have a `"confidence"`
key are skipped, results are saved every `--save-every` companies, and work
proceeds in descending market-cap order, so stopping early leaves the largest
companies scored. Failed companies are retried on the next run.

Usage (sidecar must be running on port 3001):
  julia --project=packages/CompanyConfidence scripts/run_bse_confidence_checks.jl
  julia --project=packages/CompanyConfidence scripts/run_bse_confidence_checks.jl --min-mcap 100
  julia --project=packages/CompanyConfidence scripts/run_bse_confidence_checks.jl --limit 200
  julia --project=packages/CompanyConfidence scripts/run_bse_confidence_checks.jl --rescore
"""

using CompanyConfidence, TijoriData, JSON3, Dates, Printf

const SIDECAR_PORT  = 3001
const DATA_DIR      = joinpath(@__DIR__, "..", "website", "data")
const BSE_FILE      = joinpath(DATA_DIR, "bse_companies_latest.json")
const NSE_FILE      = joinpath(DATA_DIR, "nse_companies_latest.json")
const DEFAULT_SAVE_EVERY = 25

function parse_args()
    args = Dict{String,Any}("min_mcap" => 0.0, "limit" => nothing,
                            "save_every" => DEFAULT_SAVE_EVERY, "rescore" => false)
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a in ("-h", "--help")
            println("""
Usage:
  julia --project=packages/CompanyConfidence scripts/run_bse_confidence_checks.jl [options]

Options:
  --min-mcap CR     Only companies with market cap >= CR (₹ Cr) (default: 0 = all)
  --limit N         Stop after N Tijori analyses this run (default: no limit)
  --save-every N    Write the JSON every N analyses (default: $DEFAULT_SAVE_EVERY)
  --rescore         Ignore existing scores and redo every company
  -h, --help        Show this message

Prerequisites: sidecar on port 3001; bse_companies_latest.json
(generate_bse_list.jl); nse_companies_latest.json for score reuse.
""")
            exit(0)
        elseif a == "--min-mcap" && i < length(ARGS); args["min_mcap"] = parse(Float64, ARGS[i+1]); i += 2
        elseif a == "--limit" && i < length(ARGS);    args["limit"] = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--save-every" && i < length(ARGS); args["save_every"] = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--rescore";                      args["rescore"] = true; i += 1
        else @warn "Unknown argument: $a"; i += 1
        end
    end
    return args
end

nonempty(x) = !isnothing(x) && !isempty(string(x))
mcap(c)     = Float64(something(get(c, :market_cap_cr, nothing), 0.0))

"""Existing NSE confidence dicts keyed by NSE symbol."""
function load_nse_confidence()::Dict{String,Any}
    isfile(NSE_FILE) || return Dict{String,Any}()
    out = Dict{String,Any}()
    for c in JSON3.read(read(NSE_FILE, String)).companies
        conf = get(c, :confidence, nothing)
        conf === nothing || (out[string(c.symbol)] = conf)
    end
    return out
end

function save(raw, companies)
    payload = Dict{String,Any}(
        "date"          => string(raw.date),
        "generated_at"  => string(raw.generated_at),
        "confidence_at" => string(now(UTC)),
        "total"         => Int(raw.total),
        "sources"       => collect(raw.sources),
        "companies"     => companies,
    )
    tmp = BSE_FILE * ".tmp"
    open(io -> JSON3.pretty(io, payload), tmp, "w")
    mv(tmp, BSE_FILE; force=true)
end

function main()
    args = parse_args()

    TijoriData.configure!(port=SIDECAR_PORT)
    TijoriData.is_running() || error("Sidecar not reachable on port $SIDECAR_PORT. " *
                                      "Run: node sidecar/server_http.js")
    isfile(BSE_FILE) || error("Not found: $BSE_FILE\nRun: julia --project=packages/StockSwingPredictor scripts/generate_bse_list.jl")

    raw       = JSON3.read(read(BSE_FILE, String))
    companies = [Dict{String,Any}(string(k) => v for (k, v) in pairs(c)) for c in raw.companies]
    nse_conf  = load_nse_confidence()
    isempty(nse_conf) && @warn "No NSE confidence scores found — every company will go through Tijori"

    # ── Pass 1: reuse NSE scores for dual-listed companies ────────────────────
    reused = 0
    for c in companies
        (!args["rescore"] && haskey(c, "confidence")) && continue
        if get(c, "dual_listed", false) == true && haskey(nse_conf, string(c["nse_symbol"]))
            c["confidence"] = merge(Dict{String,Any}(string(k) => v for (k, v) in pairs(nse_conf[string(c["nse_symbol"])])),
                                    Dict{String,Any}("source" => "nse"))
            reused += 1
        end
    end
    @info "Reused NSE scores for $reused dual-listed companies"

    # ── Pass 2: analyse the rest through Tijori ───────────────────────────────
    todo = filter(companies) do c
        nonempty(get(c, "slug", nothing)) &&
        (args["rescore"] || !haskey(c, "confidence")) &&
        something(get(c, "market_cap_cr", nothing), 0.0) >= args["min_mcap"]
    end
    sort!(todo, by = c -> something(get(c, "market_cap_cr", nothing), 0.0), rev=true)
    args["limit"] === nothing || (todo = first(todo, args["limit"]))
    n = length(todo)
    @info "Analysing $n companies via Tijori (estimated $(n * 6 ÷ 60) – $(n * 15 ÷ 60) minutes)"

    failed = String[]
    done   = 0
    for (i, c) in enumerate(todo)
        slug, sym, name = string(c["slug"]), string(c["symbol"]), string(c["name"])
        t0 = time()
        try
            report = analyze(slug; check_surveillance=false)
            d = report_to_dict(report)
            d["source"] = "tijori"
            c["confidence"] = d
            done += 1
            @info "[$i/$n] $name ($sym)  →  $(report.score) / 100  $(report.pass ? "PASS" : "FAIL")  ($(round(time() - t0, digits=1))s)"
        catch e
            push!(failed, "$sym ($slug): $(sprint(showerror, e))")
            @warn "[$i/$n] $name ($sym)  →  ERROR: $(sprint(showerror, e))"
        end
        i % args["save_every"] == 0 && save(raw, companies)
    end
    save(raw, companies)

    # ── Summary ───────────────────────────────────────────────────────────────
    scored   = count(c -> haskey(c, "confidence"), companies)
    with_slug = count(c -> nonempty(get(c, "slug", nothing)), companies)
    passing  = count(c -> haskey(c, "confidence") && c["confidence"]["pass"] == true, companies)
    println()
    println("BSE confidence checks complete:")
    println("  Analysed this run: $done / $n   (reused from NSE: $reused)")
    println("  Scored overall:    $scored / $with_slug companies with a slug  ·  passing (>= $(CompanyConfidence.PASS_THRESHOLD)): $passing")
    isempty(failed) || println("  Failed ($(length(failed))):\n    " * join(first(failed, 20), "\n    "))
end

main()
