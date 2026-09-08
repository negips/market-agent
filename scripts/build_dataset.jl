"""
build_dataset.jl

Assemble the training Dataset from cached daily OHLCV, 60-minute OHLCV,
and LLM scalar features.

Steps:
  1. Load confidence-scored company list → universe ordering.
  2. Build master trading calendar + (n_dates × n_companies) closes and vols matrices.
  3. For each company, slide a weekly window over the calendar:
       - Slice and normalise the hourly series (N_HOURLY_BARS bars).
       - Look up LLM scalars (most recent doc before the window date).
       - Compute 5-day hourly trajectory label.
  4. Sort all examples by date, save to BSON.

The saved Dataset is loaded directly by train_model.jl.

Prerequisites:
  - website/data/ohlcv/{SYMBOL}_daily.csv (from collect_ohlcv.jl)
  - website/data/ohlcv/{SYMBOL}_hourly.csv (from collect_ohlcv.jl)
  - website/data/llm_features/{SYMBOL}.json (from extract_llm_features.jl)
  - website/data/nse_companies_latest.json (with confidence scores)

Usage:
  julia --project=packages/StockSwingPredictor scripts/build_dataset.jl
  julia --project=packages/StockSwingPredictor scripts/build_dataset.jl 100  # top-N only
"""

using StockSwingPredictor, TijoriData, JSON3, DataFrames, CSV, Dates, Printf

const REPO_ROOT       = joinpath(@__DIR__, "..")
const COMPANIES_FILE  = joinpath(REPO_ROOT, "website", "data", "nse_companies_latest.json")
const OHLCV_DIR       = joinpath(REPO_ROOT, "website", "data", "ohlcv")
const LLM_DIR         = joinpath(REPO_ROOT, "website", "data", "llm_features")
const OUT_DIR         = joinpath(REPO_ROOT, "website", "data", "training")
const DATASET_FILE    = joinpath(OUT_DIR, "dataset.bson")

function main()
    if "--help" in ARGS || "-h" in ARGS
        println("""
Usage:
  julia --project=packages/StockSwingPredictor scripts/build_dataset.jl [N]

Arguments:
  N   Top N companies by market cap to include (default: all scored).

Output:
  website/data/training/dataset.bson
""")
        return
    end

    top_n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : typemax(Int)

    # ── Load company universe ─────────────────────────────────────────────────

    isfile(COMPANIES_FILE) || error("Not found: $COMPANIES_FILE")
    raw     = JSON3.read(read(COMPANIES_FILE, String))
    all_c   = collect(raw.companies)
    eligible = filter(all_c) do c
        !isnothing(get(c, :confidence, nothing)) &&
        !isempty(string(get(c, :symbol, "")))
    end
    sort!(eligible, by = c -> Float64(get(c, :market_cap_cr, 0.0)), rev=true)
    universe = first(eligible, top_n)
    companies = [string(c.symbol) for c in universe]

    @info "Universe: $(length(companies)) companies"

    # ── Build master market matrices ──────────────────────────────────────────

    @info "Building market matrices from daily OHLCV…"
    closes, vols, master_dates = build_market_matrices(OHLCV_DIR, companies)
    @info "  $(length(master_dates)) trading dates × $(length(companies)) companies"
    @info "  closes range: $(minimum(filter(!isnan, closes)))  →  $(maximum(filter(!isnan, closes)))"

    # ── Generate training examples ────────────────────────────────────────────

    all_examples = TrainingExample[]
    ok = missing_daily = missing_hourly = 0

    for (j, c) in enumerate(universe)
        sym  = string(get(c, :symbol, ""))
        name = string(get(c, :name, sym))

        daily_path  = joinpath(OHLCV_DIR, "$(sym)_daily.csv")
        hourly_path = joinpath(OHLCV_DIR, "$(sym)_hourly.csv")

        if !isfile(daily_path)
            missing_daily += 1
            continue
        end
        if !isfile(hourly_path)
            missing_hourly += 1
            continue
        end

        daily  = sort!(CSV.read(daily_path,  DataFrame; types=Dict(:date => Date)), :date)
        hourly = sort!(CSV.read(hourly_path, DataFrame; types=Dict(:datetime => DateTime)), :datetime)

        llm_cache = _load_llm_cache(sym, LLM_DIR)

        examples = generate_company_examples(sym, j, daily, hourly,
                                              master_dates, llm_cache)
        append!(all_examples, examples)
        ok += 1

        j % 50 == 0 && @info "[$j/$(length(universe))] $sym — $(length(examples)) examples (total: $(length(all_examples)))"
    end

    @info "Companies: $ok processed | $missing_daily no daily OHLCV | $missing_hourly no hourly OHLCV"
    @info "Total examples before sorting: $(length(all_examples))"

    # ── Sort by date (required for time-ordered split) ────────────────────────

    sort!(all_examples, by = ex -> (ex.date, ex.symbol))

    # ── Build and save Dataset ────────────────────────────────────────────────

    dataset = Dataset(closes, vols, master_dates, companies, all_examples)
    @info dataset

    mkpath(OUT_DIR)
    save_dataset(dataset, DATASET_FILE)
    println()
    @info "Done → $DATASET_FILE"
end

# ── Helpers ───────────────────────────────────────────────────────────────────

function _load_llm_cache(symbol::String, llm_dir::String)::Dict{Date, LLMFeatures}
    cache = Dict{Date, LLMFeatures}()
    path  = joinpath(llm_dir, "$(symbol).json")
    isfile(path) || return cache
    try
        d = JSON3.read(read(path, String))
        f = d.features
        date_str     = string(get(d, :extracted_at, ""))
        extracted_at = tryparse(Date, length(date_str) >= 10 ? date_str[1:10] : "")
        doc_date     = isnothing(extracted_at) ? today() : extracted_at
        cache[doc_date] = LLMFeatures(
            Float32(get(f, :management_tone,           0.0)),
            Float32(get(f, :guidance_direction,        0.0)),
            Float32(get(f, :guidance_specificity,      0.0)),
            Float32(get(f, :demand_outlook,            0.0)),
            Float32(get(f, :margin_commentary,         0.0)),
            Float32(get(f, :competitive_pressure,      0.0)),
            Float32(get(f, :new_wins_announced,        0.0)),
            Float32(get(f, :capex_expansion,           0.0)),
            Float32(get(f, :buyback_or_dividend,       0.0)),
            Float32(get(f, :mgmt_language_hedging,     0.0)),
            Float32(get(f, :auditor_concerns,          0.0)),
            Float32(get(f, :related_party_flags,       0.0)),
            Float32(get(f, :contingent_liability_flag, 0.0)),
            Float32(get(f, :extraction_confidence,     0.0)),
            Float32(get(f, :doc_age_days,              0.0)),
        )
    catch e
        @warn "LLM cache load failed for $symbol: $(sprint(showerror, e))"
    end
    return cache
end

main()
