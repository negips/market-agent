"""
build_dataset.jl

Assemble the training Dataset from the inference cache and LLM features.

All price data is read from the pre-built inference cache — no OHLCV CSVs
are touched here. The resulting dataset.bson contains only index pointers
and small feature vectors (~30 MB regardless of universe size).

Steps:
  1. Load inference cache → company universe + price matrices.
  2. For each company, slide a weekly window over the master calendar:
       - Record hourly_end_idx pointer (no hourly data stored per example).
       - Look up LLM scalars (most recent doc before the window date).
       - Compute 5-day hourly log-return trajectory label.
  3. Sort all examples by date, save to BSON.

Prerequisites:
  - website/data/inference_cache.bson    (from build_cache.jl)
  - website/data/llm_features/{SYM}.json (from extract_llm_features.jl, optional)

Usage:
  julia --project=packages/StockSwingPredictor scripts/build_dataset.jl
"""

using StockSwingPredictor, JSON3, Dates, Printf

const REPO_ROOT    = joinpath(@__DIR__, "..")
const CACHE_FILE   = joinpath(REPO_ROOT, "website", "data", "inference_cache.bson")
const LLM_DIR      = joinpath(REPO_ROOT, "website", "data", "llm_features")
const OUT_DIR      = joinpath(REPO_ROOT, "website", "data", "training")
const DATASET_FILE = joinpath(OUT_DIR, "dataset.bson")

function main()
    if "--help" in ARGS || "-h" in ARGS
        println("""
Usage:
  julia --project=packages/StockSwingPredictor scripts/build_dataset.jl

Input:   website/data/inference_cache.bson
Output:  website/data/training/dataset.bson  (~30 MB)
""")
        return
    end

    # ── Load inference cache ──────────────────────────────────────────────────

    @info "Loading inference cache…"
    cache = load_inference_cache(CACHE_FILE)
    @info "  $(length(cache.dates)) trading dates"
    @info "  $(length(cache.hourly_datetimes)) hourly bars"
    @info "  $(length(cache.companies)) companies"

    # ── Generate training examples ────────────────────────────────────────────

    all_examples = TrainingExample[]
    n_comp       = length(cache.companies)

    for (j, sym) in enumerate(cache.companies)
        llm_cache = _load_llm_cache(sym, LLM_DIR)
        examples  = generate_company_examples(sym, j, cache, llm_cache)
        append!(all_examples, examples)
        j % 100 == 0 &&
            @info "[$j/$n_comp] $sym — $(length(examples)) examples (total: $(length(all_examples)))"
    end

    @info "Total examples before sort: $(length(all_examples))"
    sort!(all_examples, by = ex -> (ex.date, ex.symbol))

    # ── Build and save Dataset ────────────────────────────────────────────────

    dataset = Dataset(cache.dates, cache.companies, all_examples)
    @info dataset

    mkpath(OUT_DIR)
    save_dataset(dataset, DATASET_FILE)
    @info "Done → $DATASET_FILE"
end

# ── Helpers ───────────────────────────────────────────────────────────────────

function _load_llm_cache(symbol::String, llm_dir::String)::Dict{Date, LLMFeatures}
    cache = Dict{Date, LLMFeatures}()
    path  = joinpath(llm_dir, "$(symbol).json")
    isfile(path) || return cache
    try
        d    = JSON3.read(read(path, String))
        f    = d.features
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
