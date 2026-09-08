"""
build_dataset.jl

Assemble training examples from cached OHLCV, LLM features, fundamentals,
and company metadata. Applies normalisation and saves the ready-to-train
dataset along with NormStats for use at inference time.

Sliding window: every 5 trading days (weekly) per company.
Label: 5-trading-day log return.
Split: 80% train / 10% val / 10% test (time-ordered, no shuffle).

Output:
  website/data/training/dataset_train.csv
  website/data/training/dataset_val.csv
  website/data/training/dataset_test.csv
  website/data/training/norm_stats.json
  website/data/training/sector_vocab.json

Usage:
  julia --project=packages/StockSwingPredictor scripts/build_dataset.jl
  julia --project=packages/StockSwingPredictor scripts/build_dataset.jl 100  # top-N companies
"""

using StockSwingPredictor, TijoriData, JSON3, DataFrames, CSV, Dates, Printf, Statistics

const REPO_ROOT        = joinpath(@__DIR__, "..")
const COMPANIES_FILE   = joinpath(REPO_ROOT, "website", "data", "nse_companies_latest.json")
const PROJECTIONS_FILE = joinpath(REPO_ROOT, "website", "data", "earnings_projections.json")
const OHLCV_DIR        = joinpath(REPO_ROOT, "website", "data", "ohlcv")
const LLM_DIR          = joinpath(REPO_ROOT, "website", "data", "llm_features")
const OUT_DIR          = joinpath(REPO_ROOT, "website", "data", "training")
const SIDECAR_PORT     = 3001

function main()
    if "--help" in ARGS || "-h" in ARGS
        println("""
Usage:
  julia --project=packages/StockSwingPredictor scripts/build_dataset.jl [N]

Arguments:
  N   Top N companies by market cap to include (default: all with OHLCV data).

Output:
  website/data/training/dataset_{train,val,test}.csv
  website/data/training/norm_stats.json
  website/data/training/sector_vocab.json
""")
        return
    end

    top_n = length(ARGS) >= 1 && ARGS[1] != "--help" ? parse(Int, ARGS[1]) : typemax(Int)

    # ── Connect sidecar for fundamentals ──────────────────────────────────────

    TijoriData.configure!(port=SIDECAR_PORT)
    if !TijoriData.is_running()
        @warn "Sidecar not running — fundamentals will be zeros. " *
              "Start with: node sidecar/server_http.js"
    end

    # ── Load companies ────────────────────────────────────────────────────────

    isfile(COMPANIES_FILE) || error("Not found: $COMPANIES_FILE")
    raw   = JSON3.read(read(COMPANIES_FILE, String))
    all_c = collect(raw.companies)

    eligible = filter(all_c) do c
        !isnothing(get(c, :confidence, nothing)) &&
        !isempty(string(get(c, :symbol, "")))
    end
    sort!(eligible, by = c -> Float64(get(c, :market_cap_cr, 0.0)), rev=true)
    todo = first(eligible, top_n)
    @info "$(length(todo)) companies to process"

    # ── Load Nifty 50 OHLCV ───────────────────────────────────────────────────

    nifty_path = joinpath(OHLCV_DIR, "IDX_NIFTY_50_daily.csv")
    isfile(nifty_path) || error("Nifty OHLCV not found: $nifty_path\nRun: julia scripts/collect_ohlcv.jl")
    nifty_ohlcv = CSV.read(nifty_path, DataFrame; types=Dict(:date => Date))
    sort!(nifty_ohlcv, :date)
    @info "Nifty 50: $(nrow(nifty_ohlcv)) days"

    # ── Load earnings projections (for days_until_earnings feature) ───────────

    earnings_by_symbol = Dict{String, Vector{Date}}()
    if isfile(PROJECTIONS_FILE)
        proj_raw = JSON3.read(read(PROJECTIONS_FILE, String))
        for (sym, val) in pairs(proj_raw.projections)
            d_str = string(get(val, :projected_date, get(val, "projected_date", nothing)))
            d = tryparse(Date, d_str)
            isnothing(d) || push!(get!(earnings_by_symbol, string(sym), Date[]), d)
        end
    end

    # ── Build sector vocabulary from all eligible companies ───────────────────

    all_sectors = unique([string(get(c, :sector, "Unknown")) for c in eligible])
    sort!(all_sectors)
    sector_vocab = all_sectors
    n_meta = 5 + length(sector_vocab)
    @info "Sector vocabulary: $(length(sector_vocab)) sectors"

    # ── Assemble examples ─────────────────────────────────────────────────────

    all_examples = Example[]

    for (i, c) in enumerate(todo)
        sym  = string(get(c, :symbol, ""))
        slug = string(get(c, :slug,   ""))
        name = string(get(c, :name,   sym))
        sector = string(get(c, :sector, "Unknown"))

        ohlcv = load_cached_ohlcv(sym, OHLCV_DIR)
        if isempty(ohlcv)
            i % 20 == 0 && @info "[$i/$(length(todo))] $sym — no OHLCV, skipping"
            continue
        end
        sort!(ohlcv, :date)

        # Sector index OHLCV
        idx_name = sector_index_name(sector)
        idx_file = joinpath(OHLCV_DIR, "IDX_$(replace(idx_name, " " => "_"))_daily.csv")
        sector_ohlcv = isfile(idx_file) ?
                       sort!(CSV.read(idx_file, DataFrame; types=Dict(:date => Date)), :date) :
                       nifty_ohlcv   # fallback to Nifty 50

        # LLM features cache: Date → LLMFeatures
        llm_cache = _load_llm_cache(sym, LLM_DIR)

        # Fundamentals cache: Date → FundamentalFeatures
        fund_cache = _load_fund_cache(slug, TijoriData.is_running())

        # Earnings dates
        earnings_dates = get(earnings_by_symbol, sym, Date[])

        # Company metadata
        conf  = get(c, :confidence, nothing)
        score = isnothing(conf) ? 0.0 : Float64(get(conf, :score, 0.0))
        pledge = isnothing(conf) ? 0.0 : begin
            pl = get(conf, :pledging, nothing)
            isnothing(pl) ? 0.0 : Float64(get(pl, :latest_pct, 0.0))
        end

        meta_nt = (
            market_cap_cr      = Float64(get(c, :market_cap_cr, 0.0)),
            confidence_score   = score,
            promoter_pledge_pct = pledge,
            is_fo              = false,   # F&O lookup not yet implemented
            sector             = sector,
        )

        examples = generate_examples(sym, name, ohlcv, nifty_ohlcv, sector_ohlcv,
                                     llm_cache, fund_cache, earnings_dates,
                                     meta_nt, sector_vocab)
        append!(all_examples, examples)

        if i % 10 == 0
            @info "[$i/$(length(todo))] $sym — $(length(examples)) examples  (total: $(length(all_examples)))"
        end
    end

    @info "Total examples before label filtering: $(length(all_examples))"

    # ── Build dataset ─────────────────────────────────────────────────────────

    feat_names = all_feature_names(sector_vocab)
    dataset = build_dataset(all_examples, feat_names, sector_vocab)
    @info dataset

    # ── Split ─────────────────────────────────────────────────────────────────

    train_idx, val_idx, test_idx = time_split(dataset)
    @info "Split: $(length(train_idx)) train / $(length(val_idx)) val / $(length(test_idx)) test"

    # ── Normalise ─────────────────────────────────────────────────────────────

    norm_stats = compute_norm_stats(dataset, train_idx)
    normalise!(dataset.X, norm_stats)

    # ── Save ──────────────────────────────────────────────────────────────────

    mkpath(OUT_DIR)

    function split_dataset(ds::Dataset, idx)
        Dataset(ds.X[:, idx], ds.y[idx], ds.feature_names,
                ds.symbols[idx], ds.dates[idx], ds.sector_vocab)
    end

    save_dataset(split_dataset(dataset, train_idx), joinpath(OUT_DIR, "dataset_train.csv"))
    save_dataset(split_dataset(dataset, val_idx),   joinpath(OUT_DIR, "dataset_val.csv"))
    save_dataset(split_dataset(dataset, test_idx),  joinpath(OUT_DIR, "dataset_test.csv"))

    save_norm_stats(norm_stats, joinpath(OUT_DIR, "norm_stats.json"))

    open(joinpath(OUT_DIR, "sector_vocab.json"), "w") do io
        JSON3.pretty(io, sector_vocab)
    end

    println()
    @info "Dataset build complete → $OUT_DIR"
    @info "  Features: $(length(feat_names))"
    @info "  Examples: $(size(dataset.X, 2))"
    @info "  Label range: $(round(minimum(dataset.y)*100, digits=2))% → $(round(maximum(dataset.y)*100, digits=2))%"
end

# ── Helpers ───────────────────────────────────────────────────────────────────

function _load_llm_cache(symbol::String, llm_dir::String)::Dict{Date, LLMFeatures}
    cache = Dict{Date, LLMFeatures}()
    path  = joinpath(llm_dir, "$(symbol).json")
    isfile(path) || return cache

    try
        d = JSON3.read(read(path, String))
        f = d.features
        extracted_at = tryparse(Date, string(get(d, :extracted_at, ""))[1:10])
        doc_date     = isnothing(extracted_at) ? today() : extracted_at

        llm = LLMFeatures(
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
        cache[doc_date] = llm
    catch e
        @warn "Could not load LLM features for $symbol: $(sprint(showerror, e))[1:60]"
    end

    return cache
end

function _load_fund_cache(slug::String, sidecar_running::Bool)::Dict{Date, FundamentalFeatures}
    cache = Dict{Date, FundamentalFeatures}()
    sidecar_running || return cache
    isempty(slug)   && return cache

    try
        # Use today as the as_of date; the function internally limits to available data.
        fund = extract_fundamentals(slug, today())
        # Key by today so it's available for all historical windows.
        # In a more rigorous reconstruction, you'd fetch per-quarter.
        cache[Date(2000, 1, 1)] = fund   # early date so it applies to all windows
    catch e
        @warn "Could not fetch fundamentals for $slug: $(sprint(showerror, e))[1:60]"
    end

    return cache
end

main()
