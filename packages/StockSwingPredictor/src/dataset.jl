"""
Sliding-window dataset assembly, normalisation, and train/val/test splitting.

Sampling strategy:
  - For each company, slide a window forward in weekly steps (every 5 trading days).
  - At each step T, assemble all features from data observable at T.
  - Label = log return of the stock from close(T) to close(T + 5 trading days).
  - Examples where any required data block is unavailable are dropped.

Train/val/test split is time-ordered (not random) to prevent look-ahead leakage:
  80% oldest data → train, 10% → val, 10% newest → test.
"""

using DataFrames, Statistics, Dates, JSON3, CSV

# ── Label computation ─────────────────────────────────────────────────────────

"""
Compute the 5-trading-day log return starting from bar at index `t` in `df`.
Returns `nothing` when fewer than 5 bars remain after `t`.
"""
function label_5d_return(df::DataFrame, t::Int)::Union{Float32, Nothing}
    t + 5 > nrow(df) && return nothing
    log_ret = log(df.close[t + 5] / df.close[t])
    return Float32(log_ret)
end

# ── Sliding window ────────────────────────────────────────────────────────────

"""
Generate all training examples for one company by sliding a weekly window.

# Arguments
- `symbol`: NSE tradingsymbol
- `company`: display name
- `ohlcv`: full daily OHLCV DataFrame for the stock (sorted ascending)
- `nifty_ohlcv`: daily OHLCV for NIFTY 50
- `sector_ohlcv`: daily OHLCV for the relevant sector index
- `llm_cache`: Dict mapping Date → LLMFeatures (date of the document)
- `fund_cache`: Dict mapping Date → FundamentalFeatures (quarter start date)
- `company_meta`: NamedTuple with market_cap_cr, confidence_score,
                  promoter_pledge_pct, is_fo, sector
- `sector_vocab`: ordered sector list for one-hot
- `step`: slide step in trading days (default 5 = weekly)
- `min_history_days`: minimum bars before first example (default 130 ≈ 6 months)

# Returns
Vector of `Example` structs (label=nothing for the last window where T+5 is future).
"""
function generate_examples(symbol::String, company::String,
                            ohlcv::DataFrame, nifty_ohlcv::DataFrame,
                            sector_ohlcv::DataFrame,
                            llm_cache::Dict{Date, LLMFeatures},
                            fund_cache::Dict{Date, FundamentalFeatures},
                            earnings_dates::Vector{Date},
                            company_meta::NamedTuple,
                            sector_vocab::Vector{String};
                            step::Int=5,
                            min_history_days::Int=130)::Vector{Example}

    examples = Example[]
    n = nrow(ohlcv)
    n < min_history_days + step && return examples

    for t in min_history_days:step:n
        date = ohlcv.date[t]

        # ── Time-series features ──
        ts_stock  = compute_ts_features(ohlcv,        find_date_index(ohlcv,        date))
        ts_nifty  = compute_ts_features(nifty_ohlcv,  find_date_index(nifty_ohlcv,  date))
        ts_sector = compute_ts_features(sector_ohlcv, find_date_index(sector_ohlcv, date))

        # ── Fundamentals: last available quarter before date ──
        fund = _latest_before(fund_cache, date, FundamentalFeatures(zeros(Float32, N_FUNDAMENTAL_FEATURES)))

        # ── LLM features: most recent document published before date ──
        llm = _latest_before(llm_cache, date, MISSING_LLM)

        # ── Days until next earnings ──
        days_until = _days_until_next(earnings_dates, date)

        # ── Meta ──
        meta = meta_to_vec(
            company_meta.market_cap_cr,
            company_meta.confidence_score,
            days_until,
            company_meta.promoter_pledge_pct,
            company_meta.is_fo,
            company_meta.sector,
            sector_vocab,
        )

        features = assemble_features(ts_stock, ts_nifty, ts_sector, fund, llm, meta)

        label = label_5d_return(ohlcv, t)

        push!(examples, Example(symbol, date, features, label))
    end

    return examples
end

# ── Dataset assembly ──────────────────────────────────────────────────────────

"""
Build a `Dataset` from a vector of `Example` structs.
Drops examples with `label === nothing` (future / incomplete data).
Rows are sorted by date (ascending) — essential for time-ordered splitting.
"""
function build_dataset(examples::Vector{Example},
                       feature_names::Vector{String},
                       sector_vocab::Vector{String})::Dataset

    labeled = filter(e -> !isnothing(e.label), examples)
    isempty(labeled) && error("No labeled examples — cannot build dataset")

    sort!(labeled, by = e -> (e.date, e.symbol))

    n = length(labeled)
    nf = length(feature_names)
    X = Matrix{Float32}(undef, nf, n)
    y = Vector{Float32}(undef, n)
    syms  = String[]
    dates = Date[]

    for (i, ex) in enumerate(labeled)
        X[:, i] = ex.features
        y[i]    = ex.label
        push!(syms,  ex.symbol)
        push!(dates, ex.date)
    end

    return Dataset(X, y, feature_names, syms, dates, sector_vocab)
end

# ── Train / val / test split ──────────────────────────────────────────────────

"""
Time-ordered split: oldest 80% → train, next 10% → val, newest 10% → test.
Returns `(train, val, test)` index ranges into `dataset`.
"""
function time_split(dataset::Dataset; train_frac=0.80, val_frac=0.10)
    n = size(dataset.X, 2)
    n_train = floor(Int, n * train_frac)
    n_val   = floor(Int, n * val_frac)
    n_test  = n - n_train - n_val

    train_idx = 1:n_train
    val_idx   = n_train+1 : n_train+n_val
    test_idx  = n_train+n_val+1 : n

    return train_idx, val_idx, test_idx
end

# ── Normalisation ─────────────────────────────────────────────────────────────

"""
Compute per-feature mean and std from the training split. Returns a `NormStats`.
Features with zero variance are left unnormalised (std clamped to 1).
"""
function compute_norm_stats(dataset::Dataset, train_idx)::NormStats
    X_train = dataset.X[:, train_idx]
    means = vec(mean(X_train, dims=2))
    stds  = vec(std(X_train,  dims=2))
    stds  = max.(stds, 1f-6)
    return NormStats(dataset.feature_names, Float32.(means), Float32.(stds))
end

"""Apply normalisation in-place to a feature matrix."""
function normalise!(X::Matrix{Float32}, stats::NormStats)
    X .= (X .- stats.means) ./ stats.stds
    return X
end

"""Apply normalisation to a single feature vector (copy)."""
function normalise(x::Vector{Float32}, stats::NormStats)::Vector{Float32}
    return (x .- stats.means) ./ stats.stds
end

# ── Serialisation ─────────────────────────────────────────────────────────────

"""Save `NormStats` to a JSON file."""
function save_norm_stats(stats::NormStats, path::String)
    open(path, "w") do io
        JSON3.pretty(io, Dict(
            "feature_names" => stats.feature_names,
            "means"         => stats.means,
            "stds"          => stats.stds,
        ))
    end
end

"""Load `NormStats` from a JSON file."""
function load_norm_stats(path::String)::NormStats
    d = JSON3.read(read(path, String))
    NormStats(
        collect(String, d.feature_names),
        collect(Float32, d.means),
        collect(Float32, d.stds),
    )
end

"""Save a `Dataset` to a CSV (one row per example, features as columns)."""
function save_dataset(dataset::Dataset, path::String)
    df = DataFrame(dataset.X', dataset.feature_names)
    df.label  = dataset.y
    df.symbol = dataset.symbols
    df.date   = dataset.dates
    CSV.write(path, df)
    @info "Dataset written: $(size(dataset.X, 2)) examples × $(size(dataset.X, 1)) features → $path"
end

"""Load a `Dataset` from a CSV saved by `save_dataset`."""
function load_dataset(path::String)::Dataset
    df = CSV.read(path, DataFrame)
    meta_cols = ["label", "symbol", "date"]
    feat_cols = setdiff(names(df), meta_cols)
    X = Matrix{Float32}(df[:, feat_cols])'
    y = Vector{Float32}(df.label)
    syms  = Vector{String}(df.symbol)
    dates = Vector{Date}(df.date)
    sector_vocab = String[]   # not stored in CSV — reload separately if needed
    return Dataset(X, y, feat_cols, syms, dates, sector_vocab)
end

# ── Helpers ───────────────────────────────────────────────────────────────────

"""Return the value from `cache` with the largest key ≤ `date`, or `default`."""
function _latest_before(cache::Dict{K,V}, date::Date, default::V)::V where {K,V}
    best_key = nothing
    for k in keys(cache)
        (k <= date) && (isnothing(best_key) || k > best_key) && (best_key = k)
    end
    isnothing(best_key) ? default : cache[best_key]
end

"""Days until the next earnings date after `date`. Returns 90 if none known."""
function _days_until_next(earnings_dates::Vector{Date}, date::Date)::Int
    future = filter(d -> d > date, earnings_dates)
    isempty(future) && return 90
    return Dates.value(minimum(future) - date)
end
