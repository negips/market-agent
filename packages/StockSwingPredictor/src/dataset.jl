"""
Dataset assembly, batch construction, train/val/test splitting, and persistence.

## Market matrix layout

For a training example predicting stock X on date T (date_idx = t):

  closes[t-N_MARKET_DAYS+1 : t, :]   — (28, N) raw closing prices
  Normalised per company: divide each column by its value at row t-N_MARKET_DAYS+1
  so every series starts at 1.0 and subsequent values are relative returns.

  vols[t-N_MARKET_DAYS+1 : t, :]     — (28, N) daily (H-L)/C, no normalisation needed.

  Column ordering in the assembled batch tensor: stock X is moved to column 1
  so `model.market_cnn` output[:, 1, :] is always the target stock's embedding.

## Hourly series

Each TrainingExample pre-stores the target stock's N_HOURLY_BARS-length normalised
closing series (close / close[1] of the 8-week window). This is small per example
(~1 KB) and avoids re-slicing the hourly CSVs during training.

## Label

5-trading-day hourly log-return trajectory after date T:
  label[h] = log(hourly_close[h] / daily_close[T])  for h in 1 … N_PRED_HOURS
"""

using DataFrames, CSV, Dates, Statistics, JSON3, Printf, BSON

# ── Label computation (reused from hourly OHLCV) ─────────────────────────────

"""
Compute the N_PRED_HOURS-length hourly log-return trajectory following daily bar
at index `t` in `daily_df`, using `hourly_df` for intraday prices.

Each value is `log(hourly_close / daily_df.close[t])`.
Returns `nothing` when N_PRED_DAYS trading days don't exist after `t`,
or when no hourly bars fall on those days.
"""
function label_5d_hourly(hourly_df::DataFrame, daily_df::DataFrame,
                          t::Int)::Union{Vector{Float32}, Nothing}
    t + N_PRED_DAYS > nrow(daily_df) && return nothing

    ref_close    = daily_df.close[t]
    target_dates = Set(daily_df.date[t+1 : t+N_PRED_DAYS])

    mask   = [Date(row.datetime) in target_dates for row in eachrow(hourly_df)]
    window = sort(hourly_df[mask, :], :datetime)
    nrow(window) == 0 && return nothing

    closes  = window.close
    n_bars  = length(closes)
    traj    = Vector{Float32}(undef, N_PRED_HOURS)
    for i in 1:min(n_bars, N_PRED_HOURS)
        traj[i] = Float32(log(closes[i] / ref_close))
    end
    if n_bars < N_PRED_HOURS
        last_val = Float32(log(closes[end] / ref_close))
        for i in n_bars+1:N_PRED_HOURS
            traj[i] = last_val
        end
    end
    return traj
end

# ── Master market-matrix construction ────────────────────────────────────────

"""
Build the (n_dates × n_companies) closing price and daily-vol matrices from
cached daily OHLCV CSVs. Missing dates for a company are forward-filled with
the last known price (correct: the previous close is the last observable price).

Returns `(closes, vols, master_dates)`.
"""
function build_market_matrices(ohlcv_dir::String,
                                companies::Vector{String})::Tuple{Matrix{Float32},
                                                                   Matrix{Float32},
                                                                   Vector{Date}}
    # ── Collect the union of all trading dates ────────────────────────────────
    all_dates = Set{Date}()
    for sym in companies
        path = joinpath(ohlcv_dir, "$(sym)_daily.csv")
        isfile(path) || continue
        df = CSV.read(path, DataFrame; types=Dict(:date => Date), select=[:date])
        union!(all_dates, df.date)
    end
    master_dates = sort(collect(all_dates))
    n_dates = length(master_dates)
    n_comp  = length(companies)
    date_index = Dict(d => i for (i, d) in enumerate(master_dates))

    closes = fill(NaN32, n_dates, n_comp)
    vols   = fill(NaN32, n_dates, n_comp)

    for (j, sym) in enumerate(companies)
        path = joinpath(ohlcv_dir, "$(sym)_daily.csv")
        isfile(path) || continue
        df = CSV.read(path, DataFrame; types=Dict(:date => Date))

        for row in eachrow(df)
            i = get(date_index, row.date, 0)
            i == 0 && continue
            closes[i, j] = Float32(row.close)
            if row.high > row.low && row.close > 0
                vols[i, j] = Float32((row.high - row.low) / row.close)
            else
                vols[i, j] = 0f0
            end
        end

        # Forward-fill missing values along the time dimension
        last_c = NaN32; last_v = 0f0
        for i in 1:n_dates
            if !isnan(closes[i, j])
                last_c = closes[i, j]
                last_v = vols[i, j]
            elseif !isnan(last_c)
                closes[i, j] = last_c
                vols[i, j]   = last_v
            end
        end
    end

    return closes, vols, master_dates
end

# ── Per-company example generation ───────────────────────────────────────────

"""
Generate all `TrainingExample`s for one company by sliding a weekly window
over the master date calendar.

Skips windows where:
  - The company has no daily data on the training date.
  - Fewer than N_HOURLY_BARS hourly bars exist up to the training date.
  - The 5-day label window extends beyond available hourly data.
"""
function generate_company_examples(
    symbol::String,
    sym_idx::Int,
    daily_df::DataFrame,
    hourly_df::DataFrame,
    master_dates::Vector{Date},
    llm_cache::Dict{Date, LLMFeatures};
    step::Int = 5,
)::Vector{TrainingExample}

    examples = TrainingExample[]
    n_master = length(master_dates)

    for (t, date) in enumerate(master_dates)
        # Weekly stride over the master calendar
        t % step != 0  && continue
        # Need N_MARKET_DAYS of market history (handled at batch time via closes matrix,
        # but we guard here to avoid incomplete windows at dataset edges)
        t < N_MARKET_DAYS + 1   && continue
        t + N_PRED_DAYS > n_master && continue

        # Company must have data on this exact trading date
        daily_idx = searchsortedlast(daily_df.date, date)
        (daily_idx == 0 || daily_df.date[daily_idx] != date) && continue

        # ── Hourly series: last N_HOURLY_BARS bars ending at `date` ──────────
        h_end  = DateTime(date, Time(23, 59, 59))
        h_mask = hourly_df.datetime .<= h_end
        h_sub  = hourly_df[h_mask, :]
        nrow(h_sub) < N_HOURLY_BARS && continue

        h_slice = h_sub[end-N_HOURLY_BARS+1:end, :]
        ref     = Float32(h_slice.close[1])
        ref <= 0 && continue
        hourly_norm = Float32.(h_slice.close) ./ ref

        # ── LLM scalars ───────────────────────────────────────────────────────
        llm_feat = latest_before(llm_cache, date, MISSING_LLM)
        llm_vec  = llm_to_vec(llm_feat)

        # ── Label: 5-day hourly trajectory ───────────────────────────────────
        label = label_5d_hourly(hourly_df, daily_df, daily_idx)
        isnothing(label) && continue

        push!(examples, TrainingExample(date, symbol, t, sym_idx,
                                        hourly_norm, llm_vec, label))
    end

    return examples
end

# ── Train / val / test split ──────────────────────────────────────────────────

"""
Time-ordered split of dataset examples. Returns (train_idx, val_idx, test_idx).
Examples must already be sorted ascending by date (guaranteed by `build_dataset.jl`).
"""
function time_split(dataset::Dataset; train_frac=0.80, val_frac=0.10)
    n       = length(dataset.examples)
    n_train = floor(Int, n * train_frac)
    n_val   = floor(Int, n * val_frac)

    train_idx = 1:n_train
    val_idx   = n_train+1 : n_train+n_val
    test_idx  = n_train+n_val+1 : n

    return train_idx, val_idx, test_idx
end

# ── Batch assembly (called inside training loop) ──────────────────────────────

"""
Assemble a mini-batch from `dataset` given a vector of example indices.

Returns `(market, hourly, llm, y)` where:
  market — `(N_MARKET_DAYS, N_MARKET_CHANNELS, N_companies, B)` Float32
  hourly — `(N_HOURLY_BARS, B)` Float32
  llm    — `(N_LLM_FEATURES, B)` Float32
  y      — `(N_PRED_HOURS, B)` Float32

The target stock is placed at column 1 of dim 3 in `market`.
Closing prices are normalised so the first day of each company's 28-day window = 1.0.
"""
function assemble_batch(dataset::Dataset, indices::AbstractVector{Int})
    B = length(indices)
    N = length(dataset.companies)

    market = Array{Float32}(undef, N_MARKET_DAYS, N_MARKET_CHANNELS, N, B)
    hourly = Matrix{Float32}(undef, N_HOURLY_BARS,   B)
    llm    = Matrix{Float32}(undef, N_LLM_FEATURES,  B)
    y      = Matrix{Float32}(undef, N_PRED_HOURS,    B)

    for (b, idx) in enumerate(indices)
        ex = dataset.examples[idx]
        t  = ex.date_idx   # row in closes/vols matrices
        k  = ex.sym_idx    # column of the target company

        # ── Market closes: 28-day window, normalised to first day = 1.0 ──────
        raw_c  = dataset.closes[t-N_MARKET_DAYS+1:t, :]   # (28, N)
        anchor = raw_c[1:1, :]                              # (1, N)
        norm_c = raw_c ./ max.(anchor, 1f-6)               # (28, N)
        norm_c[isnan.(norm_c)] .= 1f0

        raw_v  = dataset.vols[t-N_MARKET_DAYS+1:t, :]     # (28, N)
        raw_v[isnan.(raw_v)] .= 0f0

        # ── Rearrange columns: target company at position 1 ──────────────────
        others = filter(!=(k), 1:N)
        order  = [k; others]

        market[:, 1, :, b] = norm_c[:, order]
        market[:, 2, :, b] = raw_v[:, order]

        hourly[:, b] = ex.hourly
        llm[:, b]    = ex.llm
        y[:, b]      = ex.label
    end

    return market, hourly, llm, y
end

# ── Persistence ───────────────────────────────────────────────────────────────

"""
Save a `Dataset` to `path` (BSON). Serialises all arrays as primitives to
avoid struct versioning issues on load.
"""
function save_dataset(dataset::Dataset, path::String)
    closes    = dataset.closes
    vols      = dataset.vols
    dates     = dataset.dates
    companies = dataset.companies

    n = length(dataset.examples)
    ex_dates    = [ex.date     for ex in dataset.examples]
    ex_symbols  = [ex.symbol   for ex in dataset.examples]
    ex_date_idx = Int32[ex.date_idx for ex in dataset.examples]
    ex_sym_idx  = Int32[ex.sym_idx  for ex in dataset.examples]
    ex_hourly   = reduce(hcat, [ex.hourly for ex in dataset.examples])  # (N_HOURLY_BARS, n)
    ex_llm      = reduce(hcat, [ex.llm    for ex in dataset.examples])  # (N_LLM_FEATURES, n)
    ex_labels   = reduce(hcat, [ex.label  for ex in dataset.examples])  # (N_PRED_HOURS, n)

    BSON.@save path closes vols dates companies ex_dates ex_symbols ex_date_idx ex_sym_idx ex_hourly ex_llm ex_labels
    @info "Dataset saved → $path  ($(n) examples, $(length(companies)) companies, $(length(dates)) dates)"
end

"""Load a `Dataset` previously saved by `save_dataset`."""
function load_dataset(path::String)::Dataset
    BSON.@load path closes vols dates companies ex_dates ex_symbols ex_date_idx ex_sym_idx ex_hourly ex_llm ex_labels

    n = length(ex_dates)
    examples = Vector{TrainingExample}(undef, n)
    for i in 1:n
        examples[i] = TrainingExample(
            ex_dates[i], ex_symbols[i],
            Int(ex_date_idx[i]), Int(ex_sym_idx[i]),
            ex_hourly[:, i], ex_llm[:, i], ex_labels[:, i],
        )
    end

    return Dataset(closes, vols, dates, companies, examples)
end
