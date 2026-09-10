"""
Dataset assembly, batch construction, train/val/test splitting, and persistence.

All price data comes from `InferenceCache` — no CSV reads happen here.
`TrainingExample` stores only index pointers and small feature vectors;
the market and hourly price windows are sliced from the cache at batch time.

## Market matrix layout

For a training example predicting stock X on date T (date_idx = t):

  cache.closes[t-N_MARKET_DAYS+1 : t, :]   — (28, N) raw closing prices
  Normalised per company: divide each column by its value at row t-N_MARKET_DAYS+1
  so every series starts at 1.0 and subsequent values are relative returns.

  cache.vols[t-N_MARKET_DAYS+1 : t, :]     — (28, N) daily (H-L)/C, no normalisation needed.

  Column ordering in the assembled batch tensor: stock X is moved to column 1
  so `model.market_cnn` output[:, 1, :] is always the target stock's embedding.

## Hourly series

Sliced from `cache.hourly_closes[hourly_end_idx-N_HOURLY_BARS+1 : hourly_end_idx, sym_idx]`
at batch time and normalised to start at 1.0. Nothing is pre-stored per example.

## Label

5-trading-day hourly log-return trajectory after date T:
  label[h] = log(hourly_close[h] / daily_close[T])  for h in 1 … N_PRED_HOURS
"""

using DataFrames, Dates, Statistics, JSON3, Printf, BSON

# ── Label computation ─────────────────────────────────────────────────────────

"""
Compute the N_PRED_HOURS-length hourly log-return trajectory for company
`sym_idx` following master-calendar date at index `t`.

Returns `nothing` when N_PRED_DAYS trading days don't exist after `t`,
or when no hourly bars fall on those days for this company.
"""
function label_5d_hourly(cache::InferenceCache, sym_idx::Int,
                          t::Int)::Union{Vector{Float32}, Nothing}
    t + N_PRED_DAYS > length(cache.dates) && return nothing

    ref_close = cache.closes[t, sym_idx]
    (isnan(ref_close) || ref_close <= 0f0) && return nothing

    label_start = cache.dates[t + 1]
    label_end   = cache.dates[t + N_PRED_DAYS]

    h_lo = searchsortedfirst(cache.hourly_datetimes, DateTime(label_start))
    h_hi = searchsortedlast( cache.hourly_datetimes, DateTime(label_end, Time(23, 59, 59)))
    h_lo > h_hi && return nothing

    target_dates = Set(cache.dates[t+1 : t+N_PRED_DAYS])
    h_closes = Float32[]
    for h in h_lo:h_hi
        Date(cache.hourly_datetimes[h]) in target_dates || continue
        v = cache.hourly_closes[h, sym_idx]
        isnan(v) && continue
        push!(h_closes, v)
    end

    isempty(h_closes) && return nothing

    n_bars = length(h_closes)
    traj   = Vector{Float32}(undef, N_PRED_HOURS)
    for i in 1:min(n_bars, N_PRED_HOURS)
        traj[i] = Float32(log(max(h_closes[i], 1f-6) / ref_close))
    end
    n_bars < N_PRED_HOURS && (traj[n_bars+1:end] .= traj[n_bars])
    return traj
end

# ── Per-company example generation ───────────────────────────────────────────

"""
Generate all `TrainingExample`s for one company by sliding a weekly window
over the master date calendar in `cache`.

Skips windows where:
  - The company has no daily close on the training date (NaN after forward-fill).
  - Fewer than N_HOURLY_BARS hourly bars exist up to the training date, or
    the first bar of the window is NaN (company not yet listed that far back).
  - The 5-day label window extends beyond available data.
"""
function generate_company_examples(
    symbol::String,
    sym_idx::Int,
    cache::InferenceCache,
    llm_cache::Dict{Date, LLMFeatures};
    step::Int = 5,
)::Vector{TrainingExample}

    examples  = TrainingExample[]
    n_dates   = length(cache.dates)

    for t in 1:n_dates
        t % step != 0     && continue
        t < N_MARKET_DAYS + 1 && continue
        t + N_PRED_DAYS > n_dates && continue

        isnan(cache.closes[t, sym_idx]) && continue

        date = cache.dates[t]

        # Hourly guard: need N_HOURLY_BARS bars ending on this date.
        h_end = searchsortedlast(cache.hourly_datetimes,
                                  DateTime(date, Time(23, 59, 59)))
        h_end < N_HOURLY_BARS && continue

        # First bar of the window must be valid (company was listed that far back).
        isnan(cache.hourly_closes[h_end - N_HOURLY_BARS + 1, sym_idx]) && continue

        llm_feat = latest_before(llm_cache, date, MISSING_LLM)
        llm_vec  = llm_to_vec(llm_feat)

        label = label_5d_hourly(cache, sym_idx, t)
        isnothing(label) && continue

        push!(examples, TrainingExample(date, symbol, t, sym_idx, h_end, llm_vec, label))
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
    return 1:n_train, n_train+1:n_train+n_val, n_train+n_val+1:n
end

# ── Batch assembly (called inside training loop) ──────────────────────────────

"""
Assemble a mini-batch from `dataset` and `cache` given a vector of example indices.

Returns `(market, hourly, llm, y)` where:
  market — `(N_MARKET_DAYS, N_MARKET_CHANNELS, N_companies, B)` Float32
  hourly — `(N_HOURLY_BARS, B)` Float32
  llm    — `(N_LLM_FEATURES, B)` Float32
  y      — `(N_PRED_HOURS, B)` Float32

The target stock is placed at column 1 of dim 3 in `market`.
Both market closes and the hourly series are normalised so their first
sample = 1.0. All slicing is O(1) into the pre-loaded cache matrices.
"""
function assemble_batch(dataset::Dataset, cache::InferenceCache,
                         indices::AbstractVector{Int})
    B = length(indices)

    market = Array{Float32}(undef, N_MARKET_DAYS, N_MARKET_CHANNELS, N_MARKET_COMPANIES, B)
    hourly = Matrix{Float32}(undef, N_HOURLY_BARS,  B)
    llm    = Matrix{Float32}(undef, N_LLM_FEATURES, B)
    y      = Matrix{Float32}(undef, N_PRED_HOURS,   B)

    for (b, idx) in enumerate(indices)
        ex = dataset.examples[idx]
        t  = ex.date_idx
        k  = ex.sym_idx

        # ── Market columns: target at col 1, then top N_MARKET_COMPANIES-1 ───
        # Companies are sorted by market cap so top-N is always 1:N_MARKET_COMPANIES.
        # If the target falls outside the top-N, substitute it for the Nth slot.
        market_cols = if k <= N_MARKET_COMPANIES
            [k; filter(!=(k), 1:N_MARKET_COMPANIES)]
        else
            [k; collect(1:N_MARKET_COMPANIES-1)]
        end

        # ── Market context: 28-day window, normalised to first day = 1.0 ──────
        raw_c  = cache.closes[t-N_MARKET_DAYS+1:t, market_cols]
        anchor = raw_c[1:1, :]
        norm_c = raw_c ./ max.(anchor, 1f-6)
        norm_c[isnan.(norm_c)] .= 1f0

        raw_v  = cache.vols[t-N_MARKET_DAYS+1:t, market_cols]
        raw_v[isnan.(raw_v)] .= 0f0

        market[:, 1, :, b] = norm_c
        market[:, 2, :, b] = raw_v

        # ── Hourly series: O(1) slice from cache, normalised to start = 1.0 ──
        h_end   = ex.hourly_end_idx
        h_start = h_end - N_HOURLY_BARS + 1
        raw_h   = cache.hourly_closes[h_start:h_end, k]
        ref_h   = max(raw_h[1], 1f-6)
        hourly[:, b] = raw_h ./ ref_h

        llm[:, b] = ex.llm
        y[:, b]   = ex.label
    end

    return market, hourly, llm, y
end

# ── Persistence ───────────────────────────────────────────────────────────────

"""
Save a `Dataset` to `path` (BSON).

Only index pointers and small feature vectors are stored — no price matrices.
Typical file size: ~30 MB for 150k examples.
"""
function save_dataset(dataset::Dataset, path::String)
    dates     = dataset.dates
    companies = dataset.companies
    n         = length(dataset.examples)

    ex_dates        = [ex.date           for ex in dataset.examples]
    ex_symbols      = [ex.symbol         for ex in dataset.examples]
    ex_date_idx     = Int32[ex.date_idx       for ex in dataset.examples]
    ex_sym_idx      = Int32[ex.sym_idx        for ex in dataset.examples]
    ex_hourly_end   = Int32[ex.hourly_end_idx for ex in dataset.examples]
    ex_llm          = reduce(hcat, [ex.llm   for ex in dataset.examples])
    ex_labels       = reduce(hcat, [ex.label for ex in dataset.examples])

    BSON.@save path dates companies ex_dates ex_symbols ex_date_idx ex_sym_idx ex_hourly_end ex_llm ex_labels
    @info "Dataset saved → $path  ($n examples, $(length(companies)) companies, $(length(dates)) dates)"
end

"""Load a `Dataset` previously saved by `save_dataset`."""
function load_dataset(path::String)::Dataset
    BSON.@load path dates companies ex_dates ex_symbols ex_date_idx ex_sym_idx ex_hourly_end ex_llm ex_labels
    n = length(ex_dates)
    examples = Vector{TrainingExample}(undef, n)
    for i in 1:n
        examples[i] = TrainingExample(
            ex_dates[i], ex_symbols[i],
            Int(ex_date_idx[i]), Int(ex_sym_idx[i]),
            Int(ex_hourly_end[i]),
            ex_llm[:, i], ex_labels[:, i],
        )
    end
    return Dataset(dates, companies, examples)
end
