"""
StockSwingPredictor

Neural-network-based large-move predictor for NSE-listed equities.

Architecture: regularised 6-layer MLP trained on weekly sliding-window snapshots
covering all NSE equities over a rolling 3–5 year history.

Input features per example:
  • Time-series derived (stock + Nifty 50 + sector index): 39 values
  • Quarterly fundamentals (last 4 quarters, 7 metrics):   28 values
  • LLM-extracted scalars (conference call / earnings PDF): 15 values
  • Company metadata (market cap, confidence, sector …):   ~40 values
  Total: ~120 features

Label: N_PRED_HOURS (35) hourly log-return values over the next 5 trading days,
each relative to the reference daily close. The final value (trajectory[end])
is the end-of-day-5 close — the primary actionable signal.

See also: [TijoriData](@ref), [CompanyConfidence](@ref), [EarningsCalendar](@ref)

## Pipeline

```
scripts/collect_ohlcv.jl          # download daily + hourly OHLCV for all companies
scripts/extract_llm_features.jl   # LLM extraction from conference calls (resumable)
scripts/build_dataset.jl          # assemble training examples, normalise, split
scripts/train_model.jl            # train MLP, save to website/data/models/
scripts/score_watchlist.jl        # run inference on current earnings watchlist
```
"""
module StockSwingPredictor

using TijoriData
using Flux, BSON
using DataFrames, CSV
using HTTP, JSON3
using Statistics, LinearAlgebra
using Dates, Printf

include("types.jl")
include("kite_data.jl")
include("fundamentals.jl")
include("llm_extract.jl")
include("features.jl")
include("dataset.jl")
include("model.jl")
include("train.jl")
include("display.jl")

export
    # types
    OHLCVBar, TSFeatures, FundamentalFeatures, LLMFeatures, MetaFeatures,
    Example, Dataset, NormStats, SwingSignal,
    MISSING_LLM,
    N_TS_FEATURES, N_FUNDAMENTAL_FEATURES, N_LLM_FEATURES,
    FUNDAMENTAL_METRICS, N_QUARTERS,
    N_HOURS_PER_DAY, N_PRED_DAYS, N_PRED_HOURS,

    # kite_data
    load_kite_session, load_instruments, build_token_map,
    fetch_ohlcv, collect_ohlcv, load_cached_ohlcv,
    fetch_ohlcv_hourly, collect_ohlcv_hourly, load_cached_ohlcv_hourly,
    sector_index_name, NSE_INDICES,

    # fundamentals
    extract_fundamentals, fundamental_feature_names,

    # llm_extract
    extract_features, extract_features_from_kb,

    # features
    compute_ts_features, find_date_index,
    ts_to_vec, ts_feature_names,
    llm_to_vec, llm_feature_names,
    meta_to_vec, meta_feature_names,
    assemble_features, all_feature_names,

    # dataset
    label_5d_hourly, generate_examples, build_dataset,
    time_split, compute_norm_stats, normalise!, normalise,
    save_norm_stats, load_norm_stats, save_dataset, load_dataset,

    # model
    build_model, predict, save_model, load_model,

    # train
    train!, evaluate, save_training_log

end
