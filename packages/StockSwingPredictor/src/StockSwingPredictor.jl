"""
StockSwingPredictor

Neural-network-based large-move predictor for NSE-listed equities.

## Architecture

Two CNN branches feed a shared MLP head:

- **Market branch** (shared weights across the full ~500-company universe):
  Each company's 28-day normalised closing-price + daily-vol series is processed
  by the same 1D-CNN → 128-dim embedding. The target stock's embedding and the
  mean of the remaining 499 embeddings (market context) are kept separate.

- **Hourly branch** (target stock only):
  8 weeks of 60-minute normalised closes → deeper 1D-CNN → 256-dim embedding.

- **MLP head**: concat(128 + 128 + 256 + 15 LLM scalars) → 527 → … → 35 outputs.

## Label

35-step hourly log-return trajectory over the next 5 trading days, each value
relative to the reference daily close. The final bar (eod_return) is the headline
actionable signal.

## Pipeline

```
scripts/collect_ohlcv.jl        # download daily + hourly OHLCV for all companies
scripts/update_ohlcv.jl         # daily incremental update (append new bars)
scripts/build_cache.jl          # build inference_cache.bson from CSVs (run each morning)
scripts/extract_llm_features.jl # LLM extraction from conference calls (resumable, optional)
scripts/build_dataset.jl        # assemble Dataset — index pointers + labels only (~30 MB)
scripts/train_model.jl          # train SwingPredictor, save BSON
```

See also: [TijoriData](@ref), [CompanyConfidence](@ref), [EarningsCalendar](@ref)
"""
module StockSwingPredictor

using TijoriData
using Flux, BSON
using DataFrames, CSV
using HTTP, JSON3
using Statistics, LinearAlgebra
using Dates, Printf, Random

include("types.jl")
include("kite_data.jl")
include("inference_cache.jl")
include("llm_extract.jl")
include("features.jl")
include("fundamentals.jl")   # not active in current pipeline — see file header
include("dataset.jl")
include("model.jl")
include("train.jl")
include("broker.jl")
include("display.jl")

export
    # types / constants
    LLMFeatures, MISSING_LLM,
    TrainingExample, Dataset, SwingSignal,
    N_LLM_FEATURES, N_MARKET_DAYS, N_MARKET_CHANNELS, N_MARKET_COMPANIES,
    N_HOURLY_BARS, N_HOURS_PER_DAY, N_PRED_DAYS, N_PRED_HOURS,

    # kite_data
    load_kite_session, load_instruments, build_token_map,
    fetch_ohlcv, collect_ohlcv, load_cached_ohlcv,
    fetch_ohlcv_hourly, collect_ohlcv_hourly, load_cached_ohlcv_hourly,
    sector_index_name, NSE_INDICES,

    # inference_cache
    InferenceCache, build_inference_cache, load_inference_cache,
    find_hourly_end, find_date,

    # llm_extract
    extract_features, extract_features_from_kb,

    # features
    llm_to_vec, llm_feature_names, latest_before, find_date_index,

    # fundamentals (not active in current pipeline)
    FundamentalFeatures, FUNDAMENTAL_METRICS, N_QUARTERS, N_FUNDAMENTAL_FEATURES,
    extract_fundamentals, fundamental_feature_names,

    # dataset
    label_5d_hourly, generate_company_examples,
    time_split, assemble_batch,
    save_dataset, load_dataset,

    # model
    SwingArchitecture, DualCNN, DUAL_CNN_V1,
    SwingPredictor, build_model, predict, save_model, load_model,

    # train
    train!, evaluate, save_training_log,

    # broker
    get_holdings, get_positions, get_margins, get_orders

end
