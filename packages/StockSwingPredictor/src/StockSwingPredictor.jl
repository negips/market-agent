"""
StockSwingPredictor

Neural-network-based large-move predictor for NSE-listed equities.

## Architecture

Two CNN branches feed an MLP head. Three registered architectures share the same
branch structure but differ in how market context is aggregated and in output horizon:

- **Market branch** (weights shared across all companies in the universe):
  Each company's 28-day close + daily-vol + relative-vol series → 1D-CNN → 128-dim
  embedding. The target stock's embedding is separated from the context embeddings:
  mean-pooled for `DualCNN_v1`; multi-head cross-attention for `DualCNNv2` and
  `DualCNNv3` (model learns which peers are relevant per prediction).

- **Hourly branch** (target stock only):
  280-bar 60-minute normalised close → deeper 1D-CNN → 256-dim embedding.

- **MLP head**: `concat(target_emb, ctx_emb, hourly_emb)` → 512 → … → `pred_hours` outputs.

| Arch        | Context       | `pred_hours` | MLP hidden      | Loss weighting       |
|-------------|---------------|--------------|-----------------|----------------------|
| DualCNN_v1  | Mean-pool     | 35 (5-day)   | [512,256,128,64]| JUMP_THRESHOLD (4%)  |
| DualCNN_v2  | Cross-attn h=2| 35 (5-day)   | [512,256,128,64]| JUMP_THRESHOLD (4%)  |
| DualCNN_v3  | Cross-attn h=2| 70 (10-day)  | [512,256,128]   | Lee-Mykland M²       |

## Label

Dataset labels are `N_PRED_HOURS = 70` bars (built with `--pred-hours 70`).
v1/v2 slice to their first 35 bars at batch time; v3 uses all 70.
Label format: `label[h] = log(close_h / ref_close)` for h in 1…pred_hours.

## Pipeline

```
scripts/collect_ohlcv.jl        # download daily + hourly OHLCV for all companies
scripts/update_ohlcv.jl         # daily incremental update (append new bars)
scripts/build_cache.jl          # build inference_cache.bson from CSVs (run each morning)
scripts/build_dataset.jl --pred-hours 35   # dataset for v1/v2  → dataset_35.bson
scripts/build_dataset.jl --pred-hours 70   # dataset for v3     → dataset_70.bson
scripts/train_model.jl --arch v3           # train; auto-selects dataset_70.bson
```

Note: `extract_llm_features.jl` exists for future architectures but is not used
by v1/v2/v3 — LLM features are stored in TrainingExample for forward-compatibility
but zeroed out and ignored during training.

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
include("macro_data.jl")
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
    fetch_ohlcv_5min, collect_ohlcv_5min, load_cached_ohlcv_5min,
    fetch_ohlcv_15min, collect_ohlcv_15min, load_cached_ohlcv_15min,
    sector_index_name, NSE_INDICES,

    # macro_data
    YAHOO_MACRO_INSTRUMENTS, KITE_MACRO_INSTRUMENTS,
    fetch_yahoo_ohlcv, fetch_kite_macro_ohlcv,
    load_instruments_for_exchange, build_macro_kite_tokens,
    collect_macro_ohlcv, load_macro_ohlcv,
    fetch_kite_macro_5min, collect_macro_5min, load_macro_5min,
    fetch_kite_macro_15min, collect_macro_15min, load_macro_15min,

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
    SwingArchitecture, DualCNN, DUAL_CNN_V1, DualCNNv2, DUAL_CNN_V2,
    DualCNNv3, DUAL_CNN_V3,
    SwingPredictor, build_model, predict, save_model, load_model,

    # train
    train!, evaluate, save_training_log,

    # broker
    get_holdings, get_positions, get_margins, get_orders

end
