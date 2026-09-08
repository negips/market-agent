"""
All structs for StockSwingPredictor.
"""

# ── Raw data containers ───────────────────────────────────────────────────────

"""Daily OHLCV bar from Kite."""
struct OHLCVBar
    date   :: Date
    open   :: Float64
    high   :: Float64
    low    :: Float64
    close  :: Float64
    volume :: Float64
end

# ── Feature groups ────────────────────────────────────────────────────────────

"""
Derived time-series features computed from a window of daily OHLCV bars.
One instance per entity (stock, Nifty 50, sector index) per training example.
"""
struct TSFeatures
    return_1w       :: Float32   # 5-day log return
    return_2w       :: Float32   # 10-day
    return_4w       :: Float32   # 20-day
    return_8w       :: Float32   # 40-day
    return_13w      :: Float32   # 65-day
    return_26w      :: Float32   # 130-day
    vol_4w          :: Float32   # realised vol (std of daily log-returns, 20d)
    vol_13w         :: Float32   # 65-day
    vol_26w         :: Float32   # 130-day
    volume_trend_4w :: Float32   # slope of log-volume over 20 days (normalised)
    rsi_14d         :: Float32   # RSI(14), scaled to [0, 1]
    price_vs_high   :: Float32   # close / 52-week high
    price_vs_low    :: Float32   # close / 52-week low
end

const N_TS_FEATURES = 13  # features per entity; 3 entities → 39 total

"""
Quarterly fundamental features: last 4 quarters, 7 metrics each = 28 values.
Metrics are normalised by their own trailing 4Q mean so the NN sees growth rates,
not raw rupee figures that differ by orders of magnitude across market caps.
"""
struct FundamentalFeatures
    # 4 quarters × 7 metrics, stored row-major: [q1_rev, q1_ebitda_m, ..., q4_roce]
    values :: Vector{Float32}   # length 28
end

const FUNDAMENTAL_METRICS = ["revenue", "ebitda_margin", "pat_margin",
                              "cfo_margin", "eps", "debt_equity", "roce"]
const N_QUARTERS = 4
const N_FUNDAMENTAL_FEATURES = length(FUNDAMENTAL_METRICS) * N_QUARTERS  # 28

"""
Scalar features extracted from the most recent conference call or earnings
release by the LLM. Values are all in [-1, 1] or [0, 1] or {0, 1}.
"""
struct LLMFeatures
    management_tone          :: Float32   # -1 bearish … +1 bullish
    guidance_direction       :: Float32   # -1 cut / 0 none / +1 raised → mapped to {-1,0,1}
    guidance_specificity     :: Float32   # 0 none … 3 quantified → scaled to [0,1]
    demand_outlook           :: Float32   # -1 … +1
    margin_commentary        :: Float32   # -1 pressure … +1 expanding
    competitive_pressure     :: Float32   # 0 … 1
    new_wins_announced       :: Float32   # {0, 1}
    capex_expansion          :: Float32   # {0, 1}
    buyback_or_dividend      :: Float32   # {0, 1}
    mgmt_language_hedging    :: Float32   # 0 confident … 1 hedged
    auditor_concerns         :: Float32   # {0, 1}
    related_party_flags      :: Float32   # {0, 1}
    contingent_liability_flag :: Float32  # {0, 1}
    extraction_confidence    :: Float32   # 0 … 1 (LLM self-reported)
    doc_age_days             :: Float32   # days since document was published (normalised)
end

const N_LLM_FEATURES = 15

"""Missing LLM features: used when no document is available."""
const MISSING_LLM = LLMFeatures(0f0, 0f0, 0f0, 0f0, 0f0, 0f0, 0f0, 0f0, 0f0,
                                 0.5f0, 0f0, 0f0, 0f0, 0f0, 1f0)

"""
Metadata features: company-level static attributes at time T.
sector_onehot has a fixed length defined by the vocabulary built from training data.
"""
struct MetaFeatures
    log_market_cap       :: Float32   # log10(market_cap_cr)
    confidence_score     :: Float32   # 0 … 1
    days_until_earnings  :: Float32   # clipped to [0, 90], scaled to [0, 1]
    promoter_pledge_pct  :: Float32   # 0 … 100 → scaled to [0, 1]
    is_fo                :: Float32   # {0, 1} — F&O listed
    sector_onehot        :: Vector{Float32}
end

# ── Assembled input / output ──────────────────────────────────────────────────

"""
One training (or inference) example. `label` is the 5-trading-day log return
of the stock starting from `date`. It is `nothing` for inference examples
(future dates where the outcome is not yet known).
"""
struct Example
    symbol   :: String
    date     :: Date
    features :: Vector{Float32}   # fully assembled, normalised input vector
    label    :: Union{Float32, Nothing}
end

"""
The assembled, normalised dataset ready for training.
`X`: (n_features × n_examples) Float32 matrix
`y`: (n_examples,) Float32 vector
`feature_names`: length n_features — for interpretability / debugging
"""
struct Dataset
    X             :: Matrix{Float32}
    y             :: Vector{Float32}
    feature_names :: Vector{String}
    symbols       :: Vector{String}
    dates         :: Vector{Date}
    sector_vocab  :: Vector{String}
end

# ── Normalisation stats (saved alongside model) ───────────────────────────────

"""
Per-feature mean and standard deviation computed from the training split.
Applied identically at inference time.
"""
struct NormStats
    feature_names :: Vector{String}
    means         :: Vector{Float32}
    stds          :: Vector{Float32}
end

# ── Inference output ──────────────────────────────────────────────────────────

"""Model prediction for a single company at inference time."""
struct SwingSignal
    symbol          :: String
    company         :: String
    predicted_return :: Float32   # 5-day predicted log return
    percentile      :: Float32   # rank within current batch (0 … 1)
    earnings_date   :: Union{Date, Nothing}
    days_until      :: Int
    confidence_score :: Float32
    price_source    :: String
end
