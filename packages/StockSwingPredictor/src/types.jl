"""
All structs and constants for StockSwingPredictor.
"""

# ── Architecture constants ────────────────────────────────────────────────────

# Market background context: closing prices + daily vol for the full universe
const N_MARKET_DAYS     = 28   # 4 calendar weeks of trading days
const N_MARKET_CHANNELS = 2    # channel 1: normalised close, channel 2: daily (H-L)/C vol

# Individual stock fine-grained series: 8 calendar weeks of 60-minute bars
const N_HOURLY_BARS = 280      # 40 trading days × 7 bars/day

# Prediction horizon: 5 trading days at 60-minute resolution
const N_HOURS_PER_DAY = 7      # NSE: 9:15–15:30, 7 hourly bars (last bar partial)
const N_PRED_DAYS     = 5
const N_PRED_HOURS    = N_HOURS_PER_DAY * N_PRED_DAYS   # 35 output neurons

# LLM scalar count — bump N_LLM_FEATURES and add fields to LLMFeatures to expand
const N_LLM_FEATURES = 15

# ── LLM scalar features ───────────────────────────────────────────────────────

"""
Scalar features extracted from the most recent conference call or earnings
release by the LLM. All values in [-1, 1] or [0, 1] or {0, 1}.
"""
struct LLMFeatures
    management_tone           :: Float32   # -1 bearish … +1 bullish
    guidance_direction        :: Float32   # {-1, 0, +1}
    guidance_specificity      :: Float32   # 0 … 1
    demand_outlook            :: Float32   # -1 … +1
    margin_commentary         :: Float32   # -1 pressure … +1 expanding
    competitive_pressure      :: Float32   # 0 … 1
    new_wins_announced        :: Float32   # {0, 1}
    capex_expansion           :: Float32   # {0, 1}
    buyback_or_dividend       :: Float32   # {0, 1}
    mgmt_language_hedging     :: Float32   # 0 confident … 1 hedged
    auditor_concerns          :: Float32   # {0, 1}
    related_party_flags       :: Float32   # {0, 1}
    contingent_liability_flag :: Float32   # {0, 1}
    extraction_confidence     :: Float32   # 0 … 1
    doc_age_days              :: Float32   # days since document, normalised
end

"""Missing LLM features: neutral values used when no document is available."""
const MISSING_LLM = LLMFeatures(0f0, 0f0, 0f0, 0f0, 0f0, 0f0, 0f0, 0f0, 0f0,
                                 0.5f0, 0f0, 0f0, 0f0, 0f0, 1f0)

# ── Per-example training record ───────────────────────────────────────────────

"""
One labeled training (or inference) example.

`date_idx` indexes into `Dataset.dates` (the master trading calendar).
`sym_idx`  indexes into `Dataset.companies` (fixed universe ordering).
`hourly`   is normalised: each value is `close / close[1]` of the 8-week window.
`label`    is the N_PRED_HOURS trajectory of log-returns relative to close on `date`.
"""
struct TrainingExample
    date     :: Date
    symbol   :: String
    date_idx :: Int
    sym_idx  :: Int
    hourly   :: Vector{Float32}   # length N_HOURLY_BARS
    llm      :: Vector{Float32}   # length N_LLM_FEATURES
    label    :: Vector{Float32}   # length N_PRED_HOURS
end

# ── Dataset ───────────────────────────────────────────────────────────────────

"""
Assembled dataset for training and inference.

`closes[i, j]` — raw closing price of company j on date i (forward-filled).
`vols[i, j]`   — daily intraday range: (high - low) / close for company j on date i.
`dates`         — master trading calendar (sorted ascending), one row per date.
`companies`     — universe of symbols in a fixed order, one column per company.
`examples`      — all labeled training examples, sorted by date.

The market background matrix for a training example at date_idx `t` is assembled
on-the-fly in `assemble_batch`: closes[t-N_MARKET_DAYS+1:t, :] normalised so
the window's first day = 1.0, with the target company moved to column 1.
"""
struct Dataset
    closes    :: Matrix{Float32}           # (n_dates, n_companies)
    vols      :: Matrix{Float32}           # (n_dates, n_companies)
    dates     :: Vector{Date}
    companies :: Vector{String}
    examples  :: Vector{TrainingExample}
end

# ── Inference output ──────────────────────────────────────────────────────────

"""Model prediction for a single company at inference time."""
struct SwingSignal
    symbol               :: String
    company              :: String
    predicted_trajectory :: Vector{Float32}   # N_PRED_HOURS log returns vs ref close
    eod_return           :: Float32           # trajectory[end]: end-of-day-5 log return
    percentile           :: Float32           # rank by |eod_return| within batch
    earnings_date        :: Union{Date, Nothing}
    days_until           :: Int
    confidence_score     :: Float32
    price_source         :: String
end
