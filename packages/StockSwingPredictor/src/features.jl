"""
Time-series feature computation and input vector assembly.

Each training example is assembled from:
  1. TSFeatures  × 3 entities (stock, Nifty 50, sector index)  = 39 values
  2. FundamentalFeatures                                        = 28 values
  3. LLMFeatures                                                = 15 values
  4. MetaFeatures (log_mcap, confidence, days_until, pledge,
                   is_fo, sector_onehot[n_sectors])            = 5 + n_sectors

Total ≈ 120+ features depending on sector vocabulary size.
"""

using Statistics, DataFrames, Dates, LinearAlgebra

# ── Time-series feature computation ──────────────────────────────────────────

"""
Compute `TSFeatures` from a sorted ascending DataFrame of daily OHLCV bars,
evaluated at position `t` (the last bar of the feature window).

Requires at least 130 bars (≈ 6 months of trading) before `t`.
Returns a zero `TSFeatures` if insufficient data.
"""
function compute_ts_features(df::DataFrame, t::Int)::TSFeatures
    zero_ts = TSFeatures(ntuple(_ -> 0f0, N_TS_FEATURES)...)

    t < 2 && return zero_ts
    closes  = Float64.(df.close[1:t])
    volumes = Float64.(df.volume[1:t])
    n = length(closes)

    # ── Log returns ──
    log_rets = diff(log.(closes))
    n_rets = length(log_rets)

    function period_return(days::Int)::Float32
        days > n_rets && return 0f0
        Float32(closes[end] / closes[max(1, end - days)] - 1.0)
    end

    r1w  = period_return(5)
    r2w  = period_return(10)
    r4w  = period_return(20)
    r8w  = period_return(40)
    r13w = period_return(65)
    r26w = period_return(130)

    # ── Realised volatility (annualised std of daily log-returns) ──
    function realised_vol(days::Int)::Float32
        days > n_rets && return 0f0
        slice = log_rets[max(1, end - days + 1):end]
        length(slice) < 2 && return 0f0
        Float32(std(slice) * sqrt(252))
    end

    vol4  = realised_vol(20)
    vol13 = realised_vol(65)
    vol26 = realised_vol(130)

    # ── Volume trend: slope of log-volume over last 20 days (OLS) ──
    vol_trend = let
        days = 20
        if length(volumes) >= days
            lv = log.(max.(volumes[end-days+1:end], 1.0))
            x  = Float64.(1:days)
            xm = mean(x); ym = mean(lv)
            slope = sum((x .- xm) .* (lv .- ym)) / sum((x .- xm).^2)
            Float32(clamp(slope, -0.2, 0.2))   # normalised slope per day
        else
            0f0
        end
    end

    # ── RSI(14) scaled to [0, 1] ──
    rsi = let
        period = 14
        if n_rets >= period
            sl = log_rets[end-period+1:end]
            gains  = sum(r for r in sl if r > 0; init=0.0)
            losses = -sum(r for r in sl if r < 0; init=0.0)
            ag = gains  / period
            al = losses / period
            rs = al == 0 ? 100.0 : ag / al
            Float32((100.0 - 100.0 / (1.0 + rs)) / 100.0)
        else
            0.5f0
        end
    end

    # ── Price vs 52-week high / low ──
    lookback = min(252, n)
    window   = closes[end-lookback+1:end]
    hi52     = maximum(window)
    lo52     = minimum(window)
    vs_high  = Float32(closes[end] / hi52)
    vs_low   = Float32(hi52 == lo52 ? 1.0 : (closes[end] - lo52) / (hi52 - lo52))

    return TSFeatures(r1w, r2w, r4w, r8w, r13w, r26w,
                      vol4, vol13, vol26,
                      vol_trend, rsi,
                      vs_high, vs_low)
end

"""Find the row index in a daily OHLCV DataFrame for the given date (exact or earlier)."""
function find_date_index(df::DataFrame, date::Date)::Int
    n = nrow(df)
    n == 0 && return 0
    idx = searchsortedlast(df.date, date)
    return idx
end

"""
Convert `TSFeatures` to a plain `Vector{Float32}`.
"""
function ts_to_vec(f::TSFeatures)::Vector{Float32}
    [f.return_1w, f.return_2w, f.return_4w, f.return_8w, f.return_13w, f.return_26w,
     f.vol_4w, f.vol_13w, f.vol_26w, f.volume_trend_4w, f.rsi_14d,
     f.price_vs_high, f.price_vs_low]
end

"""
Build the time-series feature name list for one entity (stock / nifty / sector).
"""
function ts_feature_names(prefix::String)::Vector{String}
    ["$(prefix)_ret_1w", "$(prefix)_ret_2w", "$(prefix)_ret_4w",
     "$(prefix)_ret_8w", "$(prefix)_ret_13w", "$(prefix)_ret_26w",
     "$(prefix)_vol_4w", "$(prefix)_vol_13w", "$(prefix)_vol_26w",
     "$(prefix)_volume_trend", "$(prefix)_rsi",
     "$(prefix)_vs_high", "$(prefix)_vs_low"]
end

# ── LLM features to vector ────────────────────────────────────────────────────

function llm_to_vec(f::LLMFeatures)::Vector{Float32}
    [f.management_tone, f.guidance_direction, f.guidance_specificity,
     f.demand_outlook, f.margin_commentary, f.competitive_pressure,
     f.new_wins_announced, f.capex_expansion, f.buyback_or_dividend,
     f.mgmt_language_hedging, f.auditor_concerns, f.related_party_flags,
     f.contingent_liability_flag, f.extraction_confidence, f.doc_age_days]
end

function llm_feature_names()::Vector{String}
    ["llm_mgmt_tone", "llm_guidance_dir", "llm_guidance_spec",
     "llm_demand", "llm_margin", "llm_competitive",
     "llm_new_wins", "llm_capex", "llm_buyback",
     "llm_hedging", "llm_auditor", "llm_related_party",
     "llm_contingent", "llm_confidence", "llm_doc_age"]
end

# ── Meta features ─────────────────────────────────────────────────────────────

"""
Build meta features for a company at a given time point.

# Arguments
- `market_cap_cr`: market cap in crores (Float64)
- `confidence_score`: 0–100 CompanyConfidence score
- `days_until_earnings`: days until next earnings (0 if unknown)
- `promoter_pledge_pct`: 0–100
- `is_fo`: whether F&O listed
- `sector`: string from Tijori
- `sector_vocab`: ordered list of known sectors (for one-hot)
"""
function meta_to_vec(market_cap_cr, confidence_score, days_until_earnings,
                     promoter_pledge_pct, is_fo::Bool,
                     sector::String, sector_vocab::Vector{String})::Vector{Float32}

    mcap_log = market_cap_cr !== nothing && Float64(market_cap_cr) > 0 ?
               Float32(log10(Float64(market_cap_cr))) : 0f0

    conf_norm    = Float32(clamp(something(confidence_score, 0), 0, 100) / 100.0)
    days_norm    = Float32(clamp(something(days_until_earnings, 90), 0, 90) / 90.0)
    pledge_norm  = Float32(clamp(something(promoter_pledge_pct, 0), 0, 100) / 100.0)
    fo_flag      = Float32(is_fo ? 1.0 : 0.0)

    onehot = zeros(Float32, length(sector_vocab))
    idx = findfirst(==(sector), sector_vocab)
    !isnothing(idx) && (onehot[idx] = 1f0)

    return vcat([mcap_log, conf_norm, days_norm, pledge_norm, fo_flag], onehot)
end

function meta_feature_names(sector_vocab::Vector{String})::Vector{String}
    base = ["meta_log_mcap", "meta_confidence", "meta_days_until_earnings",
            "meta_pledge_pct", "meta_is_fo"]
    sector_names = ["sector_$(replace(s, " " => "_"))" for s in sector_vocab]
    vcat(base, sector_names)
end

# ── Full feature vector assembly ──────────────────────────────────────────────

"""
Assemble the complete flat input vector for one training example.
"""
function assemble_features(ts_stock::TSFeatures, ts_nifty::TSFeatures,
                            ts_sector::TSFeatures, fund::FundamentalFeatures,
                            llm::LLMFeatures, meta::Vector{Float32})::Vector{Float32}
    vcat(ts_to_vec(ts_stock), ts_to_vec(ts_nifty), ts_to_vec(ts_sector),
         fund.values, llm_to_vec(llm), meta)
end

"""
Build the full ordered feature name list (matches `assemble_features` output).
"""
function all_feature_names(sector_vocab::Vector{String})::Vector{String}
    vcat(ts_feature_names("stock"),
         ts_feature_names("nifty50"),
         ts_feature_names("sector"),
         fundamental_feature_names(),
         llm_feature_names(),
         meta_feature_names(sector_vocab))
end
