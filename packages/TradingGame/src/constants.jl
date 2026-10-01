"""
Named constants governing rule enforcement and (from Stage 3 onward) RL training.

Every numeric value here traces back to a specific numbered rule in
`TradingGameRules.txt` (repo root) — see the comment on each constant.
"""

# ── Rule-derived constants ──────────────────────────────────────────────────────

const FEE_RATE              = 0.005          # rule 11: 0.5% of transaction value, both legs
const SETTLEMENT_DAYS       = 2               # rule 4: reserved-cash → cash delay, in trading days
const MIN_HOLD_DAYS         = 1               # rule 10: lock-up before a voluntary sale
const MAX_HOLD_DAYS         = 10              # rule 9: forced exit after this many trading days
const DECISION_INTERVAL_MIN = 15              # rule 8: minimum minutes between decisions
const MAX_POSITION_FRACTION = 0.15            # rule 12: a single symbol can't exceed this share of portfolio value
const N_MAX_HOLDINGS        = 20              # rule 13: max distinct symbols held at once

"""Rule 14: spendable cash (excludes reserved cash — see `TradingGameRules.txt`'s
own distinct "cash"/"reserved cash" categories, rules 4 and 7) can never exceed
this share of total portfolio value. Unlike rules 12/13, this is enforced as a
*soft* constraint (a reward penalty, in `env.jl`'s `step!`), not a structural
mask — the episode starts at 100% cash (before any stock is bought) and
matured reserved cash lands back in cash passively, so a hard "never" ceiling
would require inventing an undefined forced-buy rule (which stock, how much)
with no basis in the rules text. See `step!`'s docstring for the penalty
formula and `StepResult.info["cash_ceiling_violated"]`."""
const MAX_CASH_FRACTION = 0.30

"""Weight on the rule-14 soft penalty: `step!` subtracts
`CASH_CEILING_PENALTY_COEF * max(0, cash/value - MAX_CASH_FRACTION)` from that
bar's reward. Scaled to the same order as `ENTROPY_COEF` (both are per-bar
shaping terms layered on a typically-small per-bar log-return reward) — small
enough not to swamp genuine portfolio-value signal, large enough that sitting
at 100% cash (excess=0.70) costs a reward of -0.007/bar, a real, learnable
incentive to deploy capital."""
const CASH_CEILING_PENALTY_COEF = 0.01

# ── Decision cadence proxy ───────────────────────────────────────────────────────

"""
Historical Kite 15-minute OHLCV retention caps at ~200 days and 5-minute at ~100
days (see `packages/StockSwingPredictor/src/kite_data.jl`), so multi-year RL
training cannot run at rule 8's literal 15-minute cadence. `HOURLY` trains the
simulator at one decision per hourly bar (or immediately on a news bar) as a
practical proxy — this is NOT exact rule-8 compliance, only a training-time
stand-in. `MINUTE_15` is reserved for a future live/rolling-window cache and is
not yet implemented (`is_decision_bar` raises if selected).
"""
@enum DecisionGranularity HOURLY MINUTE_15

const TRAINING_DECISION_GRANULARITY = HOURLY

# ── Candidate universe ────────────────────────────────────────────────────────

"""
Cap on the joint-action candidate universe size. Smaller than
`StockSwingPredictor.N_MARKET_COMPANIES` (150) because this bounds the actual
*action* space of the joint policy, not passive market context — see
`universe.jl` (Stage 0/2).
"""
const N_CANDIDATE_STOCKS = 60

"""
Minimum `confidence.score` (0–100) a company must have, read from
`nse_companies_latest.json`, to enter the candidate universe — matches
`CompanyConfidence.PASS_THRESHOLD`. Duplicated here (rather than depending on
`CompanyConfidence`) because `universe.jl` only ever reads this pre-computed
score from JSON, never calls `CompanyConfidence.analyze` itself — see
`universe.jl`'s module docstring for why.
"""
const MIN_CONFIDENCE_SCORE = 40.0

# ── Observation shape (observation.jl) ───────────────────────────────────────

const N_HOURLY_BARS_SHORT = 120   # ~17 trading days of hourly bars — actor-critic encoder window
const N_PRICE_CHANNELS    = 3     # normalised close, daily (H-L)/C vol, relative volume

const N_MACRO_DAYS   = 10
const N_MACRO_SERIES = 9   # SP500, US_VIX, USD_INR, INDIA_VIX, CRUDE_OIL, GOLD, SILVER, NATURAL_GAS, COPPER
const MACRO_SERIES_NAMES = ["SP500", "US_VIX", "USD_INR", "INDIA_VIX",
                             "CRUDE_OIL", "GOLD", "SILVER", "NATURAL_GAS", "COPPER"]

"""Per-stock news feature width: [decayed own-symbol signal, decayed market-wide
signal]. Zero until `news_features.jl` (Stage 0/2 backfill) lands — see
`_zero_news` in `observation.jl`."""
const N_NEWS_FEATURES = 2

"""Per-holding observation width: [held flag, quantity-weighted unrealised P&L,
minimum remaining-hold-budget fraction across lots — see `assemble_observation`]."""
const N_HOLDING_FEATURES = 3

"""Global portfolio scalars, all O(1)-scale ratios (never raw rupee amounts —
see `assemble_observation`'s portfolio-scalar section for why): [cash/value,
reserved/value, value/initial_cash, stocks_value/value]."""
const N_PORTFOLIO_SCALARS = 4

# ── RL training (used from Stage 3 onward; declared here as the single source
#    of truth so `policy.jl`/`ppo.jl`/`train.jl` never redefine them) ───────────

const GAMMA               = 0.99
const GAE_LAMBDA           = 0.95
const CLIP_EPS              = 0.2
const VALUE_LOSS_COEF       = 0.5
const ENTROPY_COEF          = 0.01
const DECAY_HALFLIFE_HOURS  = 24.0    # news-feature recency decay (observation.jl, Stage 2)
