"""
Named constants governing rule enforcement and (from Stage 3 onward) RL training.

Every numeric value here traces back to a specific numbered rule in
`TradingGameRules.txt` (repo root) — see the comment on each constant.
"""

# ── Rule-derived constants ──────────────────────────────────────────────────────

const FEE_RATE              = 0.005           # rule 11: 0.5% of transaction value, both legs
const SETTLEMENT_DAYS       = 2               # rule 4: reserved-cash → cash delay, in trading days
const MIN_HOLD_DAYS         = 1               # rule 10: lock-up before a voluntary sale
const MAX_HOLD_DAYS         = 14              # rule 9: forced exit after this many trading days
const DECISION_INTERVAL_MIN = 15              # rule 8: minimum minutes between decisions
const MAX_POSITION_FRACTION = 0.15            # rule 12: a single symbol can't exceed this share of portfolio value

"""Rule 13: the ceiling on distinct symbols held at once, N_MAX, is defined as
a fraction of the candidate universe size N (not a fixed constant) — see
`n_max_holdings`. `N_MAX_HOLDINGS_FRACTION` is the `0.25` from the rule text
itself."""
const N_MAX_HOLDINGS_FRACTION = 0.25

"""Rule 13's N_MAX for a universe of `n_candidates` symbols: `round(0.25 *
n_candidates)`, floored at 1 so a tiny universe (e.g. a 3-symbol test fixture)
still allows at least one position — the rule text doesn't specify a rounding
convention or a minimum, and zero would make the game unplayable, which
contradicts every other rule's premise that stocks can be held at all."""
n_max_holdings(n_candidates::Int) = max(1, round(Int, N_MAX_HOLDINGS_FRACTION * n_candidates))

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
`CASH_CEILING_PENALTY_COEF * max(0, cash/value - MAX_CASH_FRACTION)` from
*every* bar's reward — unlike the log-return term (see `TRAINING_REWARD_MODE`
and `REWARD_INTERVAL_DAYS`, neither of which this penalty depends on), this
penalty is never windowed, under either reward mode. Scaled to the same
order as `ENTROPY_COEF` — small enough not to swamp a reward window's
genuine portfolio-value signal, large enough that sitting at 100% cash
(excess=0.70) costs -0.007/bar, a real, learnable incentive to deploy
capital even between reward windows."""
const CASH_CEILING_PENALTY_COEF = 0.01

"""Rule 15: once a symbol is sold (voluntarily or via the rule-9 forced exit —
the rule text doesn't distinguish, and `_execute_sell!` already treats both
uniformly for rule 11's fee, so this follows the same precedent), it can't be
newly bought again for this many trading days. Structural (masked in
`resolve_actions`, like rules 5/10/12/13), not a reward penalty like rule 14
— there's no "episode start" edge case here that makes a hard ceiling
ill-defined the way rule 14's cash constraint had. Only blocks *opening a new
position*; adding to a symbol that's still currently held (a separate,
not-yet-sold lot) is unaffected — see `resolve_actions`'s docstring."""
const REBUY_COOLDOWN_DAYS = 7

"""
Which algorithm computes the reward's log-return term each bar — see `step!`
for the implementation of both. Switching modes only changes the RL reward
*signal*; it never affects rule 8's decision cadence (buy/sell decisions
still happen every hourly bar either way) or rule 14's cash-ceiling penalty
(independent of reward mode, always applied every single bar).

- `SPARSE_WINDOW`: the log-return term is 0 every bar except once every
  `REWARD_INTERVAL_DAYS` trading days, when the FULL window's return is
  reported as one lump sum: `log(value_now / value_at_window_start)`. The
  final bar of an episode also force-flushes a shorter trailing window, so
  an episode's total reward telescopes *exactly* to `log(V_final/V_initial)`
  — nothing is silently dropped, it's just reported in ~weekly (or however
  long `REWARD_INTERVAL_DAYS` is) chunks instead of hourly ones.

- `ROLLING_WINDOW`: every bar's log-return term is the trailing
  `REWARD_INTERVAL_DAYS`-day return, `log(value_now / value_N_days_ago)` (0
  until that much history exists, early in an episode). Reward is dense —
  never 0 once past the warm-up — but consecutive bars' rewards overlap
  heavily (a 14-day window shares 13 of its days with the next bar's
  window), and the episode-total reward no longer telescopes to a simple
  final/initial ratio: summing it out algebraically gives
  `log((V_{T-N+1}·...·V_T) / (V_1·...·V_N))`, a ratio of the products of the
  last N and first N daily values, not `log(V_final/V_initial)` — so
  `train_return`/`val_return` stop being directly readable as "the
  portfolio's total return" the way they are under `SPARSE_WINDOW`; check
  `portfolio_value(env)` directly instead.
"""
@enum RewardMode SPARSE_WINDOW ROLLING_WINDOW

const TRAINING_REWARD_MODE = SPARSE_WINDOW

"""The reward window length, in trading days, for whichever `TRAINING_REWARD_MODE`
is active — "once every N days" for `SPARSE_WINDOW`, "trailing N-day return"
for `ROLLING_WINDOW`. Shared between both modes so switching modes is a
one-line change (`TRAINING_REWARD_MODE`) without also having to re-tune a
separate window-length constant per mode."""
const REWARD_INTERVAL_DAYS = 7

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

"""Minimum `severity` (NewsMonitor's 0–1 scale — see `NewsMonitor.llm_classify.jl`'s
calibration text: 0.5 = moderate impact, 1.0 = major market-moving) for a
classified signal to add its hourly bar to `TradingGameEnv.news_hour_indices`
(`news_features.jl`'s `build_news_feature_cache`) — routine/noise
announcements (severity ~0.1) don't count as a news-triggered decision
point. Currently a no-op under `TRAINING_DECISION_GRANULARITY == HOURLY`
(every bar is already a decision bar — see `env.jl`'s module docstring);
kept so the set is populated correctly once `MINUTE_15` lands."""
const NEWS_DECISION_SEVERITY_THRESHOLD = 0.5

const GAMMA               = 0.99
const GAE_LAMBDA           = 0.95
const CLIP_EPS              = 0.2
const VALUE_LOSS_COEF       = 0.5
const ENTROPY_COEF          = 0.01
const DECAY_HALFLIFE_HOURS  = 24.0    # news-feature recency decay (observation.jl, Stage 2)
