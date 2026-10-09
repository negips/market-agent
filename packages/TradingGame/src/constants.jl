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
const MAX_POSITION_FRACTION = 0.30            # rule 12: a single symbol can't exceed this share of portfolio value

"""Rule 13: the ceiling on distinct symbols held at once, N_MAX, is defined as
a fraction of the candidate universe size N (not a fixed constant) — see
`n_max_holdings`. `N_MAX_HOLDINGS_FRACTION` started as the `0.25` from the rule
text and is now tuned to `0.5`."""
const N_MAX_HOLDINGS_FRACTION = 0.5

"""Rule 13's N_MAX for a universe of `n_candidates` symbols: `round(N_MAX_HOLDINGS_FRACTION
* n_candidates)`, floored at 1 so a tiny universe (e.g. a 3-symbol test fixture)
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
order of `ENTROPY_COEF` (0.01), and 5x it — small enough not to swamp a
reward window's genuine portfolio-value signal, large enough that sitting at
100% cash (excess=0.70) costs -0.035/bar, a real, learnable incentive to
deploy capital even between reward windows."""
const CASH_CEILING_PENALTY_COEF = 0.05

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

"""Game v3's default reward charged, immediately, for each *illegal move*: `0.01`,
i.e. 1% of the current portfolio value (the reward is in portfolio-fraction
units, so this is `0.01` of reward). An illegal move is one the rules refuse and
that the policy could not have been prevented from attempting — see
`resolve_actions`. The trade is rejected and the charge is taken in that same
bar's reward, not counted or windowed. Override with `--illegal-penalty`."""
const ILLEGAL_PENALTY_COEF_V3 = 0.01

"""Game v2 (see `GameRules` in `types.jl`): holding a stock past this many
trading days is no longer force-exited (v1's rule 9, `MAX_HOLD_DAYS`) but
costs a soft per-bar reward penalty instead, mirroring rule 14's cash penalty
— `hold_penalty_coef * (share of portfolio value in lots held >= this many
days)`. Kept separate from `MAX_HOLD_DAYS` so v1's forced exit stays at 14
days and v1 results remain comparable."""
const MAX_HOLD_DAYS_V2 = 7

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
Decision cadence of a training run, set by the bar length of the `InferenceCache`
it trains on (`cache.bar_minutes`) — see `decision_granularity`. `MINUTE_15`
decides at every 15-minute bar, which is rule 8's minimum interval exactly (a
cache built with `build_cache.jl --granularity 15min`). `HOURLY` decides once per
hourly bar, a coarser proxy that predates the 15-minute data (Kite's per-request
span cap once looked like a retention limit, which is why this proxy existed).
"""
@enum DecisionGranularity HOURLY MINUTE_15

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

"""Game v3's encoder window, in trading days (two weeks). v1/v2 keep the fixed
`N_HOURLY_BARS_SHORT` bars; a rules set with `obs_window_days > 0` sizes the
window as that many trading days of bars instead — see `obs_window_bars`."""
const OBS_WINDOW_DAYS_V3 = 10

"""Hourly history bars per trading session (9:15 … 15:15)."""
const HISTORY_BARS_PER_DAY = 7

"""Game v3 expresses the price history as log-returns, `x_i = LOG_RETURN_SCALE *
ln(c_i / c_{i-1})` — i.e. in percent (an hourly move of 0.5% is `0.5`) — so the
network's inputs are O(1) instead of O(0.005). See `observation.jl`."""
const LOG_RETURN_SCALE = 100f0

const N_HOURLY_BARS_SHORT = 120   # bars in the actor-critic encoder window: ~17 trading days of hourly bars, ~5 days of 15-minute bars
const N_PRICE_CHANNELS    = 2     # normalised close, PREVIOUS day's (H-L)/C vol (game v3 uses only the first; see `n_price_channels`)

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

"""Game v2's portfolio vector: the 4 above plus two cash-token scalars —
`cash/value ÷ MAX_CASH_FRACTION` (cap utilisation, 1.0 = at the cap) and the
fraction of `MAX_HOLD_DAYS_V2` that cash has now spent above the cap (the
cash analogue of a stock's days-held)."""
const N_PORTFOLIO_SCALARS_V2 = 6

"""Width of v2's cash token before `cash_encoder` embeds it: [cash/value,
reserved/value, cap utilisation, days-over-cap fraction] — the portfolio
vector's entries 1, 2, 5 and 6."""
const N_CASH_TOKEN_FEATURES = 4

# ── RL training (used from Stage 3 onward; declared here as the single source
#    of truth so `policy.jl`/`ppo.jl`/`train.jl` never redefine them) ───────────

"""Minimum `severity` (NewsMonitor's 0–1 scale — see `NewsMonitor.llm_classify.jl`'s
calibration text: 0.5 = moderate impact, 1.0 = major market-moving) for a
classified signal to add its hourly bar to `TradingGameEnv.news_hour_indices`
(`news_features.jl`'s `build_news_feature_cache`) — routine/noise
announcements (severity ~0.1) don't count as a news-triggered decision
point. Currently a no-op: every bar of an hourly or 15-minute cache is
already a decision bar (see `env.jl`'s module docstring); kept so the set is
populated correctly if a coarser-than-news cadence is ever added."""
const NEWS_DECISION_SEVERITY_THRESHOLD = 0.5

const GAMMA                 = 0.99
const GAE_LAMBDA            = 0.95

"""`GAMMA`/`GAE_LAMBDA` are defined per *hourly* step. On a cache with shorter
bars the same wall-clock horizon spans more steps, so the per-step factor is
rescaled, `x^(bar_minutes/60)` — a 15-minute run discounts and bootstraps over
the same real time as an hourly one instead of 4× faster. Identity for hourly."""
bar_scaled(x::Real, bar_minutes::Integer) = Float64(x)^(bar_minutes / 60)

"""`(gamma, gae_lambda)` PPO should use for `rules` on `cache`: the hourly constants
rescaled to the bar length (`bar_scaled`), except under a terminal reward
(`GameRules.terminal_reward`). With a single reward at the episode's last bar
(`reward_window_days == 0`) both are 1: any discount below 1 would shrink that reward to
nothing over ~20,000 bars (0.9975^20000 ≈ 1e-22). With a reward every few days the
discount keeps its bar-scaled value (a window's reward still counts ~0.65 at the window's
start for a week) and λ is 1, so each step is credited with the discounted rewards that follow."""
discount_factors(rules, cache) =
    !rules.terminal_reward ? (bar_scaled(GAMMA, cache.bar_minutes), bar_scaled(GAE_LAMBDA, cache.bar_minutes)) :
    rules.reward_window_days == 0 ? (1.0, 1.0) : (bar_scaled(GAMMA, cache.bar_minutes), 1.0)
const CLIP_EPS              = 0.2
const VALUE_LOSS_COEF       = 0.5
const ENTROPY_COEF          = 0.01
const DECAY_HALFLIFE_HOURS  = 24.0    # news-feature recency decay (observation.jl, Stage 2)
