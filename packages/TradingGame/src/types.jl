"""
All structs and exception types for TradingGame.
"""

using Dates
using StockSwingPredictor: InferenceCache

# ── Action representation ────────────────────────────────────────────────────────

"""A candidate's requested action for the current decision step."""
@enum ActionType HOLD SELL BUY

"""
One candidate's raw, unmasked action, as produced by a policy (or a baseline).

`weight` is only meaningful when `kind == BUY` — the requested share of
available cash to allocate, before cash-constraint renormalisation. Ignored
for `HOLD`/`SELL`. `sym_idx` indexes `InferenceCache.companies`/`closes` etc.
directly (not a separate candidate-universe-local index).
"""
struct RawAction
    sym_idx :: Int
    kind    :: ActionType
    weight  :: Float64
end

"""
A mask-and-normalise output of `resolve_actions` — ready to execute as-is.
`notional` is the cash amount to spend and is only meaningful for `BUY`;
`SELL` always liquidates every sellable lot of `sym_idx` (see rule 10's 1-day
lock-up in `resolve_actions`).
"""
struct ResolvedTrade
    sym_idx  :: Int
    kind     :: ActionType
    notional :: Float64
end

"""A full decision-step action: one `RawAction` per stock the caller wants to act on."""
const JointAction = Vector{RawAction}

# ── Portfolio state ───────────────────────────────────────────────────────────────

"""
A single open position (one purchase lot). Multiple concurrent lots of the same
symbol are tracked independently, each ageing against rules 9/10 on its own
`entry_date_idx` — this is what lets `resolve_actions` sell only the lots that
have cleared the 1-day lock-up while leaving newer lots of the same stock held.

`entry_date_idx`/`entry_hour_idx` index into the environment's `InferenceCache`
(`dates`/`hourly_datetimes`) rather than storing a `Date`, so day-counting for
rules 9 and 10 is exact trading-day arithmetic, not calendar-day arithmetic.
"""
Base.@kwdef struct Holding
    symbol         :: String
    sym_idx        :: Int
    entry_date_idx :: Int
    entry_hour_idx :: Int
    quantity       :: Float64
    entry_price    :: Float64
    entry_fee      :: Float64
end

"""
Sale proceeds (voluntary or forced) sitting in "reserved cash" (rule 4) until
`available_date_idx` (a trading-day index into `InferenceCache.dates`), at
which point `step!` moves `amount` into spendable cash. `source_symbol` is an
audit-trail field only — no rule computation depends on it.
"""
Base.@kwdef struct ReservedCashLot
    amount             :: Float64
    available_date_idx :: Int
    source_symbol      :: String
end

"""Mutable portfolio state: spendable cash, cash pending settlement, open positions.

`rebuy_cooldown` implements rule 15: `sym_idx => date_idx` the symbol becomes
eligible to be newly bought again (set by `_execute_sell!` to
`date_idx + REBUY_COOLDOWN_DAYS` on every sale, forced or voluntary — see
`REBUY_COOLDOWN_DAYS`'s docstring). A symbol absent from this dict has never
been sold this episode and is unrestricted. Same `available_date_idx`-style
"earliest eligible date" convention as `ReservedCashLot`, checked in
`resolve_actions`."""
Base.@kwdef mutable struct Portfolio
    cash           :: Float64
    reserved       :: Vector{ReservedCashLot} = ReservedCashLot[]
    holdings       :: Vector{Holding}         = Holding[]
    rebuy_cooldown :: Dict{Int, Int}          = Dict{Int, Int}()
end

# ── Episode / environment ─────────────────────────────────────────────────────────

"""
Which rule set an episode is played under. Carried by `EpisodeConfig` so the
same code path serves both versions; construct with `rules_v1()` (the original
game, the default everywhere) or `rules_v2(; cash_penalty, hold_penalty)`.

- `same_bar_execution`: v1 `false` — a decision made on bar t is filled at bar
  t+1's price; v2 `true` — filled at bar t's own close, which stands in for a
  live system's instantaneous price.
- `forced_exit`: v1 `true` — a lot held `max_hold_days` is sold unconditionally
  (rule 9); v2 `false` — it is only penalised, see `hold_penalty_coef`.
- `max_hold_days`: v1 14 (`MAX_HOLD_DAYS`, the forced-exit age); v2 7
  (`MAX_HOLD_DAYS_V2`, the age at which the soft penalty starts).
- `hold_penalty_coef`: per-bar reward penalty
  `coef * (share of portfolio value in lots held >= max_hold_days)`. Only used
  when `forced_exit == false`; `0.0` = off.
- `cash_penalty_coef`: rule 14's per-bar penalty
  `coef * max(0, cash/value - MAX_CASH_FRACTION)`. v1 defaults to
  `CASH_CEILING_PENALTY_COEF`; v2 to `0.0`.
- `cash_token`: v2 only — the observation carries two extra portfolio scalars
  and the policy embeds cash as one more token in its attention set, so cash is
  treated like a (pseudo-)stock holding that has its own cap and its own age.
"""
Base.@kwdef struct GameRules
    version             :: Int     = 1
    same_bar_execution  :: Bool    = false
    forced_exit         :: Bool    = true
    max_hold_days       :: Int     = MAX_HOLD_DAYS
    hold_penalty_coef   :: Float64 = 0.0
    cash_penalty_coef   :: Float64 = CASH_CEILING_PENALTY_COEF
    cash_token          :: Bool    = false
end

"""The original game. `cash_penalty` overrides rule 14's coefficient
(`nothing` keeps `CASH_CEILING_PENALTY_COEF`)."""
rules_v1(; cash_penalty::Union{Nothing, Real}=nothing) =
    GameRules(cash_penalty_coef = cash_penalty === nothing ? CASH_CEILING_PENALTY_COEF : Float64(cash_penalty))

"""Game v2: same-bar execution, no forced exit (a soft `hold_penalty` on stocks
held `MAX_HOLD_DAYS_V2`+ days instead), a cash token, and a soft `cash_penalty`.
Both penalties default to `0.0` (off)."""
rules_v2(; cash_penalty::Real=0.0, hold_penalty::Real=0.0) =
    GameRules(version=2, same_bar_execution=true, forced_exit=false, max_hold_days=MAX_HOLD_DAYS_V2,
              hold_penalty_coef=Float64(hold_penalty), cash_penalty_coef=Float64(cash_penalty), cash_token=true)

"""Width of the observation's portfolio vector under `rules` (see
`N_PORTFOLIO_SCALARS`/`N_PORTFOLIO_SCALARS_V2`)."""
n_portfolio_scalars(rules::GameRules) = rules.cash_token ? N_PORTFOLIO_SCALARS_V2 : N_PORTFOLIO_SCALARS

"""
Configuration for one simulated episode.

`candidate_universe` is the fixed, pre-capped set of tradeable symbols for this
episode (see `universe.jl`, Stage 2) — actions on any other symbol are masked
to `HOLD` by `resolve_actions`.
"""
Base.@kwdef struct EpisodeConfig
    initial_cash       :: Float64
    start_date         :: Date
    end_date           :: Date
    candidate_universe :: Vector{String}
    rules              :: GameRules = GameRules()
end

"""
Simulator state. Construct once per `InferenceCache` and reuse across many
episodes via `reset!` — this avoids re-loading the (tens-of-MB) cache per
episode during RL rollouts.

`news_hour_indices` holds hourly-bar indices at which a news event forces a
decision bar regardless of the cadence timer (rule 8's "or immediately after a
news item"). Empty by default; populated from `news_features.jl` (Stage 2).

`candidate_sym_idx` (a `Set`, for O(1) membership tests in `resolve_actions`)
and `candidate_order` (the same symbols as a `Vector`, in `config.candidate_universe`
order) are two views of the same episode-fixed universe — `candidate_order` is
what `observation.jl` uses so a given tensor column always refers to the same
stock for the whole episode.

`reward_window_start_date_idx`/`reward_window_start_value` track the current
~`REWARD_INTERVAL_DAYS`-trading-day reward window under
`TRAINING_REWARD_MODE == SPARSE_WINDOW` (see `constants.jl`): the trading-day
index and portfolio value as of the last reward checkpoint, reset by `reset!`
to the episode start and advanced by `step!` every time a window closes.

`daily_value_base_date_idx`/`daily_values` are the equivalent state for
`TRAINING_REWARD_MODE == ROLLING_WINDOW`: one portfolio-value snapshot per
distinct trading day since episode start, appended in strictly consecutive
order (the simulator never skips a trading day), which is what makes O(1)
lookup of "the value N trading days ago" possible —
`daily_values[date_idx - daily_value_base_date_idx + 1 - REWARD_INTERVAL_DAYS]`
— rather than needing a search. Maintained by `step!` regardless of which
mode is actually active (negligible cost: at most one entry per trading day,
so even a 5-year episode is only ~1,250 entries).

`cash_over_since_date_idx` is the trading-day index at which spendable cash last
rose above `MAX_CASH_FRACTION` of portfolio value and has stayed there (0 =
currently at/below the cap) — the cash analogue of a lot's `entry_date_idx`,
read by `assemble_observation!` for game v2's cash token and maintained by
`step!`.

`price_overrides` is the "instant market snapshot" mechanism: `hour_idx =>
(sym_idx => price)`, built by `news_features.jl` from 1-minute OHLCV taken at
a qualifying news event's exact timestamp, not the hourly close — see
`current_price`'s docstring (`env.jl`) for how every "price right now" read
in the simulator consults this first. Empty by default, which makes
`current_price` fall through to the plain hourly close unconditionally — the
override is purely additive, so every existing test/caller that never passes
one sees byte-for-byte the same prices as before this field existed.
"""
mutable struct TradingGameEnv
    cache             :: InferenceCache
    portfolio         :: Portfolio
    config            :: Union{Nothing, EpisodeConfig}
    current_hour_idx  :: Int
    current_date      :: Date
    end_hour_idx      :: Int
    candidate_sym_idx :: Set{Int}
    candidate_order   :: Vector{Int}
    news_hour_indices :: Set{Int}
    reward_window_start_date_idx :: Int
    reward_window_start_value    :: Float64
    daily_value_base_date_idx    :: Int
    daily_values                 :: Vector{Float64}
    price_overrides   :: Dict{Int, Dict{Int, Float32}}
    cash_over_since_date_idx :: Int
end

function TradingGameEnv(cache::InferenceCache; news_hour_indices::Set{Int}=Set{Int}(),
                         price_overrides::Dict{Int, Dict{Int, Float32}}=Dict{Int, Dict{Int, Float32}}())
    TradingGameEnv(cache, Portfolio(cash=0.0), nothing, 0, Date(1900, 1, 1), 0,
                    Set{Int}(), Int[], news_hour_indices, 0, 0.0, 0, Float64[], price_overrides, 0)
end

"""One executed trade (forced exit, voluntary sell, or buy) — `StepResult.info["trades"]`
entries and `LiveTracker.trades` entries, display/logging only, never consulted
for rule decisions. A `NamedTuple`, not a `Dict{String,Any}`: it's constructed
on essentially every bar a 60-candidate policy touches a position (measured on
a real run: several trades per bar is common, not rare), and `Dict{String,Any}`
is dramatically more expensive per instance — each `Any`-typed value is boxed
separately, so one Dict costs on the order of 15+ individual heap allocations
versus one compact allocation for a concretely-typed `NamedTuple`. JSON3
serializes both to the identical `{"kind":...,"symbol":...}` shape, so this
is transparent to `website/tradinggamelive.html`, the only external consumer."""
const TradeEvent = @NamedTuple{kind::String, symbol::String, price::Float64, quantity::Float64,
                                notional::Float64, fee::Float64, date::String, t::String,
                                entry_price::Float64, pnl::Float64, ret::Float64, days_held::Int,
                                p_hold::Float64, p_sell::Float64, p_buy::Float64}

"""`TradeEvent`'s round-trip fields, in order: `entry_price` is the lot's buy price
(a buy's own price), `pnl` the realised net profit in rupees — sale proceeds
minus fee, minus the lot's cost basis (`quantity * entry_price + entry_fee`),
zero for a buy — `ret` that profit as a fraction of the cost basis, and
`days_held` trading days between entry and exit (0 for a buy). `p_hold`/
`p_sell`/`p_buy` are the actor's probabilities for this candidate at the
event's bar; the simulator itself has no policy, so `step!` emits
`UNRECORDED_PROB` and `collect_rollout` overwrites it. A sentinel rather than
`NaN` because `JSON3` can't serialize `NaN`."""
const UNRECORDED_PROB = -1.0

"""Result of one `step!` call. `info` carries per-step diagnostics (fees paid,
forced-exit count, …) for logging — never used for rule decisions."""
struct StepResult
    reward :: Float64
    done   :: Bool
    info   :: Dict{String, Any}
end

# ── Exceptions ─────────────────────────────────────────────────────────────────────

"""
Raised by `step!` when a buy would exceed available cash. This should be
structurally impossible when actions pass through `resolve_actions` first
(rule 5's affordability check is enforced there, not here) — reaching this
error indicates a bug in the masking layer, not an invalid policy output.
"""
struct CashConstraintViolation <: Exception
    requested :: Float64
    available :: Float64
end

Base.showerror(io::IO, e::CashConstraintViolation) =
    print(io, "CashConstraintViolation: requested ", e.requested,
              ", available ", e.available,
              " — resolve_actions should have prevented this")
