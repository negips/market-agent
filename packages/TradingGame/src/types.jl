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

"""Mutable portfolio state: spendable cash, cash pending settlement, open positions."""
Base.@kwdef mutable struct Portfolio
    cash     :: Float64
    reserved :: Vector{ReservedCashLot} = ReservedCashLot[]
    holdings :: Vector{Holding}         = Holding[]
end

# ── Episode / environment ─────────────────────────────────────────────────────────

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
end

function TradingGameEnv(cache::InferenceCache; news_hour_indices::Set{Int}=Set{Int}())
    TradingGameEnv(cache, Portfolio(cash=0.0), nothing, 0, Date(1900, 1, 1), 0,
                    Set{Int}(), Int[], news_hour_indices)
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
                                notional::Float64, fee::Float64, date::String, t::String}

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
