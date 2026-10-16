# Rules and constants

```@meta
CurrentModule = TradingGame
```

Every rule is a named constant in `constants.jl` (the table below is generated
from the live values, so it is always current). Rule numbers refer to
`TradingGameRules.txt`.

```@eval
using Markdown, TradingGame
rows = [
    ("FEE_RATE",                  "Rule 11 — fee on both legs of every trade"),
    ("SETTLEMENT_DAYS",           "Rule 4 — sale proceeds sit in reserved cash this many trading days"),
    ("MIN_HOLD_DAYS",             "Rule 10 — lock-up before a voluntary sale"),
    ("MAX_HOLD_DAYS",             "Rule 9 (v1) — forced exit after this many trading days"),
    ("MAX_HOLD_DAYS_V2",          "v2/v3 — a lot this old or older is penalised instead of force-sold"),
    ("MAX_POSITION_FRACTION",     "Rule 12 — per-stock cap, enforced at purchase time"),
    ("N_MAX_HOLDINGS_FRACTION",   "Rule 13 — distinct symbols held ≤ this × candidate-universe size"),
    ("MAX_CASH_FRACTION",         "Rule 14 — spendable cash above this share of value is penalised (and, in v2, is the cash token's cap)"),
    ("CASH_CEILING_PENALTY_COEF", "v1's default cash-penalty coefficient"),
    ("REBUY_COOLDOWN_DAYS",       "Rule 15 — a sold symbol can't be reopened for this many trading days"),
    ("REWARD_INTERVAL_DAYS",      "v1/v2 reward window length (v3 takes `--reward-window-days`)"),
    ("ILLEGAL_PENALTY_COEF_V3",   "v3 — default charge per illegal move, as a share of portfolio value"),
    ("OBS_WINDOW_DAYS_V3",        "v3 — trading days of hourly history in the price window"),
    ("HISTORY_BARS_PER_DAY",      "Hourly history bars per session"),
    ("LOG_RETURN_SCALE",          "v3 — history returns are `LOG_RETURN_SCALE·ln(c_i/c_{i-1})`, i.e. percent"),
    ("GAMMA",                     "PPO discount per hourly step (rescaled by `bar_scaled` for 15-minute bars)"),
    ("GAE_LAMBDA",                "GAE bias/variance dial"),
    ("CLIP_EPS",                  "PPO clip range"),
    ("VALUE_LOSS_COEF",           "Critic loss weight"),
    ("ENTROPY_COEF",              "Default entropy bonus weight"),
    ("N_HOURLY_BARS_SHORT",       "Intraday bars in the encoder window (hourly or 15-minute)"),
    ("N_PRICE_CHANNELS",          "v1/v2 channels per bar: normalised close, previous day's (H−L)/C (v3 uses one)"),
]
io = IOBuffer()
println(io, "| Constant | Value | Meaning |\n|:---|:---|:---|")
for (name, desc) in rows
    println(io, "| `", name, "` | `", getfield(TradingGame, Symbol(name)), "` | ", desc, " |")
end
Markdown.parse(String(take!(io)))
```

## Game versions

`EpisodeConfig(...; rules=rules_v1())` (the default), `rules_v2(; cash_penalty, hold_penalty, illegal_penalty)` or
`rules_v3(; cash_penalty, hold_penalty, illegal_penalty, history_encoder, terminal_reward, reward_window_days)`
picks the rule set; `scripts/game/train_trading_policy.jl --game-version 1|2|3` does the same from the CLI.

| | v1 | v2 | v3 |
|:---|:---|:---|:---|
| Market / clock | NSE, hourly | any cache | **BSE, a decision every 15-minute bar** (`reset!` refuses any other cache) |
| Fill price | next bar's close | **same bar's close** (a live instantaneous price) | same as v2 |
| Holding limit | forced exit at `MAX_HOLD_DAYS` | no forced exit; soft `hold_penalty × (share of value in lots ≥ MAX_HOLD_DAYS_V2 days old)` | same as v2 |
| Cash ceiling | penalty `CASH_CEILING_PENALTY_COEF × max(0, cash/value − MAX_CASH_FRACTION)` | same formula, `cash_penalty`, default `0.0` | same as v2 |
| Price input | 120 clock bars × 2 channels, GRU | same | **70 completed hourly log-returns** (+ the current 15-minute price per stock), straight into the fusion layer |
| Macro / news | on | on | off |
| Cash in the model | two portfolio scalars | an extra attention token | no cash/portfolio token: the seven portfolio scalars sit beside every stock in fusion, and the critic reads them |
| Impossible sells | turned into HOLD after sampling | same | **masked before sampling** (conditional probabilities) |
| Other illegal moves | dropped | dropped (optionally charged) | **charged** `illegal_penalty ×` portfolio value at the decision |
| Reward | weekly `log(V/V_window_start)` + per-bar penalties | same | `(V_end − V_start)/V_start −` the window's accumulated penalties `/ V_start`, paid per reward window |
| Portfolio vector | 4 entries | 6 entries | 6 entries |
| Parameters | — | 43,301 | 29,129 |

A checkpoint only loads under the version it was trained with (the networks differ).

The rule set types — [`GameRules`](@ref), [`rules_v1`](@ref), [`rules_v2`](@ref), [`rules_v3`](@ref) and
[`n_portfolio_scalars`](@ref) — are documented under [Simulator](@ref).

## Constants reference

```@autodocs
Modules = [TradingGame]
Pages   = ["/constants.jl"]
```
