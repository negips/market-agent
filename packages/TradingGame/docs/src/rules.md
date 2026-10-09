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
    ("MAX_HOLD_DAYS_V2",          "v2 — a lot this old or older is penalised instead of force-sold"),
    ("MAX_POSITION_FRACTION",     "Rule 12 — per-stock cap, enforced at purchase time"),
    ("N_MAX_HOLDINGS_FRACTION",   "Rule 13 — distinct symbols held ≤ this × candidate-universe size"),
    ("MAX_CASH_FRACTION",         "Rule 14 — spendable cash above this share of value is penalised (and, in v2, is the cash token's cap)"),
    ("CASH_CEILING_PENALTY_COEF", "v1's default cash-penalty coefficient"),
    ("REBUY_COOLDOWN_DAYS",       "Rule 15 — a sold symbol can't be reopened for this many trading days"),
    ("REWARD_INTERVAL_DAYS",      "Reward window length"),
    ("GAMMA",                     "PPO discount per hourly step (rescaled by `bar_scaled` for 15-minute bars)"),
    ("GAE_LAMBDA",                "GAE bias/variance dial"),
    ("CLIP_EPS",                  "PPO clip range"),
    ("VALUE_LOSS_COEF",           "Critic loss weight"),
    ("ENTROPY_COEF",              "Default entropy bonus weight"),
    ("N_HOURLY_BARS_SHORT",       "Intraday bars in the encoder window (hourly or 15-minute)"),
    ("N_PRICE_CHANNELS",          "Channels per bar: normalised close, previous day's (H−L)/C"),
]
io = IOBuffer()
println(io, "| Constant | Value | Meaning |\n|:---|:---|:---|")
for (name, desc) in rows
    println(io, "| `", name, "` | `", getfield(TradingGame, Symbol(name)), "` | ", desc, " |")
end
Markdown.parse(String(take!(io)))
```

## Game versions

`EpisodeConfig(...; rules=rules_v1())` (the default) or `rules=rules_v2(; cash_penalty, hold_penalty)`
picks the rule set; `scripts/train_trading_policy.jl --game-version 2` does the same from the CLI.

| | v1 | v2 |
|:---|:---|:---|
| Fill price | next bar's close | **same bar's close** (stands in for a live instantaneous price) |
| Holding limit | forced exit at `MAX_HOLD_DAYS` | no forced exit; soft penalty `hold_penalty × (share of value in lots ≥ MAX_HOLD_DAYS_V2 days old)` per bar |
| Cash ceiling | penalty `CASH_CEILING_PENALTY_COEF × max(0, cash/value − MAX_CASH_FRACTION)` | same formula, coefficient `cash_penalty`, default `0.0` |
| Cash in the model | two portfolio scalars | an extra attention token (cash/value, reserved/value, cap utilisation, days over the cap) |
| Portfolio vector | 4 entries | 6 entries |

A v2 policy contains the cash token, so checkpoints only load under the version
they were trained with.

The rule set types — [`GameRules`](@ref), [`rules_v1`](@ref), [`rules_v2`](@ref) and
[`n_portfolio_scalars`](@ref) — are documented under [Simulator](@ref).

## Constants reference

```@autodocs
Modules = [TradingGame]
Pages   = ["/constants.jl"]
```
