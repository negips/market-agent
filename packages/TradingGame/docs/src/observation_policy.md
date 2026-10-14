# Observation & policy

```@meta
CurrentModule = TradingGame
```

## Observation

Game v3 builds each stock's price input from two series of the inference cache: an **hourly history axis**
(the newest 70 completed bars → 70 log-returns `LOG_RETURN_SCALE·ln(c_i/c_{i-1})`) and the **15-minute clock**
(the current bar's close, as a snapshot return from the last completed hourly close). v1/v2 read a window of the
clock's own bars. See [`obs_window_bars`](@ref) and [`n_stock_features`](@ref).

```@autodocs
Modules = [TradingGame]
Pages   = ["/observation.jl"]
```

## News features

```@autodocs
Modules = [TradingGame]
Pages   = ["/news_features.jl"]
```

## Policy

v3: `Dense(70 + 8 + 7 → 64)` fusion (history returns, per-stock state including four legality inputs, the seven portfolio scalars repeated beside every
stock), attention over the N stock tokens, an actor head, and a critic that reads the mean attended token plus the
portfolio scalars (29,129 parameters), plus an auxiliary legality head (`Dense(64 → 4)` per stock) that predicts why a buy would be refused (`buy_blockers`); it is trained with `LEGALITY_LOSS_COEF` and feeds neither the action nor V(s). v1/v2: a GRU price encoder, a portfolio token (and, v2, a cash token).

```@autodocs
Modules = [TradingGame]
Pages   = ["/policy.jl"]
```
