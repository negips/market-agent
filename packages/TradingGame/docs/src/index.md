# TradingGame.jl

Rule-exact simulator and hand-rolled PPO pipeline for the trading game in
`TradingGameRules.txt`. Three rule sets are selectable per episode: the original
game (**v1**), **v2** (cash as a pseudo-stock, same-bar fills, soft penalties
instead of a forced exit) and **v3** (v2's rules on BSE at a 15-minute cadence, with
a simpler network, illegal moves charged, and a reward paid per window) — see
[Rules and constants](@ref).

The package also holds **v4**, which is not a game: a supervised next-minute forecaster of stock returns (a
Student-t location and scale per stock) built on the 1-minute data — see [Minute forecaster (v4)](@ref).

This site is generated from the package's docstrings by
[Documenter.jl](https://documenter.juliadocs.org); regenerate it any time with

```bash
julia --project=packages/TradingGame/docs packages/TradingGame/docs/make.jl
```

and open `packages/TradingGame/docs/build/index.html`. (The first run resolves
and precompiles the docs environment, which takes a few minutes.)

## Where to start

| You want to… | Read |
|:---|:---|
| know what every rule constant is and what v2/v3 change | [Rules and constants](@ref) |
| understand exactly what happens on one decision bar (training or validation) | [A decision step](@ref) |
| look up a function, type or constant | the **API** pages in the sidebar |
| train and read the next-minute forecaster | [Minute forecaster (v4)](@ref) |
| see how this package relates to the rest of the repo | [Related packages](@ref) |

## Module overview

```@docs
TradingGame
```

## Source map

| File | Role |
|:---|:---|
| `constants.jl` | every named rule/RL constant |
| `types.jl` | `Portfolio`, `Holding`, `GameRules`, `EpisodeConfig`, `TradingGameEnv`, … |
| `action.jl` | `resolve_actions` (refuses and charges illegal moves), `sellable_mask` / `mask_action_logits` (impossible sells, before sampling) |
| `env.jl` | `reset!` / `step!` — settlement, fills, the penalty ledger and reward windows |
| `baseline_policy.jl` | random and momentum-heuristic policies |
| `observation.jl` | observation tensors: hourly-history returns + 15-minute snapshot (v3) or the clock window (v1/v2), holding state, portfolio ratios (+ macro, news for v1/v2) |
| `policy.jl` | `ActorCriticPolicy` (v3: dense fusion + attention; v1/v2: GRU encoder, portfolio/cash tokens) |
| `ppo.jl` | rollout, GAE, clipped-surrogate update |
| `timing.jl`, `run_summary.jl` | `PhaseTimer` phase timings → `timing.json`; per-run result statistics → `val_summary.json` |
| `step_log.jl` | per-decision-step validation diagnostics (`StepLog`, `load_val_steps`) |
| `train.jl`, `live.jl` | `train_policy!`, checkpointing, `live_status.json` / `val_runs/` / `val_steps/` |
| `universe.jl`, `date_windows.jl` | train/val candidate universes and date windows |
| `news_features.jl` | recency-decayed news features + 1-minute price snapshots |
| `minute_cache.jl`, `forecast_dist.jl` | v4: the return cache at any bar size and the Student-t / Gaussian likelihoods |
| `forecaster.jl`, `forecast_train.jl` | v4: `Forecaster` (stock mixing + causal time blocks, return and timestamp heads), training and evaluation |
