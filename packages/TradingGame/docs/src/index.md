# TradingGame.jl

Rule-exact simulator and hand-rolled PPO pipeline for the trading game in
`TradingGameRules.txt`. Two rule sets are selectable per episode: the original
game (**v1**) and **v2** (cash as a pseudo-stock, same-bar fills, soft penalties
instead of a forced exit) — see [Rules and constants](@ref).

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
| know what every rule constant is and what v2 changes | [Rules and constants](@ref) |
| understand exactly what happens on one decision bar (training or validation) | [A decision step](@ref) |
| look up a function, type or constant | the **API** pages in the sidebar |
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
| `action.jl` | `resolve_actions` — the only place rules mask a policy's choice |
| `env.jl` | `reset!` / `step!` — settlement, fills, penalties, reward |
| `baseline_policy.jl` | random and momentum-heuristic policies |
| `observation.jl` | observation tensors from `InferenceCache` + macro + news + portfolio |
| `policy.jl` | `ActorCriticPolicy` (GRU encoder, attention, optional cash token) |
| `ppo.jl` | rollout, GAE, clipped-surrogate update |
| `train.jl`, `live.jl` | `train_policy!`, checkpointing, `live_status.json` / `val_runs.jsonl` |
| `universe.jl`, `date_windows.jl` | train/val candidate universes and date windows |
| `news_features.jl` | recency-decayed news features + 1-minute price snapshots |
