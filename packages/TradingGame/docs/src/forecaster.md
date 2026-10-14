# Minute forecaster (v4)

```@meta
CurrentModule = TradingGame
```

v4 is not a game and not reinforcement learning. It is a supervised transformer, [`Forecaster`](@ref), that reads
the last `window` (96) one-minute states of `N` stocks and predicts, at every minute, each stock's
**distribution** of the next-minute log-return and the next timestamp. Only stock prices go in (no news, macro or
portfolio). Every stock passes through the same weights, so the training and validation companies may differ, and so
may `N`. The layers and their matrix-form equations are on `website/architecture.html` (Forecaster_v4 tab) and
`website/mathematics.html`; every plotted quantity of a run is defined on `website/tradinggamelive.html`
(Forecaster v4 tab, "What every plotted variable means").

```bash
julia -t auto --project=packages/TradingGame scripts/build_forecaster_cache.jl --config configs/bse_v3.toml
julia --project=packages/TradingGame scripts/train_minute_forecaster.jl --config configs/bse_v3.toml --device gpu
```

Both scripts read the `[forecaster_cache]` (with `granularity = "1min" | "5min" | "15min" | "hourly" | "daily"`) and `[forecaster]` sections of the run's TOML file (`device` must be set in
`[forecaster]`, not `[train]`, which belongs to the PPO script); a command-line flag overrides the file.

## Data

[`build_forecaster_cache`](@ref) reads `website/data/ohlcv/{exchange}/{granularity}/{SYMBOL}.csv` (1-minute by default) for the universe's train and val
companies and writes `website/data/forecaster_cache_{granularity}.bson`. The axis is trading days × 375 session minutes; a day joins it
when 20% of the symbols have 10 or more session bars. Missing minutes are forward-filled and flagged *untraded*; a
row with volume above zero is *traded*. Returns are `100·ln(c_t / c_{t-1})`, clipped at ±10, then divided by their
traded-minute RMS (`cache.scale`) before the model sees them.

The cache is separate from `inference_cache.bson` (the v1–v3 game's cache) because it holds returns and a traded mask on a
fixed day × minute axis, not price levels on a decision clock.

## Output and loss

For every stock and minute the network outputs a location `μ` and a bounded log-scale `log σ` (σ between about 0.05 and
7 return units; see `LOGSIGMA_CENTER`, `LOGSIGMA_HALFRANGE`), and a timestamp description from a calendar prior plus a learned
correction.

| `loss` | What is fitted | Notes |
|:---|:---|:---|
| `student_t` (default) | location and scale, ν = `nu` (default 4) | heavy tails: outliers cost little |
| `gaussian` | location and scale | light tails: one extreme minute can dominate the gradient |
| `huber` | location only | the scale head is untrained; no likelihood is reported |

The NLL terms are weighted by `σ^(2β)` held constant (`nll_beta`, default 0.5, the β-NLL), so the model cannot lower the loss by
inflating σ where its mean is wrong. The return loss counts only where the stock traded and the target is not the first bar of a
day (the overnight gap is not a one-minute move); the timestamp adds 0.1 × MSE.

## Evaluation

[`evaluate_forecaster`](@ref) scores three sets after every epoch: `val` (val companies, val dates; it decides the checkpoint),
`val_train_companies` (train companies, val dates) and `train`. For the location it reports `skill = 1 − MSE/MSE(predict zero)`,
the repeat-last-return baseline and the sign hit rate; for the scale it reports the NLL and its gain over a constant σ per stock and
over an EWMA-of-recent-returns σ, and the coverage of the model's own 68% and 95% intervals; and it reports the timestamp accuracy and,
in `val_report.json`, the skill and the predicted-against-realised standard deviation per 15-minute slot and per stock.

One-minute returns are close to unpredictable in the mean, so the likelihood gain over the baseline scales is usually the clearest sign
that the model learned something; skill of a few percent is meaningful, and negative skill means worse than predicting no move.
Absolute NLL depends on the cache's return scale, so compare gains and skill, not raw NLL, across caches.

## Linear baseline

`scripts/ar_baseline.jl` fits a linear autoregressive model (ridge regression of the next return on the last `k` returns of the same stock,
shared across stocks, and a scale regression on the time of day and recent absolute returns, scaled by each stock's level) on the training
companies and dates, and scores it on exactly the evaluation sets of the run, with the same code ([`score_forecasts`](@ref)). It writes
`ar_baseline.json` next to `val_report.json` and prints the two side by side, so the network's skill and likelihood gain can be read against
what lagged returns alone already give.

```bash
julia --project=packages/TradingGame scripts/ar_baseline.jl --config configs/bse_v3.toml [--lags 10]
```

## Running and number types

`dtype = "float32"` (default), `"float16"` or `"float64"` sets the type of the forward and backward pass. Float16 keeps Float32 master
weights, scales the loss dynamically (a step that overflows is skipped and counted in `skipped_steps`), and computes the loss in
Float32; the checkpoint is always Float32. Float8 is not supported. `--resume` (also the `resume` key; `--no-resume` overrides it), `--eval-only` and the `STOP` / `STOP_NOW` files work as in
`train_trading_policy.jl`. `--init-from PATH` (or `init_from` in `[forecaster]`) starts a *fresh* run, epoch 1 with cleared logs, from the weights of
another checkpoint ([`warm_start_forecaster`](@ref)): the architecture (`window`, `d`, `heads`, `blocks`) must match, everything else (dropout,
learning rate, epochs) may differ, and it is exclusive with `--resume` and `--eval-only`. Outputs go to `website/data/forecaster/` (`--out-dir` changes it): `forecaster.bson` (best), `epoch_log.jsonl`,
`timing.json`, `run_config.json`, `live_status.json` and `val_report.json`. `scripts/sweep_forecaster.jl` repeats a run over a list of widths `d`
with the same seed.

## Minute cache

```@autodocs
Modules = [TradingGame]
Pages   = ["/minute_cache.jl"]
```

## Predictive distribution

```@autodocs
Modules = [TradingGame]
Pages   = ["/forecast_dist.jl"]
```

## Model

```@autodocs
Modules = [TradingGame]
Pages   = ["/forecaster.jl"]
```

## Training and evaluation

```@autodocs
Modules = [TradingGame]
Pages   = ["/forecast_train.jl"]
```

## Linear baseline (API)

```@autodocs
Modules = [TradingGame]
Pages   = ["/forecast_baseline.jl"]
```
