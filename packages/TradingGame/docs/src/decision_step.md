# A decision step

```@meta
CurrentModule = TradingGame
```

A validation run is the same rollout code as a training episode, run once every
`--eval-every` iterations over the held-out window with `greedy=true` (each
candidate takes its most-likely action, nothing is learned). Its summed reward is
`val_return`, which alone decides whether `policy.bson` is checkpointed.

## Once per episode

1. [`train_policy!`](@ref) — when `iter % eval_every == 0`, tags the live tracker `val` and calls
   [`collect_rollout`](@ref)`(env, policy, val_config; greedy=true, live_cb=…)`.
2. [`collect_rollout`](@ref) — takes a CPU copy of the policy, calls [`reset!`](@ref), preallocates one observation tensor per bar.
3. [`reset!`](@ref) — cash ← initial; clears holdings, reserved cash and cooldowns; finds the first/last intraday bar; loads the candidate universe.

## Every bar (hourly or 15-minute)

| # | Call | What it does |
|:---|:---|:---|
| 1 | `assemble_observation!` | state at the **current** bar: 120-bar close window per stock (÷ first valid close) + previous day's range, news, holding state, macro, portfolio ratios |
| 2 | [`stack_observations`](@ref) | adds the batch dimension |
| 3 | `policy(…)` — [`ActorCriticPolicy`](@ref) | GRU per stock → fuse news + holding → (v2: cash token) + portfolio token → one attention pass → actor logits `(3, N)`, buy-weight logit, critic value |
| 4 | softmax → `argmax` | greedy: each candidate independently takes its most likely of HOLD/SELL/BUY; buy weight = `sigmoid(logit)`. **No rule applied yet** |
| 5 | [`step!`](@ref) | see below |
| 6 | `_annotate_trade_probs!` | stamps each trade of the bar with p(hold)/p(sell)/p(buy) |
| 7 | `RolloutStep` + `live_cb` | stored in the buffer; value curve and trades streamed to `live_status.json` |

### Inside `step!`

**v1** — advance clock → settle reserved cash → forced exits (age ≥ `MAX_HOLD_DAYS`) → [`resolve_actions`](@ref) → apply → mark value → reward − cash penalty.

**v2** — settle → [`resolve_actions`](@ref) → apply at the **current** bar's close → advance clock → settle again for the new date → mark value → reward − cash penalty − hold penalty.

[`resolve_actions`](@ref) masks, in order: SELL on a stock that is not held or still inside the lock-up → HOLD;
BUY outside the candidate universe → HOLD; BUY that would open a position inside its rebuy cooldown → HOLD;
BUYs beyond the free holdings slots dropped **at random**; surviving buys split the cash held at the start
of the bar by buy weight, net of the fee; each buy capped at `MAX_POSITION_FRACTION` of portfolio value.

## After the last bar

`val_return` = sum of per-bar rewards (telescopes to `log(V_final/V_initial)` minus penalties);
`save_val_run!` appends the run to `val_runs.jsonl`; the policy is saved if `val_return` is the best so far.

## Things worth remembering

- The network is never told which actions are blocked — masks come **after** the argmax, so a high p(sell) with no executed sell can be the lock-up/not-held mask.
- Every candidate whose argmax is BUY is bought on the same bar, splitting the cash pool.
- Training samples from the probabilities; validation is greedy, so a policy that never ranks SELL first never sells in validation.
