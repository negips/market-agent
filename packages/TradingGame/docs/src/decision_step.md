# A decision step

```@meta
CurrentModule = TradingGame
```

A validation run is the same rollout code as a training episode, run once every
`--eval-every` iterations over the held-out window with `greedy=true` (each
candidate takes its most-likely action, nothing is learned). Its summed reward is
`val_return`, which alone decides whether `policy.bson` is checkpointed. This page
follows game **v3**; v1/v2 differences are noted.

## Once per episode

1. [`train_policy!`](@ref) — when `iter % eval_every == 0`, tags the live tracker `val` and calls
   [`collect_rollout`](@ref)`(env, policy, val_config; greedy=true, step_log=…, live_cb=…)`.
2. [`collect_rollout`](@ref) — takes a CPU copy of the policy, calls [`reset!`](@ref), preallocates one observation tensor per bar (and a [`StepLog`](@ref) when requested).
3. [`reset!`](@ref) — cash ← initial; clears holdings, reserved cash, cooldowns and the penalty ledger; opens the first reward window (start value = initial cash); finds the first/last bar; loads the candidate universe; v3 refuses a cache that is not BSE 15-minute with an hourly history axis.

## Every bar (15-minute under v3)

| # | Call | What it does |
|:---|:---|:---|
| 1 | `assemble_observation!` | state at the **current** bar: per stock the 70 log-returns between the newest 71 *completed* hourly closes and the snapshot `100·ln(p_now/c_last)`; holding state; four legality inputs per stock (cooldown left, affordability, cap headroom, sellable); 7 portfolio ratios (the last is free holding slots). (v1/v2: 120-bar clock window, macro, news.) |
| 2 | [`sellable_mask`](@ref) | per stock: is there a lot that may be sold now? (held and past the lock-up). Stored for PPO. |
| 3 | [`stack_observations`](@ref) | adds the batch dimension |
| 4 | `policy(…)` — [`ActorCriticPolicy`](@ref) | per stock: 70 returns + 8 state features + the 7 portfolio scalars → `Dense 85→64` → one attention pass over the N stock tokens → actor logits `(3, N)`, buy-weight logit; the critic reads the mean attended token plus the portfolio scalars; the auxiliary legality head predicts why each buy would be refused |
| 5 | [`mask_action_logits`](@ref) → softmax → `argmax` | impossible SELL logits are removed first, so the probabilities are conditional on the move being possible; PPO's log-probabilities use the same masked distribution. Greedy: each candidate independently takes its most likely action; buy weight = `sigmoid(logit)` |
| 6 | [`step!`](@ref) | see below |
| 7 | `_annotate_trade_probs!` | stamps each trade of the bar with p(hold)/p(sell)/p(buy) |
| 8 | [`record_step!`](@ref) | when the per-step log is on: this bar's probabilities, actions, V(s), reward, book — trades or not |
| 9 | `RolloutStep` + `live_cb` | stored in the buffer; value curve and trades streamed to `live_status.json` |

### Inside `step!`

**v3 (and v2)** — settle reserved cash → [`resolve_actions`](@ref) → apply at the **current** bar's close → advance clock → settle again for the new date → mark value → add the cash/hold penalties to the ledger → pay the reward if a window ends here.

**v1** — advance clock → settle reserved cash → forced exits (age ≥ `MAX_HOLD_DAYS`) → [`resolve_actions`](@ref) → apply → mark value → reward − cash penalty.

[`resolve_actions`](@ref) handles, in order: BUY outside the candidate universe → ignored; BUY that would open a position inside its
rebuy cooldown → refused and charged; BUYs beyond the free holdings slots dropped **at random** and charged; with no cash, or not enough
for one share → refused and charged; surviving buys split the cash held at the start of the bar by buy weight, net of the fee; a buy over
`MAX_POSITION_FRACTION` of portfolio value is charged (the part within the cap fills). A "charge" adds `illegal_penalty ×` the portfolio value
at the decision to the window's ledger (rupees); under v1 refused moves are simply dropped.

### The v3 reward

0 on every bar. When a reward window ends — every `--reward-window-days` trading days, and the episode's last bar —

```math
R = \frac{V_\text{end} - V_\text{start}}{V_\text{start}} - \frac{\text{ledger}}{V_\text{start}}
```

then `V_start` ← the current value and the ledger resets. (v1/v2: the weekly `log(V/V_window_start)` minus per-bar penalties.)
PPO's discount follows ([`discount_factors`](@ref)): γ = λ = 1 for a single end-of-period window; the bar-scaled γ and λ = 1 with windows.

## After the last bar

`val_return` = sum of the window rewards (penalties already taken out — not the pure portfolio return);
`save_val_run!` writes the run to `val_runs/iter_NNNNN.json` and its summary to `val_runs/index.json`; [`save_step_log`](@ref) / [`save_step_summary`](@ref) write
`val_steps/iter_NNNNN.bson` and `.json`; the policy is saved if `val_return` is the best so far.

## Things worth remembering

- History bars enter only once complete, and a decision fills at its own bar's close — nothing after the decision is visible.
- Impossible sells cost nothing (they can't be chosen); every other refused move is paid for, but only when the reward window ends.
- Every candidate whose argmax is BUY is bought on the same bar, splitting the cash pool — with many BUYs most are over the position cap or short of a share.
- Training samples from the probabilities; validation is greedy, so a policy that never ranks SELL first never sells in validation.
- `val_return` mixes gain and penalties; the real portfolio return is in the run's value curve.
