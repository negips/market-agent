"""
TradingGame test suite.

`test_rules.jl` (Stage 1): rule-compliance tests against a small synthetic
`InferenceCache` (no real OHLCV data or sidecar required) — also defines the
`make_test_cache`/`make_test_env` fixtures reused by `test_policy.jl`.

`test_policy.jl` (Stage 2): observation-assembly and `ActorCriticPolicy`
forward-pass shape/differentiability tests; the GPU sub-test is skipped
automatically when `CUDA.functional()` is false.

`test_train.jl` (Stage 3): PPO rollout/update mechanics and `train_policy!`
conventions (checkpoint round-trip, `episode_log.jsonl`, `STOP`/`STOP_NOW`).
Fast, mechanical tests only — the actual "does the policy learn" sanity check
(episode return trending upward over many iterations) is a manual run; see
the TradingGame module docstring's quick-start.
"""

using Test, TradingGame, StockSwingPredictor, Flux, CUDA, JSON3, Dates, Random

include("test_rules.jl")
include("test_policy.jl")
include("test_train.jl")
