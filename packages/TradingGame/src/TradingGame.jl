"""
    TradingGame

Rule-exact market simulator and hand-rolled PPO training pipeline for the
trading game defined in `TradingGameRules.txt` (repo root): start with cash,
buy/sell NSE/BSE stocks in whole-share counts, sale proceeds settle into
spendable cash after `SETTLEMENT_DAYS`, a bought lot cannot be voluntarily
sold before `MIN_HOLD_DAYS` and is force-sold at `MAX_HOLD_DAYS`, every trade
costs `FEE_RATE`, no symbol may exceed `MAX_POSITION_FRACTION` of portfolio
value (enforced at purchase time only — see `resolve_actions`'s docstring for
why organic price drift above the cap is not force-trimmed) and no more than
`n_max_holdings(N)` distinct symbols (a fraction, `N_MAX_HOLDINGS_FRACTION`,
of the candidate universe size `N`) may be held at once, spendable cash should
not exceed `MAX_CASH_FRACTION` of portfolio value (rule 14, enforced as a
reward penalty rather than a mask — see `MAX_CASH_FRACTION`'s docstring), a
sold symbol can't be newly bought again for `REBUY_COOLDOWN_DAYS` trading days
(rule 15), decisions happen at the `TRAINING_DECISION_GRANULARITY` cadence,
reward is reported every `REWARD_INTERVAL_DAYS` trading days rather than
every bar, via either of two switchable algorithms (`TRAINING_REWARD_MODE`
— `SPARSE_WINDOW` or `ROLLING_WINDOW`, see their docstring for the formula/
tradeoffs of each; decision cadence is unaffected either way), and the
objective is to maximise the total value of the portfolio, evaluated every
`REWARD_INTERVAL_DAYS` days (rule 16).

## Stages

1. Simulator + rule-compliance tests (`env.jl`, `action.jl`, `baseline_policy.jl`) — done.
2. Observation assembly + actor-critic network (`observation.jl`, `policy.jl`) — done.
3. Hand-rolled PPO + a small-universe training sanity check (`ppo.jl`, `train.jl`) — done.
4. Full-scale training + historical backtest validation — in progress:
   candidate universe (`universe.jl`, `scripts/build_market_universe_snapshot.jl`) done;
   historical news-signal backfill (`scripts/backfill_news_signals.jl`,
   `news_features.jl`) done.
5. Live execution — explicitly out of scope for this package; see
   `StockSwingPredictor.broker.jl` for the read-only Kite portfolio functions a
   future live-execution follow-up would extend (no order-placement function
   exists anywhere in this repo yet).

## Quick start: simulate

```julia
using TradingGame, StockSwingPredictor

cache = load_inference_cache("website/data/inference_cache.bson")
env   = TradingGameEnv(cache)
reset!(env, EpisodeConfig(
    initial_cash       = 1_000_000.0,
    start_date         = Date(2024, 1, 1),
    end_date           = Date(2024, 6, 30),
    candidate_universe = ["INFY", "TCS", "RELIANCE"],
))

result = step!(env, heuristic_policy(env))   # or random_policy(env), or your own JointAction
result.reward         # usually 0 — nonzero only once every REWARD_INTERVAL_DAYS trading days
                       # (plus rule 14's small per-bar cash-ceiling penalty, every step)
portfolio_value(env)
```

## Quick start: candidate universe

```julia
pool = eligible_candidates(cache, "website/data/nse_companies_latest.json")

strategy = DisjointTopMarketCap(n_train=60, n_val=20, seed=42)   # or SharedTopMarketCap(n=60),
train, val = build_universes(strategy, pool)                     # RandomUniverse(...), BucketedRandom(...)

save_universe_snapshot(strategy, train, val, "website/data/trading_game/universe_latest.json")
universe = load_universe_snapshot("website/data/trading_game/universe_latest.json")
universe.train   # candidate_universe for train_config
universe.val      # candidate_universe for val_config
```

## Quick start: train

```julia
policy = ActorCriticPolicy()
train_config = EpisodeConfig(initial_cash=1_000_000.0, start_date=Date(2024,1,1),
                              end_date=Date(2024,3,31), candidate_universe=["INFY","TCS","RELIANCE"])
policy, log = train_policy!(policy, env, train_config;
                      iterations=100, checkpoint_path="policy.bson",
                      episode_log_path="episode_log.jsonl", stop_file="STOP")
log["train_return"]   # Σ log-returns per training-rollout episode — should trend upward
```

See also: [StockSwingPredictor](@ref)
"""
module TradingGame

include("constants.jl")
include("types.jl")
include("action.jl")
include("env.jl")
include("baseline_policy.jl")
include("observation.jl")
include("news_features.jl")
include("policy.jl")
include("ppo.jl")
include("live.jl")
include("train.jl")
include("universe.jl")
include("date_windows.jl")
include("display.jl")

export
    # constants
    FEE_RATE, SETTLEMENT_DAYS, MIN_HOLD_DAYS, MAX_HOLD_DAYS, DECISION_INTERVAL_MIN,
    MAX_POSITION_FRACTION, N_MAX_HOLDINGS_FRACTION, n_max_holdings,
    MAX_CASH_FRACTION, CASH_CEILING_PENALTY_COEF, REBUY_COOLDOWN_DAYS,
    RewardMode, SPARSE_WINDOW, ROLLING_WINDOW, TRAINING_REWARD_MODE, REWARD_INTERVAL_DAYS,
    DecisionGranularity, HOURLY, MINUTE_15, TRAINING_DECISION_GRANULARITY,
    N_CANDIDATE_STOCKS, MIN_CONFIDENCE_SCORE, NEWS_DECISION_SEVERITY_THRESHOLD,
    GAMMA, GAE_LAMBDA, CLIP_EPS,
    VALUE_LOSS_COEF, ENTROPY_COEF, DECAY_HALFLIFE_HOURS, N_HOURLY_BARS_SHORT,
    N_PRICE_CHANNELS, N_MACRO_DAYS, N_MACRO_SERIES, MACRO_SERIES_NAMES,
    N_NEWS_FEATURES, N_HOLDING_FEATURES, N_PORTFOLIO_SCALARS,

    # types
    ActionType, HOLD, SELL, BUY, RawAction, ResolvedTrade, JointAction,
    Holding, ReservedCashLot, Portfolio, EpisodeConfig, TradingGameEnv, StepResult,
    CashConstraintViolation,

    # action
    resolve_actions,

    # env
    reset!, step!, portfolio_value, portfolio_breakdown, is_decision_bar, current_price,

    # baseline_policy
    random_policy, heuristic_policy,

    # observation
    MacroCache, build_macro_cache, Observation, assemble_observation, stack_observations,

    # news_features
    NewsFeatureCache, build_news_feature_cache, news_feature_fn, build_news_snapshots,
    NEWS_SNAPSHOT_MAX_LAG_MINUTES,

    # policy
    ActorCriticPolicy, save_policy, load_policy,

    # ppo
    RolloutStep, collect_rollout, compute_gae, ppo_update!,

    # live
    LiveTracker, start_episode!, make_live_callback, start_update!, make_update_callback, save_val_run!,

    # train
    train_policy!, save_policy_training_log,

    # universe
    UniverseEntry, eligible_candidates, build_candidate_universe,
    UniverseStrategy, SharedTopMarketCap, DisjointTopMarketCap, RandomUniverse, BucketedRandom,
    build_universes, save_universe_snapshot, load_universe_snapshot,

    # date_windows
    resolve_date_windows, save_date_window, load_date_window

end
