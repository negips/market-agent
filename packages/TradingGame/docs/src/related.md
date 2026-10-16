# Related packages

`TradingGame` is one package of the `market-agent` repo. It builds on two of the
others and deliberately keeps the coupling light.

## [StockSwingPredictor](@id StockSwingPredictor)

Provides `InferenceCache` (aligned daily and intraday price matrices that every
observation is sliced from in O(1) — for game v3 a 15-minute clock plus an hourly
history axis resampled from it, built by `scripts/ohlcv/build_cache.jl --exchange bse
--granularity 15min`), the macro-series loaders, and the Kite data client. `TradingGame` depends on it as a package; see
`packages/StockSwingPredictor/src/` — and `broker.jl` there for the read-only
portfolio functions a future live-execution follow-up would extend (no order
placement exists yet).

## CompanyConfidence

Not a package dependency. `universe.jl` reads the *precomputed* confidence
scores from `nse_companies_latest.json` or `bse_companies_latest.json` (whichever
matches the cache's exchange) to pre-filter the candidate universe
(`MIN_CONFIDENCE_SCORE`); see [`eligible_candidates`](@ref).

## NewsMonitor

Not a package dependency either. `news_features.jl` reads the classified-signal
database written by `scripts/news/backfill_news_signals.jl` directly through SQLite;
see [`build_news_feature_cache`](@ref). News is NSE-only and game v3 does not use it.
