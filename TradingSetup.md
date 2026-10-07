# Trading Setup

Everything that has to run, in order, before `train_trading_policy.jl` can start.
Steps 1–6 build the data; steps 7–9 build the training inputs. Daily steps (marked
**daily**) are the only ones you repeat routinely; the rest are one-off or occasional.

All commands run from the repo root. `--project` matters: each script uses the
package environment shown.

```
Kite login ─► sidecar ─► company list ─► confidence scores ─┐
                   └──► OHLCV collect/update ─► macro ──────┼─► inference cache ─► universe ─► (news) ─► training
                                                             │
                                          NSE announcements ─► news signals ─► 1-min snapshots
```

## 0. Session prerequisites (daily)

| Step | Command | What it does |
|---|---|---|
| Kite login | `node sidecar/kite_login.js` | Logs in to Kite Connect with the `.env` credentials + TOTP and writes `sidecar/kite_session.json`. The token lasts one trading day. Add `--account 2` to log in the second API key. |
| Sidecar | `node sidecar/server_http.js` | Starts the Node.js wrapper around Tijori Finance on port 3001. Needed by every confidence/company-list script. |

Long jobs that cross the daily token expiry re-login automatically on a 403.

## 1. Company list (occasional, per trading day you want a fresh snapshot)

```bash
julia --project=packages/CompanyConfidence scripts/generate_nse_list.jl
```
Builds `website/data/nse_companies_latest.json`: every NSE EQ-series company with closing
price, market cap and Tijori slug. Sidecar required.

Optional, for BSE:
```bash
julia --project=packages/StockSwingPredictor scripts/generate_bse_list.jl
```
Builds `bse_companies_latest.json` (Kite BSE equities joined to Tijori by BSE scripcode)
and tags companies also listed on NSE.

## 2. Confidence scores (occasional)

```bash
julia --project=packages/CompanyConfidence scripts/run_confidence_checks.jl 2305
```
Runs the five fraud/reliability signals (Beneish M-Score, cash-flow vs. net income,
promoter pledging, Tijori forensics flags) on the top N companies and writes a
`confidence` score (0–100, pass ≥ 40) into `nse_companies_latest.json`. No LLM involved;
about 6–15 s per company. The training universe is filtered on this score, so unscored
companies never enter it. Resumable.

BSE counterpart (reuses NSE scores for dual-listed companies, analyses the rest):
```bash
julia --project=packages/CompanyConfidence scripts/run_bse_confidence_checks.jl --min-mcap 100
```

## 3. OHLCV price history

First-time download, one per exchange and granularity (daily / hourly / 5min / 15min / 1min):

```bash
julia --project=packages/StockSwingPredictor scripts/collect_nse_ohlcv.jl
julia --project=packages/StockSwingPredictor scripts/collect_bse_ohlcv.jl   # optional
```
Downloads bars from Kite into `website/data/ohlcv/{nse,bse}/{granularity}/{SYMBOL}.csv`.
Use `--daily-only`, `--hourly-only`, `--1min-only`, etc. to restrict, `--refresh` to
re-download. **Training runs at hourly granularity**, so daily + hourly are the required
ones; 1-minute is only for the news-snapshot feature.

**Two jobs in parallel:** with a second Kite key, run each on different data with
`--account 1` / `--account 2` (e.g. NSE on one, BSE on the other). Don't point both at the
same exchange.

Routine upkeep (**daily**, after Kite login):
```bash
julia --project=packages/StockSwingPredictor scripts/update_ohlcv.jl
```
Appends only the bars missing since the last run, for NSE + BSE and the macro series.
`--nse-only`, `--dry-run`, `--daily-only` are available.

Optional, extend history backward (hourly bars don't exist before ~2015-02):
```bash
julia --project=packages/StockSwingPredictor scripts/backfill_ohlcv.jl --dry-run
```

## 4. Macro series (once, then yearly)

```bash
julia --project=packages/StockSwingPredictor scripts/collect_macro_ohlcv.jl
```
Downloads daily history for S&P 500, US VIX, crude, gold, silver, natural gas, copper
(Yahoo) plus India VIX and USD/INR (Kite) into `website/data/ohlcv/macro/`. This is the
macro channel of the policy's observation. `update_ohlcv.jl` keeps it current.

## 5. Inference cache (**daily**, after `update_ohlcv.jl`)

```bash
julia --project=packages/StockSwingPredictor scripts/build_cache.jl
```
Aligns every confidence-passing company's daily and hourly CSVs onto shared time axes,
forward-fills gaps, and writes `website/data/inference_cache.bson`. The simulator reads
only this file during training (no CSV reads in the rollout loop), so training sees
nothing newer than the last rebuild.

## 6. News data (only if training with news; skip with `--no-news`)

```bash
julia --project=packages/NewsMonitor scripts/fetch_nse_history.jl
julia --project=packages/NewsMonitor scripts/backfill_news_signals.jl
```
- `fetch_nse_history.jl` downloads every NSE corporate announcement since 2010 into
  `nse_announcements.db` (SQLite, resumable, ~15 min).
- `backfill_news_signals.jl` classifies those announcements with a local Ollama model
  (`qwen3:latest`, no API cost) into sentiment/severity/event type rows in
  `news_signals.db`. Scoped to the universe's symbols, so run step 7 first. Ollama must
  be running. Resumable.

## 7. Candidate universe

Pick the train/val company split. The recipe in `build_market_universe.flags` is:

```bash
julia --project=packages/TradingGame scripts/build_market_universe_snapshot.jl \
    --strategy random --n 40 --disjoint
```
Filters to confidence ≥ 40 companies present in the inference cache, then draws the train
and val candidate lists (here: 40 random companies each, disjoint). Writes
`website/data/trading_game/universe_latest.json`. Other strategies: `shared-topcap`
(default), `disjoint-topcap`, `random-bucketed`.

## 8. Training data preparation

```bash
julia --project=packages/TradingGame scripts/prepare_training_data.jl \
    --skip-news --skip-1min --train-start 2023-01-03 --train-end 2026-04-30 \
    --val-start 2026-06-01 --val-end 2026-08-31
```
(the flags above are the contents of `prepare_training.flags`). It:
1. Skips the universe if `universe_latest.json` exists (so step 7's deliberate choice is kept).
2. Resolves the train/val date windows against the **hourly** axis and saves them to
   `website/data/trading_game/date_window.json`. `train_trading_policy.jl` reads that file
   as a fallback, so the two always agree.
3. Unless skipped, runs the news backfill and the 1-minute news snapshots
   (`fetch_news_snapshot_ohlcv.jl`, Kite) — the second one gives training a real
   at-the-news-instant price instead of a stale hourly close.

`--dry-run` prints the resolved windows and commands without side effects.

## 9. Start training

```bash
julia --project=packages/TradingGame scripts/train_trading_policy.jl \
    --no-news --iterations 500 --minibatch 64 --eval-every 1 --entropy 0.01 --resume
```
(contents of `train_trading_policy.flags`; drop `--resume` for a fresh run). Watch it at
`website/tradinggamelive.html` (`./serve.sh website/tradinggamelive.html`), check progress with
`julia --project=packages/TradingGame scripts/training_status.jl`, and stop with
`touch website/data/trading_game/STOP` (saves a checkpoint) or `STOP_NOW` (no save).

## Minimum path

For a first run with no news (as in the current flags), the required steps are:

1. Kite login + sidecar
2. `generate_nse_list.jl`
3. `run_confidence_checks.jl`
4. `collect_nse_ohlcv.jl` (daily + hourly) and `collect_macro_ohlcv.jl`
5. `build_cache.jl`
6. `build_market_universe_snapshot.jl`
7. `prepare_training_data.jl --skip-news --skip-1min ...`
8. `train_trading_policy.jl --no-news ...`

Each morning before a new run: login → `update_ohlcv.jl` → `build_cache.jl`.

## Not part of this pipeline

`build_dataset.jl`, `train_model.jl`, `run_inference.jl`, `extract_llm_features.jl`,
`identify_jump_events.jl` belong to the separate swing-predictor (DualCNN) track.
`generate_earnings_watchlist.jl`, `enrich_earnings_dates.jl` and `monitor_news.jl` feed the
website and live monitoring, not training.
