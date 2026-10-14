# Trading Setup

Everything that has to run, in order, before `train_trading_policy.jl` can start.
Steps 1–6 build the data; steps 7–9 build the training inputs. Daily steps (marked
**daily**) are the only ones you repeat routinely; the rest are one-off or occasional.

There are two tracks, selected by the cache you build in step 5:

| Track | Market · bars | Game | Needs |
|---|---|---|---|
| **v3 (current)** | BSE · a decision every 15-minute bar, hourly history | `--game-version v3` | BSE company list + confidence, BSE `daily` + `15min` OHLCV. No macro, no news. |
| v1 / v2 (earlier) | NSE · hourly bars | `--game-version 1` or `2` | NSE list + confidence, NSE `daily` + `hourly`, macro series, optionally news |

All commands run from the repo root. `--project` matters: each script uses the
package environment shown.

```
Kite login ─► sidecar ─► company list ─► confidence scores ─┐
                   └──► OHLCV collect/update ─► (macro) ────┼─► inference cache ─► universe ─► dates ─► training
                                                             │      (BSE: 15-min clock + hourly history)
                              (v1/v2 only) NSE announcements ─► news signals ─► 1-min snapshots
```

## 0. Session prerequisites (daily)

| Step | Command | What it does |
|---|---|---|
| Kite login | `node sidecar/kite_login.js` | Logs in to Kite Connect with the `.env` credentials + TOTP and writes `sidecar/kite_session.json`. The token lasts one trading day. Add `--account 2` to log in the second API key. |
| Sidecar | `node sidecar/server_http.js` | Starts the Node.js wrapper around Tijori Finance on port 3001. Needed by every confidence/company-list script. |

Long jobs that cross the daily token expiry re-login automatically on a 403. Training itself needs
neither a login nor the sidecar — only the data already on disk.

## 1. Company list (occasional, per trading day you want a fresh snapshot)

```bash
julia --project=packages/CompanyConfidence scripts/generate_nse_list.jl
julia --project=packages/StockSwingPredictor scripts/generate_bse_list.jl
```
- NSE: builds `website/data/nse_companies_latest.json` — every NSE EQ-series company with closing
  price, market cap and Tijori slug.
- BSE: builds `bse_companies_latest.json` (Kite BSE company equities joined to Tijori by BSE
  scripcode), drops G-Sec/SDL/corporate-bond records, and tags companies also listed on NSE.

Sidecar required.

## 2. Confidence scores (occasional)

```bash
julia --project=packages/CompanyConfidence scripts/run_confidence_checks.jl 2305            # NSE
julia --project=packages/CompanyConfidence scripts/run_bse_confidence_checks.jl --min-mcap 100   # BSE
```
Runs the five fraud/reliability signals (Beneish M-Score, cash-flow vs. net income,
promoter pledging, Tijori forensics flags) and writes a `confidence` score (0–100, pass ≥ 40)
into the company file. No LLM involved; about 6–15 s per company. The training universe is
filtered on this score, so unscored companies never enter it. Resumable. The BSE run reuses NSE
scores for dual-listed companies and analyses the rest.

## 3. OHLCV price history

First-time download, one per exchange and granularity (daily / hourly / 5min / 15min / 1min):

```bash
julia --project=packages/StockSwingPredictor scripts/collect_nse_ohlcv.jl
julia --project=packages/StockSwingPredictor scripts/collect_bse_ohlcv.jl
```
Downloads bars from Kite into `website/data/ohlcv/{nse,bse}/{granularity}/{SYMBOL}.csv`.
Use `--daily-only`, `--hourly-only`, `--15min-only`, `--1min-only`, etc. to restrict, `--refresh` to
re-download.

- **v3 needs BSE `daily` and `15min`** only. BSE `hourly` is not required: the hourly history
  axis is *resampled from the 15-minute bars* (checked identical to Kite's own hourly closes),
  because the BSE hourly files cover only about a quarter of the companies.
- v1/v2 need NSE `daily` + `hourly`. 1-minute is only for the news-snapshot feature.

**Two jobs in parallel:** with a second Kite key, run each on different data with
`--account 1` / `--account 2` (e.g. NSE on one, BSE on the other). Don't point both at the
same exchange, and don't run `update_ohlcv.jl` for an exchange while a collector for the same
granularity is still writing it.

Routine upkeep (**daily**, after Kite login):
```bash
julia --project=packages/StockSwingPredictor scripts/update_ohlcv.jl
```
Appends only the bars missing since the last run, for NSE + BSE and the macro series.
`--nse-only`, `--bse-only`, `--dry-run`, `--daily-only` are available.

Optional, extend history backward (hourly bars don't exist before ~2015-02; BSE 15-minute starts
2019-01):
```bash
julia --project=packages/StockSwingPredictor scripts/backfill_ohlcv.jl --dry-run
```

## 4. Macro series (v1/v2 only; once, then yearly)

```bash
julia --project=packages/StockSwingPredictor scripts/collect_macro_ohlcv.jl
```
Downloads daily history for S&P 500, US VIX, crude, gold, silver, natural gas, copper
(Yahoo) plus India VIX and USD/INR (Kite) into `website/data/ohlcv/macro/`. This is the
macro channel of the v1/v2 policy's observation; **game v3 ignores it** (not loaded, no layers).

## 5. Inference cache (**daily**, after `update_ohlcv.jl`)

```bash
# v3: BSE, 15-minute clock + hourly history axis
julia --project=packages/StockSwingPredictor scripts/build_cache.jl --exchange bse --granularity 15min --min-mcap 500

# v1/v2: NSE, hourly
julia --project=packages/StockSwingPredictor scripts/build_cache.jl
```
Aligns every confidence-passing company's bars onto shared time axes, forward-fills gaps, and
writes `website/data/inference_cache.bson` (it **overwrites** the previous cache). The simulator reads
only this file during training (no CSV reads in the rollout loop), so training sees nothing newer than
the last rebuild.

- The cache records its exchange and bar length, and every later step reads them from it.
- `--history resample|hourly|none` (default `resample` with `--granularity 15min`) controls the hourly
  history axis; v3 refuses a cache without it. Bars outside 09:15–15:15 (e.g. BSE Muhurat sessions) are dropped.
- `--min-mcap CR` matters on BSE: most small caps barely trade, so their 15-minute bars are mostly
  forward-filled and not realistically tradable at those prices. `500` keeps about 1,900 companies; no
  filter means about 4,700 companies and ~1 GB. A positional `N` keeps only the top N by market cap.
- A checkpoint trained on one cache cannot be resumed on another (different bar length / inputs):
  start a fresh run after switching.

## 6. News data (v1/v2 only; skip with `--no-news`)

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

News is NSE-only (NSE announcements keyed by NSE symbols), and **game v3 does not use it**.

## 7. Candidate universe

Pick the train/val company split. The recipe is the `[universe]` section of `configs/bse_v3.toml`:

```bash
julia --project=packages/TradingGame scripts/build_market_universe_snapshot.jl --config configs/bse_v3.toml
# the same thing as flags: --strategy random --n 40 --disjoint
```
Filters to confidence ≥ 40 companies present in the inference cache (reading the company file of the
cache's exchange), then draws the train and val candidate lists (here: 40 random companies each,
disjoint). Writes `website/data/trading_game/universe_latest.json` and **overwrites** the previous one —
rebuild it after changing the cache's exchange. Other strategies: `shared-topcap` (default),
`disjoint-topcap`, `random-bucketed`. `--min-mcap CR` drops small companies before the strategy runs;
`--seed N` makes the draw reproducible.

## 8. Training date windows

```bash
julia --project=packages/TradingGame scripts/prepare_training_data.jl --config configs/bse_v3.toml
```
(reads `[dates]`, `[prepare]` and `[universe]`; the flags `--skip-news --skip-1min --train-start …` still work and override the file). It:
1. Skips the universe if `universe_latest.json` exists (so step 7's deliberate choice is kept).
2. Resolves the train/val date windows against the cache's **intraday clock axis** and saves them to
   `website/data/trading_game/date_window.json`. `train_trading_policy.jl` reads that file
   as a fallback, so the two always agree. v3 needs about 10 trading days of history before
   `--train-start` (the 70-bar window).
3. For NSE caches, unless skipped, runs the news backfill and the 1-minute news snapshots
   (`fetch_news_snapshot_ohlcv.jl`, Kite). For a BSE cache both stages are skipped automatically.

`--dry-run` prints the resolved windows and commands without side effects.

## 9. Start training

```bash
julia --project=packages/TradingGame scripts/train_trading_policy.jl --config configs/bse_v3.toml
```
(reads `[dates]`, `[game]` and `[train]`; every setting is a flag too, e.g. `--device gpu --iterations 5` overrides the file for one run. Add `--resume` only to continue an existing run of the same
version — a first v3 run has no checkpoint, and v1/v2 checkpoints don't fit the v3 network).

The flags that define v3:
- `--game-version v3` — v2's rules on BSE 15-minute bars; errors out on any other cache.
- v3 shows the network why a move would be refused: four legality inputs per stock (rebuy cooldown left, affordability of one share, position-cap headroom, sellable), a free-slots portfolio scalar, and an auxiliary legality head trained against the rules' own answer (`legality_loss` in `episode_log.jsonl`). Checkpoints from before this (28,485 parameters) cannot be used with `--init-from`/`--resume`.
- `--illegal-penalty X` — each illegal move costs X × the portfolio value at the decision (default 0.01).
  Impossible sells are masked before the policy samples and cost nothing.
- `--reward-mode terminal|stepwise` (default terminal) and `--reward-window-days N` (default 0): the reward is
  0 on every bar and, at the end of each N-trading-day window (0 = the whole period), the window's gain minus
  its accumulated penalties, both over the portfolio value at the window's start. With 0, PPO uses discount 1;
  with N > 0 the bar-scaled discount and λ = 1.
- `--history-encoder direct|gru` — the 70 hourly returns go straight into the fusion layer (default) or through a GRU.
- `--val-steps-every N` (default 1; 0 off) — write the full per-decision-bar log of every Nth validation run.
- v1/v2 flags: `--game-version 1|2`, `--cash-penalty X`, `--hold-penalty X`.

Expect roughly a minute per iteration on 40 stocks for a 3-year 15-minute training window (about 22,000 bars),
mostly the PPO update; the GPU only accelerates that update.

Watch it at `website/tradinggamelive.html` (`./serve.sh website/tradinggamelive.html`), check progress with
`julia --project=packages/TradingGame scripts/training_status.jl`, and stop with
`touch website/data/trading_game/STOP` (saves a checkpoint) or `STOP_NOW` (no save).

### Studying what it learned

Every validation run also writes `website/data/trading_game/val_steps/iter_NNNNN.bson` (the full per-bar,
per-stock log: probabilities, chosen actions, V(s), book) plus a small `iter_NNNNN.json` the live page reads
to list the bars on which nothing was traded. Summarise one with

```bash
julia --project=packages/TradingGame scripts/val_steps_summary.jl            # newest run
julia --project=packages/TradingGame scripts/val_steps_summary.jl 42         # iteration 42
```
A fresh (non-`--resume`) run clears the previous run's `episode_log.jsonl`, `val_runs/` and `val_steps/`.

## Minimum path (v3)

1. Kite login + sidecar (only if the data or lists need refreshing)
2. `generate_bse_list.jl` and `run_bse_confidence_checks.jl`
3. `collect_bse_ohlcv.jl` (daily + 15-minute)
4. `build_cache.jl --exchange bse --granularity 15min --min-mcap 500`
5. `build_market_universe_snapshot.jl --strategy random --n 40 --disjoint`
6. `prepare_training_data.jl --skip-news --skip-1min ...`
7. `train_trading_policy.jl --game-version v3 ...`

Each morning before a new run: login → `update_ohlcv.jl` → `build_cache.jl`.

## Minimum path (v1/v2, NSE)

Kite login + sidecar → `generate_nse_list.jl` → `run_confidence_checks.jl` → `collect_nse_ohlcv.jl`
(daily + hourly) and `collect_macro_ohlcv.jl` → `build_cache.jl` → `build_market_universe_snapshot.jl` →
`prepare_training_data.jl --skip-news --skip-1min ...` → `train_trading_policy.jl --no-news ...`

## Not part of this pipeline

`build_dataset.jl`, `train_model.jl`, `run_inference.jl`, `extract_llm_features.jl`,
`identify_jump_events.jl` belong to the separate swing-predictor (DualCNN) track — which still expects the
hourly NSE cache, so don't point it at a BSE 15-minute cache.
`generate_earnings_watchlist.jl`, `enrich_earnings_dates.jl` and `monitor_news.jl` feed the
website and live monitoring, not training.
