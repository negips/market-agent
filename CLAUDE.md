# market-agent

Self-improving trading agent for Indian stock markets (NSE/BSE), built entirely in Julia.

## Project goal

Identify stocks likely to make large price moves (swing prediction), using a
multi-source data pipeline: OHLCV price history, a company fraud/confidence filter,
real-time news signals classified by an LLM, and a CNN-based swing predictor.
Eventually self-improves by analyzing its own trade history.

## Repository layout

```
market-agent/
├── website/                        # browser-based UI (served via serve.sh)
│   ├── index.html                  # redirects to watchlist.html
│   ├── setup.html                  # environment checklist + roadmap
│   ├── companies.html              # full NSE company list with confidence scores
│   ├── watchlist.html              # upcoming earnings × confidence filter
│   └── assets/
│       ├── style.css               # shared design system (dark theme, sidebar layout)
│       ├── nav.js                  # sidebar navigation component
│       └── utils.js                # shared formatters (₹, confidence badges, dates)
│
├── sidecar/                        # Node.js HTTP wrapper around Tijori Finance
│   ├── server_http.js              # Express server — exposes Tijori tools as REST
│   ├── kite_login.js               # Daily Kite Connect OAuth + TOTP token acquisition
│   ├── decode_migration_qr.js      # One-off: decode Google Authenticator migration QR
│   ├── package.json
│   └── tijori-finance-mcp/         # clone from github.com/LaZZy0v0/tijori-finance-mcp
│
├── packages/                       # standalone Julia packages
│   ├── TijoriData/                 # Tijori Finance data client
│   ├── CompanyConfidence/          # Fraud/reliability scoring module
│   ├── EarningsCalendar/           # NSE earnings event fetcher (no sidecar)
│   ├── NewsMonitor/             # BSE + RSS news poller → LLM-classified signals
│   │   ├── Project.toml
│   │   └── src/
│   │       ├── NewsMonitor.jl      # module entry + exports
│   │       ├── types.jl            # NewsItem, NewsSignal, PollerConfig
│   │       ├── nse.jl              # NSE corporate announcements API (20+ yr history)
│   │       ├── bse.jl              # BSE corporate announcements RSS (today only)
│   │       ├── rss.jl              # RSS feed fetcher + XML parser
│   │       ├── llm_classify.jl     # Claude API → NewsSignal (symbol, sentiment, severity)
│   │       └── poller.jl           # concurrent polling loop + JSONL writer
│   ├── StockSwingPredictor/     # Neural network large-move predictor + broker client
│   │   ├── Project.toml
│   │   └── src/
│   │       ├── StockSwingPredictor.jl  # module entry + exports
│   │       ├── types.jl               # all structs (LLMFeatures, TrainingExample, Dataset, …)
│   │       ├── kite_data.jl           # instrument lookup, daily + hourly OHLCV fetch + cache
│   │       ├── macro_data.jl          # macro instrument OHLCV: Yahoo Finance + Kite CDS/NSE
│   │       ├── inference_cache.jl     # InferenceCache: aligned price matrices for O(1) batch slicing
│   │       ├── broker.jl              # Kite portfolio/funds: get_holdings, get_positions, get_margins, get_orders
│   │       ├── fundamentals.jl        # quarterly P&L feature extraction via TijoriData (not active)
│   │       ├── llm_extract.jl         # Claude API → 15 scalar signals from PDFs
│   │       ├── features.jl            # TS derived features, vector assembly
│   │       ├── dataset.jl             # sliding-window examples, normalisation, split
│   │       ├── model.jl               # Flux.jl DualCNN, save/load
│   │       ├── train.jl               # training loop, early stopping, chunked eval
│   │       └── display.jl             # Base.show overrides
│   └── TradingGame/             # RL-trained portfolio trading policy (simulator + training)
│       ├── Project.toml
│       └── src/
│           ├── TradingGame.jl      # module entry + exports
│           ├── constants.jl        # FEE_RATE, SETTLEMENT_DAYS, MIN_HOLD_DAYS, MAX_HOLD_DAYS, …
│           ├── types.jl            # Portfolio, Holding, ReservedCashLot, TradingGameEnv, …
│           ├── action.jl           # joint action space + cash-constraint/lock-up masking
│           ├── env.jl              # reset!/step! — exact TradingGameRules.txt enforcement
│           ├── baseline_policy.jl  # random + momentum-heuristic policies (rule-compliance validation)
│           ├── observation.jl      # obs tensor assembly: InferenceCache + MacroCache + news + portfolio
│           ├── policy.jl           # recurrent/transformer actor-critic (ActorCriticPolicy, Flux)
│           ├── ppo.jl              # hand-rolled PPO + GAE (collect_rollout, ppo_update!)
│           ├── live.jl             # streams the current episode to live_status.json (website/tradinggamelive.html)
│           ├── train.jl            # training loop: train_policy! — checkpoint, episode_log.jsonl, STOP
│           ├── universe.jl         # candidate universe: confidence-filtered pool + pluggable train/val split strategies
│           ├── date_windows.jl     # resolve_date_windows: shared by train_trading_policy.jl + prepare_training_data.jl
│           ├── news_features.jl    # NewsFeatureCache: decayed news_fn + 1-minute instant-price snapshots at news bars
│           └── display.jl          # Base.show overrides
│       └── docs/                   # LaTeX architecture write-ups (.tex + .bib + built .pdf, all tracked)
│           └── tradinggame_scaling.tex  # anchor-token cross-attention scaling proposal — equations,
│                                         # TikZ schematics, measured-GPU-memory regression, references
│
├── scripts/                        # standalone Julia scripts (not packages)
│   ├── generate_nse_list.jl              # builds data/nse_companies_latest.json
│   ├── run_confidence_checks.jl          # runs CompanyConfidence on top-N by market cap
│   ├── enrich_earnings_dates.jl          # projects next earnings date via Tijori history (run every 2 weeks)
│   ├── generate_earnings_watchlist.jl    # merges NSE calendar + projections → watchlist JSON
│   ├── kite_relogin.jl                   # force a fresh Kite login from Julia (shells out to kite_login.js)
│   ├── collect_nse_ohlcv.jl              # download all NSE OHLCV: daily/hourly/5min/15min/1min (Kite)
│   ├── collect_bse_ohlcv.jl              # download all BSE OHLCV: daily/hourly/5min/15min/1min (Kite)
│   ├── collect_macro_ohlcv.jl            # download macro instrument OHLCV (Yahoo + Kite CDS/NSE)
│   ├── update_ohlcv.jl                   # incremental update: append only missing bars since last run
│   ├── backfill_ohlcv.jl                 # extend existing OHLCV CSVs backward to an earlier --from date
│   ├── extract_llm_features.jl           # Claude API → 14 scalar signals per company (resumable)
│   ├── monitor_news.jl                   # real-time BSE + RSS news monitor daemon
│   ├── backfill_news_signals.jl          # classify historical NSE announcements via local Ollama (resumable)
│   ├── fetch_news_snapshot_ohlcv.jl      # fetch 1-min OHLCV from Kite for news-event days only (resumable)
│   ├── build_cache.jl                    # build inference_cache.bson from OHLCV CSVs (--exchange nse|bse, --granularity hourly|15min)
│   ├── build_dataset.jl                  # sliding-window dataset assembly; --pred-hours 35|70
│   ├── train_model.jl                    # train SwingPredictor (v1/v2/v3); auto-selects dataset
│   ├── build_market_universe_snapshot.jl # TradingGame candidate universe: confidence-filtered, pluggable train/val split
│   ├── prepare_training_data.jl          # orchestrates universe + news backfill + 1-min snapshots, one date resolution
│   ├── train_trading_policy.jl           # drives TradingGame.train_policy! — see website/tradinggamelive.html
│   └── training_status.jl                # one-shot read-only snapshot of a running train_trading_policy.jl
│
│   ├── data/                           # generated artifacts (gitignored)
│   │   ├── nse_companies_latest.json   # latest snapshot (read by companies.html)
│   │   ├── nse_companies_YYYYMMDD.json # dated snapshots
│   │   └── earnings_watchlist_latest.json # read by watchlist.html
│
├── nse_companies.html              # browser viewer — search, sort, confidence badges
├── serve.sh                        # start local HTTP server and open browser
└── CLAUDE.md                       # this file
```

## Language

**Everything is Julia** except the sidecar, which must be Node.js because
`tijori-finance-mcp` uses Playwright for browser automation and has no Julia equivalent.
Do not introduce Python.

## Code conventions

- All public functions have Documenter.jl-compatible docstrings (triple-quoted, with
  `# Arguments`, `# Returns`, `# Example` sections)
- No inline comments unless the WHY is genuinely non-obvious
- Structs go in `types.jl`; `Base.show` overrides go in `display.jl`
- Each package is independently usable from the Julia REPL — do not create tight
  coupling between packages
- Named constants instead of magic numbers (e.g. `const BENEISH_THRESHOLD = -1.78`)
- Explicit error types (subtype `Exception`) rather than generic `error()` strings
  where the caller may want to catch specific failures

## Documentation

Every package has its own `docs/` directory, added as packages grow — for detailed
architecture write-ups (design proposals, complexity analysis, anything that wants
equations/diagrams/citations), not day-to-day API docs. Written in LaTeX; `.tex`,
`.bib`, and the built `.pdf` are all tracked in git (unlike `papers/`, which is
downloaded reference material and is gitignored). First instance:
`packages/TradingGame/docs/tradinggame_scaling.tex`. `TradingGame` also has a
Documenter.jl site generated from its docstrings (`docs/make.jl`, `docs/src/*.md`,
`docs/Project.toml`; output in the gitignored `docs/build/`) — build it with
`julia --project=packages/TradingGame/docs packages/TradingGame/docs/make.jl` and open
`packages/TradingGame/docs/build/index.html`. The API pages are `@autodocs` blocks per
source file, so new docstrings appear automatically; add a new source file to the
matching `docs/src/*.md` page. Module-level docstrings describe
each package's role in the pipeline and link to related packages with
`See also: [OtherModule](@ref)`.

## Package dependency rules

```
TijoriData              — data only, no trading logic
CompanyConfidence       — depends on TijoriData
EarningsCalendar        — NSE data only, no dependencies on other packages
NewsMonitor             — BSE/RSS news polling + LLM classification; no dependencies on other packages
StockSwingPredictor     — depends on TijoriData; Kite used directly via HTTP
                          includes broker.jl (portfolio, positions, funds, orders)
TradingGame             — depends on StockSwingPredictor (InferenceCache, macro_data)
                          and CompanyConfidence (universe pre-filter, read as
                          precomputed JSON, not a package dependency — see
                          universe.jl). No package dependency on NewsMonitor:
                          news_features.jl reads scripts/backfill_news_signals.jl's
                          news_signals.db directly via SQLite.jl, the same
                          light-coupling pattern. Simulator + RL training only — no
                          order placement (see StockSwingPredictor/broker.jl for
                          the future live-execution follow-up)
```

## Using CompanyConfidence

```julia
using CompanyConfidence, TijoriData

TijoriData.start!()                          # start the sidecar first

report = analyze("infosys-limited")
report.score   # 0–100 (higher = more trustworthy)
report.pass    # false if score < 40

# Individual signals
report.beneish.m_score          # Beneish M-Score (nothing for banks)
report.cashflow.years_cfo_lt_ni # years where CFO < Net Income
report.pledging.latest_pct      # promoter pledging %
report.forensics.flags          # Tijori red flags
report.surveillance.on_asm      # on NSE ASM list?

# Signals work standalone (no sidecar needed)
pl = get_financials("satyam-computer-services-limited", :pl)
bs = get_financials("satyam-computer-services-limited", :bs)
cf = get_financials("satyam-computer-services-limited", :cf)
beneish_score(pl, bs, cf)

# Skip NSE network check (offline / testing)
analyze("yes-bank-limited"; check_surveillance=false)
```

### One-time setup for CompanyConfidence

```julia
using Pkg
Pkg.develop(path="packages/TijoriData")
Pkg.develop(path="packages/CompanyConfidence")
```

## Using EarningsCalendar

```julia
using EarningsCalendar, Dates

# No sidecar needed — hits NSE directly
events = upcoming_earnings()          # next 30 days
events = upcoming_earnings(7)         # next week
events = fetch_earnings_calendar(Date(2026, 10, 1), Date(2026, 10, 31))

events[1].symbol    # "INFY"
events[1].company   # "Infosys Limited"
events[1].date      # Date(2026, 10, 17)
events[1].purpose   # "Quarterly Results"

# Filter to symbols with slugs for CompanyConfidence
symbols = [e.symbol for e in events]
```

### One-time setup for EarningsCalendar

```julia
using Pkg
Pkg.develop(path="packages/EarningsCalendar")
```

## Sidecar setup (one-time)

```bash
cd sidecar
git clone https://github.com/LaZZy0v0/tijori-finance-mcp.git
cd tijori-finance-mcp && node setup.js   # opens browser for Tijori login
cd ..
npm install                              # installs express
```

## Using NewsMonitor

```julia
using NewsMonitor

# Run manually from the REPL
items = fetch_nse_announcements()                    # today's NSE corporate announcements
items = fetch_bse_announcements()                    # today's BSE corporate announcements (RSS, today only)
items = fetch_rss("https://economictimes.indiatimes.com/markets/rss.cms", "ET")

sig = classify_item(items[1]; api_key=ENV["ANTHROPIC_API_KEY"])
sig.symbol      # "RELIANCE"
sig.event_type  # "results"
sig.sentiment   # +0.8
sig.severity    # 0.9
sig.summary     # "Reliance Q2 PAT beats estimates by 12% — strong refining margins"
```

### Run the live daemon

```bash
# Polls BSE every 60s + 3 RSS feeds every 5 min, classifies with Claude Haiku
julia scripts/monitor_news.jl

# Flags
julia scripts/monitor_news.jl --all-hours   # don't restrict to 09:00–16:30 IST
julia scripts/monitor_news.jl --bse-only    # skip RSS, BSE announcements only
```

Output: `website/data/news_signals.jsonl` — one JSON line per classified item.

### Build the historical announcements database

```bash
# Fetch all NSE corporate announcements from 2010 to yesterday (~830 API calls, ~15 min)
julia --project=packages/NewsMonitor scripts/fetch_nse_history.jl

# Custom range or delay
julia --project=packages/NewsMonitor scripts/fetch_nse_history.jl --from 2015-01-01 --delay 0.5

# Resumable — re-run after interruption to continue from last completed week
julia --project=packages/NewsMonitor scripts/fetch_nse_history.jl
```

Output: `website/data/nse_announcements.db` (SQLite)
Schema: `announcements(guid, symbol, an_dt, desc, attchmnt_text, attchmnt_file, has_xbrl, raw_json)`
Query example:
```julia
using SQLite
db = SQLite.DB("website/data/nse_announcements.db")
rows = collect(DBInterface.execute(db,
    "SELECT symbol, an_dt, desc, attchmnt_text FROM announcements
     WHERE symbol = ? AND an_dt >= ? ORDER BY an_dt",
    ["TCS", "2024-01-01 00:00:00"]))
```

### One-time setup

```julia
using Pkg; Pkg.develop(path="packages/NewsMonitor")
```

## Using broker functions (StockSwingPredictor)

```julia
using StockSwingPredictor, DataFrames

# Requires a valid kite_session.json — run kite_login.js first
session = load_kite_session(pwd())

# Long-term demat holdings
holdings = get_holdings(session)
# → DataFrame: symbol, exchange, isin, quantity, average_price,
#              last_price, close_price, pnl, day_change, day_change_pct

# Open intraday and overnight positions
positions = get_positions(session)             # net view (default)
positions = get_positions(session; kind=:day)  # intraday only

# Available funds
margins = get_margins(session)                         # equity segment (default)
margins = get_margins(session; segment=:commodity)
margins.cash             # uninvested cash (₹)
margins.net              # total available including collateral
margins.debits           # margin currently utilised

# Today's order book
orders = get_orders(session)
# → DataFrame: order_id, symbol, exchange, transaction_type, product,
#              quantity, price, status, filled_quantity, placed_at
```

Note: the Kite Historical API session (`kite_session.json`) is a full Kite Connect
session and works for all endpoints — no separate trading API key is needed.

## Daily session workflow

```bash
# 1. Acquire a fresh Kite Connect access token (valid for the trading day)
node sidecar/kite_login.js              # reads .env, writes sidecar/kite_session.json
# equivalently, from Julia:
julia --project=packages/StockSwingPredictor scripts/kite_relogin.jl

# 2. Start the sidecar (Julia will start it automatically via start!(), but you can also run it manually)
node sidecar/server_http.js              # default port 3001
PORT=3002 node sidecar/server_http.js   # custom port
```

### Two API keys, two parallel jobs

Kite's rate limits are per API key. `.env` can hold a second Kite Connect app
(`KITE_HISTORICAL2_API_KEY`/`_SECRET`; a third would be `KITE_HISTORICAL3_*`),
selected with `--account N` (default 1) on `collect_nse_ohlcv.jl`,
`collect_bse_ohlcv.jl`, `update_ohlcv.jl`, `backfill_ohlcv.jl`,
`fetch_news_snapshot_ohlcv.jl` and `kite_relogin.jl`. Same trading user
(id/password/TOTP); each account has its own session file
(`sidecar/kite_session.json`, `kite_session2.json`, …, all gitignored) and its
own automatic 403 relogin. Log in once per account per day:

```bash
node sidecar/kite_login.js              # account 1
node sidecar/kite_login.js --account 2  # account 2
julia --project=packages/StockSwingPredictor scripts/collect_nse_ohlcv.jl --1min-only --account 1
julia --project=packages/StockSwingPredictor scripts/collect_bse_ohlcv.jl --1min-only --account 2
```

Run the two jobs on **different data** — different exchanges or granularities.
Two jobs writing the same CSVs or the same `earliest_trading_day.json`/
`confirmed_floor.json` files (e.g. both `update_ohlcv.jl` runs on the same
exchange) would race on them.

### Kite token expiry during long-running jobs

The Kite access token is valid for one trading day. `load_kite_session`
returns a mutable `KiteSession` (not a plain tuple), and every Kite call in
`StockSwingPredictor` (`fetch_ohlcv*`, `load_instruments`, `broker.jl`'s
portfolio/positions/margins/orders, `macro_data.jl`'s Kite-sourced series)
goes through one shared `_kite_get` helper instead of raw `HTTP.get`. On a
403 (expired token), `_kite_get` automatically runs `relogin_kite!` —
shells out to `sidecar/kite_login.js` (Playwright-driven; Kite's login has
no plain REST endpoint) and mutates the session's `api_key`/`access_token`
**in place** — then retries the one failed request. Because the session
object is mutable and shared, every other call anywhere in the same script
immediately sees the refreshed token too, not just the retried one.

This means an overnight job (`collect_nse_ohlcv.jl`, `update_ohlcv.jl`,
`backfill_ohlcv.jl`, `fetch_news_snapshot_ohlcv.jl`, …) that spans the daily
expiry boundary recovers on its own, mid-run — **no script-level changes
are needed anywhere**; they all already just load one `session` at startup
and pass it through every call. Requires a display (the relogin launches a
real, visible browser) and `.env` credentials present, same as running
`node sidecar/kite_login.js` directly. To force a refresh proactively
instead of waiting for a 403 — e.g. right before starting a job you know
will run long — use `scripts/kite_relogin.jl` (see below).

Two failure modes this does NOT turn into a retry loop:
- **Network actually down**: a connection failure throws straight out of
  `HTTP.get` before `_kite_get` ever sees a status code, so relogin is
  never considered at all — it propagates to the calling function's
  existing per-chunk `try/catch` (warn, skip, continue to the next chunk),
  exactly like any other transient network error, unrelated to Kite tokens.
- **Login itself is broken** (not just an expired token, but e.g. the
  network drops mid-relogin, TOTP timing drifts, Zerodha changes their
  login page): a 403 means Kite's server DID respond, so relogin is
  attempted — but at most once per `KITE_RELOGIN_COOLDOWN_SECONDS` (60s),
  globally, regardless of how many chunks/symbols/functions hit 403 in that
  window. A multi-year, multi-symbol backfill hitting hundreds of 403s from
  a genuinely broken login degrades to one logged relogin attempt followed
  by "skipping relogin, cooldown" warnings and normal per-chunk failures —
  never a cascade of repeated browser launches.

## Using TijoriData in the Julia REPL

```julia
using TijoriData

TijoriData.start!("/path/to/market-agent/sidecar")   # starts Node.js sidecar
is_running()                                          # verify

results = search_company("HDFC Bank")
ov  = get_overview(results[1].slug)
pl  = get_financials(results[1].slug, :pl)            # DataFrame
sh  = get_shareholding(results[1].slug)               # DataFrame with pledging
kb  = get_knowledge_base(results[1].slug)
doc = fetch_document(kb.conference_calls[1].url)      # full PDF text
```

## Running tests

```bash
# Unit tests only (no sidecar required)
cd packages/TijoriData
julia --project=. test/runtests.jl

# With integration tests (requires running sidecar)
TIJORI_SIDECAR_DIR=/path/to/market-agent/sidecar julia --project=. test/runtests.jl
```

## Scripts

Scripts live in `scripts/` and are run directly with `julia` — they are not packages.
All scripts accept `--help` / `-h`.

### generate_nse_list.jl

Builds the JSON snapshot of all NSE-listed EQ companies with closing price, market cap,
and Tijori slug. Run once per trading day before `run_confidence_checks.jl`.

```bash
# Sidecar must be running first
julia scripts/generate_nse_list.jl              # uses today's date
julia scripts/generate_nse_list.jl 2026-09-07  # explicit date (for past bhavcopy)
```

Writes `website/data/nse_companies_YYYYMMDD.json` and updates `website/data/nse_companies_latest.json`.

### run_confidence_checks.jl

Runs all 5 CompanyConfidence signals on the top N companies by market cap and merges
the results back into `data/nse_companies_latest.json` as a `"confidence"` key.
No LLMs involved — purely arithmetic and Tijori data.

```bash
# Sidecar must be running first; use CompanyConfidence project
julia --project=packages/CompanyConfidence scripts/run_confidence_checks.jl        # top 100
julia --project=packages/CompanyConfidence scripts/run_confidence_checks.jl 500   # top 500
julia --project=packages/CompanyConfidence scripts/run_confidence_checks.jl 2305  # all with slugs
```

Estimated time: ~6–15 seconds per company via the Tijori sidecar.

### generate_earnings_watchlist.jl

Fetches upcoming NSE earnings events and joins them with confidence data from
`nse_companies_latest.json`. Only companies with confidence ≥ 40 appear.
Writes `data/earnings_watchlist_latest.json` consumed by `website/watchlist.html`.

```bash
julia --project=packages/EarningsCalendar scripts/generate_earnings_watchlist.jl        # 30 days
julia --project=packages/EarningsCalendar scripts/generate_earnings_watchlist.jl 60     # 60 days
```

### update_ohlcv.jl

Incrementally updates all existing OHLCV CSVs with bars added since the last run.
Reads the last date from each CSV and fetches only the gap to yesterday — much
faster than `collect_nse_ohlcv.jl` / `collect_bse_ohlcv.jl` for routine maintenance. Appends rows in-place.
Updates both NSE and BSE by default; pass `--nse-only` or `--bse-only` to restrict
to one exchange (mutually exclusive). Macro instrument CSVs update once regardless
of exchange selection.

OHLCV data lives one subfolder per granularity under each exchange:
`website/data/ohlcv/{nse,bse}/{daily,hourly,5min,15min,1min}/{SYMBOL}.csv`
(`collect_nse_ohlcv.jl`/`collect_bse_ohlcv.jl` write this layout; `--daily-only`,
`--hourly-only`, `--5min-only`, `--15min-only`, and `--1min-only` on either script
restrict to one granularity). Macro instrument CSVs are the one exception — too
few files to need subfolders, so they stay flat and suffixed at
`website/data/ohlcv/macro/{NAME}_{5min,15min,daily}.csv`.

Kite's per-interval day limits (60/100/200/400/2000 days for
1min/5min/15min/60min/day) are a single-request span cap, not a retention
cliff — verified live against the real API, every intraday interval still
returns genuine multi-year-old data today. `collect_nse_ohlcv.jl`/
`collect_bse_ohlcv.jl`'s `--from` flag applies to every granularity, not
just daily, and a deep `--from` (e.g. 2010-01-01) works for all of them —
it just means far more chunked API calls and disk the finer the
granularity, especially `--1min-only`.

```bash
# Run every trading day after kite_login.js (no arguments needed — updates NSE + BSE)
julia --project=packages/StockSwingPredictor scripts/update_ohlcv.jl

# Restrict to a single exchange
julia --project=packages/StockSwingPredictor scripts/update_ohlcv.jl --nse-only
julia --project=packages/StockSwingPredictor scripts/update_ohlcv.jl --bse-only

# Preview what would be fetched without hitting the API
julia --project=packages/StockSwingPredictor scripts/update_ohlcv.jl --dry-run

# Update only daily bars (skip the slower hourly pass)
julia --project=packages/StockSwingPredictor scripts/update_ohlcv.jl --daily-only
```

Both `collect_nse_ohlcv.jl`/`collect_bse_ohlcv.jl` and `update_ohlcv.jl` also take
`--skip-daily`/`--skip-hourly`/`--skip-5min`/`--skip-15min`/`--skip-1min` (the
collect scripts added `--skip-daily`/`--skip-hourly` later than the others, for
symmetry — `update_ohlcv.jl` only ever needed `--skip-5min`/`--skip-15min`/
`--skip-1min` since its daily/hourly passes were never worth skipping on their own).

### backfill_ohlcv.jl

The mirror image of `update_ohlcv.jl`: extends existing OHLCV CSVs *backward* to
an earlier `--from` date, instead of forward to yesterday. For each existing
`{SYMBOL}.csv`, reads the earliest date/datetime already on disk and fetches only
the older gap down to `--from`, merging it in (de-duplicated, re-sorted) rather
than overwriting the whole file the way `collect_nse_ohlcv.jl --refresh` would.
A symbol with no existing CSV is skipped — this tool only extends, it doesn't do
initial collection (use `collect_nse_ohlcv.jl`/`collect_bse_ohlcv.jl` for a
brand-new granularity, e.g. NSE `15min`, which has zero files today).

This exists because most of the real archive was collected back when the
"N-day retention" figures above were believed to be hard limits — most symbols'
files therefore start much later than Kite can actually provide. `--from`
defaults to `2010-01-04` (not `-01-01`, a Friday NSE holiday that would make
the skip check below permanently unsatisfiable).

```bash
# Extend everything (both exchanges, all granularities) back to 2010-01-04
julia --project=packages/StockSwingPredictor scripts/backfill_ohlcv.jl

# See the call-count estimate first — no API calls made
julia --project=packages/StockSwingPredictor scripts/backfill_ohlcv.jl --dry-run

# Scope a trial run
julia --project=packages/StockSwingPredictor scripts/backfill_ohlcv.jl --symbol RELIANCE
julia --project=packages/StockSwingPredictor scripts/backfill_ohlcv.jl --nse-only --hourly-only
julia --project=packages/StockSwingPredictor scripts/backfill_ohlcv.jl --skip-1min
julia --project=packages/StockSwingPredictor scripts/backfill_ohlcv.jl --from 2015-01-01 --log-file /tmp/custom.log
```

Same `--nse-only`/`--bse-only`, `--daily-only`/`--hourly-only`/`--5min-only`/
`--15min-only`/`--1min-only`, matching `--skip-*` flags, and `--symbol` as its
siblings. Macro is out of scope here too (same reasoning as `update_ohlcv.jl`).
Going back to 2010 for the finer granularities across the full symbol universe
is a genuinely large job (many chunked API calls per symbol) — `--dry-run` first
to see the estimate, then scope with `--symbol`/`--*-only` before committing to
a full run.

Full per-symbol detail (every backfill, every "no older bars" confirmation)
goes to `website/data/ohlcv/logs/backfill_ohlcv.log` (`--log-file PATH` to
override) via the shared `ScriptLog` mechanism — see `update_ohlcv.jl`'s
entry below. The terminal only gets stage headers, a progress heartbeat
every 100 symbols, and warnings; `tail -f` the log file for live status.
Kite relogin attempts (`_kite_get`'s automatic re-authenticate-on-403, see
below) are also routed there via a module-level active-log pointer, since a
relogin can stall for a long time with no other visible sign of why.

Two layers of "don't re-probe a gap already proven empty" keep repeat runs
cheap:
- **`{nse,bse}/earliest_trading_day.json`** (one per exchange, at the
  exchange root): `{symbol => earliest date daily has ever found}`,
  rebuilt from the daily CSVs after every daily pass. Every finer
  granularity fetches from `max(--from, earliest_trading_day[symbol])`
  instead of `--from` alone — daily's 2000-day span cap makes it by far
  the cheapest granularity to have already settled "nothing before an
  IPO date" for a symbol.
- **`{nse,bse}/{hourly,5min,15min,1min}/confirmed_floor.json`** (one per
  granularity, inside that granularity's own subfolder): `{symbol =>
  earliest date THAT granularity has confirmed has no older bars}`,
  written the first time a symbol hits "no older bars" at that specific
  granularity. This exists because each intraday granularity has its own
  Kite retention floor, shallower than daily's and shared across symbols
  regardless of real IPO date — observed live, hourly has nothing before
  ~2015-02-02 for virtually every NSE symbol, so without this a symbol
  with a pre-2015 IPO has its full multi-chunk gap re-probed and
  re-confirmed empty on every single run, forever.

### collect_macro_ohlcv.jl

Downloads historical daily OHLCV for macro instruments (global indices, commodities,
FX, volatility). Run once after `collect_nse_ohlcv.jl`; no incremental update needed as
macro history is stable — re-run yearly or with `--refresh` to extend.

| Instrument   | Source | Ticker  | Description                   |
|---|---|---|---|
| SP500        | Yahoo  | ^GSPC   | S&P 500                       |
| US_VIX       | Yahoo  | ^VIX    | CBOE Volatility Index         |
| CRUDE_OIL    | Yahoo  | CL=F    | WTI crude oil futures         |
| GOLD         | Yahoo  | GC=F    | Gold futures                  |
| SILVER       | Yahoo  | SI=F    | Silver futures                |
| NATURAL_GAS  | Yahoo  | NG=F    | Henry Hub natural gas         |
| COPPER       | Yahoo  | HG=F    | Copper futures                |
| INDIA_VIX    | Kite   | NSE idx | India VIX (local fear gauge)  |
| USD_INR      | Kite   | CDS FUT | USD/INR continuous futures    |

```bash
# One-time historical fetch (2010–yesterday)
julia --project=packages/StockSwingPredictor scripts/collect_macro_ohlcv.jl

# Custom date range
julia --project=packages/StockSwingPredictor scripts/collect_macro_ohlcv.jl --from 2015-01-01

# Force re-fetch all (e.g. to extend history after updating --from)
julia --project=packages/StockSwingPredictor scripts/collect_macro_ohlcv.jl --refresh
```

Output: `website/data/ohlcv/macro/{NAME}_daily.csv` — same schema as equity OHLCV.

### build_dataset.jl

Assembles the `Dataset` from the inference cache. Slides a weekly window over
every company's history, records index pointers and labels. No OHLCV CSVs are
read here — all price data comes from the pre-built cache.

Output file is named `dataset_{pred_hours}.bson` so multiple label lengths can
coexist. `train_model.jl` automatically selects the right file based on the arch.

```bash
# Build 35-bar labels for v1 / v2 (5-day horizon)
julia --project=packages/StockSwingPredictor scripts/build_dataset.jl --pred-hours 35

# Build 70-bar labels for v3 (10-day horizon, default)
julia --project=packages/StockSwingPredictor scripts/build_dataset.jl --pred-hours 70
julia --project=packages/StockSwingPredictor scripts/build_dataset.jl  # same as above
```

Output: `website/data/training/dataset_{pred_hours}.bson` (~30 MB per file)

### train_model.jl

Trains a `SwingPredictor` on the assembled dataset. Automatically loads
`dataset_{arch.pred_hours}.bson` (e.g. `dataset_70.bson` for `--arch v3`).
Saves checkpoint on every validation improvement; supports clean stop via sentinel files.

```bash
# Train v3 (10-day, LM weighting) — requires dataset_70.bson
julia --project=packages/StockSwingPredictor scripts/train_model.jl --arch v3

# Common flags
julia --project=packages/StockSwingPredictor scripts/train_model.jl \
      --arch v3 --epochs 200 --lr 1e-4 --batch 128 --l2 1e-5

# GPU training (falls back to cpu with a warning if CUDA.functional() is false)
julia --project=packages/StockSwingPredictor scripts/train_model.jl --arch v3 --device gpu

# Resume from checkpoint
julia --project=packages/StockSwingPredictor scripts/train_model.jl --arch v3 --resume

# Overfit diagnostic: train on N examples only — confirms gradients flow (default N=64)
julia --project=packages/StockSwingPredictor scripts/train_model.jl --arch v3 --overfit
julia --project=packages/StockSwingPredictor scripts/train_model.jl --arch v3 --overfit 128
```

Outputs (under `website/data/models/{arch.name}/`):
- `swing_predictor.bson` — best weights
- `training_log.json` — full train/val MSE history
- `model_card.json` — hyperparams, dataset stats, test metrics
- `epoch_log.jsonl` — per-epoch and per-batch loss (streamed during training)

To stop training cleanly: `touch website/data/models/DualCNN_v3/STOP` (saves checkpoint).
Hard stop without save: `touch website/data/models/DualCNN_v3/STOP_NOW`.

### build_cache.jl — NSE/hourly or BSE/15-minute

`--exchange nse|bse` (default `nse`) picks the OHLCV tree and the `{nse,bse}_companies_latest.json`
confidence list; `--granularity hourly|15min` (default `hourly`) picks the intraday bars;
`--min-mcap CR` and a positional top-N limit the universe; `--out PATH` writes elsewhere.
The cache records its `exchange` and `bar_minutes`, and everything downstream reads them from it:
`build_market_universe_snapshot.jl`/`prepare_training_data.jl` use the matching companies file, and
`TradingGame` decides at **every bar** (`decision_granularity(cache)`: `MINUTE_15` = rule 8's minimum
interval, or `HOURLY`). Intraday bars outside 09:15–15:15 (BSE Muhurat evening sessions) are dropped.
The cache's `hourly_closes`/`hourly_datetimes` fields keep their old names but hold whatever bar length
`bar_minutes` says. For a 15-minute run, `GAMMA`/`GAE_LAMBDA` are rescaled to `x^(bar_minutes/60)` so
the discount covers the same wall-clock time, and `N_HOURLY_BARS_SHORT` (120) spans ~5 trading days
instead of ~17. Most BSE small caps barely trade (their 15-minute bars are mostly forward-filled), so
build with `--min-mcap`, e.g.:

```bash
julia --project=packages/StockSwingPredictor scripts/build_cache.jl --exchange bse --granularity 15min --min-mcap 500
julia --project=packages/TradingGame scripts/build_market_universe_snapshot.jl --strategy random --n 40 --disjoint
julia --project=packages/TradingGame scripts/prepare_training_data.jl --skip-universe
julia --project=packages/TradingGame scripts/train_trading_policy.jl --game-version v2 --no-news
```

Select it with `--game-version 3` — v2's rules, but the run errors out unless the cache is BSE 15-minute.

News features and 1-minute snapshots are NSE-only (NSE announcements keyed by NSE symbols), so use
`--no-news` with a BSE cache; `prepare_training_data.jl` skips both news stages automatically for BSE.
`--history resample|hourly|none` adds the hourly history axis (default `resample` with `--granularity 15min`: hourly bars aggregated from the 15-minute bars, close = last 15-minute close in the slot — checked equal to Kite's own hourly closes on all 1,882 RELIANCE overlap bars; BSE's hourly files cover only ~25% of companies and many start in 2025, so resampling is the default; `hourly` reads `ohlcv/bse/hourly` instead). Game v3 refuses a cache without it. `build_cache.jl` overwrites `inference_cache.bson`, and a checkpoint trained on one cache cannot sensibly
be resumed on another (different bar length), so start a fresh run after switching.

### build_market_universe_snapshot.jl

Builds the `TradingGame` candidate universe: filters `nse_companies_latest.json`'s
already-computed confidence scores (>= 40, no live sidecar calls) and restricts to
symbols with a cached `InferenceCache` entry — this filtered, market-cap-sorted
pool (`TradingGame.eligible_candidates`) is then split into a **train** candidate
list and a **val** candidate list by one of several pluggable `--strategy` recipes
(`TradingGame.UniverseStrategy`, in `packages/TradingGame/src/universe.jl`). Train
and val are free to use different company counts and/or entirely different
companies — `TradingGameEnv`, rule 13's holdings cap, and `ActorCriticPolicy` all
operate purely on each episode's own candidate content (no symbol-identity
embedding or positional encoding across candidates), so nothing downstream needs
to change to support that; see the module docstring in `universe.jl` for why.

```bash
# shared-topcap (default): top --n by market cap, identical list for train and val
julia --project=packages/TradingGame scripts/build_market_universe_snapshot.jl        # top 60
julia --project=packages/TradingGame scripts/build_market_universe_snapshot.jl --n 100

# disjoint-topcap: top (n_train+n_val) by market cap, split disjoint and
# stratified by market-cap decile so neither side skews large/small-cap
julia --project=packages/TradingGame scripts/build_market_universe_snapshot.jl \
    --strategy disjoint-topcap --n-train 60 --n-val 20 --seed 42

# random: uniformly random --n from the WHOLE confidence-passing pool, not
# restricted to top-market-cap; --disjoint draws independent train/val sets
julia --project=packages/TradingGame scripts/build_market_universe_snapshot.jl \
    --strategy random --n 60 --disjoint --seed 42

# random-bucketed: random selection from named market-cap bands (HI may be 'inf')
julia --project=packages/TradingGame scripts/build_market_universe_snapshot.jl \
    --strategy random-bucketed --band 0:5000:15 --band 5000:inf:15
```

Prerequisites: `nse_companies_latest.json` (`generate_nse_list.jl` then
`run_confidence_checks.jl`) and `inference_cache.bson` (`build_cache.jl`).
Output: `website/data/trading_game/universe_latest.json` — `{"strategy": ...,
"strategy_params": ..., "train_candidates": [...], "val_candidates": [...]}`, load
with `TradingGame.load_universe_snapshot` → `(train=Vector{String},
val=Vector{String})`, directly usable as `train_config`/`val_config`'s
`EpisodeConfig.candidate_universe` respectively (see `train_trading_policy.jl`).

### backfill_news_signals.jl

Classifies historical NSE corporate announcements (`nse_announcements.db`, raw
text only — no sentiment/severity) into `NewsSignal`-shaped rows using a
**local Ollama model** (`NewsMonitor.classify_item_ollama`), not the paid
Claude API — classifying every historical announcement for even a modest
universe is tens of thousands of calls, free and fine for an overnight local
job, expensive for a per-call API. `classify_item_ollama` reuses the exact
same tool schema/calibration text as `classify_item` (the live daemon's
Claude path), so backfilled and live signals are classified identically.

Scope defaults to every symbol in `universe_latest.json`'s train+val
candidates (not the whole market — the confidence-passing pool alone is
~1960 symbols / ~1.08M announcements, ~12+ days serial even for free; the
actual training universe is what matters, and it only grows as new symbols
enter it). `--symbol` overrides this with an explicit list instead.

Resumable like `fetch_nse_history.jl`: results are written keyed by `guid`
into `website/data/news_signals.db`, `INSERT OR IGNORE`'d, so a re-run (new
symbol, or an interrupted run) only classifies rows not already present. NSE's
own `symbol` column from `nse_announcements.db` is always trusted over
whatever symbol the model guesses from free text — observed the model
occasionally return a near-miss variant (e.g. `LICI` → `LICIND`/`LICINDIA`) or
name a different company mentioned in the announcement; this only matters for
this historical path, since the live daemon's RSS/BSE sources often have no
structured symbol at all and genuinely need the model's guess.

```bash
# Requires Ollama running locally (qwen3:latest by default) — no API key needed
julia --project=packages/NewsMonitor scripts/backfill_news_signals.jl
julia --project=packages/NewsMonitor scripts/backfill_news_signals.jl --dry-run
julia --project=packages/NewsMonitor scripts/backfill_news_signals.jl --symbol RELIANCE --symbol TCS
julia --project=packages/NewsMonitor scripts/backfill_news_signals.jl --from 2021-01-01
julia --project=packages/NewsMonitor scripts/backfill_news_signals.jl --ollama-model qwen3:latest
```

Prerequisites: `nse_announcements.db` (`fetch_nse_history.jl`) and, for the
default symbol scope, `universe_latest.json` (`build_market_universe_snapshot.jl`).
Output: `website/data/news_signals.db`, table `news_signals(guid, symbol,
event_type, sentiment, severity, summary, source, headline, url,
published_at, classified_at)` — loaded by `TradingGame.build_news_feature_cache`
(`news_features.jl`), which `train_trading_policy.jl` reads by default (see
below; `--no-news` opts out).

### fetch_news_snapshot_ohlcv.jl

Fetches 1-minute OHLCV from Kite for just the few minutes around EVERY
classified news event — not a continuous historical backfill, and not a
whole trading day per event either. `news_features.jl`'s instant-price
snapshot mechanism only ever looks up the first 1-minute bar at-or-after a
news timestamp within `TradingGame.NEWS_SNAPSHOT_MAX_LAG_MINUTES` (30 min),
for *every* candidate symbol (the whole market universe's state at that
instant, not just whichever symbol the news was about) — so fetching a
whole day (~375 bars) per (symbol, event) pair would be ~12x more than ever
gets read. Nothing on disk provides even the narrow need today —
`website/data/ohlcv/nse/1min/` starts out empty.

Deliberately **not** filtered by severity — every distinct `published_at`
in `news_signals.db` gets fetched, regardless of how routine the event was.
Whether a signal is severe enough to actually act on is a training-time
judgement call (`TradingGame.build_news_feature_cache`'s
`severity_threshold` argument, applied when training reads the DB), not a
fetch-time one — keeping the two separate means tuning that threshold later
(or just trying a lower one) never requires re-fetching anything; the data's
already on disk either way. Symbols default to `universe_latest.json`'s
train+val candidates (or `--symbol`) — genuinely every candidate, not just
whichever symbol the news was about: this mimics what live operation will
actually do (a severe-enough event triggers a check of the market
universe's full instantaneous state, combined with hourly/daily history, to
decide buy/sell/hold across the whole book), so narrowing to the newsy
symbol alone would be cheaper but would stop simulating that. For each
(symbol, event) pair not already covered by a bar within the same
30-minute window in that symbol's existing 1-minute CSV, fetches a short
window (`timestamp - 1 min` … `timestamp + 30 min`) via `StockSwingPredictor.
fetch_ohlcv_1min_window` (one Kite call per pair, not chunked — that
function is for short windows only, unlike `fetch_ohlcv_1min`'s whole-day
chunking) and merges the handful of returned bars in (de-duplicated by
`datetime`, re-sorted) — never overwrites the file, so this composes
cleanly with a later full `--1min-only` backfill or `update_ohlcv.jl`'s
daily runs touching the same CSVs.

`--role {train,val,both}` (default `both`) is the real lever for cutting
the job down — not narrowing symbols within a role (see above for why
that's the wrong cut). Train and val are genuinely independent: usually
different symbols, and (under the default `trailing` `--val-window`)
different, non-overlapping date windows. Fetching one combined (train ∪ val
symbols) × (every event, full range) job wastes calls on pairs that can
never matter — a val-only symbol's price during train's date range is
never read by any val episode, and vice versa. Running `--role train
--from <train_start> --to <train_end>` and `--role val --from <val_start>
--to <val_end>` as two separate passes only fetches what each role can
actually use. On a real 20/20 split with a ~4.5-year train window and
~4-month val window, this cut a combined 910,640-pair job down to 290,500
(train) + 29,420 (val) = 319,920 — about a 2.85x reduction, with val
alone dropping to roughly 1/15th of its undifferentiated size.

`--from`/`--to` narrow the event timestamps by date. Unlike daily/hourly/
5-min/15-min, Kite's 1-minute coverage is NOT reliably available arbitrarily
far back — verified live (RELIANCE/TCS/INFY, weekday, mid-market-hours):
2015-01 returns nothing, 2016-01 returns real bars, 2016-07 returns nothing
again, 2017-01 onward returns bars consistently. It's patchy, not one clean
cutoff, so no default filter is applied — an unfiltered run keeps
re-attempting pre-2017-ish events on every future run too, since a missing
bar looks identical to "not yet fetched" and is never treated as permanent.
`--from 2017-01-01` is a reasonable starting point to skip that wasted
effort, not a guarantee every later date succeeds.

```bash
julia --project=packages/TradingGame scripts/fetch_news_snapshot_ohlcv.jl
julia --project=packages/TradingGame scripts/fetch_news_snapshot_ohlcv.jl --dry-run
julia --project=packages/TradingGame scripts/fetch_news_snapshot_ohlcv.jl --symbol RELIANCE --symbol TCS
julia --project=packages/TradingGame scripts/fetch_news_snapshot_ohlcv.jl --from 2017-01-01
julia --project=packages/TradingGame scripts/fetch_news_snapshot_ohlcv.jl --role train --from 2021-09-09 --to 2026-05-11
julia --project=packages/TradingGame scripts/fetch_news_snapshot_ohlcv.jl --role val   --from 2026-05-12
```

Prerequisites: `news_signals.db` (`backfill_news_signals.jl`) and, for the
default symbol scope, `universe_latest.json` (`build_market_universe_snapshot.jl`).
Requires a fresh `kite_session.json` (`node sidecar/kite_login.js`) — this
hits the live Kite historical-data API, unlike `backfill_news_signals.jl`
(local Ollama, no Kite dependency). Output: merged into
`website/data/ohlcv/nse/1min/{SYMBOL}.csv`, read by
`TradingGame.build_news_feature_cache`'s snapshot builder the same way
`train_trading_policy.jl` already expects (see below) — that function's
`severity_threshold` is where severity actually gets applied.

### prepare_training_data.jl

Single entry point for the three scripts above — universe, news backfill,
1-minute snapshots — instead of running each one separately and manually
keeping their train/val dates in sync. The date math is `TradingGame.
resolve_date_windows`, moved into the package specifically so this script
and `train_trading_policy.jl` call the exact same function rather than two
independent implementations that could quietly drift apart — whatever this
prepares for is guaranteed to be what a training run with the same
`--val-window`/`--val-days`/`--train-*`/`--val-*` flags will actually use.

Stages, in order: (1) universe — `TradingGame.eligible_candidates`/
`build_universes`/`save_universe_snapshot` run in-process (same functions
`build_market_universe_snapshot.jl` itself calls), skipped if
`universe_latest.json` already exists (an existing, possibly deliberately-
built universe is never silently clobbered — pass `--rebuild-universe` to
force); only `SharedTopMarketCap(n=...)` is available here, run
`build_market_universe_snapshot.jl` yourself first for any other strategy,
then `--skip-universe` picks it up as-is. (2) news backfill — one combined
pass over `min(train_start,val_start)…max(train_end,val_end)` (symbol-
scoped only, no role split needed — see `backfill_news_signals.jl`).
(3) 1-minute snapshots — two separate `--role train`/`--role val` passes,
each scoped to that role's own window (see `fetch_news_snapshot_ohlcv.jl`
for why role, not symbol, is the right split). Stages 2 and 3 run as real
subprocesses (`--project=packages/NewsMonitor` / `--project=packages/
TradingGame` respectively — TradingGame has no package dependency on
NewsMonitor, so stage 2 can't be an in-process call), so expect their
startup cost (Flux/CUDA for stage 3, a minute or so each call) on top of
this script's own — small next to how long stages 2/3 themselves run.

```bash
julia --project=packages/TradingGame scripts/prepare_training_data.jl
julia --project=packages/TradingGame scripts/prepare_training_data.jl --dry-run
julia --project=packages/TradingGame scripts/prepare_training_data.jl --val-days 120
julia --project=packages/TradingGame scripts/prepare_training_data.jl --val-window same
julia --project=packages/TradingGame scripts/prepare_training_data.jl --train-start 2021-09-09 --val-start 2026-05-12
julia --project=packages/TradingGame scripts/prepare_training_data.jl --skip-universe --skip-news
julia --project=packages/TradingGame scripts/prepare_training_data.jl --n 100 --rebuild-universe
```

`--dry-run` prints the resolved windows and the exact commands it would
run — no universe build, no Ollama calls, no Kite calls.

The windows are resolved against `cache.hourly_datetimes`, not
`cache.dates` — `TradingGameEnv` trains at hourly granularity only
(`TRAINING_DECISION_GRANULARITY` in `env.jl`), and hourly's real Kite
floor is shallower than daily's (daily can reach back to 2010-01-04 after
`backfill_ohlcv.jl`'s fixes; hourly has nothing before ~2015-02-02 for
virtually every NSE symbol — see `backfill_ohlcv.jl`'s entry above).
Resolving against the daily axis would default `train_start` to a date no
hourly bar can satisfy, crashing `TradingGameEnv.reset!` with "no hourly
bars at or before episode start."

The resolved window is also saved to `website/data/trading_game/
date_window.json` (not run in `--dry-run`, which has no side effects at
all). `train_trading_policy.jl` reads this back automatically as a
fallback default for any of `--train-start`/`--train-end`/`--val-start`/
`--val-end` not given explicitly on its own command line — run this script
once with your intended date flags, then `train_trading_policy.jl` with
none at all, and it reproduces the exact same window. An explicit flag on
either script, or a `--resume`d run's saved `run_config.json`, still takes
priority over this file.

### train_trading_policy.jl

Drives `TradingGame.train_policy!`: loads the cache + the train/val candidate
universes (`build_market_universe_snapshot.jl` — a `UniverseStrategy` may give
train and val different companies, not just different dates), resolves the
train/val date windows, and trains a fresh `ActorCriticPolicy`. Streams
`live_status.json` by default — watch the run at `website/tradinggamelive.html`.

Builds a real `MacroCache` (`website/data/ohlcv/macro`) and `NewsFeatureCache`
(`website/data/news_signals.db`, via `news_features.jl`) by default and passes
both through to `train_policy!` — before this, `macro_cache`/`news_fn` were
plumbed all the way through `train_policy!`/`collect_rollout` but this script
never actually built or passed either, so every run trained on the all-zero
placeholders for both channels regardless of what data existed on disk. A
missing macro dir or `news_signals.db` degrades to that same zero placeholder
with a warning, rather than erroring — both are optional enrichments, not
hard prerequisites. `--no-macro`/`--no-news` silence the warning by opting out
explicitly; `--news-db PATH` points at a non-default classified-signals DB.

Also builds `TradingGameEnv.price_overrides`: at every bar with a qualifying
news event, every candidate's price is replaced with its real **1-minute
open** at-or-just-after the news's exact timestamp (read from
`website/data/ohlcv/nse/1min/{SYMBOL}.csv`), instead of the hourly bar's
close — "look at the instant market state the moment news lands, trade off
that," not off a close that could be up to an hour stale. `current_price`
(`env.jl`) is the single choke point every price-right-now read in the
simulator goes through, so this is consistent across execution, mark-to-
market, and the observation's current-bar price feature. A symbol with no
1-minute CSV yet (or whose nearest 1-minute bar is too stale —
`NEWS_SNAPSHOT_MAX_LAG_MINUTES`, 30 min) simply falls back to the hourly
close for that bar — run `fetch_news_snapshot_ohlcv.jl` first (see above) to
populate exactly the days this needs; without it this mechanism silently
degrades to "no override," which
is why it's additive rather than required.

```bash
julia --project=packages/TradingGame scripts/train_trading_policy.jl
julia --project=packages/TradingGame scripts/train_trading_policy.jl --iterations 500 --val-days 40
julia --project=packages/TradingGame scripts/train_trading_policy.jl --resume --iterations 100
julia --project=packages/TradingGame scripts/train_trading_policy.jl --seed 42
julia --project=packages/TradingGame scripts/train_trading_policy.jl --init-from other_run/policy.bson
julia --project=packages/TradingGame scripts/train_trading_policy.jl --device gpu
julia --project=packages/TradingGame scripts/train_trading_policy.jl --entropy 0.02
julia --project=packages/TradingGame scripts/train_trading_policy.jl --val-window same
julia --project=packages/TradingGame scripts/train_trading_policy.jl --val-start 2024-06-01 --val-end 2024-12-31
julia --project=packages/TradingGame scripts/train_trading_policy.jl --no-macro --no-news
julia --project=packages/TradingGame scripts/train_trading_policy.jl --game-version 2 --cash-penalty 0.02 --hold-penalty 0.02
```

`--game-version {1,2,3}` (default `1`) selects the rule set. **v3** is the v2 rules (same flags) restricted to a BSE 15-minute cache, with the **macro context and news features switched
off** (`GameRules.use_macro = use_news = false`): they are not loaded or computed and the policy has no
macro GRU or news inputs. Its price input is **two distinct sets**: (1) a **14-day hourly history** — the newest 70 *completed* hourly bars (`obs_window_days = 10` trading days × 7), as **log-returns** `x_i = 100·ln(c_i/c_{i-1})` over 71 completed closes (`use_volatility = false`, one channel); and (2) an **instantaneous snapshot** per stock — the same return one step on, `s = 100·ln(p_now/c_last)` from the last completed hourly close to the current 15-minute close, joined to the holding features (4 per-stock features, `n_stock_features`; the tensor keeps its `holding` name). The decision clock steps in 15-minute bars and decisions fill at that bar's close; the history axis only advances when an hourly bar completes (`history_end_idx`), so nothing runs ahead. v1/v2 keep the fixed 120-bar, 2-channel window. 28,485 parameters vs 43,301 for v2 (no cash token and no portfolio token either: the six portfolio scalars are repeated beside every stock and concatenated onto its fusion input, `Dense(70+4+6 → 64)`, attention runs over the N stock tokens alone, and the critic reads their mean **plus the six portfolio scalars directly** (`GameRules.portfolio_in_fusion`, `portfolio_to_critic`; the critic MLP is 70 → 64 → 32 → 1). With `--history-encoder direct` (the v3 default; `rules_v3(history_encoder=:direct)`) the 70 returns go **straight into the fusion layer** (no GRU); `--history-encoder gru` keeps the recurrent encoder for comparison (40,389 parameters). The code for both stays; other versions still
use them. v3 checkpoints are not interchangeable with v2's (different network). v3 also splits rule
enforcement in two. (1) **Impossible moves are masked before the policy samples**: a SELL on a stock with nothing
sellable (not held, or inside the 1-day lock-up) is removed from that stock's distribution (`GameRules.premask`,
`sellable_mask`/`mask_action_logits`), so the HOLD/SELL/BUY probabilities are conditional and PPO's log-probs use
the same masked distribution (the mask is stored per step in the rollout). (2) **Every other refused move is charged**: `--illegal-penalty` (default `ILLEGAL_PENALTY_COEF_V3` = 0.01, i.e. 1% of portfolio value; v2 default 0,
v1 ignores it) is subtracted from that bar's reward for each buy with no or too little cash (not enough for one share),
in rebuy cooldown, beyond the holdings cap (each candidate that gets no slot), or over the 30% position cap (the part
within the cap still fills); the rest is rejected. Nothing is counted or logged as a rate; the charge is reward only,
the portfolio itself is untouched. **The v3 reward is paid per reward window** (`GameRules.terminal_reward`, `--reward-mode terminal|stepwise`, default terminal;
`--reward-window-days N`, default 0): 0 on every bar, and at the end of each window
`R = (V_end − V_start)/V_start − (penalties accumulated in the window, in rupees)/V_start`, with `V_start` the portfolio value when that window
began (so each window is normalised by its own start, not by the initial cash). A window is `N` trading days, with a shorter last one flushed
at the episode's end; `N = 0` makes the whole train/val period one window. Each illegal move adds `--illegal-penalty × (portfolio value at the
moment of the decision)` to the window's ledger (`env.penalty_accum`); the cash/hold penalties, if set, add their share of the value on each
bar; the ledger and `V_start` reset at every payout. `train_return`/`val_return` = the sum of the window rewards. PPO's discount follows
(`discount_factors`): with `N = 0`, `γ = λ = 1` (any discount would shrink an end-of-episode reward to nothing over ~20,000 bars); with `N > 0`
the bar-scaled `γ` and `λ = 1`. `stepwise` restores the weekly log-return plus per-bar penalties. Caveat: with one episode per iteration and
few payouts, the advantage is mostly `R − V(s)`; shorter windows (or several episodes per update) give the baseline more to compare v3 ignores `--macro`/`--news`
settings and loads neither: `reset!` refuses any
other cache (`build_cache.jl --exchange bse --granularity 15min`), carried per episode by
`EpisodeConfig.rules` (`GameRules`; `rules_v1()`/`rules_v2()`). **v2**: cash is a
pseudo-stock — an extra attention token built from cash/value, reserved/value, cap
utilisation (`cash/value ÷ MAX_CASH_FRACTION`) and days spent over the cap; decisions
fill at the **same bar's close** instead of the next bar's; there is **no forced exit**
— a lot held `MAX_HOLD_DAYS_V2` (7)+ days costs `--hold-penalty × (share of portfolio
value in such lots)` per bar; and the cash-ceiling penalty is `--cash-penalty` (both
default `0.0`, i.e. off; under v1 `--cash-penalty` overrides `CASH_CEILING_PENALTY_COEF`
and `--hold-penalty` is ignored with a warning). The version, `--cash-penalty` and
`--hold-penalty` are `--resume`-restored via `run_config.json`. v2 policies carry the
cash token, so a checkpoint can only be resumed / warm-started under the version it
was trained with (a mismatch stops with an error). v1 behaviour is unchanged.

`--val-window MODE` (default `trailing`) picks how the train/val date windows are
derived: `trailing` is the original behavior — val is the last `--val-days` of the
cache, train is everything before it, sound even with disjoint train/val
companies since there's no leakage risk left to guard against. `same` has train
and val both span the full cache date range — only meaningful once train/val
use different companies (see `build_market_universe_snapshot.jl`), and makes
full use of the cache's data on both sides instead of carving out a held-out
tail. Explicit
`--train-start`/`--train-end`/`--val-start`/`--val-end` (any subset, ISO
`yyyy-mm-dd`) override whichever bound `--val-window` would otherwise have
picked, e.g. for deliberately validating against a specific regime. Like
`prepare_training_data.jl`, these resolve against `cache.hourly_datetimes`,
not `cache.dates` — see that script's entry above for why.

For any of the four date bounds not given explicitly, the fallback (below
`--resume`'s saved `run_config.json`, above the raw cache-bounds default)
is `website/data/trading_game/date_window.json`, written by
`prepare_training_data.jl`. So `prepare_training_data.jl --train-start
2021-09-09` once, then `train_trading_policy.jl` with no date flags at
all, trains on that same `2021-09-09` start with nothing to repeat.

Prerequisites: `inference_cache.bson` (`build_cache.jl`) and
`universe_latest.json` (`build_market_universe_snapshot.jl`).
Outputs (under `website/data/trading_game/`): `policy.bson`, `episode_log.jsonl`,
`live_status.json`, `val_runs.jsonl` (every held-out episode's full portfolio value/
stock value/cash value curves, trades, and final holdings snapshot, appended —
never overwritten — once per evaluation; `tradinggamelive.html` plots each run as
its own color with a show/hide toggle, portfolio value as a thick line and
stock/cash as a separate linked sub-chart below it, plus a table of the latest
run's final positions), `run_config.json`. Stop cleanly with
`touch website/data/trading_game/STOP`
(checkpoint saved) or hard-stop with `STOP_NOW` (no save), same convention as
`train_model.jl`. To restart after either: re-run with `--resume` — loads
`policy.bson` instead of a fresh policy and continues `episode_log.jsonl`'s
iteration numbering (`--iterations` then means "how many more", not a new
total), same `--resume` convention as `train_model.jl`.

Every run writes its effective `--initial-cash`/`--val-days`/`--eval-every`/
`--lr`/`--entropy`/`--seed`/`--device`/`--minibatch`/`--val-window`/
`--train-start`/`--train-end`/`--val-start`/`--val-end` to `run_config.json`.
`--resume` reads it back and applies those values for any of those flags not
*also* given explicitly on the resume command line — an explicit flag always
wins over the saved one. This is what makes a bare `--resume` reproduce the
original run's config instead of silently reverting to script defaults (e.g.
`--val-days` snapping back to 60, corrupting the train/val split relative to
what the checkpoint was actually trained on). Delete `run_config.json`, or
pass the flags explicitly, to intentionally change config on resume.

`run_config.json` also records informational-only fields not restored on
`--resume` (the candidate universe always comes from `universe_latest.json`
itself at load time; the rule constants come from `constants.jl`, not a
flag) — `n_candidates_train`/`n_candidates_val` (how many companies each
role's universe held), `n_max_holdings_train`/`n_max_holdings_val` (rule 13's
cap, which now differs per role since train/val can have different candidate
counts), `resolved_device`/`resolved_minibatch`, `resolved_train_start`/
`resolved_train_end`/`resolved_val_start`/`resolved_val_end` (the dates that
actually ran — distinct from the plain `train_start`/etc. keys, which hold
the *raw* `--train-start`/etc. override if one was given, `null` otherwise),
and `rule_constants` (every named constant in `constants.jl`). This is what
`tradinggamelive.html`'s "Training configuration" card displays — a quick way
to see a checkpoint's actual training parameters without console access.

`--seed N` makes a fresh policy's initial weights reproducible
(`ActorCriticPolicy(seed=...)`, via `Random.seed!`) and also seeds the PPO
rollout's action sampling. `--init-from PATH` is a *fresh* run (iteration 1,
cleared log — unlike `--resume`) that warm-starts the policy's weights from an
existing checkpoint at `PATH` instead of random init; mutually exclusive with
`--resume`.

`--entropy N` (default `0.01`, `TradingGame.ENTROPY_COEF`) weights PPO's
entropy bonus — higher keeps the policy's per-candidate action distribution
spread out for longer before it collapses onto a single preferred action,
at the cost of noisier rollouts.

`--device gpu` moves the policy to GPU once, up front (falls back to `cpu`
with a warning if `CUDA.functional()` is false, same as `train_model.jl`).
`--minibatch` defaults to 256 on `gpu` / 32 on `cpu` when not given
explicitly, same convention as `train_model.jl`'s batch-size default.
`collect_rollout` always runs its own forward pass on CPU regardless of
`--device` — the rollout can't be batched across time steps (each bar's
action depends on the simulator state left by the previous one), and
measured one-bar-at-a-time GPU calls run roughly 10x slower than CPU for
this network (host round-trip + 120 individual GRU-step kernel launches per
call dominate the tiny per-call compute). `--device gpu` therefore only
accelerates `ppo_update!`'s minibatched passes, where batching actually
helps.

### Per-step validation logs (`val_steps/`) and `val_steps_summary.jl`

Every held-out validation rollout also writes `website/data/trading_game/val_steps/iter_NNNNN.bson` (`--val-steps-every N`,
default 1; 0 disables; wiped by a fresh run): one column for **every decision bar, trade or not** — HOLD/SELL/BUY
probabilities per stock after the sell mask (`probs`, `(3, N, T)`), the SELL probability before the mask (`p_sell_raw`), buy
weight, chosen action, `sell_ok`/`held` flags, the fill price, the critic's V(s), reward and its illegal-move part, and the
book after the step (portfolio value, cash, stocks value, holdings, trades). Probabilities are `Float16`; a year of 15-minute
bars for 40 stocks is about 5 MB per run. Load with `TradingGame.load_val_steps(path)`, or run
`julia --project=packages/TradingGame scripts/val_steps_summary.jl [iteration | file ...]` for averages by held/not held, by
15-minute slot of the day, per stock, and the critic against the reward that followed. Each run also gets a small `iter_NNNNN.json` per-bar summary (mean and top probabilities, picks, V(s), penalty, book — about 0.5 MB) that
`tradinggamelive.html`'s "Held-out validation — trades" table reads to fill in the bars on which nothing was traded ("No action" rows, with the
probabilities the policy had on that bar); runs without it show those rows with dashes. The validation rollout is greedy
(argmax), so the probabilities show what the policy believed, not a sample.

### training_status.jl

One-shot, read-only snapshot of the current `train_trading_policy.jl` run —
process status, `run_config.json`, `live_status.json` (iteration, phase,
portfolio value, PPO update progress), and the last 5 logged iterations from
`episode_log.jsonl` with their rollout/val/update timing breakdown. No
arguments, no prompts; prints and exits. Meant for a quick check from outside
a full session (e.g. Remote Control on the mobile app).

```bash
julia --project=packages/TradingGame scripts/training_status.jl
```

### Website

The `website/` folder is a static multi-page site with a shared dark-theme sidebar.
Serve from the repo root so `data/` is accessible:

```bash
./serve.sh                              # opens website/watchlist.html on :8080
./serve.sh website/setup.html           # open a specific page
./serve.sh website/companies.html 9000  # custom port
```

| Page | URL | Data |
|---|---|---|
| Watchlist | `website/watchlist.html` | `website/data/earnings_watchlist_latest.json` |
| Companies | `website/companies.html` | `website/data/nse_companies_latest.json` |
| Setup     | `website/setup.html`     | localStorage (checklist state) |
| Trading game | `website/tradinggamelive.html` | `website/data/trading_game/live_status.json`, `val_runs.jsonl` |

## Data sources

| Source | What it provides | Access |
|---|---|---|
| Tijori Finance (via sidecar) | Financials, shareholding, forensics score, PDFs, screener | `TijoriData` package |
| NSE direct download | ASM/GSM surveillance lists, bulk deals | HTTP (planned) |
| SEBI website | Enforcement orders against companies/promoters | HTTP scrape (planned) |
| IBC portal | Insolvency proceedings | HTTP scrape (planned) |

## Pipeline (planned — not yet built)

```
EarningsCalendar
      ↓
CompanyConfidence scorer     ← score < 40 → skip
      ↓
StockSwingPredictor       ← options implied move + multi-source signals
      ↓
LLM synthesis (Claude API)
      ↓
SignalDatabase
      ↓
Self-improvement loop        ← analyze outcomes, retrain parameters
```

`TradingGame` branches off `StockSwingPredictor` (its `InferenceCache`/macro data)
and `CompanyConfidence` (its candidate-universe pre-filter) as a parallel RL
training track, not a downstream stage of the swing-prediction pipeline above —
see the dependency table and `TradingGame`'s module docstring. Unlike the rest of
this diagram it's actively in progress, not planned: simulator, PPO training loop,
live training/validation visualization (`tradinggamelive.html`), and the
historical news-signal backfill (`backfill_news_signals.jl`, `news_features.jl`)
all exist and run today.

## Environment variables

Store all secrets in a `.env` file in the repo root (gitignored). `kite_login.js`
loads it automatically; Julia code can use `DotEnv.jl` or read it manually.

| Variable | Used by | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | LLM calls (planned) | Claude API authentication |
| `KITE_HISTORICAL_API_KEY` | `kite_login.js`, `kite_data.jl` | Kite Historical Data app key |
| `KITE_HISTORICAL_API_SECRET` | `kite_login.js` | Kite Historical Data app secret |
| `KITE_USER_ID` | `kite_login.js` | Zerodha trading account client ID (e.g. AB1234) |
| `KITE_PASSWORD` | `kite_login.js` | Zerodha trading account password |
| `KITE_TOTP_SECRET` | `kite_login.js` | Base32 TOTP secret from authenticator app |
| `KITE_HISTORICAL2_API_KEY` | `kite_login.js --account 2`, `kite_data.jl` | Second Kite app key (separate rate limit) |
| `KITE_HISTORICAL2_API_SECRET` | `kite_login.js --account 2` | Second Kite app secret |
| `KITE_CONNECT_ID` | — | Kite developer portal login (not used in code) |
| `KITE_CONNECT_PASSWORD` | — | Kite developer portal password (not used in code) |
| `PORT` | sidecar | HTTP port (default 3001) |

The daily access token is **not** an env var — `kite_login.js` writes it to
`sidecar/kite_session.json` (gitignored). `load_kite_session(pwd())` reads it in Julia.
The token is valid for one trading day and works for all Kite Connect endpoints
(historical data, portfolio, positions, funds, orders).
