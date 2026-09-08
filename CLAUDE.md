# market-agent

Self-improving trading agent for Indian stock markets (NSE/BSE), built entirely in Julia.

## Project goal

Identify stocks likely to make large price moves around earnings events, using a
multi-source data pipeline, a company fraud/confidence filter, and an LLM reasoning
layer. Eventually self-improves by analyzing its own trade history.

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
│   ├── kite_setup.js               # Daily Kite Connect OAuth + TOTP token acquisition
│   ├── decode_migration_qr.js      # One-off: decode Google Authenticator migration QR
│   ├── package.json
│   └── tijori-finance-mcp/         # clone from github.com/LaZZy0v0/tijori-finance-mcp
│
├── packages/                       # standalone Julia packages
│   ├── TijoriData/                 # Tijori Finance data client
│   ├── CompanyConfidence/          # Fraud/reliability scoring module
│   └── EarningsCalendar/           # NSE earnings event fetcher (no sidecar)
│       ├── Project.toml
│       ├── src/
│       │   ├── TijoriData.jl       # module entry point + exports
│       │   ├── types.jl            # all structs
│       │   ├── client.jl           # HTTP client, sidecar lifecycle, parsing helpers
│       │   ├── display.jl          # Base.show overrides for REPL
│       │   ├── company.jl          # search_company, get_overview, get_knowledge_base
│       │   ├── financials.jl       # get_financials, get_operational_metrics, get_fund_flow
│       │   ├── shareholding.jl     # get_shareholding (includes promoter pledging)
│       │   ├── documents.jl        # fetch_document (PDF text extraction)
│       │   └── screener.jl         # screen_companies, list_screens, search_fields
│       └── test/
│           └── runtests.jl         # unit tests (no sidecar) + integration tests
│
├── scripts/                        # standalone Julia scripts (not packages)
│   ├── generate_nse_list.jl           # builds data/nse_companies_latest.json
│   ├── run_confidence_checks.jl       # runs CompanyConfidence on top-N by market cap
│   ├── enrich_earnings_dates.jl       # projects next earnings date via Tijori history (run every 2 weeks)
│   └── generate_earnings_watchlist.jl # merges NSE calendar + projections → watchlist JSON
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

Every package has its own `docs/` directory (not yet created — add as packages grow).
Module-level docstrings describe the package's role in the pipeline and link to related
packages with `See also: [OtherModule](@ref)`.

## Package dependency rules

```
TijoriData        — data only, no trading logic
CompanyConfidence — depends on TijoriData
EarningsCalendar  — NSE data only, no dependencies on other packages
EarningsPredictor — depends on TijoriData + CompanyConfidence + EarningsCalendar (planned)
Backtest          — no external data dependencies (planned)
BrokerClient      — Kite Connect REST wrapper (planned)
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

## Daily session workflow

```bash
# 1. Acquire a fresh Kite Connect access token (valid for the trading day)
node sidecar/kite_setup.js              # reads .env, writes sidecar/kite_session.json

# 2. Start the sidecar (Julia will start it automatically via start!(), but you can also run it manually)
node sidecar/server_http.js              # default port 3001
PORT=3002 node sidecar/server_http.js   # custom port
```

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
EarningsSwingPredictor       ← options implied move + multi-source signals
      ↓
LLM synthesis (Claude API)
      ↓
SignalDatabase
      ↓
Self-improvement loop        ← analyze outcomes, retrain parameters
```

## Environment variables

Store all secrets in a `.env` file in the repo root (gitignored). `kite_setup.js`
loads it automatically; Julia code can use `DotEnv.jl` or read it manually.

| Variable | Used by | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | LLM calls (planned) | Claude API authentication |
| `KITE_API_KEY` | `kite_setup.js`, BrokerClient | Zerodha Kite Connect app key |
| `KITE_API_SECRET` | `kite_setup.js`, BrokerClient | Zerodha Kite Connect app secret |
| `KITE_USER_ID` | `kite_setup.js` | Zerodha client ID (e.g. AB1234) |
| `KITE_PASSWORD` | `kite_setup.js` | Zerodha login password |
| `KITE_TOTP_SECRET` | `kite_setup.js` | Base32 TOTP secret from authenticator app |
| `PORT` | sidecar | HTTP port (default 3001) |

The daily access token is **not** an env var — `kite_setup.js` writes it to
`sidecar/kite_session.json` (gitignored), and the sidecar exposes it at
`GET /kite/token`. The token is valid for one trading day.
