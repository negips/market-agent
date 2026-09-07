# market-agent

Self-improving trading agent for Indian stock markets (NSE/BSE), built entirely in Julia.

## Project goal

Identify stocks likely to make large price moves around earnings events, using a
multi-source data pipeline, a company fraud/confidence filter, and an LLM reasoning
layer. Eventually self-improves by analyzing its own trade history.

## Repository layout

```
market-agent/
├── sidecar/                        # Node.js HTTP wrapper around Tijori Finance
│   ├── server_http.js              # Express server — exposes Tijori tools as REST
│   ├── package.json
│   └── tijori-finance-mcp/         # clone from github.com/LaZZy0v0/tijori-finance-mcp
│
├── packages/                       # standalone Julia packages
│   └── TijoriData/                 # Tijori Finance data client
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
TijoriData       — data only, no trading logic
CompanyConfidence — depends on TijoriData (planned)
EarningsPredictor — depends on TijoriData + CompanyConfidence (planned)
Backtest          — no external data dependencies (planned)
BrokerClient      — Kite Connect REST wrapper (planned)
```

## Sidecar setup (one-time)

```bash
cd sidecar
git clone https://github.com/LaZZy0v0/tijori-finance-mcp.git
cd tijori-finance-mcp && node setup.js   # opens browser for Tijori login
cd ..
npm install                              # installs express
```

Start the sidecar:
```bash
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

| Variable | Used by | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | LLM calls (planned) | Claude API authentication |
| `ZERODHA_API_KEY` | BrokerClient (planned) | Zerodha Kite Connect app key |
| `ZERODHA_API_SECRET` | BrokerClient (planned) | Zerodha Kite Connect app secret |
| `ZERODHA_ACCESS_TOKEN` | BrokerClient (planned) | Daily session token (regenerated each trading day via OAuth) |
| `TIJORI_EMAIL` | sidecar setup | Tijori Finance login |
| `TIJORI_PASSWORD` | sidecar setup | Tijori Finance login |
| `PORT` | sidecar | HTTP port (default 3001) |
