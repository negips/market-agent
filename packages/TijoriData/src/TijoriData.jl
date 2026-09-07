"""
    TijoriData

Julia client for Tijori Finance data via the HTTP sidecar.

# Quick start

```julia
using TijoriData

# 1. Start the sidecar (one-time per Julia session)
TijoriData.start!("/path/to/Agents/sidecar")

# 2. Search for a company
results = search_company("HDFC Bank")
slug = results[1].slug        # "hdfc-bank-limited"

# 3. Fetch data
ov   = get_overview(slug)     # ratios, forensics score
pl   = get_financials(slug, :pl)   # P&L as DataFrame
sh   = get_shareholding(slug)      # promoter/FII/DII/pledging
kb   = get_knowledge_base(slug)    # annual reports, concalls
doc  = fetch_document(kb.conference_calls[1].url)  # full PDF text

# 4. Screen companies
r = screen_companies("( ROCE > 20 ) and ( Market Capitalization > 5000 )")
r.data   # DataFrame of matching companies
```

# Sidecar setup (one-time)

```bash
cd sidecar
git clone https://github.com/LaZZy0v0/tijori-finance-mcp.git
cd tijori-finance-mcp && node setup.js   # authenticates and saves session
cd ..
npm install                              # installs express
```

# If the sidecar is already running externally

```julia
using TijoriData
TijoriData.configure!(port=3001)                         # non-default port
TijoriData.configure!(sidecar_dir="/other/path/sidecar") # moved repo
is_running()                                              # true if reachable
```
"""
module TijoriData

using HTTP
using JSON3
using DataFrames
using Dates
using OrderedCollections
using Printf
using PrettyTables

# ── Includes (order matters: types and client must come before callers) ────────

include("types.jl")
include("client.jl")
include("display.jl")
include("company.jl")
include("financials.jl")
include("shareholding.jl")
include("documents.jl")
include("screener.jl")

# ── Public API ────────────────────────────────────────────────────────────────

export TijoriError

# Lifecycle
export start!, stop!, is_running, configure!

# Company
export search_company, get_overview, get_knowledge_base, resolve_id

# Financials
export get_financials, get_operational_metrics, get_fund_flow
export get_revenue_mix, get_market_share

# Shareholding
export get_shareholding

# Documents
export fetch_document

# Screener
export screen_companies, list_screens, search_fields

# Market & macro
export get_markets, get_sector_stocks, get_conglomerate_stocks
export get_macro_indicators, get_raw_materials

end # module TijoriData
