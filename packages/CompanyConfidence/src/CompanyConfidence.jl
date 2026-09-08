"""
    CompanyConfidence

Scores a company's reliability and flags fraud/manipulation signals before the
StockSwingPredictor spends effort on it. A company scoring below
`PASS_THRESHOLD` (40) should be skipped.

# Signals

| Signal            | Method                           | What it detects |
|:------------------|:---------------------------------|:----------------|
| Beneish M-Score   | 8-factor accrual ratio model     | Earnings manipulation via accounting distortions |
| CFO vs NI         | Cash-flow-vs-earnings divergence | Revenue inflation not backed by real cash |
| Promoter pledging | Shareholding trend analysis      | Promoter financial stress |
| Tijori forensics  | Quick-look assessment parser     | Audit concerns, related-party risks |
| ASM / GSM lists   | NSE API                          | Exchange surveillance flags |

# Quick start

```julia
using CompanyConfidence, TijoriData

TijoriData.start!()                          # start the sidecar first
report = analyze("infosys-limited")
report.score    # 0–100 (higher = more trustworthy)
report.pass     # false if score < PASS_THRESHOLD (40)

# Inspect individual signals
report.beneish            # BeneishResult
report.cashflow           # CashflowResult
report.pledging           # PledgingResult
report.forensics          # ForensicsResult
report.surveillance       # SurveillanceResult

# Run just one check (pure — no sidecar needed)
pl = get_financials("satyam-computer-services-limited", :pl)
bs = get_financials("satyam-computer-services-limited", :bs)
cf = get_financials("satyam-computer-services-limited", :cf)
beneish_score(pl, bs, cf)
```

# Setup for local development

Since TijoriData is a sibling package, run this once per Julia environment:

```julia
using Pkg
Pkg.develop(path="path/to/packages/TijoriData")
Pkg.develop(path="path/to/packages/CompanyConfidence")
```

# Constants

- `BENEISH_THRESHOLD = -1.78`  M-Score above this → likely earnings manipulator
- `PASS_THRESHOLD    = 40.0`   Companies below this score are skipped by StockSwingPredictor

See also: [TijoriData](@ref)
"""
module CompanyConfidence

using DataFrames
using Dates
using HTTP
using JSON3
using Printf
using Statistics
using TijoriData

include("types.jl")
include("helpers.jl")
include("beneish.jl")
include("cashflow.jl")
include("pledging.jl")
include("forensics.jl")
include("surveillance.jl")
include("score.jl")
include("display.jl")

# ── Constants ─────────────────────────────────────────────────────────────────

"""M-Score threshold from Beneish (1999): above this → likely earnings manipulator."""
const BENEISH_THRESHOLD = -1.78

"""Companies scoring below this are considered too risky for StockSwingPredictor."""
const PASS_THRESHOLD = 40.0

# ── Exports ───────────────────────────────────────────────────────────────────

export ConfidenceReport, BeneishResult, CashflowResult, PledgingResult
export ForensicsResult, SurveillanceResult
export BENEISH_THRESHOLD, PASS_THRESHOLD

export analyze
export beneish_score, cashflow_check, pledging_check, forensics_check, surveillance_check

end # module CompanyConfidence
