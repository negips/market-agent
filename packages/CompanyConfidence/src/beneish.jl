"""
Beneish M-Score: 8-factor earnings manipulation detection model.

Reference: Beneish (1999), "The Detection of Earnings Manipulation".
Probit model coefficients and -1.78 threshold are from that paper.

Not applicable to banking and NBFC companies, whose balance sheets omit
traditional COGS, receivables, and fixed-asset conventions.
"""

# ── Line item candidates ──────────────────────────────────────────────────────
# Each vector lists Tijori metric label substrings (case-insensitive) to try
# in order. The first matching row wins.

const _REVENUE    = String["net revenue", "revenue from operations", "total revenue",
                            "net sales", "total income from operations", "sales"]
const _COGS       = String["cost of revenue", "cost of goods sold", "material cost",
                            "raw material", "direct cost", "cost of sales",
                            "purchase of stock", "cost of production"]
const _SGA        = String["selling and distribution", "selling & distribution",
                            "sg&a", "general and administrative",
                            "employee benefit", "personnel cost", "staff cost",
                            "selling, general", "employee cost"]
const _DEPR       = String["depreciation and amortization", "depreciation & amortization",
                            "depreciation", "amortization"]
const _NET_INCOME = String["profit after tax", "net profit", "pat",
                            "profit for the year", "net income", "net earnings"]
const _RECV       = String["trade receivables", "debtors", "accounts receivable",
                            "receivables"]
const _CURR_ASSETS = String["total current assets", "current assets"]
const _PPE        = String["net fixed assets", "property, plant", "tangible assets",
                            "net block", "fixed assets"]
const _TOT_ASSETS = String["total assets", "balance sheet total", "assets"]
const _LTD        = String["long term borrowings", "long-term borrowings",
                            "non-current borrowings", "long term debt",
                            "secured loans", "unsecured loans"]
const _CURR_LIAB  = String["total current liabilities", "current liabilities"]
const _CFO        = String["cash from operations", "net cash from operating",
                            "operating cash flow", "cash generated from operations",
                            "cash flow from operations", "cash from operating"]

# ── Main function ─────────────────────────────────────────────────────────────

"""
    beneish_score(pl, bs, cf; is_banking=false) -> BeneishResult

Compute the Beneish M-Score for a company using its annual financial statements.

`pl`, `bs`, `cf` are DataFrames as returned by `TijoriData.get_financials`.
Uses the two most recent fiscal years found in `pl` as year t and t-1.

Returns a `BeneishResult` with `applicable=false` for banking/NBFC companies.

# M-Score formula (Beneish 1999)

```
M = -4.84 + 0.920·DSRI + 0.528·GMI + 0.404·AQI + 0.892·SGI
         + 0.115·DEPI  − 0.172·SGAI + 4.679·TATA − 0.327·LVGI
```

M > -1.78 → likely manipulator

# Example
```julia
pl = get_financials("satyam-computer-services-limited", :pl)
bs = get_financials("satyam-computer-services-limited", :bs)
cf = get_financials("satyam-computer-services-limited", :cf)
r  = beneish_score(pl, bs, cf)
r.m_score        # e.g. -1.54 (above threshold → flagged)
r.missing_items  # line items that could not be found
```
"""
function beneish_score(pl::DataFrame, bs::DataFrame, cf::DataFrame;
                       is_banking::Bool=false)::BeneishResult

    if is_banking
        return BeneishResult(false, nothing, nothing, false,
                             nothing, nothing, nothing, nothing,
                             nothing, nothing, nothing, nothing,
                             nothing, nothing, String[])
    end

    # Identify annual year columns (newest first); excludes mid-year quarterly columns
    pl_years = _annual_year_cols(pl)
    if length(pl_years) < 2
        return BeneishResult(true, nothing, nothing, false,
                             nothing, nothing, nothing, nothing,
                             nothing, nothing, nothing, nothing,
                             nothing, nothing, ["Need ≥ 2 years of P&L data"])
    end

    yt  = pl_years[1]   # year t   (most recent)
    yt1 = pl_years[2]   # year t-1 (prior year)

    miss = String[]

    # ── P&L items ─────────────────────────────────────────────────────────────
    rev_t  = _extract(pl, _REVENUE,    yt,  "Revenue",          miss)
    rev_t1 = _extract(pl, _REVENUE,    yt1, "Revenue (prior)",  miss)
    cgs_t  = _extract(pl, _COGS,       yt,  "COGS",             miss)
    cgs_t1 = _extract(pl, _COGS,       yt1, "COGS (prior)",     miss)
    sga_t  = _extract(pl, _SGA,        yt,  "SGA",              miss)
    sga_t1 = _extract(pl, _SGA,        yt1, "SGA (prior)",      miss)
    dep_t  = _extract(pl, _DEPR,       yt,  "Depreciation",     miss)
    dep_t1 = _extract(pl, _DEPR,       yt1, "Depreciation (prior)", miss)
    ni_t   = _extract(pl, _NET_INCOME, yt,  "Net Income",       miss)

    # ── BS items ──────────────────────────────────────────────────────────────
    rec_t  = _extract(bs, _RECV,       yt,  "Receivables",       miss)
    rec_t1 = _extract(bs, _RECV,       yt1, "Receivables (prior)", miss)
    ca_t   = _extract(bs, _CURR_ASSETS, yt,  "Current Assets",   miss)
    ca_t1  = _extract(bs, _CURR_ASSETS, yt1, "Current Assets (prior)", miss)
    ppe_t  = _extract(bs, _PPE,        yt,  "PPE",              miss)
    ppe_t1 = _extract(bs, _PPE,        yt1, "PPE (prior)",      miss)
    ta_t   = _extract(bs, _TOT_ASSETS, yt,  "Total Assets",     miss)
    ta_t1  = _extract(bs, _TOT_ASSETS, yt1, "Total Assets (prior)", miss)
    ltd_t  = _extract(bs, _LTD,        yt,  "LT Debt",          miss)
    ltd_t1 = _extract(bs, _LTD,        yt1, "LT Debt (prior)",  miss)
    cl_t   = _extract(bs, _CURR_LIAB,  yt,  "Current Liabilities", miss)
    cl_t1  = _extract(bs, _CURR_LIAB,  yt1, "Current Liabilities (prior)", miss)

    # ── CF items — match to yt by year number ─────────────────────────────────
    cf_years = _annual_year_cols(cf)
    cfo_t = if isempty(cf_years)
        push!(miss, "CFO")
        nothing
    else
        _extract(cf, _CFO, cf_years[1], "CFO", miss)
    end

    # ── Compute 8 indices ─────────────────────────────────────────────────────

    # DSRI: Days Sales Receivable Index
    dsri = _sdiv(_sdiv(rec_t, rev_t), _sdiv(rec_t1, rev_t1))

    # GMI: Gross Margin Index  (deteriorating margin → more pressure to manipulate)
    gm_t  = _sdiv(_ssub(rev_t,  cgs_t),  rev_t)
    gm_t1 = _sdiv(_ssub(rev_t1, cgs_t1), rev_t1)
    gmi   = _sdiv(gm_t1, gm_t)

    # AQI: Asset Quality Index  (increase → more intangibles / deferred costs)
    hard_t  = _sadd(ca_t,  ppe_t)
    hard_t1 = _sadd(ca_t1, ppe_t1)
    nca_t   = isnothing(hard_t)  || isnothing(ta_t)  ? nothing : 1.0 - hard_t  / ta_t
    nca_t1  = isnothing(hard_t1) || isnothing(ta_t1) ? nothing : 1.0 - hard_t1 / ta_t1
    aqi     = _sdiv(nca_t, nca_t1)

    # SGI: Sales Growth Index  (high growth correlates with manipulation pressure)
    sgi = _sdiv(rev_t, rev_t1)

    # DEPI: Depreciation Index  (>1 → assets depreciated more slowly → inflate earnings)
    dep_rate_t  = _sdiv(dep_t,  _sadd(dep_t,  ppe_t))
    dep_rate_t1 = _sdiv(dep_t1, _sadd(dep_t1, ppe_t1))
    depi = _sdiv(dep_rate_t1, dep_rate_t)

    # SGAI: SGA Expenses Index
    sgai = _sdiv(_sdiv(sga_t, rev_t), _sdiv(sga_t1, rev_t1))

    # LVGI: Leverage Index  (rising debt relative to assets)
    lev_t  = _sdiv(_sadd(ltd_t,  cl_t),  ta_t)
    lev_t1 = _sdiv(_sadd(ltd_t1, cl_t1), ta_t1)
    lvgi   = _sdiv(lev_t, lev_t1)

    # TATA: Total Accruals to Total Assets  (positive → accrual-inflated earnings)
    tata = _sdiv(_ssub(ni_t, cfo_t), ta_t)

    # ── M-Score ───────────────────────────────────────────────────────────────
    indices = [dsri, gmi, aqi, sgi, depi, sgai, lvgi, tata]
    m_score = if any(isnothing, indices)
        nothing
    else
        -4.840 + 0.920*dsri + 0.528*gmi + 0.404*aqi + 0.892*sgi +
         0.115*depi - 0.172*sgai + 4.679*tata - 0.327*lvgi
    end

    is_manip = isnothing(m_score) ? nothing : m_score > BENEISH_THRESHOLD

    return BeneishResult(
        true, m_score, is_manip, is_manip === true,
        dsri, gmi, aqi, sgi, depi, sgai, lvgi, tata,
        string(yt), string(yt1),
        miss,
    )
end
