"""
Types for all data returned by the Tijori Finance sidecar.

Each type corresponds to one endpoint. Numeric fields that Tijori may not
populate for all companies (e.g. PE for banks) are Union{T, Nothing}.
"""

# ── Exceptions ────────────────────────────────────────────────────────────────

"""
    TijoriError(message)

Raised when the sidecar returns `{ ok: false, error: "..." }` or when the
HTTP connection to the sidecar fails. Check `err.message` for details.
"""
struct TijoriError <: Exception
    message::String
end

Base.showerror(io::IO, e::TijoriError) = print(io, "TijoriError: ", e.message)

# ── Search ────────────────────────────────────────────────────────────────────

"""
    SearchResult

One company returned from `search_company`. Use `slug` as the identifier for
all subsequent data calls.

Fields:
- `name`     — display name (e.g. "HDFC Bank Ltd")
- `slug`     — URL slug used as the company identifier (e.g. "hdfc-bank-limited")
- `symbol`   — NSE/BSE ticker if available
- `exchange` — "NSE", "BSE", or nothing
"""
struct SearchResult
    name::String
    slug::String
    symbol::Union{String, Nothing}
    exchange::Union{String, Nothing}
end

# ── Company overview ──────────────────────────────────────────────────────────

"""
    CompanyOverview

Snapshot of a company: key ratios, Tijori's forensics score, and basic
identifiers. Returned by `get_overview(slug)`.

Fields:
- `slug`       — company identifier
- `company`    — full company name
- `symbol`     — NSE ticker
- `company_id` — Tijori's internal numeric ID (needed for `get_fund_flow`)
- `is_banking` — true for banks and NBFCs (affects which ratios are relevant)
- `mcap`       — market cap in ₹ Cr (nothing if unavailable)
- `pe`         — trailing P/E ratio (nothing for banks, loss-making companies)
- `ratios`     — dict of label → value string as shown on Tijori
- `forensics`  — Tijori's quick_look fraud/forensics assessment (nothing if absent)
"""
struct CompanyOverview
    slug::String
    company::String
    symbol::Union{String, Nothing}
    company_id::Union{Int, Nothing}
    is_banking::Bool
    mcap::Union{Float64, Nothing}
    pe::Union{Float64, Nothing}
    ratios::OrderedDict{String, String}
    forensics::Union{Dict{String, Any}, Nothing}
end

# ── Knowledge base ────────────────────────────────────────────────────────────

"""
    Document

A single document from Tijori's knowledge base.

Fields:
- `period` — human-readable period label (e.g. "FY24", "Oct 2024")
- `url`    — authenticated CDN URL; pass directly to `fetch_document`
"""
struct Document
    period::String
    url::String
end

"""
    KnowledgeBase

All investor documents available for a company on Tijori.
Returned by `get_knowledge_base(slug)`.

Fields:
- `slug`                   — company identifier
- `annual_reports`         — Vector{Document}
- `earnings_releases`      — Vector{Document}
- `investor_presentations` — Vector{Document}
- `conference_calls`       — Vector{Document}

Use `fetch_document(doc.url)` to retrieve the full text of any document.
"""
struct KnowledgeBase
    slug::String
    annual_reports::Vector{Document}
    earnings_releases::Vector{Document}
    investor_presentations::Vector{Document}
    conference_calls::Vector{Document}
end

# ── Document text ─────────────────────────────────────────────────────────────

"""
    DocumentText

Full extracted text of a Tijori PDF document.
Returned by `fetch_document(url)`.

Fields:
- `url`   — source URL
- `pages` — page count
- `text`  — full plain-text content of the PDF
"""
struct DocumentText
    url::String
    pages::Int
    text::String
end

# ── Operational metric ────────────────────────────────────────────────────────

"""
    MetricSeries

One operational KPI with its full historical time series.

Fields:
- `name`         — metric name (e.g. "Passenger Load Factor")
- `unit`         — unit string (e.g. "%", "₹ Cr")
- `latest_value` — most recent value (nothing if unavailable)
- `latest_date`  — YYYY-MM date string of the latest data point
- `history`      — DataFrame with columns [:date, :value] sorted ascending
"""
struct MetricSeries
    name::String
    unit::Union{String, Nothing}
    latest_value::Union{Float64, Nothing}
    latest_date::Union{String, Nothing}
    history::DataFrame
end

# ── Screen result ─────────────────────────────────────────────────────────────

"""
    ScreenResult

Result of running `screen_companies`. The full match count is in
`total_results`; the returned slice (up to the requested `limit`) is in `data`.

Fields:
- `total_results` — total number of companies matching the query
- `returned`      — number of rows in `data` (≤ total_results)
- `offset`        — starting row index of this page (0-based)
- `query`         — the query string that was run (nothing for presets)
- `data`          — DataFrame with one row per company
"""
struct ScreenResult
    total_results::Int
    returned::Int
    offset::Int
    query::Union{String, Nothing}
    data::DataFrame
end
