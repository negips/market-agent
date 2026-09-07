"""
REPL display methods for TijoriData types.

Julia calls `Base.show(io, mime, obj)` when printing in the REPL.
These methods produce concise, readable output without printing the raw struct.
DataFrames use their own built-in display — no overrides needed there.
"""

# ── SearchResult ──────────────────────────────────────────────────────────────

function Base.show(io::IO, r::SearchResult)
    sym = isnothing(r.symbol) ? "" : " [$(r.symbol)]"
    print(io, r.name, sym, "  →  slug: \"", r.slug, "\"")
end

function Base.show(io::IO, ::MIME"text/plain", results::Vector{SearchResult})
    println(io, length(results), " result(s):")
    for (i, r) in enumerate(results)
        sym = isnothing(r.symbol) ? "" : "  [$(r.symbol)]"
        exch = isnothing(r.exchange) ? "" : " · $(r.exchange)"
        println(io, "  $i. $(r.name)$sym$exch")
        println(io, "     slug: \"$(r.slug)\"")
    end
end

# ── CompanyOverview ───────────────────────────────────────────────────────────

function Base.show(io::IO, ::MIME"text/plain", ov::CompanyOverview)
    sym  = isnothing(ov.symbol) ? "" : " [$(ov.symbol)]"
    bank = ov.is_banking ? " · Banking/NBFC" : ""
    println(io, "─"^60)
    println(io, ov.company, sym, bank)
    println(io, "slug: $(ov.slug)", isnothing(ov.company_id) ? "" : "  |  id: $(ov.company_id)")
    println(io, "─"^60)

    # Market cap & PE
    mcap_str = isnothing(ov.mcap) ? "N/A" : "₹$(round(ov.mcap, digits=0)) Cr"
    pe_str   = isnothing(ov.pe)   ? "N/A" : string(round(ov.pe, digits=1))
    println(io, "Market Cap: $mcap_str   P/E: $pe_str")

    # Key ratios (show up to 8)
    if !isempty(ov.ratios)
        println(io)
        println(io, "Key Ratios:")
        pairs_list = collect(ov.ratios)
        for (label, val) in first(pairs_list, 8)
            @printf(io, "  %-28s %s\n", label * ":", val)
        end
        length(pairs_list) > 8 && println(io, "  … $(length(pairs_list) - 8) more ratios")
    end

    # Forensics score
    if !isnothing(ov.forensics)
        println(io)
        println(io, "Tijori Forensics:")
        for (k, v) in ov.forensics
            @printf(io, "  %-28s %s\n", string(k) * ":", string(v))
        end
    end
    print(io, "─"^60)
end

function Base.show(io::IO, ov::CompanyOverview)
    sym = isnothing(ov.symbol) ? "" : " [$(ov.symbol)]"
    print(io, "CompanyOverview(", ov.company, sym, ")")
end

# ── KnowledgeBase ─────────────────────────────────────────────────────────────

function Base.show(io::IO, ::MIME"text/plain", kb::KnowledgeBase)
    println(io, "KnowledgeBase — $(kb.slug)")
    println(io, "─"^50)
    _show_docs(io, "Annual Reports",          kb.annual_reports)
    _show_docs(io, "Earnings Releases",        kb.earnings_releases)
    _show_docs(io, "Investor Presentations",   kb.investor_presentations)
    _show_docs(io, "Conference Calls",         kb.conference_calls)
    println(io)
    print(io, "Use fetch_document(doc.url) to retrieve full text.")
end

function _show_docs(io::IO, title::String, docs::Vector{Document})
    isempty(docs) && return
    println(io, "\n  $title ($(length(docs))):")
    for (i, d) in enumerate(docs)
        println(io, "    $i. $(d.period)")
    end
end

function Base.show(io::IO, kb::KnowledgeBase)
    total = length(kb.annual_reports) + length(kb.earnings_releases) +
            length(kb.investor_presentations) + length(kb.conference_calls)
    print(io, "KnowledgeBase($(kb.slug), $total documents)")
end

# ── DocumentText ──────────────────────────────────────────────────────────────

function Base.show(io::IO, ::MIME"text/plain", doc::DocumentText)
    println(io, "DocumentText — $(doc.pages) pages")
    println(io, "URL: $(doc.url)")
    println(io, "─"^50)
    preview = first(doc.text, 500)
    println(io, preview)
    length(doc.text) > 500 && print(io, "\n… $(length(doc.text)) chars total. Access via doc.text")
end

function Base.show(io::IO, doc::DocumentText)
    print(io, "DocumentText($(doc.pages) pages, $(length(doc.text)) chars)")
end

# ── MetricSeries ──────────────────────────────────────────────────────────────

function Base.show(io::IO, ::MIME"text/plain", m::MetricSeries)
    unit = isnothing(m.unit) ? "" : " ($(m.unit))"
    val  = isnothing(m.latest_value) ? "N/A" : string(round(m.latest_value, digits=2))
    date = isnothing(m.latest_date) ? "" : " as of $(m.latest_date)"
    println(io, m.name, unit)
    println(io, "  Latest: $val$date")
    println(io, "  $(nrow(m.history)) data points in history")
end

function Base.show(io::IO, m::MetricSeries)
    val = isnothing(m.latest_value) ? "N/A" : string(round(m.latest_value, digits=2))
    print(io, "MetricSeries(\"$(m.name)\", latest=$(val))")
end

# ── ScreenResult ──────────────────────────────────────────────────────────────

function Base.show(io::IO, ::MIME"text/plain", r::ScreenResult)
    q = isnothing(r.query) ? "" : "  Query: \"$(r.query)\"\n"
    println(io, "ScreenResult: $(r.total_results) total match(es)")
    print(io, q)
    println(io, "Showing rows $(r.offset + 1)–$(r.offset + r.returned)")
    println(io, "─"^50)
    isempty(r.data) || show(io, MIME"text/plain"(), r.data)
end

function Base.show(io::IO, r::ScreenResult)
    print(io, "ScreenResult($(r.total_results) total, $(r.returned) returned)")
end
