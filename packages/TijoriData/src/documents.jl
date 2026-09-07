"""
PDF document fetching.

Tijori PDFs (annual reports, concall transcripts, investor presentations) are
hosted on an authenticated CDN. This module fetches them through the sidecar's
live browser session, which holds the required Tijori credentials.
"""

"""
    fetch_document(url) -> DocumentText

Fetch and extract the full text of a Tijori Finance PDF.

`url` must be an authenticated files.tijorifinance.com URL as returned by
`get_knowledge_base`. Attempting to use this URL in a regular browser or HTTP
client without the active Tijori session will return 403.

The full text is suitable for passing to an LLM for analysis (auditor
qualifications, management guidance, related-party disclosures, etc.).

Note: Large annual reports (200+ pages) can take 10–20 seconds to extract.

# Example
```julia
kb   = get_knowledge_base("satyam-computer-services-limited")
ar   = kb.annual_reports[end]
doc  = fetch_document(ar.url)
println("Pages: ", doc.pages)
println(first(doc.text, 2000))   # first 2000 chars
```
"""
function fetch_document(url::String)::DocumentText
    isempty(strip(url)) && error("url must not be empty")
    raw = _post("/document", Dict("url" => url))
    return DocumentText(
        string(raw.url),
        Int(raw.pages),
        string(raw.text),
    )
end
