"""Types for the NewsMonitor package."""

using Dates

"""
Raw news item fetched from any source, before LLM classification.
"""
Base.@kwdef struct NewsItem
    guid::String
    source::String          # "BSE" | "RSS:ET" | "RSS:BS" | "RSS:MC"
    headline::String
    body::String            # announcement text or RSS description (HTML stripped)
    url::String
    published_at::DateTime
    bse_code::String = ""   # BSE scripcode when available (BSE source only)
end

"""
LLM-classified news item. Written as one JSON line to `news_signals.jsonl`.
Used at inference time to attach recent news context to a price window.
"""
Base.@kwdef struct NewsSignal
    guid::String
    source::String
    headline::String
    url::String
    published_at::DateTime
    classified_at::DateTime
    symbol::String          # NSE tradingsymbol, or "" for market-wide / unresolved
    event_type::String      # see CLASSIFY_TOOL for the full enum
    sentiment::Float32      # -1.0 (bad) … +1.0 (good) expected price impact
    severity::Float32       # 0 (noise) … 1 (major market-moving)
    summary::String         # one-sentence description of the event and expected move
end

"""Configuration for `run_poller`."""
Base.@kwdef struct PollerConfig
    api_key::String
    output_file::String
    bse_interval::Int = 60          # seconds between BSE polls
    rss_interval::Int = 300         # seconds between each RSS feed poll
    rss_feeds::Vector{Tuple{String,String}} = [
        ("ET",  "https://economictimes.indiatimes.com/markets/rss.cms"),
        ("BS",  "https://www.business-standard.com/rss/markets-106.rss"),
        ("MC",  "https://www.moneycontrol.com/rss/marketreports.xml"),
    ]
    market_hours_only::Bool = true
end
