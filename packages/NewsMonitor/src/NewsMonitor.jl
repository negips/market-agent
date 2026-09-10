"""
    NewsMonitor

Real-time news polling and LLM-based market-impact classification for Indian
equities. Monitors BSE corporate announcements and financial news RSS feeds,
classifying each item with Claude to produce a structured `NewsSignal`.

Signals are written as newline-delimited JSON to `website/data/news_signals.jsonl`
and consumed by the `StockSwingPredictor` feature assembly pipeline.

## Quick start

```julia
using NewsMonitor

config = PollerConfig(
    api_key     = ENV["ANTHROPIC_API_KEY"],
    output_file = "website/data/news_signals.jsonl",
)
run_poller(config)   # blocks; kill with Ctrl-C
```

See also: [StockSwingPredictor](@ref)
"""
module NewsMonitor

include("types.jl")
include("bse.jl")
include("rss.jl")
include("llm_classify.jl")
include("poller.jl")

export NewsItem, NewsSignal, PollerConfig
export fetch_bse_announcements, fetch_rss
export classify_item
export run_poller, is_market_hours, append_signal

end
