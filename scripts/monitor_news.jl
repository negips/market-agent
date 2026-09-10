"""
monitor_news.jl

Real-time news monitor for Indian equity markets.

Polls BSE corporate announcements (every 60 s) and financial news RSS feeds
(every 5 min), classifies each new item with Claude, and appends the result to
`website/data/news_signals.jsonl` as newline-delimited JSON.

The process runs until killed (Ctrl-C). It is safe to restart — already-seen
items are deduplicated in memory (not re-classified on restart, but the JSONL
file is append-only so no data is lost).

Usage:
  julia scripts/monitor_news.jl
  julia scripts/monitor_news.jl --all-hours    # don't restrict to market hours
  julia scripts/monitor_news.jl --bse-only     # skip RSS feeds
"""

using NewsMonitor, Dates

const REPO_ROOT = joinpath(@__DIR__, "..")

function load_api_key()::String
    key = get(ENV, "ANTHROPIC_API_KEY", "")
    !isempty(key) && return key
    env_path = joinpath(REPO_ROOT, ".env")
    isfile(env_path) || return ""
    for line in readlines(env_path)
        m = match(r"^ANTHROPIC_API_KEY\s*=\s*(.+)$", strip(line))
        isnothing(m) || return strip(string(m[1]))
    end
    return ""
end

function main()
    if "--help" in ARGS || "-h" in ARGS
        println("""
Usage:
  julia scripts/monitor_news.jl [FLAGS]

Flags:
  --all-hours   Poll outside of 09:00-16:30 IST market hours too.
  --bse-only    Skip RSS feeds; poll BSE corporate announcements only.

Output:
  website/data/news_signals.jsonl  — one JSON object per line
""")
        return
    end

    api_key = load_api_key()
    isempty(api_key) && error("ANTHROPIC_API_KEY not set. Add it to .env or export it.")

    all_hours = "--all-hours" in ARGS
    bse_only  = "--bse-only"  in ARGS

    rss_feeds = bse_only ? Tuple{String,String}[] : [
        ("ET", "https://economictimes.indiatimes.com/markets/rss.cms"),
        ("BS", "https://www.business-standard.com/rss/markets-106.rss"),
        ("MC", "https://www.moneycontrol.com/rss/marketreports.xml"),
    ]

    output_file = joinpath(REPO_ROOT, "website", "data", "news_signals.jsonl")

    config = PollerConfig(
        api_key            = api_key,
        output_file        = output_file,
        bse_interval       = 60,
        rss_interval       = 300,
        rss_feeds          = rss_feeds,
        market_hours_only  = !all_hours,
    )

    @info "Starting news monitor"
    @info "Output → $output_file"
    bse_only && @info "Mode: BSE only (RSS disabled)"
    all_hours && @info "Mode: all hours (not restricted to market hours)"

    run_poller(config)
end

main()
