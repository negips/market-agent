"""
Main polling loop — coordinates BSE and RSS pollers with the LLM classifier.

Architecture: two concurrent task types feed a shared `Channel{NewsItem}`.
The main task drains the channel, calling Claude for each new item and
appending results to a JSONL file.

Tasks share a `seen_guids` set for deduplication. This is safe without a lock
because Julia's `@async` tasks are cooperatively scheduled — they only switch
at I/O or sleep boundaries, so the Set operations are never interleaved.
"""

using Dates, JSON3

const MAX_SEEN_GUIDS = 50_000  # ring-clear when exceeded to bound memory

"""
Check whether the current wall-clock time falls within Indian market hours.
Covers 09:00–16:30 IST (the regular session plus post-close announcements).
"""
function is_market_hours()::Bool
    u = now(UTC)
    dayofweek(u) ∈ (Saturday, Sunday) && return false
    # IST = UTC + 5:30 = UTC + 330 min
    t = (hour(u) * 60 + minute(u) + 330) % (24 * 60)
    return 540 <= t <= 990   # 09:00–16:30 IST
end

"""
Append one `NewsSignal` as a JSON line to `path`.
"""
function append_signal(sig::NewsSignal, path::String)
    mkpath(dirname(path))
    open(path, "a") do io
        JSON3.write(io, Dict(
            "guid"          => sig.guid,
            "source"        => sig.source,
            "headline"      => sig.headline,
            "url"           => sig.url,
            "published_at"  => string(sig.published_at),
            "classified_at" => string(sig.classified_at),
            "symbol"        => sig.symbol,
            "event_type"    => sig.event_type,
            "sentiment"     => sig.sentiment,
            "severity"      => sig.severity,
            "summary"       => sig.summary,
        ))
        write(io, '\n')
    end
end

"""
Run the news polling daemon. Blocks until the process is killed.

Starts one BSE poller task and one task per RSS feed, all feeding a shared
`Channel`. The main thread drains the channel: classifies each item with
Claude and writes the result to `config.output_file`.

# Arguments
- `config`: `PollerConfig` with API key, intervals, and feed list.
"""
function run_poller(config::PollerConfig)
    seen = Set{String}()
    ch   = Channel{NewsItem}(256)

    function dedup_and_send(items)
        for item in items
            item.guid ∈ seen && continue
            push!(seen, item.guid)
            length(seen) > MAX_SEEN_GUIDS && empty!(seen)
            put!(ch, item)
        end
    end

    # ── BSE poller ────────────────────────────────────────────────────────────
    @async while true
        try
            if !config.market_hours_only || is_market_hours()
                items = fetch_bse_announcements()
                dedup_and_send(items)
            end
        catch e
            @warn "BSE poller error: $(sprint(showerror, e))"
        end
        sleep(config.bse_interval)
    end

    # ── RSS pollers (one task per feed) ───────────────────────────────────────
    for (label, url) in config.rss_feeds
        @async while true
            try
                if !config.market_hours_only || is_market_hours()
                    items = fetch_rss(url, label)
                    dedup_and_send(items)
                end
            catch e
                @warn "RSS poller error ($label): $(sprint(showerror, e))"
            end
            sleep(config.rss_interval)
        end
    end

    @info "News monitor started — BSE every $(config.bse_interval)s, " *
          "RSS every $(config.rss_interval)s"
    config.market_hours_only && @info "Polling restricted to 09:00–16:30 IST"

    # ── LLM classifier (drains channel) ───────────────────────────────────────
    for item in ch
        sig = try
            classify_item(item; api_key=config.api_key)
        catch e
            @warn "Classification error for $(item.guid): $(sprint(showerror, e))"
            nothing
        end

        isnothing(sig) && continue

        append_signal(sig, config.output_file)

        sym_tag  = isempty(sig.symbol) ? "market" : sig.symbol
        sent_str = sig.sentiment >= 0 ? "+$(round(sig.sentiment, digits=2))" :
                                         "$(round(sig.sentiment, digits=2))"
        @info "[$(sig.source)] $sym_tag  $(sig.event_type)  " *
              "sent=$sent_str  sev=$(round(sig.severity, digits=2))  — $(sig.summary)"
    end
end
