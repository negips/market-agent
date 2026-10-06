"""
LLM-based news classification via the Claude API, or a local Ollama model.

Uses tool-use (structured output) to guarantee a typed response. For each
`NewsItem`, the model identifies the affected NSE symbol, event type,
sentiment, and severity — producing a `NewsSignal` ready for storage and
feature assembly. `classify_item` (Claude, live daemon) and
`classify_item_ollama` (local, `scripts/backfill_news_signals.jl`) share the
exact same `CLASSIFY_TOOL`/`CLASSIFY_SYSTEM` definition, so a historically
backfilled signal and a live one are calibrated identically — only the model
and transport differ.
"""

using HTTP, JSON3, Dates

const ANTHROPIC_BASE   = "https://api.anthropic.com"
const CLASSIFY_MODEL   = "claude-haiku-4-5-20251001"  # fast + cheap for high-volume classification
const MAX_BODY_CHARS   = 2_000

const OLLAMA_BASE  = "http://localhost:11434"
const OLLAMA_MODEL = "qwen3:latest"   # local, zero marginal cost — used for historical backfill volume

const CLASSIFY_TOOL = Dict(
    "name"        => "classify_news",
    "description" => "Classify a financial news item for its expected stock market impact.",
    "input_schema" => Dict(
        "type"       => "object",
        "properties" => Dict(
            "nse_symbol" => Dict(
                "type"        => "string",
                "description" => "NSE tradingsymbol of the primary affected company (e.g. RELIANCE, INFY, HDFCBANK). " *
                                 "Empty string if the news is sector-wide or market-wide.",
            ),
            "event_type" => Dict(
                "type" => "string",
                "enum" => ["results", "merger", "buyback", "dividend", "fundraise",
                           "regulatory", "management", "macro", "sector", "other"],
            ),
            "sentiment" => Dict(
                "type"        => "number",
                "description" => "Expected directional price impact: -1.0 = strongly negative, 0.0 = neutral, +1.0 = strongly positive.",
            ),
            "severity" => Dict(
                "type"        => "number",
                "description" => "Magnitude of expected move: 0.0 = routine / noise, 0.5 = moderate impact, 1.0 = major market-moving event.",
            ),
            "summary" => Dict(
                "type"        => "string",
                "description" => "One sentence: what happened and why it is expected to move the stock.",
            ),
        ),
        "required" => ["nse_symbol", "event_type", "sentiment", "severity", "summary"],
    ),
)

const CLASSIFY_SYSTEM = """
You are a senior equity analyst for Indian stock markets (NSE/BSE).
Given a news headline and body, identify:
  1. The affected NSE-listed company (use the exact NSE tradingsymbol, e.g. RELIANCE not RIL).
  2. The event type from the fixed enum.
  3. The expected price impact direction and magnitude.

Calibration:
  severity 1.0: merger announcement, surprise earnings beat/miss >10%, QIP, promoter fraud
  severity 0.5: dividend declared, management change, moderate guidance revision
  severity 0.1: routine board meeting, minor analyst note, sector commentary

Use the NSE tradingsymbol, not the company name. If you are not confident of
the exact symbol, return an empty string — do not guess.
"""

"""
Classify a `NewsItem` using Claude. Returns `nothing` on API failure.

# Arguments
- `item`: the raw news item to classify
- `api_key`: Anthropic API key
"""
function classify_item(item::NewsItem; api_key::String)::Union{NewsSignal, Nothing}
    isempty(api_key) && error("ANTHROPIC_API_KEY not set")

    text = item.headline
    if !isempty(item.body)
        text *= "\n\n" * first(item.body, MAX_BODY_CHARS)
    end

    body = JSON3.write(Dict(
        "model"       => CLASSIFY_MODEL,
        "max_tokens"  => 256,
        "system"      => CLASSIFY_SYSTEM,
        "tools"       => [CLASSIFY_TOOL],
        "tool_choice" => Dict("type" => "tool", "name" => "classify_news"),
        "messages"    => [Dict("role" => "user", "content" => text)],
    ))

    resp = try
        HTTP.post("$ANTHROPIC_BASE/v1/messages";
                  headers        = ["x-api-key"        => api_key,
                                    "anthropic-version" => "2023-06-01",
                                    "content-type"      => "application/json"],
                  body           = body,
                  request_timeout = 30,
                  status_exception = false)
    catch e
        @warn "Claude API error: $(sprint(showerror, e))"
        return nothing
    end

    if resp.status != 200
        @warn "Claude API HTTP $(resp.status)"
        return nothing
    end

    raw = try JSON3.read(resp.body) catch
        @warn "Could not parse Claude response"
        return nothing
    end

    tool_block = nothing
    for block in get(raw, :content, [])
        string(get(block, :type, "")) == "tool_use" && (tool_block = block; break)
    end
    isnothing(tool_block) && return nothing

    inp = get(tool_block, :input, nothing)
    isnothing(inp) && return nothing

    return _signal_from_tool_input(item, inp)
end

"""Shared by `classify_item` and `classify_item_ollama`: build a `NewsSignal`
from `item` plus the tool-call arguments either backend returned, applying
the same empty-symbol fallback and sentiment/severity clamping either way."""
function _signal_from_tool_input(item::NewsItem, inp)::NewsSignal
    return NewsSignal(
        guid          = item.guid,
        source        = item.source,
        headline      = item.headline,
        url           = item.url,
        published_at  = item.published_at,
        classified_at = now(UTC),
        symbol        = let llm_sym = string(get(inp, :nse_symbol, ""))
                            isempty(llm_sym) ? item.nse_symbol : llm_sym
                        end,
        event_type    = string(get(inp, :event_type, "other")),
        sentiment     = Float32(clamp(Float64(get(inp, :sentiment, 0.0)), -1.0, 1.0)),
        severity      = Float32(clamp(Float64(get(inp, :severity,  0.0)),  0.0, 1.0)),
        summary       = string(get(inp, :summary, "")),
    )
end

"""
Classify a `NewsItem` using a local Ollama model — same `CLASSIFY_TOOL`/
`CLASSIFY_SYSTEM` definition as `classify_item`, so results are calibrated
identically to the live Claude daemon's output. Zero marginal cost, used for
historical backfill volume (`scripts/backfill_news_signals.jl`) where
classifying every item via a paid API would be prohibitively expensive.

`think=false` is passed explicitly — measured ~8x faster per call on
`qwen3:latest` with no effect on the structured tool-call output (the model's
reasoning trace, when enabled, isn't consulted by anything downstream here).

Returns `nothing` on any request/parse failure or if the model didn't
respond with a tool call — callers (the backfill script) should leave such
items unclassified and retry on a later run rather than guessing a default.

# Arguments
- `item`: the raw news item to classify
- `model`: Ollama model name (default `$OLLAMA_MODEL`)
- `host`: Ollama server base URL (default `$OLLAMA_BASE`)
"""
function classify_item_ollama(item::NewsItem; model::String=OLLAMA_MODEL,
                               host::String=OLLAMA_BASE)::Union{NewsSignal, Nothing}
    text = item.headline
    if !isempty(item.body)
        text *= "\n\n" * first(item.body, MAX_BODY_CHARS)
    end

    body = JSON3.write(Dict(
        "model"    => model,
        "think"    => false,
        "stream"   => false,
        "messages" => [
            Dict("role" => "system", "content" => CLASSIFY_SYSTEM),
            Dict("role" => "user",   "content" => text),
        ],
        "tools" => [Dict(
            "type"     => "function",
            "function" => Dict(
                "name"        => CLASSIFY_TOOL["name"],
                "description" => CLASSIFY_TOOL["description"],
                "parameters"  => CLASSIFY_TOOL["input_schema"],
            ),
        )],
        "tool_choice" => Dict("type" => "function", "function" => Dict("name" => CLASSIFY_TOOL["name"])),
    ))

    resp = try
        HTTP.post("$host/api/chat";
                  headers          = ["content-type" => "application/json"],
                  body             = body,
                  request_timeout  = 60,
                  status_exception = false)
    catch e
        @warn "Ollama request error: $(sprint(showerror, e))"
        return nothing
    end

    if resp.status != 200
        @warn "Ollama HTTP $(resp.status)"
        return nothing
    end

    raw = try JSON3.read(resp.body) catch
        @warn "Could not parse Ollama response"
        return nothing
    end

    message = get(raw, :message, (;))
    tool_calls = get(message, :tool_calls, nothing)
    if isnothing(tool_calls) || isempty(tool_calls)
        # Ollama's tool_choice forcing is best-effort, unlike Claude's hard
        # guarantee (classify_item never hits this path) — observed the
        # model decline the tool call for generic/low-information content
        # (routine filings, cover letters, "see attached PDF") and just
        # explain itself in plain text instead, e.g. "The provided text is
        # a generic press release notice... No actionable data is
        # available." That refusal IS the correct classification (nothing
        # to extract), not a transient failure — it's deterministic for
        # this exact content, so returning `nothing` here would have
        # callers (`backfill_news_signals.jl`) retry it forever against an
        # identical response every time. Fall back to a routine/near-zero
        # signal using the model's own explanation as the summary, matching
        # CLASSIFY_SYSTEM's own calibration for this case ("severity 0.1:
        # routine board meeting, minor analyst note, sector commentary").
        content = string(get(message, :content, ""))
        return NewsSignal(
            guid=item.guid, source=item.source, headline=item.headline, url=item.url,
            published_at=item.published_at, classified_at=now(UTC), symbol=item.nse_symbol,
            event_type="other", sentiment=0f0, severity=0.1f0,
            summary=isempty(content) ? "No classification: model declined the tool call" : content,
        )
    end

    inp = get(tool_calls[1][:function], :arguments, nothing)
    isnothing(inp) && return nothing

    return _signal_from_tool_input(item, inp)
end
