"""
LLM-based news classification via the Claude API.

Uses tool-use (structured output) to guarantee a typed response. For each
`NewsItem`, Claude identifies the affected NSE symbol, event type, sentiment,
and severity — producing a `NewsSignal` ready for storage and feature assembly.
"""

using HTTP, JSON3, Dates

const ANTHROPIC_BASE   = "https://api.anthropic.com"
const CLASSIFY_MODEL   = "claude-haiku-4-5-20251001"  # fast + cheap for high-volume classification
const MAX_BODY_CHARS   = 2_000

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
        text *= "\n\n" * item.body[1:min(MAX_BODY_CHARS, length(item.body))]
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

    return NewsSignal(
        guid          = item.guid,
        source        = item.source,
        headline      = item.headline,
        url           = item.url,
        published_at  = item.published_at,
        classified_at = now(UTC),
        symbol        = string(get(inp, :nse_symbol, "")),
        event_type    = string(get(inp, :event_type, "other")),
        sentiment     = Float32(clamp(Float64(get(inp, :sentiment, 0.0)), -1.0, 1.0)),
        severity      = Float32(clamp(Float64(get(inp, :severity,  0.0)),  0.0, 1.0)),
        summary       = string(get(inp, :summary, "")),
    )
end
