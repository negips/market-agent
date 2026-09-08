"""
Extract scalar features from financial documents using the Claude API.

Uses Claude's tool-use (structured output) mode to guarantee a well-typed
JSON response matching the `LLMFeatures` schema. The LLM reads conference
call transcripts or earnings releases and outputs 14 scalar signals.

The prompt is anchored with examples so that scores remain consistent across
different quarters and companies — this is essential for the NN to learn
stable cross-sectional relationships.
"""

using HTTP, JSON3, Dates

const ANTHROPIC_BASE = "https://api.anthropic.com"
const EXTRACT_MODEL  = "claude-sonnet-4-6"
const MAX_DOC_CHARS  = 14_000   # truncate long PDFs to keep prompt within token budget

# ── Tool schema (structured output via Anthropic tool-use) ───────────────────

const EXTRACT_TOOL = Dict(
    "name"        => "extract_signals",
    "description" => "Extract scalar trading signals from a financial document. " *
                     "Return exact values — do not round or summarise.",
    "input_schema" => Dict(
        "type" => "object",
        "properties" => Dict(
            "management_tone" => Dict(
                "type"        => "number",
                "description" => "Overall management sentiment. -1.0 = deeply bearish / crisis mode. 0.0 = neutral / cautious. +1.0 = confident / optimistic about growth.",
            ),
            "guidance_direction" => Dict(
                "type"        => "number",
                "description" => "-1 if guidance was cut / revenue/profit outlook lowered. 0 if no guidance given. +1 if guidance was raised or reiterated positively.",
            ),
            "guidance_specificity" => Dict(
                "type"        => "number",
                "description" => "0 = no guidance. 1 = vague ('we expect growth'). 2 = directional with ranges. 3 = specific numbers given (e.g. '15-18% revenue growth in FY27').",
            ),
            "demand_outlook" => Dict(
                "type"        => "number",
                "description" => "Management's commentary on demand environment. -1 = deteriorating demand. 0 = stable. +1 = strong / accelerating demand.",
            ),
            "margin_commentary" => Dict(
                "type"        => "number",
                "description" => "Margin trajectory mentioned by management. -1 = significant pressure / expected contraction. 0 = stable. +1 = expanding / structural improvement.",
            ),
            "competitive_pressure" => Dict(
                "type"        => "number",
                "description" => "Intensity of competitive threats mentioned. 0 = none mentioned. 0.5 = some headwinds. 1.0 = severe / pricing war / market share loss.",
            ),
            "new_wins_announced" => Dict(
                "type"        => "number",
                "description" => "1 if major new orders, contracts, or customer wins were announced. 0 otherwise.",
            ),
            "capex_expansion" => Dict(
                "type"        => "number",
                "description" => "1 if significant new capital expenditure / capacity expansion was announced. 0 otherwise.",
            ),
            "buyback_or_dividend" => Dict(
                "type"        => "number",
                "description" => "1 if a buyback, special dividend, or material dividend increase was announced. 0 otherwise.",
            ),
            "mgmt_language_hedging" => Dict(
                "type"        => "number",
                "description" => "0 = management speaks confidently with direct statements. 1 = heavily hedged language, many caveats, uncertainty acknowledged throughout.",
            ),
            "auditor_concerns" => Dict(
                "type"        => "number",
                "description" => "1 if the document mentions auditor qualifications, CARO observations, or going-concern doubts. 0 otherwise.",
            ),
            "related_party_flags" => Dict(
                "type"        => "number",
                "description" => "1 if concerning related-party transactions or inter-company loans are discussed. 0 otherwise.",
            ),
            "contingent_liability_flag" => Dict(
                "type"        => "number",
                "description" => "1 if material contingent liabilities (tax disputes, litigation) are highlighted as a risk. 0 otherwise.",
            ),
            "extraction_confidence" => Dict(
                "type"        => "number",
                "description" => "Your confidence in the above estimates given the document quality and relevance. 0 = guessing / document not useful. 1 = very clear signals throughout.",
            ),
        ),
        "required" => ["management_tone", "guidance_direction", "guidance_specificity",
                       "demand_outlook", "margin_commentary", "competitive_pressure",
                       "new_wins_announced", "capex_expansion", "buyback_or_dividend",
                       "mgmt_language_hedging", "auditor_concerns", "related_party_flags",
                       "contingent_liability_flag", "extraction_confidence"],
    ),
)

const SYSTEM_PROMPT = """
You are a senior equity analyst extracting structured signals from Indian company
financial documents (conference call transcripts, earnings releases, annual reports).

Calibration anchors — use these as reference points when scoring:
  management_tone +0.8: "We are extremely pleased with our performance. Demand is
    robust across segments and we are raising our revenue guidance."
  management_tone  0.0: "We delivered results broadly in line with expectations.
    The outlook remains uncertain and we are monitoring developments closely."
  management_tone -0.7: "We faced significant headwinds this quarter. Volume was
    impacted and we see continued pressure in the near term."

  guidance_direction +1: Management explicitly says they are raising guidance or
    reiterating above-consensus growth targets.
  guidance_direction  0: No forward guidance is provided in the document.
  guidance_direction -1: Management lowers revenue or profit guidance.

Be precise. Return only values in the ranges described. If information is not
present in the document, return 0 for boolean fields and 0.0 for continuous fields.
"""

# ── API call ──────────────────────────────────────────────────────────────────

"""
Call Claude API to extract `LLMFeatures` from a text document.

# Arguments
- `text`: document text (will be truncated to MAX_DOC_CHARS)
- `doc_type`: "conference_call" | "earnings_release" | "annual_report"
- `doc_date`: when this document was published (used to compute doc_age_days)
- `api_key`: Anthropic API key (reads ENV["ANTHROPIC_API_KEY"] if not provided)
- `as_of`: the date of the training example (default: today())

# Returns
`LLMFeatures` or `MISSING_LLM` on failure.
"""
function extract_features(text::String, doc_type::String, doc_date::Date;
                          api_key::String = get(ENV, "ANTHROPIC_API_KEY", ""),
                          as_of::Date = today())::LLMFeatures

    isempty(api_key) && error("ANTHROPIC_API_KEY not set")
    isempty(strip(text)) && return MISSING_LLM

    truncated = text[1:min(MAX_DOC_CHARS, length(text))]
    user_msg  = "Document type: $doc_type\n\n$truncated"

    body = JSON3.write(Dict(
        "model"       => EXTRACT_MODEL,
        "max_tokens"  => 512,
        "system"      => SYSTEM_PROMPT,
        "tools"       => [EXTRACT_TOOL],
        "tool_choice" => Dict("type" => "tool", "name" => "extract_signals"),
        "messages"    => [Dict("role" => "user", "content" => user_msg)],
    ))

    resp = try
        HTTP.post("$ANTHROPIC_BASE/v1/messages";
                  headers = ["x-api-key"         => api_key,
                             "anthropic-version"  => "2023-06-01",
                             "content-type"       => "application/json"],
                  body    = body,
                  request_timeout = 60,
                  status_exception = false)
    catch e
        @warn "Claude API request failed: $(sprint(showerror, e))"
        return MISSING_LLM
    end

    if resp.status != 200
        @warn "Claude API HTTP $(resp.status): $(String(resp.body)[1:200])"
        return MISSING_LLM
    end

    raw = try
        JSON3.read(resp.body)
    catch
        @warn "Could not parse Claude API response"
        return MISSING_LLM
    end

    # Find the tool_use block in the response content array.
    tool_block = nothing
    for block in get(raw, :content, [])
        if string(get(block, :type, "")) == "tool_use"
            tool_block = block
            break
        end
    end

    isnothing(tool_block) && return MISSING_LLM
    inp = get(tool_block, :input, nothing)
    isnothing(inp) && return MISSING_LLM

    doc_age = Float32(Dates.value(as_of - doc_date))

    return LLMFeatures(
        Float32(clamp(get(inp, :management_tone,           0.0), -1.0, 1.0)),
        Float32(clamp(get(inp, :guidance_direction,        0.0), -1.0, 1.0)),
        Float32(clamp(get(inp, :guidance_specificity,      0.0),  0.0, 3.0) / 3.0),
        Float32(clamp(get(inp, :demand_outlook,            0.0), -1.0, 1.0)),
        Float32(clamp(get(inp, :margin_commentary,         0.0), -1.0, 1.0)),
        Float32(clamp(get(inp, :competitive_pressure,      0.0),  0.0, 1.0)),
        Float32(get(inp, :new_wins_announced,    0) != 0 ? 1.0 : 0.0),
        Float32(get(inp, :capex_expansion,       0) != 0 ? 1.0 : 0.0),
        Float32(get(inp, :buyback_or_dividend,   0) != 0 ? 1.0 : 0.0),
        Float32(clamp(get(inp, :mgmt_language_hedging,     0.0),  0.0, 1.0)),
        Float32(get(inp, :auditor_concerns,      0) != 0 ? 1.0 : 0.0),
        Float32(get(inp, :related_party_flags,   0) != 0 ? 1.0 : 0.0),
        Float32(get(inp, :contingent_liability_flag, 0) != 0 ? 1.0 : 0.0),
        Float32(clamp(get(inp, :extraction_confidence,     0.5),  0.0, 1.0)),
        Float32(min(doc_age / 365.0, 3.0)),  # cap at 3 years, scale to [0,3]
    )
end

"""
Find and extract features from the most recent document in a knowledge base
that was published on or before `as_of`.

# Arguments
- `kb`: result of TijoriData.get_knowledge_base(slug)
- `as_of`: only use documents published before this date
- `api_key`: Anthropic API key

# Returns
`LLMFeatures` — falls back to `MISSING_LLM` if no usable documents found.
"""
function extract_features_from_kb(kb, as_of::Date; api_key::String="")::LLMFeatures
    isempty(api_key) && (api_key = get(ENV, "ANTHROPIC_API_KEY", ""))

    # Priority: conference call > earnings release > investor presentation
    for (doc_type, docs) in [("conference_call",       kb.conference_calls),
                              ("earnings_release",      kb.earnings_releases),
                              ("investor_presentation", kb.investor_presentations)]
        isempty(docs) && continue
        for doc in reverse(collect(docs))   # most recent first
            doc_date = _parse_doc_date(string(get(doc, :period, "")))
            isnothing(doc_date) && continue
            doc_date > as_of && continue    # don't leak future information

            text = try
                TijoriData.fetch_document(string(doc.url)).text
            catch e
                @warn "Could not fetch document: $(sprint(showerror, e))"
                continue
            end

            isempty(strip(text)) && continue
            return extract_features(text, doc_type, doc_date; api_key=api_key, as_of=as_of)
        end
    end

    return MISSING_LLM
end

# ── Helpers ───────────────────────────────────────────────────────────────────

const MONTH_ABBR = Dict("Jan"=>1,"Feb"=>2,"Mar"=>3,"Apr"=>4,"May"=>5,"Jun"=>6,
                        "Jul"=>7,"Aug"=>8,"Sep"=>9,"Oct"=>10,"Nov"=>11,"Dec"=>12)

function _parse_doc_date(period::String)::Union{Date, Nothing}
    # Patterns: "Jan 2026", "FY25", "Q3 FY26"
    m = match(r"([A-Za-z]{3})\s+(\d{4})", period)
    if !isnothing(m)
        month = get(MONTH_ABBR, m[1], nothing)
        year  = tryparse(Int, m[2])
        isnothing(month) || isnothing(year) && return nothing
        return Date(year, month, 1)
    end
    m2 = match(r"FY(\d{2})", period)
    if !isnothing(m2)
        yy = parse(Int, m2[1])
        return Date(2000 + yy, 3, 31)  # end of Indian fiscal year
    end
    return nothing
end
