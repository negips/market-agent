"""
NSE ASM / GSM surveillance list checker.

NSE publishes two surveillance lists:
  ASM (Additional Surveillance Measure)  — unusual price/volume activity
  GSM (Graded Surveillance Measure)      — weak fundamentals; 6 stages of severity

Both require an authenticated session. This module establishes a throwaway
browser-like session against nseindia.com before querying the API. All
errors are caught: the check returns `checked=false` rather than crashing.
"""

const _NSE_BASE     = "https://www.nseindia.com"
const _NSE_ASM_URL  = "https://www.nseindia.com/api/reportASM"
const _NSE_GSM_URL  = "https://www.nseindia.com/api/reportGSM"
const _NSE_HEADERS  = [
    "User-Agent"      => "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 " *
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language" => "en-US,en;q=0.9",
    "Accept-Encoding" => "gzip, deflate, br",
]

"""
    surveillance_check(symbol) -> SurveillanceResult

Check whether `symbol` appears on NSE's ASM or GSM surveillance lists.

Requires a live internet connection. If NSE's API is unreachable or returns
an unexpected response, returns `SurveillanceResult(false, false, false, false)`
and emits a `@warn`. This is intentional: a network failure should not block
the rest of the confidence analysis.

`symbol` should be the NSE ticker (e.g. "YESBANK"). Pass `nothing` if unknown
(returns unchecked result immediately).

# Example
```julia
r = surveillance_check("YESBANK")
r.on_asm   # false
r.on_gsm   # false
r.checked  # true if NSE was reachable
```
"""
function surveillance_check(symbol::Union{String, Nothing})::SurveillanceResult
    isnothing(symbol) && return SurveillanceResult(false, false, false, false)

    sym = uppercase(strip(symbol))
    try
        # Establish NSE session to get cookies (force HTTP/1.1 — NSE's HTTP/2 uses server push)
        init_resp = HTTP.get(_NSE_BASE;
            headers = [_NSE_HEADERS..., "Accept" => "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8"],
            request_timeout = 15,
            connect_timeout = 8,
            version = v"1.1",
        )
        cookies = join(
            [c.name * "=" * c.value for c in HTTP.cookies(init_resp)],
            "; ",
        )
        api_headers = [_NSE_HEADERS...,
            "Accept"  => "application/json, text/javascript, */*; q=0.01",
            "Cookie"  => cookies,
            "Referer" => _NSE_BASE * "/",
        ]

        asm_data = _nse_fetch(_NSE_ASM_URL, api_headers)
        gsm_data = _nse_fetch(_NSE_GSM_URL, api_headers)

        on_asm = _in_list(sym, asm_data)
        on_gsm = _in_list(sym, gsm_data)

        return SurveillanceResult(on_asm, on_gsm, true, on_asm || on_gsm)
    catch e
        @warn "NSE surveillance check could not complete ($(sprint(showerror, e))). Treating as not flagged."
        return SurveillanceResult(false, false, false, false)
    end
end

# ── Internal ──────────────────────────────────────────────────────────────────

function _nse_fetch(url::String, headers::Vector)::Vector
    resp = HTTP.get(url; headers=headers, request_timeout=15, connect_timeout=8, version=v"1.1")
    body = JSON3.read(resp.body)
    return get(body, :data, [])
end

function _in_list(sym::String, records::Vector)::Bool
    any(records) do r
        for k in (:symbol, :Symbol, :SYMBOL)
            haskey(r, k) && uppercase(string(r[k])) == sym && return true
        end
        false
    end
end
