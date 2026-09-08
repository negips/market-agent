"""
    EarningsEvent

A single upcoming earnings announcement or board meeting from the NSE event calendar.

# Fields
- `symbol`  — NSE ticker symbol (e.g. `"INFY"`)
- `company` — full company name as returned by NSE
- `date`    — scheduled event date
- `purpose` — event description as returned by NSE (e.g. `"Quarterly Results"`)
"""
struct EarningsEvent
    symbol::String
    company::String
    date::Date
    purpose::String
end

"""Thrown when the NSE event-calendar API is unreachable or returns an unexpected response."""
struct NSEFetchError <: Exception
    msg::String
end
Base.showerror(io::IO, e::NSEFetchError) = print(io, "NSEFetchError: ", e.msg)
