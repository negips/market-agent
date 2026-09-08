"""
    EarningsCalendar

Fetches upcoming earnings announcements and board meetings from the NSE event-calendar
API and returns them as typed `EarningsEvent` values.

No sidecar required — hits `nseindia.com` directly over HTTP.

# Quick start

```julia
using EarningsCalendar, Dates

# Next 30 days (default)
events = upcoming_earnings()

# Custom window
events = upcoming_earnings(7)      # this week
events = upcoming_earnings(90)     # next quarter

# Explicit date range
events = fetch_earnings_calendar(Date(2026, 10, 1), Date(2026, 10, 31))

# Each event
e = events[1]
e.symbol   # "INFY"
e.company  # "Infosys Limited"
e.date     # Date(2026, 10, 17)
e.purpose  # "Quarterly Results"
```

# Filtering

Use standard Julia `filter` — no wrapper needed:

```julia
# Only "Quarterly Results" (excludes Board Meetings without results)
quarterly = filter(e -> occursin("Quarterly", e.purpose), events)

# Symbols only, for passing to CompanyConfidence
symbols = [e.symbol for e in events]
```

See also: [CompanyConfidence](@ref), [TijoriData](@ref)
"""
module EarningsCalendar

using Dates, HTTP, JSON3

include("types.jl")
include("fetch.jl")
include("display.jl")

export EarningsEvent, NSEFetchError
export upcoming_earnings, fetch_earnings_calendar

end # module EarningsCalendar
