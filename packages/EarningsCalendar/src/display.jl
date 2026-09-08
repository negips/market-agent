function Base.show(io::IO, e::EarningsEvent)
    print(io, Dates.format(e.date, "dd u yyyy"), "  ",
          rpad(e.symbol, 12), "  ", e.purpose)
end

function Base.show(io::IO, ::MIME"text/plain", events::Vector{EarningsEvent})
    n = length(events)
    if n == 0
        print(io, "EarningsEvent[] (no events found)")
        return
    end

    println(io, "$n upcoming earnings event$(n == 1 ? "" : "s"):\n")
    println(io, "  ", rpad("Date", 14), rpad("Symbol", 14), "Purpose")
    println(io, "  ", "─"^62)

    prev_date = typemin(Date)
    for e in events
        e.date != prev_date && prev_date != typemin(Date) && println(io)
        prev_date = e.date
        println(io, "  ",
                rpad(Dates.format(e.date, "dd u yyyy"), 14),
                rpad(e.symbol, 14),
                e.purpose)
    end
end
