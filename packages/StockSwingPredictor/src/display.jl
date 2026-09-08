"""Base.show overrides for StockSwingPredictor types."""

using Printf, Statistics

function Base.show(io::IO, s::SwingSignal)
    dir = s.eod_return > 0 ? "▲" : "▼"
    @printf(io, "SwingSignal %-12s %s %+.2f%% (eod day5)  pct=%.0f%%  conf=%.0f  days=%d",
            s.symbol, dir, s.eod_return * 100,
            s.percentile * 100, s.confidence_score * 100, s.days_until)
end

function Base.show(io::IO, ::MIME"text/plain", signals::Vector{SwingSignal})
    isempty(signals) && (println(io, "No signals"); return)
    sorted = sort(signals, by = s -> abs(s.eod_return), rev=true)
    println(io, "SwingSignal results ($(length(signals)) companies):")
    println(io, "  ", "-"^70)
    for s in sorted
        dir = s.eod_return > 0 ? "▲" : "▼"
        @printf(io, "  %-14s %s %+6.2f%% (eod5)  pct=%3.0f%%  conf=%2.0f  in %2dd\n",
                s.symbol, dir, s.eod_return * 100,
                s.percentile * 100, s.confidence_score * 100, s.days_until)
    end
end

function Base.show(io::IO, d::Dataset)
    n_ex   = length(d.examples)
    final_h = [ex.label[end] for ex in d.examples]
    y_mean  = round(mean(final_h) * 100, digits=3)
    y_std   = round(std(final_h)  * 100, digits=3)
    println(io, "Dataset: $(n_ex) examples | $(length(d.companies)) companies | $(length(d.dates)) trading dates")
    println(io, "  Final-bar return (eod day5): mean=$(y_mean)%  std=$(y_std)%")
    if !isempty(d.examples)
        println(io, "  Date range: $(d.examples[1].date) → $(d.examples[end].date)")
    end
end
