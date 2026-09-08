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
    n_ex   = size(d.X, 2)
    n_feat = size(d.X, 1)
    # Show stats for the final-hour bar (end-of-day-5 close vs reference).
    final_h = d.y[end, :]
    y_mean  = round(mean(final_h) * 100, digits=3)
    y_std   = round(std(final_h)  * 100, digits=3)
    println(io, "Dataset: $n_ex examples × $n_feat features  →  $(N_PRED_HOURS)-step trajectory")
    println(io, "  Final-bar return: mean=$(y_mean)%  std=$(y_std)%")
    println(io, "  Symbols: $(length(unique(d.symbols)))  " *
                "Date range: $(minimum(d.dates)) → $(maximum(d.dates))")
end

function Base.show(io::IO, stats::NormStats)
    println(io, "NormStats: $(length(stats.feature_names)) features")
end
