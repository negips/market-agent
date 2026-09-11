"""
run_inference.jl

Assemble model inputs for a (symbol, date) pair, run the trained SwingPredictor,
and write the result to website/data/prediction_result.json for display in
website/predict.html.

Usage:
  julia --project=packages/StockSwingPredictor scripts/run_inference.jl \\
        --symbol INFY --date 15-01-2024

Arguments:
  --symbol SYMBOL   NSE ticker (e.g. INFY, RELIANCE, TCS)
  --date   DATE     Trading date in YYYY-MM-DD format (must be a date in the cache)
  --model  PATH     Path to model BSON (default: website/data/models/DualCNN_v1/swing_predictor.bson)
  --output PATH     Output JSON path (default: website/data/prediction_result.json)

Output format: JSON with keys:
  symbol, date, ref_close, model, generated_at,
  predicted[{datetime, log_return, price}],
  actual[{datetime, close}],
  history{dates, closes}
"""

using StockSwingPredictor, Flux, JSON3, Dates

const REPO_ROOT = joinpath(@__DIR__, "..")

function _parse_date(s::String)::Date
    # Accept DD-MM-YYYY (preferred) or YYYY-MM-DD (fallback)
    if occursin(r"^\d{2}-\d{2}-\d{4}$", s)
        d, m, y = split(s, '-')
        return Date(parse(Int, y), parse(Int, m), parse(Int, d))
    end
    return Date(s)
end

function parse_args()
    opts = Dict{String,Any}(
        "symbol" => nothing,
        "date"   => nothing,
        "model"  => joinpath(REPO_ROOT, "website", "data", "models",
                             "DualCNN_v1", "swing_predictor.bson"),
        "output" => joinpath(REPO_ROOT, "website", "data", "prediction_result.json"),
    )
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a in ("--help", "-h")
            println("""
Usage:
  julia --project=packages/StockSwingPredictor scripts/run_inference.jl \\
        --symbol INFY --date 15-01-2024

Options:
  --symbol SYMBOL   NSE ticker symbol (required)
  --date DATE       Trading date DD-MM-YYYY (required, must be in inference cache)
  --model PATH      Model BSON path (default: website/data/models/DualCNN_v1/swing_predictor.bson)
  --output PATH     Output JSON path (default: website/data/prediction_result.json)

Output: JSON file readable by website/predict.html
""")
            exit(0)
        elseif a == "--symbol"; opts["symbol"] = uppercase(strip(ARGS[i+1])); i += 2
        elseif a == "--date";   opts["date"]   = _parse_date(ARGS[i+1]);      i += 2
        elseif a == "--model";  opts["model"]  = ARGS[i+1];                   i += 2
        elseif a == "--output"; opts["output"] = ARGS[i+1];                   i += 2
        else i += 1
        end
    end
    isnothing(opts["symbol"]) && error("--symbol is required")
    isnothing(opts["date"])   && error("--date is required")
    return opts
end

"""
Assemble market + hourly + llm tensors for a single (sym_idx, date_idx) pair.
Mirrors the logic in `assemble_batch` but for batch size 1.
LLM features are set to MISSING_LLM (zero / neutral) — use extract_features
separately if you have a recent conference call document.
"""
function assemble_single(cache::InferenceCache, sym_idx::Int,
                          t::Int)::Tuple{Array{Float32,4}, Matrix{Float32}, Matrix{Float32}}
    k = sym_idx

    # ── Market context (28-day window, N_MARKET_COMPANIES columns) ───────────
    market_cols = if k <= N_MARKET_COMPANIES
        [k; filter(!=(k), 1:N_MARKET_COMPANIES)]
    else
        [k; collect(1:N_MARKET_COMPANIES-1)]
    end

    raw_c  = cache.closes[t-N_MARKET_DAYS+1:t, market_cols]
    anchor = raw_c[1:1, :]
    norm_c = raw_c ./ max.(anchor, 1f-6)
    norm_c[isnan.(norm_c)] .= 1f0

    raw_v = cache.vols[t-N_MARKET_DAYS+1:t, market_cols]
    raw_v[isnan.(raw_v)] .= 0f0

    market = Array{Float32}(undef, N_MARKET_DAYS, N_MARKET_CHANNELS, N_MARKET_COMPANIES, 1)
    market[:, 1, :, 1] = norm_c
    market[:, 2, :, 1] = raw_v

    # ── Hourly series (target stock, N_HOURLY_BARS ending on date t) ────────
    h_end   = searchsortedlast(cache.hourly_datetimes,
                               DateTime(cache.dates[t], Time(23, 59, 59)))
    h_end < N_HOURLY_BARS && error(
        "Not enough hourly history for $(cache.companies[k]) up to $(cache.dates[t]): " *
        "found $h_end bars, need $N_HOURLY_BARS")

    h_start = h_end - N_HOURLY_BARS + 1
    raw_h   = cache.hourly_closes[h_start:h_end, k]
    ref_h   = max(raw_h[1], 1f-6)
    hourly  = Matrix{Float32}(undef, N_HOURLY_BARS, 1)
    hourly[:, 1] = raw_h ./ ref_h

    # ── LLM features (neutral — no document provided at inference time) ──────
    llm = Matrix{Float32}(undef, N_LLM_FEATURES, 1)
    llm[:, 1] = llm_to_vec(MISSING_LLM)

    return market, hourly, llm
end

function main()
    opts   = parse_args()
    symbol = opts["symbol"]::String
    date   = opts["date"]::Date

    cache_file = joinpath(REPO_ROOT, "website", "data", "inference_cache.bson")
    isfile(cache_file) ||
        error("Cache not found: $cache_file\nRun: julia --project=packages/StockSwingPredictor scripts/build_cache.jl")

    @info "Loading inference cache…"
    cache = load_inference_cache(cache_file)
    @info "  $(length(cache.dates)) dates × $(length(cache.companies)) companies"

    sym_idx = get(cache.sym_index, symbol, 0)
    sym_idx == 0 && error(
        "Symbol not in model universe: $symbol\n" *
        "Check website/data/market_universe.json for available symbols.")

    t = get(cache.date_index, date, 0)
    t == 0 && error(
        "Date is not a trading day in the cache: $date\n" *
        "Use a trading day (Mon–Fri, market open).")
    t <= N_MARKET_DAYS && error(
        "Not enough daily history before $date — need at least $N_MARKET_DAYS prior trading days.")

    ref_close = cache.closes[t, sym_idx]
    isnan(ref_close) && error("No closing price for $symbol on $date.")

    model_path = opts["model"]
    isfile(model_path) ||
        error("Model not found: $model_path\nRun: julia --project=packages/StockSwingPredictor scripts/train_model.jl")

    @info "Loading model from $model_path…"
    model, model_companies, model_meta = load_model(model_path)

    @info "Assembling inputs for $symbol on $date…"
    market, hourly, llm = assemble_single(cache, sym_idx, t)

    @info "Running inference…"
    pred_lr = predict(model, market, hourly, llm)[:, 1]   # (N_PRED_HOURS,) log-returns

    # ── Predicted timeline ───────────────────────────────────────────────────
    n_future = min(t + N_PRED_DAYS, length(cache.dates)) - t
    pred_days = cache.dates[t+1 : t+n_future]

    # Walk the hourly timeline to find the actual bars for those trading days
    pred_dts = DateTime[]
    for d in pred_days
        lo = searchsortedfirst(cache.hourly_datetimes, DateTime(d))
        hi = searchsortedlast( cache.hourly_datetimes, DateTime(d, Time(23, 59, 59)))
        for h in lo:hi
            Date(cache.hourly_datetimes[h]) == d &&
                push!(pred_dts, cache.hourly_datetimes[h])
        end
    end
    n_pred = min(N_PRED_HOURS, length(pred_dts))

    predicted = [Dict{String,Any}(
        "datetime"   => string(pred_dts[i]),
        "log_return" => Float64(pred_lr[i]),
        "price"      => Float64(ref_close * exp(pred_lr[i])),
    ) for i in 1:n_pred]

    # ── Actual hourly closes (from cache — may be empty if date is recent) ───
    actual = Dict{String,Any}[]
    for i in 1:n_pred
        dt = pred_dts[i]
        h  = searchsortedlast(cache.hourly_datetimes, dt)
        (h == 0 || cache.hourly_datetimes[h] != dt) && continue
        v = cache.hourly_closes[h, sym_idx]
        isnan(v) && continue
        push!(actual, Dict{String,Any}("datetime" => string(dt), "close" => Float64(v)))
    end

    # ── 28-day daily close history (leading context for the chart) ───────────
    hist_start = t - N_MARKET_DAYS + 1
    history = Dict{String,Any}(
        "dates"  => [string(cache.dates[i]) for i in hist_start:t],
        "closes" => [Float64(cache.closes[i, sym_idx]) for i in hist_start:t],
    )

    # ── Write output ─────────────────────────────────────────────────────────
    result = Dict{String,Any}(
        "symbol"       => symbol,
        "date"         => string(date),
        "ref_close"    => Float64(ref_close),
        "model"        => get(model_meta, "arch", "DualCNN_v1"),
        "generated_at" => string(now()),
        "predicted"    => predicted,
        "actual"       => actual,
        "history"      => history,
    )

    output_path = opts["output"]
    mkpath(dirname(output_path))
    open(output_path, "w") do io
        JSON3.pretty(io, result)
    end

    day5_pct = round(Float64(pred_lr[min(N_PRED_HOURS, n_pred)]) * 100, digits=2)
    dir      = day5_pct >= 0 ? "▲" : "▼"
    @info "Written → $output_path"
    @info "  $symbol @ $date  ref=₹$(round(ref_close, digits=2))  day-5 return: $dir $(abs(day5_pct))%"
    @info "  $(length(actual)) actual bars found in cache ($(length(predicted)) predicted)"
end

main()
