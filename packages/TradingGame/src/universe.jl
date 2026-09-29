"""
Candidate-universe construction: `CompanyConfidence`-filtered, market-cap-ranked,
capped at `N_CANDIDATE_STOCKS`.

This reads confidence scores already computed by
`scripts/run_confidence_checks.jl` into `website/data/nse_companies_latest.json`
— a fast, offline JSON filter/sort — rather than calling
`CompanyConfidence.analyze` directly. That call hits the Tijori sidecar at
~6–15s/company (per `CLAUDE.md`), which is far too slow to run inside (or even
once per) a training script; the existing daily pipeline
(`generate_nse_list.jl` → `run_confidence_checks.jl`) already produces
everything this file needs. `TradingGame` therefore does not depend on
`CompanyConfidence` as a package — see `MIN_CONFIDENCE_SCORE` in `constants.jl`.

A symbol whose confidence hasn't been computed yet (present in the JSON with a
`null`/missing `confidence` field — `run_confidence_checks.jl` only covers the
top N by market cap on a given run) is excluded, not treated as a pass.
"""

using JSON3, Dates

# ── Types ─────────────────────────────────────────────────────────────────────────

"""One candidate's snapshot at the time the universe was built."""
struct UniverseEntry
    symbol           :: String
    name             :: String
    market_cap_cr    :: Float64
    confidence_score :: Float64
end

# ── Build ─────────────────────────────────────────────────────────────────────────

"""
Build a capped, confidence-filtered, market-cap-ranked candidate universe,
restricted to symbols that actually have price history in `cache` (a symbol
can be confidence-checked but have no cached OHLCV, or vice versa).

# Arguments
- `cache`: an `InferenceCache` — only symbols in `cache.sym_index` are eligible.
- `nse_companies_path`: path to `nse_companies_latest.json`.
- `n_candidates`: cap on the returned universe size.
- `min_confidence`: minimum `confidence.score` to pass.

# Returns
`Vector{UniverseEntry}`, length `<= n_candidates`, sorted by market cap
descending. Symbols with missing/`null` confidence, missing market cap, or no
price history in `cache` are silently excluded.
"""
function build_candidate_universe(cache::InferenceCache, nse_companies_path::String;
                                   n_candidates::Int=N_CANDIDATE_STOCKS,
                                   min_confidence::Float64=MIN_CONFIDENCE_SCORE)::Vector{UniverseEntry}
    isfile(nse_companies_path) || error(
        "TradingGame.build_candidate_universe: not found: $nse_companies_path\n" *
        "Run: julia scripts/generate_nse_list.jl && " *
        "julia --project=packages/CompanyConfidence scripts/run_confidence_checks.jl")

    raw = JSON3.read(read(nse_companies_path, String))

    entries = UniverseEntry[]
    for c in raw.companies
        sym = String(c.symbol)
        haskey(cache.sym_index, sym) || continue

        conf = get(c, :confidence, nothing)
        conf === nothing && continue
        score = Float64(conf.score)
        score >= min_confidence || continue

        mcap = get(c, :market_cap_cr, nothing)
        mcap === nothing && continue

        push!(entries, UniverseEntry(sym, String(get(c, :name, sym)), Float64(mcap), score))
    end

    sort!(entries, by = e -> e.market_cap_cr, rev = true)
    return entries[1:min(n_candidates, length(entries))]
end

# ── Persistence ────────────────────────────────────────────────────────────────────

"""
Save a built universe (rank order preserved) as JSON at `path` — so
`scripts/train_trading_policy.jl` (and any REPL session) can load a stable
candidate list without re-reading/re-filtering `nse_companies_latest.json`
on every run.
"""
function save_universe_snapshot(entries::Vector{UniverseEntry}, path::String)
    mkpath(dirname(path))
    payload = (
        generated_at = string(now(UTC)),
        n_candidates = length(entries),
        candidates   = [(symbol=e.symbol, name=e.name, market_cap_cr=e.market_cap_cr,
                          confidence_score=e.confidence_score) for e in entries],
    )
    open(path, "w") do io
        JSON3.pretty(io, payload)
    end
    return nothing
end

"""
Load a previously-saved universe snapshot's symbols, in rank order — directly
usable as `EpisodeConfig.candidate_universe`.
"""
function load_universe_snapshot(path::String)::Vector{String}
    isfile(path) || error(
        "TradingGame.load_universe_snapshot: not found: $path\n" *
        "Run: julia --project=packages/TradingGame scripts/build_universe_snapshot.jl")
    raw = JSON3.read(read(path, String))
    return [String(c.symbol) for c in raw.candidates]
end
