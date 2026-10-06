"""
Candidate-universe construction: `CompanyConfidence`-filtered, then split into
a training candidate list and a held-out validation candidate list by one of
several pluggable `UniverseStrategy` recipes (see the "Strategies" section
below) — from today's default (`SharedTopMarketCap`, train and val identical,
top-N by market cap) up to fully disjoint, randomized, or market-cap-bucketed
splits, aimed at testing/improving generalization beyond a single fixed
62-ish-company universe.

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

`TradingGameEnv`, `resolve_actions`'s rule-13 cap, and `ActorCriticPolicy` all
operate purely on each episode's own candidate *content* — no symbol-identity
embedding, no positional encoding across the candidate dimension (see
`policy.jl`'s `ActorCriticPolicy` callable: the cross-candidate attention
sequence gets no positional embedding added, and every per-candidate layer is
weight-shared via a batched reshape, not indexed by symbol identity). So train
and val are free to use different company counts and/or entirely different
companies — nothing downstream of this file needs to change to support that.
"""

using JSON3, Dates, Random

# ── Types ─────────────────────────────────────────────────────────────────────────

"""One candidate's snapshot at the time the universe was built."""
struct UniverseEntry
    symbol           :: String
    name             :: String
    market_cap_cr    :: Float64
    confidence_score :: Float64
end

# ── Eligible pool ─────────────────────────────────────────────────────────────────

"""
The confidence-filtered, cache-available candidate pool, sorted by market cap
descending — the shared first stage every `UniverseStrategy` operates on.
Strategies never re-read `nse_companies_latest.json` or re-apply the
confidence/cache filters themselves; they only decide how to select from (or
split) this already-filtered, already-ranked pool.

# Arguments
- `cache`: an `InferenceCache` — only symbols in `cache.sym_index` are eligible.
- `nse_companies_path`: path to `nse_companies_latest.json`.
- `min_confidence`: minimum `confidence.score` to pass.

# Returns
`Vector{UniverseEntry}`, sorted by market cap descending. Symbols with
missing/`null` confidence, missing market cap, or no price history in `cache`
are silently excluded. Uncapped — callers (`build_candidate_universe`,
`UniverseStrategy` subtypes) decide how many of these to actually use.
"""
function eligible_candidates(cache::InferenceCache, nse_companies_path::String;
                              min_confidence::Float64=MIN_CONFIDENCE_SCORE)::Vector{UniverseEntry}
    isfile(nse_companies_path) || error(
        "TradingGame.eligible_candidates: not found: $nse_companies_path\n" *
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
    return entries
end

"""
Build a capped, confidence-filtered, market-cap-ranked candidate universe —
the simple single-list case, equivalent to `build_universes(SharedTopMarketCap(n=n_candidates),
eligible_candidates(...))` but without needing to construct a strategy. Kept
as a direct entry point since it's the common case for quick REPL use and
Stage 1–3 tests that don't care about train/val separation.

# Returns
`Vector{UniverseEntry}`, length `<= n_candidates`, sorted by market cap
descending.
"""
function build_candidate_universe(cache::InferenceCache, nse_companies_path::String;
                                   n_candidates::Int=N_CANDIDATE_STOCKS,
                                   min_confidence::Float64=MIN_CONFIDENCE_SCORE)::Vector{UniverseEntry}
    entries = eligible_candidates(cache, nse_companies_path; min_confidence=min_confidence)
    return entries[1:min(n_candidates, length(entries))]
end

# ── Strategies: train/val composition ────────────────────────────────────────────

"""
How `scripts/build_market_universe_snapshot.jl` splits `eligible_candidates`'s pool
into a training candidate list and a held-out validation candidate list.
Every subtype implements `build_universes(strategy, pool) -> (train=.., val=..)`.

Adding a new strategy (a new selection criterion, e.g. sector quotas once
`UniverseEntry` carries a sector field) means adding one new `UniverseStrategy`
subtype plus one `build_universes` method — no existing strategy, call site,
or downstream code needs to change.
"""
abstract type UniverseStrategy end

"""Split `pool` (from `eligible_candidates`) into `(train, val)` per `strategy`.
Every `UniverseStrategy` subtype implements this."""
function build_universes end

"""Draw an `AbstractRNG` from a `Union{Nothing,Int}` seed the same way the
rest of `TradingGame` does (`ActorCriticPolicy`, `train_trading_policy.jl
--seed`) — `nothing` uses the shared global RNG (non-reproducible across
runs), an `Int` gives a fresh, reproducible `MersenneTwister`."""
_rng_for(seed::Union{Nothing, Int}) = seed === nothing ? Random.default_rng() : MersenneTwister(seed)

"""
Today's default and the simplest case: both train and val draw from the
identical top-`n` pool by market cap. Not a generalization test on its own —
val overlaps train entirely in company identity, so only the train/val
*date* split (see `scripts/train_trading_policy.jl`'s `--val-window`)
separates the two roles.
"""
Base.@kwdef struct SharedTopMarketCap <: UniverseStrategy
    n :: Int = N_CANDIDATE_STOCKS
end

function build_universes(s::SharedTopMarketCap, pool::Vector{UniverseEntry})
    entries = pool[1:min(s.n, length(pool))]
    return (train=entries, val=entries)
end

"""
Train and val draw from disjoint company sets, both still confined to the top
`n_train + n_val` by market cap — keeps the liquidity/data-quality floor plain
market-cap ranking provides, rather than opening up to the full
confidence-passing pool (see `RandomUniverse` for that).

The combined top `n_train + n_val` pool is split stratified by market-cap
decile (rank-based bins; each bin contributes the same train:val ratio as the
whole) so neither role skews toward the large- or small-cap end purely from
where a plain random split happened to land — both sets span a similar
market-cap distribution. Per-bin counts use largest-remainder apportionment,
so `length(train) == min(n_train, n_total)` exactly (not just approximately),
where `n_total = min(n_train + n_val, length(pool))`.

`seed` makes the split reproducible; `nothing` (default) draws from the
shared global RNG.
"""
Base.@kwdef struct DisjointTopMarketCap <: UniverseStrategy
    n_train :: Int
    n_val   :: Int
    seed    :: Union{Nothing, Int} = nothing
end

function build_universes(s::DisjointTopMarketCap, pool::Vector{UniverseEntry})
    n_total = min(s.n_train + s.n_val, length(pool))
    n_total == 0 && return (train=UniverseEntry[], val=UniverseEntry[])
    combined = pool[1:n_total]

    rng = _rng_for(s.seed)
    n_train_target = round(Int, s.n_train / (s.n_train + s.n_val) * n_total)

    n_bins = min(10, n_total)
    shuffled_bins = [shuffle(rng, collect(b)) for b in Iterators.partition(combined, cld(n_total, n_bins))]

    # Largest-remainder apportionment: give each bin floor(its proportional
    # share), then hand the leftover seats to the bins with the largest
    # fractional remainder — guarantees sum(bin train counts) ==
    # n_train_target exactly, not just approximately (plain per-bin rounding
    # can drift by a few either way).
    exact     = [n_train_target * length(b) / n_total for b in shuffled_bins]
    counts    = floor.(Int, exact)
    remainder = n_train_target - sum(counts)
    for i in sortperm(exact .- counts; rev=true)[1:remainder]
        counts[i] += 1
    end

    train = UniverseEntry[]
    val   = UniverseEntry[]
    for (b, k) in zip(shuffled_bins, counts)
        append!(train, b[1:k])
        append!(val, b[k+1:end])
    end
    return (train=train, val=val)
end

"""
Uniformly random `n` companies from the WHOLE confidence-passing,
cache-available pool — no market-cap ranking at all, so this can reach
companies far outside the usual top-`N_CANDIDATE_STOCKS`.

`disjoint=true` draws train and val as two non-overlapping random subsets of
size `n` each (errors if the pool has fewer than `2n` eligible companies);
`disjoint=false` (default) draws one random `n` and uses it for both roles,
so only the train/val date split separates them — same relationship as
`SharedTopMarketCap`, just drawn randomly rather than market-cap-ranked.
`seed` as in `DisjointTopMarketCap`.
"""
Base.@kwdef struct RandomUniverse <: UniverseStrategy
    n        :: Int
    disjoint :: Bool = false
    seed     :: Union{Nothing, Int} = nothing
end

function build_universes(s::RandomUniverse, pool::Vector{UniverseEntry})
    rng = _rng_for(s.seed)
    if s.disjoint
        2 * s.n <= length(pool) || error(
            "RandomUniverse(n=$(s.n), disjoint=true): needs $(2 * s.n) eligible companies, pool has $(length(pool))")
        drawn = shuffle(rng, pool)[1:2 * s.n]
        return (train=drawn[1:s.n], val=drawn[s.n+1:2*s.n])
    else
        s.n <= length(pool) || error(
            "RandomUniverse(n=$(s.n)): needs $(s.n) eligible companies, pool has $(length(pool))")
        entries = shuffle(rng, pool)[1:s.n]
        return (train=entries, val=entries)
    end
end

"""
Random selection drawn from named market-cap bands with an explicit quota per
band — e.g. `quotas = [(0.0, 5_000.0) => 15, (5_000.0, Inf) => 15]` for 15
random sub-₹5,000 Cr companies plus 15 random ≥₹5,000 Cr companies. Bands are
`(lo, hi]` on `market_cap_cr`, checked independently (a company matching
multiple bands is eligible for each; a company matching none is excluded from
all); a quota larger than its band's membership errors rather than silently
under-filling.

`disjoint` has the same meaning as `RandomUniverse`: `true` draws train and
val as independent random draws per band (each band needs `2 * quota`
members); `false` (default) draws once per band and shares it between train
and val. This is how a below/above-threshold mixed universe gets built today;
a future criterion (sector, volatility band, …) reuses this same shape once
`UniverseEntry` carries the field to bucket on.
"""
Base.@kwdef struct BucketedRandom <: UniverseStrategy
    quotas   :: Vector{Pair{Tuple{Float64, Float64}, Int}}
    disjoint :: Bool = false
    seed     :: Union{Nothing, Int} = nothing
end

function build_universes(s::BucketedRandom, pool::Vector{UniverseEntry})
    rng = _rng_for(s.seed)
    if s.disjoint
        train = UniverseEntry[]
        val   = UniverseEntry[]
        for ((lo, hi), quota) in s.quotas
            members = filter(e -> lo < e.market_cap_cr <= hi, pool)
            2 * quota <= length(members) || error(
                "BucketedRandom: band ($lo, $hi] needs $(2 * quota) members for disjoint train/val, has $(length(members))")
            drawn = shuffle(rng, members)[1:2 * quota]
            append!(train, drawn[1:quota])
            append!(val, drawn[quota+1:2*quota])
        end
        return (train=train, val=val)
    else
        entries = UniverseEntry[]
        for ((lo, hi), quota) in s.quotas
            members = filter(e -> lo < e.market_cap_cr <= hi, pool)
            quota <= length(members) || error(
                "BucketedRandom: band ($lo, $hi] needs $quota members, has $(length(members))")
            append!(entries, shuffle(rng, members)[1:quota])
        end
        return (train=entries, val=entries)
    end
end

# ── Persistence ────────────────────────────────────────────────────────────────────

"""Field-reflection dump of a `UniverseStrategy`'s parameters, for the
snapshot file's `strategy_params` — a new strategy serializes automatically,
no change needed here. `quotas` (a `Vector{Pair}`, which `JSON3` has no native
encoding for) is flattened to an array of `[[lo, hi], quota]` triples.

`hi` is commonly `Inf` for an open-ended band (e.g. `--band 5000:inf:15`,
see `build_market_universe_snapshot.jl`) — raw JSON has no representation
for infinity and `JSON3.write` errors on it outright, so `_json_safe_bound`
writes it as the string `"inf"`/`"-inf"` instead (matching the CLI's own
spelling). Write-only: `strategy_params` is never read back by
`load_universe_snapshot` (informational/provenance only — see this file's
module docstring), so there's no corresponding parse path to keep in sync."""
_strategy_params(s::UniverseStrategy) =
    Dict(string(f) => _strategy_param_value(getfield(s, f)) for f in fieldnames(typeof(s)))
_strategy_param_value(v::Vector{<:Pair}) =
    [[[_json_safe_bound(p.first[1]), _json_safe_bound(p.first[2])], p.second] for p in v]
_strategy_param_value(v) = v

_json_safe_bound(x::Float64) = isinf(x) ? (x > 0 ? "inf" : "-inf") : x

"""
Save a built train/val universe pair (rank/draw order preserved) as JSON at
`path`, tagged with the `strategy` that produced it — so
`scripts/train_trading_policy.jl` (and any REPL session) can load a stable
pair of candidate lists without re-reading/re-filtering
`nse_companies_latest.json`, and so the snapshot is self-documenting about how
it was built.
"""
function save_universe_snapshot(strategy::UniverseStrategy, train::Vector{UniverseEntry},
                                 val::Vector{UniverseEntry}, path::String)
    mkpath(dirname(path))
    entry_json(e) = (symbol=e.symbol, name=e.name, market_cap_cr=e.market_cap_cr,
                      confidence_score=e.confidence_score)
    payload = (
        generated_at       = string(now(UTC)),
        strategy           = string(nameof(typeof(strategy))),
        strategy_params    = _strategy_params(strategy),
        n_candidates_train = length(train),
        n_candidates_val   = length(val),
        train_candidates   = entry_json.(train),
        val_candidates     = entry_json.(val),
    )
    open(path, "w") do io
        JSON3.pretty(io, payload)
    end
    return nothing
end

"""
Load a previously-saved universe snapshot's train/val symbol lists, in
rank/draw order — directly usable as `EpisodeConfig.candidate_universe` for
`train_config`/`val_config` respectively. `SharedTopMarketCap` writes the
identical list to both, so this is a safe, uniform return shape regardless of
which strategy actually built the snapshot.

# Returns
`(train=Vector{String}, val=Vector{String})`.
"""
function load_universe_snapshot(path::String)
    isfile(path) || error(
        "TradingGame.load_universe_snapshot: not found: $path\n" *
        "Run: julia --project=packages/TradingGame scripts/build_market_universe_snapshot.jl")
    raw = JSON3.read(read(path, String))
    return (train = [String(c.symbol) for c in raw.train_candidates],
            val   = [String(c.symbol) for c in raw.val_candidates])
end
