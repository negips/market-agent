"""
Phase timing for the rollout, the PPO update and the training loop, and the two
small JSON files the live page reads from it: `timing.json` (where each
iteration's time went) and `val_summary.json` (one result row per held-out
validation episode).

A `PhaseTimer` accumulates wall-clock seconds and call counts per named phase.
Phase names are dotted: `observe.history` is a child of `observe`, whose own
time already includes it, so a reader sums only the top-level names to get a
total and reads the children as a breakdown. Every instrumented function takes
`timer::Union{Nothing, PhaseTimer}=nothing`; with `nothing`, `tic`/`toc!` do no
clock reads at all.
"""

using JSON3

"""Accumulated seconds and call count per phase name, in first-seen order."""
mutable struct PhaseTimer
    secs   :: Dict{String, Float64}
    counts :: Dict{String, Int}
    order  :: Vector{String}
end

PhaseTimer() = PhaseTimer(Dict{String, Float64}(), Dict{String, Int}(), String[])

"""Start a phase: the current monotonic clock in nanoseconds (0 without a timer)."""
tic(::Nothing)    = UInt64(0)
tic(::PhaseTimer) = time_ns()

"""
End the phase `name` that began at `t0` and return the new clock reading, so
consecutive phases chain: `t = toc!(timer, "a", t); t = toc!(timer, "b", t)`.
A `nothing` timer returns `t0` untouched.
"""
toc!(::Nothing, ::AbstractString, t0::UInt64) = t0
function toc!(pt::PhaseTimer, name::AbstractString, t0::UInt64)
    t1 = time_ns()
    if haskey(pt.secs, name)
        pt.secs[name]   += (t1 - t0) * 1e-9
        pt.counts[name] += 1
    else
        pt.secs[name]   = (t1 - t0) * 1e-9
        pt.counts[name] = 1
        push!(pt.order, name)
    end
    return t1
end

"""Add `secs` seconds (and one call) to `name` directly, for a duration measured elsewhere."""
function add_phase!(pt::PhaseTimer, name::AbstractString, secs::Real)
    if haskey(pt.secs, name)
        pt.secs[name]   += secs
        pt.counts[name] += 1
    else
        pt.secs[name]   = secs
        pt.counts[name] = 1
        push!(pt.order, name)
    end
    return nothing
end

"""Seconds accumulated under `name` (0 if never timed)."""
phase_secs(pt::PhaseTimer, name::AbstractString) = get(pt.secs, name, 0.0)

"""Sum of the top-level phases (names without a `.`)."""
total_secs(pt::PhaseTimer) = sum((s for (n, s) in pt.secs if !occursin('.', n)); init=0.0)

"""A timer's phases as `[(name, secs, n), …]` in first-seen order, rounded for JSON."""
function phase_rows(pt::PhaseTimer)
    return [(name=n, secs=round(pt.secs[n], digits=4), n=pt.counts[n]) for n in pt.order]
end

phase_rows(::Nothing) = NamedTuple{(:name, :secs, :n), Tuple{String, Float64, Int}}[]

"""
Insert `row` (a NamedTuple with an `iteration` field) into the `{"runs": [...]}`
file at `path`, replacing a row of the same iteration, sorted by iteration,
written atomically (temp file + rename) so a page polling it never reads half a
file. A no-op for an empty `path`.
"""
function _upsert_run_row(path::String, row)
    isempty(path) && return nothing
    mkpath(dirname(path))
    rows = Any[]
    if isfile(path)
        try
            doc = JSON3.read(read(path, String))
            rows = Any[r for r in doc.runs if Int(r.iteration) != row.iteration]
        catch
            rows = Any[]
        end
    end
    push!(rows, row)
    sort!(rows; by = r -> Int(r.iteration))
    tmp = path * ".tmp"
    open(tmp, "w") do io
        JSON3.write(io, (runs=rows,))
    end
    mv(tmp, path; force=true)
    return nothing
end

"""
Append one iteration's timing to `timing.json` (see `_upsert_run_row`): the
validation rollout's phases (`val`), the training rollout's (`train_rollout`),
the PPO update's (`update`) and the loop-level phases (`iteration`, which also
holds the checkpoint and log writes). Each is a list of `{name, secs, n}`.
"""
function save_timing!(path::String; iteration::Int, bars_val::Int, bars_train::Int,
                      val::PhaseTimer, train_rollout::PhaseTimer, update::PhaseTimer, loop::PhaseTimer,
                      iter_secs::Real)
    _upsert_run_row(path, (iteration=iteration, iter_secs=round(iter_secs, digits=2),
                           bars_val=bars_val, bars_train=bars_train,
                           val=phase_rows(val), train_rollout=phase_rows(train_rollout),
                           update=phase_rows(update), iteration_phases=phase_rows(loop)))
end

"""Append one validation run's result row to `val_summary.json` (see `_upsert_run_row`)."""
save_val_summary!(path::String, row::NamedTuple) = _upsert_run_row(path, row)
