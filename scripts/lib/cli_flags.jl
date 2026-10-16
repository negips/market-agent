"""
cli_flags.jl

Shared check for scripts that read their flags with `"--x" in ARGS` and so never see a mistyped one. Include it and call
`reject_unknown_flags([...])` with every flag the script accepts; any other argument that starts with a dash stops the
script with an error that names the closest valid flag. Positional arguments and numbers (including negative ones) pass.

    include(joinpath(@__DIR__, "..", "lib", "cli_flags.jl"))
    reject_unknown_flags(["--help", "-h", "--dry-run", "--symbol"])
"""

"""
    flag_distance(a, b) -> Int

Levenshtein distance between two flag spellings.
"""
function flag_distance(a::AbstractString, b::AbstractString)::Int
    x, y = collect(a), collect(b)
    prev = collect(0:length(y))
    for i in 1:length(x)
        cur = [i]
        for j in 1:length(y)
            push!(cur, min(prev[j + 1] + 1, cur[j] + 1, prev[j] + (x[i] == y[j] ? 0 : 1)))
        end
        prev = cur
    end
    return prev[end]
end

"""
    reject_unknown_flags(known; args=ARGS)

Throw an `ErrorException` when `args` holds a dash-prefixed token that is not in `known`. `--flag=value` is checked as
`--flag`. The message suggests the nearest known flag when one is within three edits.

# Arguments
- `known`: every accepted flag, including `-h` and `--help`
- `args`: the argument vector (default `ARGS`)
"""
function reject_unknown_flags(known::AbstractVector{<:AbstractString}; args::AbstractVector{<:AbstractString}=ARGS)
    for raw in args
        (startswith(raw, "-") && length(raw) > 1 && !occursin(r"^-\d", raw)) || continue
        flag = first(split(raw, '='))
        flag in known && continue
        best = isempty(known) ? nothing : argmin(k -> flag_distance(flag, k), known)
        hint = best !== nothing && flag_distance(flag, best) <= 3 ? " (did you mean $best?)" : ""
        error("Unknown argument: $raw$hint. See --help.")
    end
    return nothing
end
