"""
scan_ohlcv_availability.jl

Summarise the OHLCV CSVs on disk, per exchange and bar size (and the macro series), into
website/data/ohlcv_availability.json for website/dataavailability.html. Only the head and the last 4 KiB of each file are read, so
the scan takes seconds even for the 1-minute tree. Row counts are estimates (file size / mean row bytes of the first 64 KiB).

Usage:
  julia scripts/ohlcv/scan_ohlcv_availability.jl [--root website/data/ohlcv] [--out website/data/ohlcv_availability.json]
"""

const REPO_ROOT = joinpath(@__DIR__, "..", "..")
using Dates
const EXCHANGES = ["nse", "bse"]
const GRANULARITIES = ["daily", "hourly", "15min", "5min", "1min"]
const TAIL_BYTES = 4096
const HEAD_BYTES = 64 * 1024
const FRESH_DAYS = 3
const AGE_BUCKETS = [("current", 3), ("recent", 30), ("stale", 365), ("dormant", typemax(Int))]

function _date_of(line::AbstractString)
    length(line) >= 10 || return nothing
    try Date(line[1:10]) catch; nothing end
end

"""
    scan_file(path) -> NamedTuple | nothing

First and last date, size and estimated row count of one OHLCV CSV; `nothing` for a file with no data row.
"""
function scan_file(path::String)
    n = filesize(path)
    n == 0 && return nothing
    head, tail = open(path) do io
        h = read(io, min(n, HEAD_BYTES))
        seek(io, max(0, n - TAIL_BYTES))
        h, read(io)
    end
    hl = split(String(filter(!=(0x00), head)), '\n'; keepempty=false)
    length(hl) >= 2 || return nothing
    tl = split(String(filter(!=(0x00), tail)), '\n'; keepempty=false)
    first_d = _date_of(hl[2])
    last_d = nothing
    for l in Iterators.reverse(tl)
        last_d = _date_of(l)
        last_d === nothing || break
    end
    (first_d === nothing || last_d === nothing) && return nothing
    row_bytes = sum(ncodeunits, hl[2:end]) / (length(hl) - 1) + 1
    return (first=first_d, last=last_d, bytes=n, rows=round(Int, n / row_bytes))
end

_median_date(v) = isempty(v) ? nothing : sort(v)[cld(length(v), 2)]
_iso(d) = d === nothing ? nothing : string(d)

function summarise(dir::String, today::Date)
    isdir(dir) || return nothing
    files = filter(f -> endswith(f, ".csv"), readdir(dir))
    infos = Any[]; empty_files = 0
    for f in files
        r = scan_file(joinpath(dir, f))
        r === nothing ? (empty_files += 1) : push!(infos, merge(r, (symbol=f[1:end-4],)))
    end
    isempty(infos) && return Dict("files" => length(files), "empty_files" => empty_files)
    ages = [(today - r.last).value for r in infos]
    buckets = Dict(name => count(a -> a <= hi && a > lo, ages) for ((name, hi), lo) in zip(AGE_BUCKETS, [-typemax(Int); [b[2] for b in AGE_BUCKETS[1:end-1]]]))
    firsts = [r.first for r in infos]; lasts = [r.last for r in infos]
    stalest = sort(infos; by=r -> r.last)[1:min(5, length(infos))]
    Dict(
        "files" => length(files), "empty_files" => empty_files,
        "bytes" => sum(r.bytes for r in infos), "rows_est" => sum(r.rows for r in infos),
        "first_min" => _iso(minimum(firsts)), "first_median" => _iso(_median_date(firsts)), "first_max" => _iso(maximum(firsts)),
        "last_min" => _iso(minimum(lasts)), "last_median" => _iso(_median_date(lasts)), "last_max" => _iso(maximum(lasts)),
        "age_buckets" => buckets,
        "oldest_last" => [Dict("symbol" => r.symbol, "last" => _iso(r.last)) for r in stalest],
    )
end

function main()
    root = joinpath(REPO_ROOT, "website", "data", "ohlcv")
    out = joinpath(REPO_ROOT, "website", "data", "ohlcv_availability.json")
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a in ("-h", "--help"); println(match(r"\"\"\"\n(.*?)\"\"\""s, read(@__FILE__, String)).captures[1]); return
        elseif a == "--root"; root = abspath(ARGS[i+1]); i += 2
        elseif a == "--out"; out = abspath(ARGS[i+1]); i += 2
        else error("Unknown argument: $a (see --help)") end
    end
    today = Dates.today()
    result = Dict{String,Any}("generated" => string(Dates.now()), "today" => string(today), "fresh_days" => FRESH_DAYS, "exchanges" => Dict{String,Any}())
    for ex in EXCHANGES
        result["exchanges"][ex] = Dict{String,Any}()
        for g in GRANULARITIES
            s = summarise(joinpath(root, ex, g), today)
            s === nothing && continue
            result["exchanges"][ex][g] = s
            println(rpad("$ex/$g", 12), s["files"], " files")
        end
    end
    macro_dir = joinpath(root, "macro")
    if isdir(macro_dir)
        rows = Any[]
        for f in sort(filter(f -> endswith(f, ".csv"), readdir(macro_dir)))
            r = scan_file(joinpath(macro_dir, f))
            r === nothing && continue
            push!(rows, Dict("name" => f[1:end-4], "first" => _iso(r.first), "last" => _iso(r.last), "rows_est" => r.rows, "bytes" => r.bytes))
        end
        result["macro"] = rows
    end
    mkpath(dirname(out))
    open(out, "w") do io
        print(io, repr_json(result))
    end
    println("wrote ", relpath(out, REPO_ROOT))
end

function repr_json(x, ind=0)
    pad = "  " ^ (ind + 1); end_pad = "  " ^ ind
    if x isa AbstractDict
        isempty(x) && return "{}"
        return "{\n" * join((pad * repr(string(k)) * ": " * repr_json(v, ind + 1) for (k, v) in sort(collect(x); by=first)), ",\n") * "\n" * end_pad * "}"
    elseif x isa AbstractVector
        isempty(x) && return "[]"
        return "[\n" * join((pad * repr_json(v, ind + 1) for v in x), ",\n") * "\n" * end_pad * "]"
    elseif x isa AbstractString; return repr(String(x))
    elseif x === nothing; return "null"
    else return string(x) end
end

main()
