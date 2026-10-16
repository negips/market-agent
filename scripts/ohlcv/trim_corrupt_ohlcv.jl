"""
trim_corrupt_ohlcv.jl

Remove the NUL-byte tail that an interrupted write (a crash or power loss while a collect/update script was appending)
leaves at the end of an OHLCV CSV. The tail is cut back to the last complete, newline-terminated row, so the partial row
before the NULs goes too; `update_ohlcv.jl` then refetches the missing bars from the last date left in the file.

Dry run by default: it only lists what it would do. Nothing is changed without --apply.

Usage:
  julia scripts/ohlcv/trim_corrupt_ohlcv.jl                      # dry run over every CSV under website/data/ohlcv
  julia scripts/ohlcv/trim_corrupt_ohlcv.jl --apply              # truncate the damaged files
  julia scripts/ohlcv/trim_corrupt_ohlcv.jl --dir website/data/ohlcv/bse/1min

Options:
  --apply      Truncate the files (default: dry run)
  --dir PATH   Scan this directory tree instead of website/data/ohlcv
  -h, --help   This text

Only the last 64 KiB of each file is read. A file with NUL bytes somewhere other than a pure tail (a NUL followed by real
data) is reported and left alone: that is a different kind of damage and needs a look.
"""

const REPO_ROOT = joinpath(@__DIR__, "..", "..")
const TAIL_BYTES = 64 * 1024

"""
    inspect_tail(path) -> (:ok | :tail | :interior, keep_bytes, nul_bytes)

`:tail` means the file ends in a run of NULs; `keep_bytes` is the length to truncate to (just after the last newline
that precedes the run). `:interior` means NULs exist in the last 64 KiB with real data after them.
"""
function inspect_tail(path::String)
    n = filesize(path)
    n == 0 && return (:ok, 0, 0)
    start = max(0, n - TAIL_BYTES)
    buf = open(path) do io
        seek(io, start)
        read(io)
    end
    first_nul = findfirst(==(0x00), buf)
    first_nul === nothing && return (:ok, n, 0)
    last_data = findlast(!=(0x00), buf)
    (last_data !== nothing && last_data > first_nul) && return (:interior, n, count(==(0x00), buf))
    nuls = length(buf) - first_nul + 1
    nl = findlast(==(UInt8('\n')), @view buf[1:first_nul - 1])
    nl === nothing && return (:interior, n, nuls)
    return (:tail, start + nl, n - (start + nl))
end

function main()
    apply = false
    root = joinpath(REPO_ROOT, "website", "data", "ohlcv")
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a in ("-h", "--help")
            println(match(r"\"\"\"\n(.*?)\"\"\""s, read(@__FILE__, String)).captures[1])
            return
        elseif a == "--apply"; apply = true; i += 1
        elseif a == "--dir"
            i < length(ARGS) || error("--dir needs a path")
            root = abspath(ARGS[i+1]); i += 2
        else
            error("Unknown argument: $a (see --help)")
        end
    end
    isdir(root) || error("Not a directory: $root")

    scanned = 0; fixed = 0; odd = 0
    for (dir, _, files) in walkdir(root), f in files
        endswith(f, ".csv") || continue
        path = joinpath(dir, f)
        scanned += 1
        kind, keep, nuls = inspect_tail(path)
        kind === :ok && continue
        rel = relpath(path, root)
        if kind === :interior
            odd += 1
            println("SKIP  $rel: NUL bytes with data after them (not a pure tail); left alone")
            continue
        end
        drop = filesize(path) - keep
        lastrow = open(path) do io
            seek(io, max(0, keep - 200)); String(read(io, min(200, keep)))
        end
        println("$(apply ? "TRIM " : "WOULD") $rel: cut $drop bytes ($nuls NULs$(drop > nuls ? " + a partial row" : "")) -> $keep; new last row: $(last(split(rstrip(lastrow), '\n')))")
        if apply
            open(path, "r+") do io
                truncate(io, keep)
            end
        end
        fixed += 1
    end
    println("\nScanned $scanned CSVs: $fixed with a NUL tail$(apply ? " (truncated)" : " (dry run, nothing changed; re-run with --apply)"), $odd with other NUL damage.")
end

main()
