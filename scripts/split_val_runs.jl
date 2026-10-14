"""
split_val_runs.jl

Convert the old append-only `website/data/trading_game/val_runs.jsonl` (one
held-out run per line, 1.5 GB after a long run) into the layout
`tradinggamelive.html` now reads: `val_runs/index.json` (per-run summaries with a
downsampled curve) plus one `val_runs/iter_NNNNN.json` per run (full curve, trades,
final holdings). The source file is left in place; delete it once the page works.

Usage:
  julia --project=packages/TradingGame scripts/split_val_runs.jl
  julia --project=packages/TradingGame scripts/split_val_runs.jl --src PATH --out DIR
"""

using TradingGame

const DATA_DIR = joinpath(@__DIR__, "..", "website", "data", "trading_game")

function main(args)
    if any(a -> a in ("-h", "--help"), args)
        println("usage: split_val_runs.jl [--src val_runs.jsonl] [--out val_runs/]")
        return
    end
    opt(name, default) = (i = findfirst(==(name), args); i === nothing ? default : args[i + 1])
    src = opt("--src", joinpath(DATA_DIR, "val_runs.jsonl"))
    out = opt("--out", joinpath(DATA_DIR, "val_runs"))
    isfile(src) || error("No such file: $src")
    n = split_val_runs_jsonl(src, out)
    println("Converted $n runs into $out")
end

main(ARGS)
