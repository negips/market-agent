"""
sweep_forecaster.jl

Train the v4 forecaster once per value of one hyperparameter and print a comparison table.
Each run is a separate `train_minute_forecaster.jl` process (same config, same seed) with its own
output directory under `website/data/forecaster_sweep/`, so the runs never overwrite each other and
a finished run is skipped on a re-run (`--force` redoes it).

Usage:
  julia --project=packages/TradingGame scripts/sweep_forecaster.jl --config configs/bse_v3.toml \\
      --param d --values 32,64,128 --seed 42 --device gpu
  julia --project=packages/TradingGame scripts/sweep_forecaster.jl --config configs/bse_v3.toml \\
      --param window --values 48,96,192 --epochs 10 --device gpu
  julia --project=packages/TradingGame scripts/sweep_forecaster.jl --table-only --param d --values 32,64,128

Options:
  --config PATH       TOML run file passed to every run
  --param NAME        Any train_minute_forecaster.jl option without its leading dashes: d, heads, blocks, window, dropout,
                      lr, batch, nu, loss, ... (underscores become dashes)
  --values A,B,C      The values to try, in order
  --tag NAME          Output directory name prefix (default: the parameter name)
  --force             Re-run a value whose val_report.json already exists
  --table-only        Do not train; print the table from the existing runs
  anything else       Passed through to every run (e.g. --seed 42 --device gpu --epochs 10)

Prints, per value: parameters, best epoch, val NLL and its gain over a constant σ and over an EWMA σ,
the 68%/95% coverage, the mean-forecast skill and hit rate, and seconds per epoch. With one seed per value,
differences of a few thousandths of a nat are within run-to-run noise.
"""

using JSON3, Printf

const REPO_ROOT = joinpath(@__DIR__, "..")
const SWEEP_DIR = joinpath(REPO_ROOT, "website", "data", "forecaster_sweep")

function parse_sweep_args()
    opts = Dict{String, Any}("param" => nothing, "values" => String[], "tag" => nothing, "force" => false,
                             "table_only" => false, "passthrough" => String[])
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a in ("-h", "--help")
            println(match(r"\"\"\"\n(.*?)\"\"\""s, read(@__FILE__, String)).captures[1])
            exit(0)
        elseif a == "--param"; opts["param"] = ARGS[i+1]; i += 2
        elseif a == "--values"; opts["values"] = String.(split(ARGS[i+1], ",")); i += 2
        elseif a == "--tag"; opts["tag"] = ARGS[i+1]; i += 2
        elseif a == "--force"; opts["force"] = true; i += 1
        elseif a == "--table-only"; opts["table_only"] = true; i += 1
        else; push!(opts["passthrough"], a); i += 1
        end
    end
    (opts["param"] === nothing || isempty(opts["values"])) && error("--param and --values are required (see --help)")
    return opts
end

function run_dir(opts, value)
    tag = something(opts["tag"], opts["param"])
    return joinpath(SWEEP_DIR, "$(tag)_$(value)")
end

function fmt(x; digits=4)
    x === nothing && return "—"
    return @sprintf("%.*f", digits, x)
end

function print_table(opts)
    rows = []
    for v in opts["values"]
        dir = run_dir(opts, v)
        report = joinpath(dir, "val_report.json")
        if !isfile(report)
            push!(rows, (v, nothing)); continue
        end
        rep = JSON3.read(read(report, String)).val
        cfg = JSON3.read(read(joinpath(dir, "run_config.json"), String))
        log = [JSON3.read(l) for l in eachline(joinpath(dir, "epoch_log.jsonl")) if !isempty(strip(l))]
        best = findlast(r -> r.best, log)
        push!(rows, (v, (params=cfg.model.parameters, best_epoch=best === nothing ? 0 : log[best].epoch, epochs=length(log),
                         nll=rep.nll, gain_const=rep.nll_gain_vs_const_sigma, gain_ewma=rep.nll_gain_vs_ewma_sigma,
                         cov68=rep.coverage_68, cov95=rep.coverage_95, skill=rep.skill_vs_zero, hit=rep.sign_hit_rate,
                         secs=sum(r.secs for r in log) / max(length(log), 1))))
    end
    println()
    @printf("%-10s %9s %6s %9s %10s %10s %7s %7s %8s %7s %9s\n", opts["param"], "params", "best", "val NLL", "gain const", "gain EWMA",
            "cov68", "cov95", "skill", "hit", "s/epoch")
    for (v, r) in rows
        if r === nothing
            @printf("%-10s  (no result)\n", v)
        else
            @printf("%-10s %9d %3d/%-2d %9s %10s %10s %7s %7s %8s %7s %9.0f\n", v, r.params, r.best_epoch, r.epochs, fmt(r.nll), fmt(r.gain_const),
                    fmt(r.gain_ewma), fmt(r.cov68, digits=3), fmt(r.cov95, digits=3), fmt(r.skill), fmt(r.hit, digits=3), r.secs)
        end
    end
    println("\nLower val NLL is better; gains are in nats per target (positive = better than the baseline).")
end

function main()
    opts = parse_sweep_args()
    if !opts["table_only"]
        script = joinpath(@__DIR__, "train_minute_forecaster.jl")
        flag = "--" * replace(opts["param"], "_" => "-")
        for v in opts["values"]
            dir = run_dir(opts, v)
            if isfile(joinpath(dir, "val_report.json")) && !opts["force"]
                @info "[$(opts["param"]) = $v] already done → $dir (use --force to redo)"
                continue
            end
            @info "[$(opts["param"]) = $v] training → $dir"
            cmd = `julia --project=$(Base.active_project()) $script $(opts["passthrough"]) $flag $v --out-dir $dir`
            run(cmd)
        end
    end
    print_table(opts)
end

main()
