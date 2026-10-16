"""
run_pipeline.jl

The whole v4 forecaster pipeline in one command: build the company universe, build the forecaster
return cache for it, and start training. Each stage is the existing script, run as a subprocess with
the same `--config`, so the three always agree on the bar size, dates and companies:

  1. universe  scripts/game/build_market_universe_snapshot.jl   -> website/data/trading_game/universe_latest.json
  2. cache     scripts/forecaster/build_forecaster_cache.jl     -> website/data/forecaster_cache_{granularity}.bson
  3. train     scripts/forecaster/train_minute_forecaster.jl    -> website/data/forecaster[_{granularity}]/

`--rotate` is for training in rounds on changing companies: the val companies of the current
universe are kept (so validation losses stay comparable from round to round), a new train set is
drawn from the rest of the pool with a fresh random seed, and the previous universe file is copied
to website/data/trading_game/universes/ first. Combine it with `--resume` (passed to the trainer
after `--`) to continue the same weights on the new companies.

Usage:
  julia scripts/forecaster/run_pipeline.jl --config configs/bse_v3.toml
  julia scripts/forecaster/run_pipeline.jl --config configs/bse_v3.toml --rotate -- --resume --epochs 15 --device gpu
  julia scripts/forecaster/run_pipeline.jl --config configs/bse_v3.toml --skip-universe -- --device gpu
  julia scripts/forecaster/run_pipeline.jl --config configs/bse_v3.toml --rotate --dry-run
  julia scripts/forecaster/run_pipeline.jl --config configs/bse_v3.toml --rotate --repeat 6 -- --epochs 15 --device gpu

Options:
  --config PATH     TOML run file, passed to all three stages (required)
  --rotate          Keep the val companies, draw a new train set (build_market_universe_snapshot.jl --keep-val)
  --seed N          Seed of the universe draw (default with --rotate: a random one per round, printed; without it the
                    config's). With --repeat, round k uses N + k - 1, so the rounds differ.
  --repeat N        Run the whole pipeline N times (default 1): rebuild the universe and cache, then train, each round.
                    Rounds after the first add --resume to the trainer (and drop --init-from / --no-resume), so the
                    weights carry on; round 1 uses your trainer arguments as given. Without --rotate, every round redraws
                    the val companies too, so the validation losses of different rounds are not comparable.
                    A failed stage ends everything. `touch website/data/STOP_PIPELINE` ends after the current round
                    (the file is removed when seen; the trainer's own STOP ends the current round sooner).
  --skip-universe   Reuse the existing universe_latest.json
  --skip-cache      Reuse the existing forecaster cache (needs the universe to be unchanged)
  --skip-train      Stop after the cache
  --dry-run         Print the commands and run nothing
  --                Everything after it goes to train_minute_forecaster.jl (e.g. --resume --epochs 15 --device gpu)

Prerequisites: website/data/inference_cache.bson (build_cache.jl; the universe stage reads it), the companies list
of the cache's exchange, and the OHLCV CSVs of the configured bar size for every company drawn.
"""

using Dates, Random

include(joinpath(@__DIR__, "..", "lib", "cli_flags.jl"))

const REPO_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const TRADING_GAME_PROJECT = "--project=packages/TradingGame"
const UNIVERSE_FILE = joinpath(REPO_ROOT, "website", "data", "trading_game", "universe_latest.json")
const UNIVERSE_ARCHIVE_DIR = joinpath(REPO_ROOT, "website", "data", "trading_game", "universes")
const MAX_SEED = 10^9
const STOP_PIPELINE_FILE = joinpath(REPO_ROOT, "website", "data", "STOP_PIPELINE")
const PIPELINE_FLAGS = ["-h", "--help", "--config", "--rotate", "--seed", "--repeat", "--skip-universe", "--skip-cache",
                        "--skip-train", "--dry-run"]

"""One stage of the pipeline: its name and the command that runs it."""
struct Stage
    name::String
    cmd::Cmd
    shown::String
end

"""
    split_args(args) -> (own, trainer_args)

Split `args` at the first `--`: the options of this script and the arguments for the trainer.
"""
function split_args(args::Vector{String})
    i = findfirst(==("--"), args)
    i === nothing && return args, String[]
    return args[1:i-1], args[i+1:end]
end

"""
    option_value(args, flag) -> Union{String,Nothing}

The value after `flag` in `args`, `nothing` if the flag is absent; an error if it is the last argument.
"""
function option_value(args::Vector{String}, flag::String)
    i = findfirst(==(flag), args)
    i === nothing && return nothing
    i < length(args) || error("$flag needs a value. See --help.")
    return args[i+1]
end

"""
    julia_stage(name, script, args; threads=nothing) -> Stage

A stage that runs `script` under the TradingGame project with `args`, in the repo root.
"""
function julia_stage(name::String, script::String, args::Vector{String}; threads::Union{Nothing,String}=nothing)
    thread_flag = threads === nothing ? String[] : ["--threads=$threads"]
    words = [thread_flag; TRADING_GAME_PROJECT; joinpath("scripts", script); args]
    return Stage(name, setenv(`$(Base.julia_cmd()) $words`; dir=REPO_ROOT), join(["julia"; words], " "))
end

"""
    repeat_count(own) -> Int

The value of `--repeat` (1 when absent); an error if it is not a positive integer.
"""
function repeat_count(own::Vector{String})::Int
    text = option_value(own, "--repeat")
    text === nothing && return 1
    n = tryparse(Int, text)
    (n === nothing || n < 1) && error("--repeat needs a positive integer, got '$text'. See --help.")
    return n
end

"""
    round_trainer_args(trainer_args, round) -> Vector{String}

The trainer arguments for round `round`: as given for round 1; later rounds continue the same weights, so `--init-from PATH` and
`--no-resume` are dropped and `--resume` is added.
"""
function round_trainer_args(trainer_args::Vector{String}, round::Int)::Vector{String}
    round == 1 && return trainer_args
    args = String[]
    skip_next = false
    for a in trainer_args
        if skip_next
            skip_next = false
        elseif a == "--init-from"
            skip_next = true
        elseif a != "--no-resume" && !startswith(a, "--init-from=")
            push!(args, a)
        end
    end
    "--resume" in args || push!(args, "--resume")
    return args
end

"""
    build_stages(own, trainer_args; round=1) -> (Vector{Stage}, Union{Int,Nothing})

The stages to run in round `round` for the options `own`, and the seed used for a rotated universe (`nothing` if the draw is
not rotated). The config path is required.
"""
function build_stages(own::Vector{String}, trainer_args::Vector{String}; round::Int=1)
    config = option_value(own, "--config")
    config === nothing && error("--config PATH is required. See --help.")
    isfile(config) || error("Config not found: $config")
    config = abspath(config)
    rotate = "--rotate" in own
    seed_text = option_value(own, "--seed")
    seed = seed_text !== nothing ? parse(Int, seed_text) + round - 1 : (rotate ? rand(RandomDevice(), 1:MAX_SEED) : nothing)

    stages = Stage[]
    if !("--skip-universe" in own)
        universe_args = ["--config", config]
        rotate && push!(universe_args, "--keep-val")
        seed === nothing || append!(universe_args, ["--seed", string(seed)])
        push!(stages, julia_stage("universe", joinpath("game", "build_market_universe_snapshot.jl"), universe_args))
    end
    "--skip-cache" in own || push!(stages, julia_stage("cache", joinpath("forecaster", "build_forecaster_cache.jl"),
                                                       ["--config", config]; threads="auto"))
    "--skip-train" in own || push!(stages, julia_stage("train", joinpath("forecaster", "train_minute_forecaster.jl"),
                                                       ["--config", config, round_trainer_args(trainer_args, round)...]))
    return stages, rotate ? seed : nothing
end

"""
    archive_universe() -> Union{String,Nothing}

Copy the current `universe_latest.json` into `universes/` under a timestamped name; the new path, or `nothing` if there
is no universe yet.
"""
function archive_universe()
    isfile(UNIVERSE_FILE) || return nothing
    mkpath(UNIVERSE_ARCHIVE_DIR)
    path = joinpath(UNIVERSE_ARCHIVE_DIR, "universe_" * Dates.format(now(), "yyyymmdd_HHMMSS") * ".json")
    cp(UNIVERSE_FILE, path)
    return path
end

"""Run the stages in order and stop at the first failure."""
function run_stages(stages::Vector{Stage}, rotate::Bool)
    for (i, stage) in enumerate(stages)
        println("\n══ stage $i/$(length(stages)): $(stage.name) ══")
        if stage.name == "universe" && rotate
            archived = archive_universe()
            archived === nothing || println("previous universe saved to ", relpath(archived, REPO_ROOT))
        end
        t0 = time()
        try
            run(stage.cmd)
        catch err
            err isa ProcessFailedException || rethrow()
            println(stderr, "\nStage '$(stage.name)' failed; the pipeline stopped there.")
            exit(1)
        end
        println("stage '$(stage.name)' done in $(round(Int, time() - t0)) s")
    end
end

function main()
    own, trainer_args = split_args(String.(ARGS))
    if any(a -> a in ("-h", "--help"), own)
        println(match(r"\"\"\"\n(.*?)\"\"\""s, read(@__FILE__, String)).captures[1])
        return
    end
    reject_unknown_flags(PIPELINE_FLAGS; args=own)
    rounds = repeat_count(own)
    dry_run = "--dry-run" in own
    rotate = "--rotate" in own
    for round in 1:rounds
        stages, seed = build_stages(own, trainer_args; round=round)
        isempty(stages) && error("Nothing to do: every stage is skipped.")
        rounds > 1 && println("\n████ round $round of $rounds ████")
        seed === nothing || println("rotating the train companies, universe seed $seed")
        if dry_run
            foreach(stage -> println(stage.name, ": ", stage.shown), stages)
        else
            run_stages(stages, rotate)
        end
        if !dry_run && round < rounds && isfile(STOP_PIPELINE_FILE)
            rm(STOP_PIPELINE_FILE)
            println("\nSTOP_PIPELINE found: stopping after round $round of $rounds.")
            return
        end
    end
end

main()
