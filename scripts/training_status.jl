"""
training_status.jl

Prints a one-shot, human-readable snapshot of the current `train_trading_policy.jl`
run: whether a process is active, its config (`run_config.json`), its live
state (`live_status.json` — iteration, phase, portfolio value, update
progress), and the last few logged iterations (`episode_log.jsonl` — return,
loss, and the rollout/val/update phase-timing breakdown). Read-only — never
touches any file a live run is writing to.

Meant for a quick check from outside a full session (e.g. Remote Control on
the mobile app): no arguments, no prompts, exits immediately after printing.
Color is used when the output looks like a terminal (disabled automatically
when piped/redirected, or with `NO_COLOR` set) — see `_colorize`.

Usage:
  julia --project=packages/TradingGame scripts/training_status.jl
"""

using JSON3, Dates, Printf

const REPO_ROOT = joinpath(@__DIR__, "..")
const DATA_DIR  = joinpath(REPO_ROOT, "website", "data", "trading_game")

const USE_COLOR = get(ENV, "NO_COLOR", "") == "" && (isa(stdout, Base.TTY) || get(ENV, "FORCE_COLOR", "") != "")

const COLORS = Dict(
    :bold => "\e[1m", :dim => "\e[2m", :reset => "\e[0m",
    :red => "\e[31m", :green => "\e[32m", :yellow => "\e[33m",
    :blue => "\e[34m", :cyan => "\e[36m", :magenta => "\e[35m",
)

"""Wrap `s` in the ANSI code for `color` (a key of `COLORS`), or return `s`
unchanged when `USE_COLOR` is false — the single place every colored string
in this script passes through, so disabling color never requires touching
call sites."""
_colorize(s, color::Symbol) = USE_COLOR ? COLORS[color] * s * COLORS[:reset] : s

_header(title::String) = println("\n", _colorize(title, :bold), "\n", _colorize("─"^60, :dim))

function _read_json(path::String)
    isfile(path) || return nothing
    try
        return JSON3.read(read(path, String))
    catch e
        return nothing
    end
end

function _process_info()
    try
        output = read(`pgrep -af train_trading_policy.jl`, String)
        lines = filter(!isempty, split(output, '\n'))
        return lines
    catch
        return String[]
    end
end

function _fmt_rupee(v)
    v === nothing && return "—"
    return "₹" * replace(@sprintf("%.0f", v), r"(\d)(?=(\d{3})+(?!\d))" => s"\1,")
end

function _fmt_pct(v)
    v === nothing && return "—"
    s = @sprintf("%+.2f%%", v)
    color = v > 0 ? :green : (v < 0 ? :red : :dim)
    return _colorize(s, color)
end

function _fmt_age(updated_at::String)
    try
        t = DateTime(replace(updated_at, "Z" => ""))
        secs = Dates.value(now(UTC) - t) / 1000
        secs < 60 && return @sprintf("%.0fs ago", secs)
        secs < 3600 && return @sprintf("%.1fm ago", secs / 60)
        return @sprintf("%.1fh ago", secs / 3600)
    catch
        return "unknown"
    end
end

function _fmt_duration(secs)
    secs === nothing && return "—"
    secs < 60 && return @sprintf("%.1fs", secs)
    secs < 3600 && return @sprintf("%dm %02ds", Int(fld(secs, 60)), Int(round(secs % 60)))
    return @sprintf("%dh %02dm", Int(fld(secs, 3600)), Int(fld(secs % 3600, 60)))
end

"""Build a fixed-width `[####------]` bar from a `done/total` fraction —
used for the PPO update's minibatch progress."""
function _progress_bar(done::Real, total::Real; width::Int=20)
    total <= 0 && return "[" * "-"^width * "]"
    frac = clamp(done / total, 0.0, 1.0)
    filled = round(Int, frac * width)
    bar = "█"^filled * "-"^(width - filled)
    return "[" * _colorize(bar, :cyan) * "]" * @sprintf(" %5.1f%%", frac * 100)
end

function main()
    println(_colorize("═"^60, :bold))
    println(_colorize("  TradingGame training status", :bold), "  ", _colorize(string(now(UTC)) * " UTC", :dim))
    println(_colorize("═"^60, :bold))

    _header("Process")
    procs = _process_info()
    if isempty(procs)
        println("  ", _colorize("● NOT RUNNING", :red))
    else
        println("  ", _colorize("● RUNNING", :green))
        for p in procs
            m = match(r"^\s*(\d+)\s+(.*train_trading_policy\.jl.*)$", p)
            if m === nothing
                println("  ", p)
            else
                println("  ", _colorize("pid " * m[1], :dim), "  ", m[2])
            end
        end
    end

    _header("Config")
    cfg = _read_json(joinpath(DATA_DIR, "run_config.json"))
    if cfg !== nothing
        @printf("  %-14s %s\n", "device:", get(cfg, :device, "?"))
        @printf("  %-14s %s\n", "minibatch:", something(get(cfg, :minibatch, nothing), "auto"))
        @printf("  %-14s %s\n", "val_days:", string(get(cfg, :val_days, "?")))
        @printf("  %-14s %s\n", "n_candidates:", string(get(cfg, :n_candidates, "?")))
        @printf("  %-14s %s\n", "initial_cash:", _fmt_rupee(get(cfg, :initial_cash, nothing)))
        @printf("  %-14s %s\n", "lr:", string(get(cfg, :lr, "?")))
        @printf("  %-14s %s\n", "entropy_coef:", string(get(cfg, :entropy_coef, "?")))
        @printf("  %-14s %s\n", "seed:", something(get(cfg, :seed, nothing), "none"))
    else
        println("  ", _colorize("no run_config.json found — no run has started yet", :dim))
    end

    _header("Live state")
    live = _read_json(joinpath(DATA_DIR, "live_status.json"))
    if live !== nothing
        println("  ", _colorize("updated " * _fmt_age(string(get(live, :updated_at, ""))), :dim))
        @printf("  %-14s %s\n", "iteration:", string(get(live, :iteration, "?")))
        @printf("  %-14s %s\n", "phase:", _colorize(string(get(live, :phase, "?")), :magenta))
        @printf("  %-14s %s\n", "sim. date:", string(get(live, :current_date, "?")))
        @printf("  %-14s %s\n", "universe:", string(get(live, :n_candidates, "?")) * " candidates")

        value = get(live, :portfolio_value, nothing)
        initial = get(live, :initial_cash, nothing)
        pnl = (value !== nothing && initial !== nothing && initial > 0) ? (value / initial - 1) * 100 : nothing
        @printf("  %-14s %s  (%s)\n", "portfolio:", _fmt_rupee(value), _fmt_pct(pnl))
        @printf("  %-14s %s\n", "cash:", _fmt_rupee(get(live, :cash, nothing)))
        @printf("  %-14s %s\n", "reserved:", _fmt_rupee(get(live, :reserved_cash, nothing)))

        holdings = get(live, :holdings, nothing)
        holdings !== nothing && @printf("  %-14s %d open position%s\n",
                                         "holdings:", length(holdings), length(holdings) == 1 ? "" : "s")

        up = get(live, :update_progress, nothing)
        if up !== nothing
            epoch, k_epochs = get(up, :epoch, 0), get(up, :k_epochs, 0)
            mb, total_mb = get(up, :minibatch, 0), get(up, :total_minibatches, 0)
            @printf("  %-14s epoch %s/%s\n", "PPO update:", string(epoch), string(k_epochs))
            println("  ", " "^14, _progress_bar(mb, total_mb), "  mb ", mb, "/", total_mb)
            @printf("  %-14s %.4f\n", "loss:", Float64(get(up, :loss, 0.0)))
        end
    else
        println("  ", _colorize("no live_status.json found", :dim))
    end

    _header("Recent iterations")
    log_path = joinpath(DATA_DIR, "episode_log.jsonl")
    if isfile(log_path)
        rows = []
        for line in eachline(log_path)
            isempty(strip(line)) && continue
            try
                push!(rows, JSON3.read(line))
            catch
            end
        end
        if !isempty(rows)
            println("  ", _colorize("last " * string(min(5, length(rows))) * " of " * string(length(rows)), :dim))
            println()
            hdr = @sprintf("  %6s  %12s  %10s  %9s  %9s  %9s  %9s", "iter", "train_ret", "val_ret", "rollout", "val", "update", "total")
            println(_colorize(hdr, :dim))
            start_idx = max(1, length(rows) - 4)
            for i in start_idx:length(rows)
                r = rows[i]
                improved = get(r, :improved, false) === true
                val_return = get(r, :val_return, nothing)
                elapsed = get(r, :elapsed_secs, nothing)
                prev_elapsed = i > 1 ? get(rows[i-1], :elapsed_secs, nothing) : nothing
                iter_total = (elapsed !== nothing && prev_elapsed !== nothing) ?
                             Float64(elapsed) - Float64(prev_elapsed) : elapsed
                train_ret = Float64(get(r, :train_return, 0.0))
                train_ret_str = _colorize(@sprintf("%12.4f", train_ret), train_ret >= 0 ? :green : :red)
                line = @sprintf("  %6s  %s  %10s  %9s  %9s  %9s  %9s",
                        string(get(r, :iteration, "?")), train_ret_str,
                        val_return === nothing ? "—" : @sprintf("%.4f", Float64(val_return)),
                        _fmt_duration(get(r, :rollout_secs, nothing)),
                        _fmt_duration(get(r, :val_rollout_secs, nothing)),
                        _fmt_duration(get(r, :update_secs, nothing)),
                        _fmt_duration(iter_total))
                line *= improved ? "  " * _colorize("★ best", :yellow) : ""
                println(line)
            end
            last_best = get(rows[end], :best_return, nothing)
            improved_rows = filter(r -> get(r, :improved, false) === true, rows)
            best_iter_num = isempty(improved_rows) ? nothing : get(improved_rows[end], :iteration, nothing)
            if last_best !== nothing && best_iter_num !== nothing
                println()
                @printf("  %s %s\n", _colorize("Best so far:", :bold),
                        @sprintf("return %.4f (iteration %s)", Float64(last_best), string(best_iter_num)))
            end
        end
    else
        println("  ", _colorize("no episode_log.jsonl found", :dim))
    end

    println()
end

main()
