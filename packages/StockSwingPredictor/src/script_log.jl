"""
Dual-output logging for the long-running OHLCV scripts (`collect_nse_ohlcv.jl`,
`collect_bse_ohlcv.jl`, `update_ohlcv.jl`, `backfill_ohlcv.jl`): every event —
including routine per-symbol detail that would otherwise flood a terminal —
goes to a FILE immediately, flushed after every single line. The terminal
only receives a much sparser stream (stage headers, periodic progress
counters, warnings/errors), via `logboth`.

The flush-per-line behavior matters on its own: Julia's own stdout, once
redirected away from a real terminal (a log redirect, a background job),
buffers heavily and can sit without writing anything for many minutes even
under continuous `@info` calls — observed live, checking a redirected
`backfill_ohlcv.jl` run's log file showed nothing for 20+ minutes of real
work. A `ScriptLog` file is never subject to that: `tail -f` or a fresh
`read` always reflects the true current state.
"""
struct ScriptLog
    io::IO
end

"""
Module-level pointer to whichever `ScriptLog` the currently-running script
has open, if any. `_kite_get` (`kite_data.jl`) reads this via
`active_script_log()` to route Kite relogin attempts to the log file
without threading a `ScriptLog` through every call site between
`collect_ohlcv*`/`backfill_*!`/`update_*!` and `_kite_get` — relogin
triggers deep inside `_kite_get`, shared by every Kite call regardless of
which script or granularity is running, so a global pointer (set by
`open_script_log`, cleared by `close_script_log`) matches the same
module-level granularity `kite_data.jl` already uses for
`KITE_RELOGIN_COOLDOWN_SECONDS`'s global cooldown. `nothing` when no
script has opened a log (direct REPL usage) — `_log_summary`/`_log_detail`
fall back to plain `@info`/`@warn` in that case, same as always.
"""
const _ACTIVE_SCRIPT_LOG = Ref{Union{ScriptLog,Nothing}}(nothing)

"""Currently-active `ScriptLog`, or `nothing` if no script has opened one."""
active_script_log() = _ACTIVE_SCRIPT_LOG[]

"""
Default log path for a Kite `account` slot: account 1 keeps `path` unchanged;
account N ≥ 2 inserts `.accountN` before the extension (`collect_bse_ohlcv.log`
→ `collect_bse_ohlcv.account2.log`). Two parallel jobs of the same script on
different accounts therefore never interleave lines in one file. Only for
DEFAULT paths — an explicit `--log-file` is always used as given.

# Arguments
- `path`: the script's default log file
- `account`: Kite API-key slot (see `kite_account_from_args`)

# Returns
- `String`: the path to open
"""
function account_log_path(path::String, account::Int)::String
    account == 1 && return path
    base, ext = splitext(path)
    return "$base.account$account$ext"
end

"""
Open (append) a script's persistent log file and write a run-start banner
recording the timestamp and `ARGS`. Append, not overwrite — a script's log
accumulates across runs so past runs stay inspectable; the banner is what
makes one run's slice of the file easy to find (e.g. `grep -A 1000000
"RUN STARTED" file.log | tail`, or just read from the end). Also registers
this log as the active one for `_kite_get`'s relogin logging — see
`_ACTIVE_SCRIPT_LOG`.
"""
function open_script_log(path::String)::ScriptLog
    mkpath(dirname(path))
    io = open(path, "a")
    println(io, "")
    println(io, "=== RUN STARTED $(now()) — args: $(join(ARGS, " ")) ===")
    flush(io)
    slog = ScriptLog(io)
    _ACTIVE_SCRIPT_LOG[] = slog
    return slog
end

"""Write a run-end banner, close the log file, and clear it as the active
log (see `_ACTIVE_SCRIPT_LOG`)."""
function close_script_log(slog::ScriptLog, summary::String="")
    suffix = isempty(summary) ? "" : " — $summary"
    println(slog.io, "=== RUN FINISHED $(now())$suffix ===")
    flush(slog.io)
    close(slog.io)
    _ACTIVE_SCRIPT_LOG[] === slog && (_ACTIVE_SCRIPT_LOG[] = nothing)
end

"""Write one timestamped line to the log FILE only — for routine per-symbol
detail (a successful backfill, a confirmed-empty result, a plain skip) that
would otherwise flood the terminal at ~2000 symbols per granularity."""
function logf(slog::ScriptLog, msg::String)
    println(slog.io, "[$(Dates.format(now(), "HH:MM:SS"))] $msg")
    flush(slog.io)
end

"""Write to BOTH the log file and the terminal (via `@info`/`@warn`) — stage
headers, periodic progress heartbeats, final summaries, and anything that
genuinely needs attention while the job is running."""
function logboth(slog::ScriptLog, msg::String; warn::Bool=false)
    logf(slog, msg)
    warn ? (@warn msg) : (@info msg)
end

"""
`_log_detail`/`_log_summary` let a function like `collect_ohlcv_hourly`
accept an OPTIONAL `slog::Union{ScriptLog, Nothing}` — `nothing` (the
default) falls back to plain `@info`/`@warn` so the function stays directly
usable from the REPL without first opening a `ScriptLog`; a real `ScriptLog`
routes routine per-symbol detail to the file only (`_log_detail`) and
stage/summary/warning messages to both outputs (`_log_summary`), same as
`logf`/`logboth`.
"""
_log_detail(::Nothing, msg::String) = (@info msg)
_log_detail(slog::ScriptLog, msg::String) = logf(slog, msg)

_log_summary(::Nothing, msg::String; warn::Bool=false) = warn ? (@warn msg) : (@info msg)
_log_summary(slog::ScriptLog, msg::String; warn::Bool=false) = logboth(slog, msg; warn)
