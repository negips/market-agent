"""
ops_server.jl

A small local server that serves website/ (a drop-in for serve.sh's python server) and lets the pages run a fixed
whitelist of actions through /api. A page can never send a command: it names an action from ACTIONS below, and only that
action's own command runs. Edit ACTIONS to add or remove what the pages may do.

Safety:
  - binds to 127.0.0.1 only (--host to change it, which you should not);
  - a request whose Host is not this server's own localhost:PORT is refused (DNS rebinding);
  - a POST whose Origin is not this server's own origin is refused (another website cannot trigger actions);
  - a job action runs at most once at a time; a second start returns 409.

Usage:
  julia --project=packages/StockSwingPredictor scripts/ops_server.jl [--port 8080] [--open website/dataavailability.html]

API (all JSON):
  GET  /api/actions          every action with its current state
  POST /api/run/NAME         start the job / perform the action NAME
  GET  /api/job/NAME         state of NAME plus the last lines of its log (jobs only)

Job logs go to website/data/ops_logs/NAME.log.
"""

using HTTP, JSON3, Dates

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))
const WEB_ROOT = joinpath(REPO_ROOT, "website")
const LOG_DIR = joinpath(WEB_ROOT, "data", "ops_logs")
const LOG_TAIL_LINES = 40
const DEFAULT_PORT = 8080

"""
    OpsAction

One whitelisted action. `kind = :job` runs `cmd` in the background (one at a time) and logs its output; `kind = :touch` creates
the file `path` (relative to the repo root) and returns at once.
"""
struct OpsAction
    name::String
    label::String
    description::String
    kind::Symbol
    cmd::Vector{String}
    path::String
end

job(name, label, description, cmd) = OpsAction(name, label, description, :job, cmd, "")
touch_file(name, label, description, path) = OpsAction(name, label, description, :touch, String[], path)

const JULIA = Base.julia_cmd().exec[1]

const ACTIONS = OpsAction[
    job("scan_availability", "Rescan data files", "Re-read every OHLCV CSV's first and last row into data/ohlcv_availability.json (about 20 s).",
        [JULIA, "scripts/scan_ohlcv_availability.jl"]),
    touch_file("stop_forecaster_1min", "Stop after this epoch", "Creates website/data/forecaster/STOP: the 1-minute run saves and stops after the current epoch.",
               "website/data/forecaster/STOP"),
    touch_file("stop_forecaster_1min_now", "Stop now", "Creates website/data/forecaster/STOP_NOW: the 1-minute run stops at once (the best checkpoint is already on disk).",
               "website/data/forecaster/STOP_NOW"),
    touch_file("stop_forecaster_15min", "Stop after this epoch", "Creates website/data/forecaster_15min/STOP: the 15-minute run saves and stops after the current epoch.",
               "website/data/forecaster_15min/STOP"),
    touch_file("stop_forecaster_15min_now", "Stop now", "Creates website/data/forecaster_15min/STOP_NOW: the 15-minute run stops at once.",
               "website/data/forecaster_15min/STOP_NOW"),
]

mutable struct JobState
    running::Bool
    started::Union{DateTime, Nothing}
    finished::Union{DateTime, Nothing}
    exit_code::Union{Int, Nothing}
end

const JOBS = Dict(a.name => JobState(false, nothing, nothing, nothing) for a in ACTIONS if a.kind === :job)
const JOBS_LOCK = ReentrantLock()

const MIME = Dict(".html" => "text/html; charset=utf-8", ".js" => "text/javascript; charset=utf-8", ".css" => "text/css; charset=utf-8",
                  ".json" => "application/json", ".jsonl" => "text/plain; charset=utf-8", ".svg" => "image/svg+xml", ".png" => "image/png",
                  ".ico" => "image/x-icon", ".csv" => "text/csv; charset=utf-8", ".toml" => "text/plain; charset=utf-8", ".txt" => "text/plain; charset=utf-8")

json_response(status::Int, obj) = HTTP.Response(status, ["Content-Type" => "application/json", "Cache-Control" => "no-store"], JSON3.write(obj))

"""
    job_log(name) -> Vector{String}

The last `LOG_TAIL_LINES` lines of the job's log file, empty when it has not run.
"""
function job_log(name::String)
    path = joinpath(LOG_DIR, name * ".log")
    isfile(path) || return String[]
    lines = readlines(path)
    return lines[max(1, end - LOG_TAIL_LINES + 1):end]
end

function action_state(a::OpsAction)
    base = Dict{String, Any}("name" => a.name, "label" => a.label, "description" => a.description, "kind" => String(a.kind))
    if a.kind === :job
        s = JOBS[a.name]
        lock(JOBS_LOCK) do
            base["running"] = s.running
            base["started"] = s.started === nothing ? nothing : string(s.started)
            base["finished"] = s.finished === nothing ? nothing : string(s.finished)
            base["exit_code"] = s.exit_code
        end
    else
        base["exists"] = isfile(joinpath(REPO_ROOT, a.path))
    end
    return base
end

"""
    start_job(a) -> Bool

Start the job in the background; `false` when it is already running.
"""
function start_job(a::OpsAction)
    s = JOBS[a.name]
    claimed = lock(JOBS_LOCK) do
        s.running && return false
        s.running = true; s.started = now(); s.finished = nothing; s.exit_code = nothing
        true
    end
    claimed || return false
    mkpath(LOG_DIR)
    logfile = joinpath(LOG_DIR, a.name * ".log")
    @async begin
        code = try
            p = run(pipeline(Cmd(Cmd(a.cmd); dir=REPO_ROOT, ignorestatus=true); stdout=logfile, stderr=logfile); wait=true)
            p.exitcode
        catch err
            open(logfile, "a") do io; println(io, "ops_server: could not run: ", err); end
            -1
        end
        lock(JOBS_LOCK) do
            s.running = false; s.finished = now(); s.exit_code = code
        end
    end
    return true
end

function api(req::HTTP.Request, path::AbstractString, port::Int)
    if req.method == "GET" && path == "/api/actions"
        return json_response(200, [action_state(a) for a in ACTIONS])
    elseif req.method == "GET" && startswith(path, "/api/job/")
        name = path[length("/api/job/") + 1:end]
        i = findfirst(a -> a.name == name && a.kind === :job, ACTIONS)
        i === nothing && return json_response(404, Dict("error" => "no such job: $name"))
        return json_response(200, merge(action_state(ACTIONS[i]), Dict("log" => job_log(name))))
    elseif req.method == "POST" && startswith(path, "/api/run/")
        origin = HTTP.header(req, "Origin")
        origin in ("http://localhost:$port", "http://127.0.0.1:$port") || return json_response(403, Dict("error" => "bad origin"))
        name = path[length("/api/run/") + 1:end]
        i = findfirst(a -> a.name == name, ACTIONS)
        i === nothing && return json_response(404, Dict("error" => "not a whitelisted action: $name"))
        a = ACTIONS[i]
        if a.kind === :job
            start_job(a) || return json_response(409, merge(action_state(a), Dict("error" => "already running")))
        else
            target = joinpath(REPO_ROOT, a.path)
            isdir(dirname(target)) || return json_response(404, Dict("error" => "no such run directory: $(dirname(a.path))"))
            write(target, string(now(), "\n"))
        end
        return json_response(200, action_state(a))
    end
    return json_response(404, Dict("error" => "unknown endpoint"))
end

function static_file(path::AbstractString)
    rel = lstrip(path == "/" ? "/index.html" : path, '/')
    full = abspath(joinpath(WEB_ROOT, rel))
    startswith(full, WEB_ROOT * "/") || return HTTP.Response(403, "forbidden")
    isfile(full) || return HTTP.Response(404, "not found")
    return HTTP.Response(200, ["Content-Type" => get(MIME, lowercase(splitext(full)[2]), "application/octet-stream"), "Cache-Control" => "no-store"], read(full))
end

function make_handler(port::Int)
    own_hosts = ("localhost:$port", "127.0.0.1:$port")
    return function (req::HTTP.Request)
        HTTP.header(req, "Host") in own_hosts || return HTTP.Response(403, "bad host")
        path = String(HTTP.URI(req.target).path)
        try
            startswith(path, "/api/") && return api(req, path, port)
            req.method in ("GET", "HEAD") || return HTTP.Response(405, "method not allowed")
            return static_file(path)
        catch err
            @warn "request failed" path exception = (err, catch_backtrace())
            return json_response(500, Dict("error" => sprint(showerror, err)))
        end
    end
end

function main()
    port = DEFAULT_PORT; host = "127.0.0.1"; open_page = ""
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a in ("-h", "--help"); println(match(r"\"\"\"\n(.*?)\"\"\""s, read(@__FILE__, String)).captures[1]); return
        elseif a == "--port"; port = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--host"; host = ARGS[i+1]; i += 2
        elseif a == "--open"; open_page = ARGS[i+1]; i += 2
        else error("Unknown argument: $a (see --help)") end
    end
    url = "http://localhost:$port/" * replace(open_page, r"^website/" => "")
    println("Serving website/ and /api on http://localhost:$port  (Ctrl+C to stop)")
    println("Whitelisted actions: ", join((a.name for a in ACTIONS), ", "))
    isempty(open_page) || (try run(pipeline(`xdg-open $url`; stderr=devnull)) catch; println("Open $url in your browser") end)
    HTTP.serve(make_handler(port), host, port)
end

main()
