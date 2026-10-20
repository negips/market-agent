"""
ops_server.jl

A small local server that serves website/ (a drop-in for serve.sh's python server) and lets the pages run a fixed
whitelist of actions through /api. A page can never send a command: it names an action from ACTIONS below, and only that
action's own command runs. Edit ACTIONS to add or remove what the pages may do.

Safety:
  - binds to every interface (reachable from other machines) unless `--local` is given, which binds 127.0.0.1 only; `--host ADDR`
    binds one address. There is no login: anyone who can reach the port can read everything under website/data. Pages opened
    as localhost can use the actions, but a request addressed to any other name (the machine's IP or hostname) gets every POST
    action refused unless `--allow-remote-actions` is also given. That is a guard against accidental clicks, not a boundary: a
    client can send any Host header, so use this on a network you trust (or `--local`);
  - with `--local`, a request whose Host is not this server's own localhost:PORT is refused (DNS rebinding); otherwise any Host
    on this port is accepted;
  - a POST whose Origin is not the page's own origin (http://HOST) is refused (another website cannot trigger actions);
  - a job action runs at most once at a time; a second start returns 409.

Usage:
  julia --project=packages/StockSwingPredictor scripts/tools/ops_server.jl [--port 8080] [--local | --host ADDR] [--allow-remote-actions] [--open website/dataavailability.html]

API (all JSON):
  GET  /api/actions          every action with its current state
  POST /api/run/NAME         start the job / perform the action NAME
  GET  /api/job/NAME         state of NAME plus the last lines of its log (jobs only)
  GET  /api/series?granularity=G&symbol=S   one company's return series from website/data/forecaster_cache_G.bson (read-only)
  GET  /api/symbols?granularity=G           the cache's companies, date range and bars per day (read-only)
  GET  /api/legendre?granularity=G&symbol=S&n=N[&kind=price|logprice][&from=D&to=D]
                                            the symbol's price projected on N Legendre polynomials (legendre_projection.jl; read-only)

Job logs go to website/data/ops_logs/NAME.log.
"""

using HTTP, JSON3, Dates, BSON, Sockets

include(joinpath(@__DIR__, "legendre_projection.jl"))

const REPO_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const WEB_ROOT = joinpath(REPO_ROOT, "website")
const LOG_DIR = joinpath(WEB_ROOT, "data", "ops_logs")
const LOG_TAIL_LINES = 40
const DEFAULT_PORT = 8080
const LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
const EXPOSED = Ref(false)
const ALLOW_REMOTE_ACTIONS = Ref(false)

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
        [JULIA, "scripts/ohlcv/scan_ohlcv_availability.jl"]),
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

const CACHE_GRANULARITIES = ("1min", "5min", "15min", "hourly", "daily")
const SERIES_DECIMALS = 4
const PRICE_DECIMALS = 2

"""One forecaster cache read from disk: its file's modification time and the arrays the series endpoint serves."""
struct LoadedCache
    mtime::Float64
    symbols::Vector{String}
    dates::Vector{Date}
    logret::Matrix{Float32}
    traded::Matrix{Bool}
    price::Union{Nothing,Matrix{Float32}}
end

const CACHE_LOCK = ReentrantLock()
const LOADED = Ref{Union{Nothing,Tuple{String,LoadedCache}}}(nothing)

"""
    loaded_cache(granularity) -> LoadedCache

The arrays of `forecaster_cache_{granularity}.bson`, read as plain BSON (no TradingGame needed). One cache is kept in memory; it is
read again when another bar size is asked for or the file has changed (the pipeline rewrites it every round).
"""
function loaded_cache(granularity::AbstractString)
    path = joinpath(WEB_ROOT, "data", "forecaster_cache_$(granularity).bson")
    isfile(path) || error("no forecaster cache for $granularity: $path")
    lock(CACHE_LOCK) do
        held = LOADED[]
        if held !== nothing && held[1] == granularity && held[2].mtime == mtime(path)
            return held[2]
        end
        d = BSON.load(path)
        c = LoadedCache(mtime(path), Vector{String}(d[:symbols]), Vector{Date}(d[:dates]), Matrix{Float32}(d[:logret]), Matrix{Bool}(d[:traded]),
                        haskey(d, :price) && !isempty(d[:price]) ? Matrix{Float32}(d[:price]) : nothing)
        LOADED[] = (String(granularity), c)
        return c
    end
end

"""A symbol's `logret` (percent), `traded` and, when the cache holds them, `price` (close, as in the CSV; `null` before the first price) columns from the cache of `granularity`, with the dates and bars per day."""
function series_payload(granularity::AbstractString, symbol::AbstractString)
    granularity in CACHE_GRANULARITIES || return nothing, "unknown granularity: $granularity"
    c = loaded_cache(granularity)
    k = findfirst(==(symbol), c.symbols)
    k === nothing && return nothing, "$symbol is not in the $granularity cache"
    payload = Dict{String,Any}("granularity" => granularity, "symbol" => symbol, "bars_per_day" => size(c.logret, 1) ÷ length(c.dates),
                "dates" => string.(c.dates), "logret" => round.(c.logret[:, k]; digits=SERIES_DECIMALS), "traded" => Int.(c.traded[:, k]))
    c.price === nothing || (payload["price"] = [isfinite(p) ? round(Float64(p); digits=PRICE_DECIMALS) : nothing for p in c.price[:, k]])
    return payload, ""
end

const BAR_MINUTES = Dict(375 => 1, 75 => 5, 25 => 15, 7 => 60, 1 => 375)
const SESSION_START_MINUTE = 9 * 60 + 15

"""The companies, first and last date and bars per day of the cache of `granularity`."""
function symbols_payload(granularity::AbstractString)
    granularity in CACHE_GRANULARITIES || return nothing, "unknown granularity: $granularity"
    c = loaded_cache(granularity)
    return Dict{String,Any}("granularity" => granularity, "symbols" => c.symbols, "first_date" => string(first(c.dates)), "last_date" => string(last(c.dates)),
                            "bars_per_day" => size(c.logret, 1) ÷ length(c.dates), "has_price" => c.price !== nothing), ""
end

"""Label of bar `i` (0-based) of the cache axis: the date, plus the HH:MM start of the bar unless there is one bar a day."""
function bar_label(dates::Vector{Date}, bars_per_day::Int, i::Int)
    day = string(dates[i ÷ bars_per_day + 1])
    bars_per_day == 1 && return day
    m = SESSION_START_MINUTE + (i % bars_per_day) * get(BAR_MINUTES, bars_per_day, 1)
    return string(day, " ", lpad(m ÷ 60, 2, '0'), ":", lpad(m % 60, 2, '0'))
end

"""
`symbol`'s price in `from…to` (ISO dates, either may be empty) projected on `n` Legendre polynomials ([`legendre_report`](@ref)),
with a label for every returned point.
"""
function legendre_payload(granularity::AbstractString, symbol::AbstractString, n_text::AbstractString, kind::AbstractString, from::AbstractString, to::AbstractString)
    granularity in CACHE_GRANULARITIES || return nothing, "unknown granularity: $granularity"
    n = tryparse(Int, n_text)
    n === nothing && return nothing, "n must be an integer"
    c = loaded_cache(granularity)
    k = findfirst(==(symbol), c.symbols)
    k === nothing && return nothing, "$symbol is not in the $granularity cache"
    c.price === nothing && return nothing, "the $granularity cache holds no prices: rebuild it (build_forecaster_cache.jl --granularity $granularity)"
    bpd = size(c.logret, 1) ÷ length(c.dates)
    d_from = isempty(from) ? first(c.dates) : tryparse(Date, from)
    d_to = isempty(to) ? last(c.dates) : tryparse(Date, to)
    (d_from === nothing || d_to === nothing) && return nothing, "from and to must be ISO dates"
    days = findall(d -> d_from <= d <= d_to, c.dates)
    isempty(days) && return nothing, "no bars in this range"
    lo, hi = (first(days) - 1) * bpd + 1, last(days) * bpd
    rep = try
        legendre_report(Float64.(c.price[lo:hi, k]), n; kind=kind)
    catch e
        e isa LegendreError || rethrow()
        return nothing, e.msg
    end
    rep["t"] = [bar_label(c.dates, bpd, lo - 1 + i) for i in rep["index"]]
    rep["granularity"] = granularity; rep["symbol"] = symbol
    rep["actual"] = round.(rep["actual"]; digits=PRICE_DECIMALS); rep["fit"] = round.(rep["fit"]; digits=PRICE_DECIMALS)
    return rep, ""
end

function api(req::HTTP.Request, path::AbstractString, port::Int)
    if req.method == "GET" && path == "/api/actions"
        return json_response(200, [action_state(a) for a in ACTIONS])
    elseif req.method == "GET" && path == "/api/series"
        q = HTTP.queryparams(HTTP.URI(req.target))
        payload, why = series_payload(get(q, "granularity", ""), get(q, "symbol", ""))
        return payload === nothing ? json_response(404, Dict("error" => why)) : json_response(200, payload)
    elseif req.method == "GET" && path == "/api/symbols"
        q = HTTP.queryparams(HTTP.URI(req.target))
        payload, why = symbols_payload(get(q, "granularity", ""))
        return payload === nothing ? json_response(404, Dict("error" => why)) : json_response(200, payload)
    elseif req.method == "GET" && path == "/api/legendre"
        q = HTTP.queryparams(HTTP.URI(req.target))
        payload, why = legendre_payload(get(q, "granularity", ""), get(q, "symbol", ""), get(q, "n", ""), get(q, "kind", "price"), get(q, "from", ""), get(q, "to", ""))
        return payload === nothing ? json_response(404, Dict("error" => why)) : json_response(200, payload)
    elseif req.method == "GET" && startswith(path, "/api/job/")
        name = path[length("/api/job/") + 1:end]
        i = findfirst(a -> a.name == name && a.kind === :job, ACTIONS)
        i === nothing && return json_response(404, Dict("error" => "no such job: $name"))
        return json_response(200, merge(action_state(ACTIONS[i]), Dict("log" => job_log(name))))
    elseif req.method == "POST" && startswith(path, "/api/run/")
        addressed_locally = HTTP.header(req, "Host") in ("localhost:$port", "127.0.0.1:$port")
        addressed_locally || ALLOW_REMOTE_ACTIONS[] || return json_response(403, Dict("error" => "actions only work from a page opened as localhost:$port (or start the server with --allow-remote-actions)"))
        HTTP.header(req, "Origin") == "http://" * HTTP.header(req, "Host") || return json_response(403, Dict("error" => "bad origin"))
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
        host_header = HTTP.header(req, "Host")
        (EXPOSED[] ? endswith(host_header, ":$port") : host_header in own_hosts) || return HTTP.Response(403, "bad host")
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
    port = DEFAULT_PORT; host = "0.0.0.0"; open_page = ""
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a in ("-h", "--help"); println(match(r"\"\"\"\n(.*?)\"\"\""s, read(@__FILE__, String)).captures[1]); return
        elseif a == "--port"; port = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--local"; host = "127.0.0.1"; i += 1
        elseif a == "--host"; host = ARGS[i+1]; i += 2
        elseif a == "--allow-remote-actions"; ALLOW_REMOTE_ACTIONS[] = true; i += 1
        elseif a == "--open"; open_page = ARGS[i+1]; i += 2
        else error("Unknown argument: $a (see --help)") end
    end
    EXPOSED[] = !(host in LOOPBACK_HOSTS)
    url = "http://localhost:$port/" * replace(open_page, r"^website/" => "")
    println("Serving website/ and /api on http://localhost:$port  (Ctrl+C to stop)")
    if EXPOSED[]
        println("Reachable from other machines: ", join(("http://$ip:$port/" for ip in Sockets.getipaddrs() if ip isa Sockets.IPv4), "  "))
        println("Actions (Stop, Rescan): ", ALLOW_REMOTE_ACTIONS[] ? "enabled for every client" : "only from a page opened as localhost:$port", ". Use --local to serve this machine only.")
    end
    println("Whitelisted actions: ", join((a.name for a in ACTIONS), ", "))
    isempty(open_page) || (try run(pipeline(`xdg-open $url`; stderr=devnull)) catch; println("Open $url in your browser") end)
    HTTP.serve(make_handler(port), host, port)
end

main()
