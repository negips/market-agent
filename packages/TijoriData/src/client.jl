"""
HTTP client for the Tijori sidecar and sidecar lifecycle management.

Public API:
  start!(sidecar_dir)  — launch the Node.js sidecar as a subprocess
  stop!()              — terminate the sidecar subprocess
  is_running()         — check whether the sidecar is reachable
  configure!(; port)   — change port without restarting (if sidecar is external)
"""

# ── Configuration ─────────────────────────────────────────────────────────────

const _PORT    = Ref{Int}(3001)
const _PROCESS = Ref{Union{Base.Process, Nothing}}(nothing)
const _BASE    = Ref{String}("http://localhost:3001")

function _update_base!()
    _BASE[] = "http://localhost:$(_PORT[])"
end

"""
    configure!(; port=3001)

Change the port used to reach the sidecar. Call this if you started the
sidecar on a non-default port or if it is running externally (e.g. on another
machine). Does not start or stop any process.

# Example
```julia
TijoriData.configure!(port=3002)
```
"""
function configure!(; port::Int=3001)
    _PORT[] = port
    _update_base!()
    return nothing
end

# ── Sidecar lifecycle ─────────────────────────────────────────────────────────

"""
    start!(sidecar_dir; port=3001, timeout=30)

Launch the Tijori HTTP sidecar as a background subprocess.

`sidecar_dir` must be the directory containing `server_http.js` and the
`tijori-finance-mcp/` clone. The sidecar's `npm install` must have been run
and the Tijori session must have been set up (`node setup.js`) beforehand.

Blocks until the sidecar responds on the health endpoint or `timeout` seconds
elapse (the first call opens a Chromium browser, which takes a few seconds).

# Example
```julia
TijoriData.start!("/path/to/Agents/sidecar")
```
"""
function start!(sidecar_dir::String; port::Int=3001, timeout::Int=30)
    configure!(port=port)

    if is_running()
        @info "Tijori sidecar already reachable on port $port"
        return nothing
    end

    server_script = joinpath(sidecar_dir, "server_http.js")
    isfile(server_script) || error("server_http.js not found at: $server_script")

    env = copy(ENV)
    env["PORT"] = string(port)

    proc = run(
        Cmd(`node server_http.js`; dir=sidecar_dir, env=env),
        devnull, stdout, stderr;
        wait=false
    )
    _PROCESS[] = proc

    # Ensure the sidecar is killed when Julia exits, even without an explicit stop!()
    atexit(() -> process_running(proc) && kill(proc))

    # Poll until the health endpoint responds
    deadline = time() + timeout
    while time() < deadline
        sleep(1)
        is_running() && begin
            @info "Tijori sidecar started on port $port"
            return nothing
        end
    end

    # Timed out — kill the process and surface an error
    kill(proc)
    _PROCESS[] = nothing
    error("Sidecar did not become ready within $(timeout)s. Check that " *
          "tijori-finance-mcp is set up and the Tijori session is valid.")
end

"""
    stop!()

Terminate the sidecar subprocess started by `start!`. No-op if `start!` was
never called or if the process has already exited.
"""
function stop!()
    proc = _PROCESS[]
    if !isnothing(proc) && process_running(proc)
        kill(proc)
        @info "Tijori sidecar stopped"
    end
    _PROCESS[] = nothing
    return nothing
end

"""
    is_running() -> Bool

Return true if the sidecar is reachable on the configured port.
"""
function is_running()::Bool
    try
        r = HTTP.get("$(_BASE[])/health"; readtimeout=3, connect_timeout=2)
        return r.status == 200
    catch
        return false
    end
end

# ── HTTP helpers ──────────────────────────────────────────────────────────────

"""
    _get(path; params...) -> JSON3.Object

GET `_BASE[]/path?params` and return the parsed JSON body.
Raises `TijoriError` if the sidecar returns `{ ok: false }`.
"""
function _get(path::String; params...)::Any
    url = "$(_BASE[])$path"
    if !isempty(params)
        qs = join(["$(k)=$(HTTP.escapeuri(string(v)))" for (k, v) in params], "&")
        url = "$url?$qs"
    end
    resp = HTTP.get(url; readtimeout=60, connect_timeout=5)
    body = JSON3.read(resp.body)
    body.ok || throw(TijoriError(string(body.error)))
    return body.data
end

"""
    _post(path, payload) -> JSON3.Object

POST JSON `payload` to `_BASE[]/path` and return the parsed body.
Raises `TijoriError` if the sidecar returns `{ ok: false }`.
"""
function _post(path::String, payload::Dict)::Any
    url  = "$(_BASE[])$path"
    body = JSON3.write(payload)
    resp = HTTP.post(url, ["Content-Type" => "application/json"], body;
                     readtimeout=120, connect_timeout=5)
    parsed = JSON3.read(resp.body)
    parsed.ok || throw(TijoriError(string(parsed.error)))
    return parsed.data
end

# ── Parsing utilities ─────────────────────────────────────────────────────────

"""
    _parse_number(s) -> Union{Float64, Nothing}

Parse a Tijori number string (e.g. "1,23,456.78", "18.4%") to Float64.
Returns nothing for missing values ("—", "", null).
"""
function _parse_number(s)::Union{Float64, Nothing}
    isnothing(s) && return nothing
    str = string(s)
    str in ("—", "-", "", "null") && return nothing
    cleaned = replace(str, "," => "", "%" => "")
    v = tryparse(Float64, cleaned)
    return v
end

"""
    _financials_to_df(raw) -> DataFrame

Convert the `{ type, headers, rows }` financials response to a DataFrame.
Column `metric` holds the row label; remaining columns are period headers.
Numeric strings are parsed to Float64 (missing where Tijori shows "—").
"""
function _financials_to_df(raw)::DataFrame
    rows = raw.rows
    isempty(rows) && return DataFrame()

    # Collect all period headers (everything after "metric")
    all_keys = unique(vcat(["metric"], [string(k) for row in rows for k in keys(row)]))
    periods  = filter(k -> k != "metric", all_keys)

    df = DataFrame()
    df[!, :metric] = [string(row.metric) for row in rows]
    for p in periods
        df[!, Symbol(p)] = [_parse_number(get(row, Symbol(p), nothing)) for row in rows]
    end
    return df
end

"""
    _shareholding_to_df(raw) -> DataFrame

Convert the shareholding `{ quarters: [...] }` response to a DataFrame.
Each row is a quarter; columns are shareholding categories (Promoter, FII, etc.).
"""
function _shareholding_to_df(raw)::DataFrame
    quarters = raw.quarters
    isempty(quarters) && return DataFrame()

    all_keys = unique(vcat(["period"], [string(k) for q in quarters for k in keys(q)]))
    categories = filter(k -> k != "period", all_keys)

    df = DataFrame()
    df[!, :period] = [string(get(q, :period, "")) for q in quarters]
    for cat in categories
        col = Symbol(cat)
        df[!, col] = [_parse_number(get(q, Symbol(cat), nothing)) for q in quarters]
    end
    return df
end
