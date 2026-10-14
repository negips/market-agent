"""
run_config.jl

Shared loader for the one-file-per-run TOML configuration read by
`build_cache.jl`, `build_market_universe_snapshot.jl`, `build_dataset.jl`,
`prepare_training_data.jl` and `train_trading_policy.jl` (see `configs/`).

Each script `include`s this file, calls `RunConfigs.take_config_arg!(ARGS)`
to strip `--config PATH` before its own parser sees `ARGS`, then
`RunConfigs.load(path)` and `RunConfigs.section_opts(cfg, "train")` for the
sections it owns. Precedence in every script: built-in default < config file
< explicit command-line flag.

Only the standard library is used (`TOML`, `Dates`), so the file loads under
any of the packages' projects.
"""
module RunConfigs

using TOML, Dates

export ConfigError

"""Raised for a config file that cannot be read, has an unknown section or
key, or holds a value of the wrong type."""
struct ConfigError <: Exception
    msg::String
end
Base.showerror(io::IO, e::ConfigError) = print(io, "ConfigError: ", e.msg)

"""`section => (config key => (script option name, value kind))`. The kinds are
`:int`, `:float` (an integer is accepted), `:bool`, `:string`, `:date`,
`:strings` (array of strings) and `:version` (`3`, `"3"` or `"v3"`)."""
const SCHEMA = Dict{String, Dict{String, Tuple{String, Symbol}}}(
    "cache" => Dict(
        "exchange"    => ("exchange",    :string),
        "granularity" => ("granularity", :string),
        "history"     => ("history",     :string),
        "min_mcap"    => ("min_mcap",    :float),
        "top_n"       => ("top_n",       :int),
        "out"         => ("out",         :string)),
    "universe" => Dict(
        "strategy" => ("strategy", :string),
        "n"        => ("n",        :int),
        "n_train"  => ("n_train",  :int),
        "n_val"    => ("n_val",    :int),
        "disjoint" => ("disjoint", :bool),
        "min_mcap" => ("min_mcap", :float),
        "seed"     => ("seed",     :int),
        "bands"    => ("bands",    :strings)),
    "dataset" => Dict(
        "pred_hours" => ("pred_hours", :int)),
    "dates" => Dict(
        "val_window"  => ("val_window",  :string),
        "val_days"    => ("val_days",    :int),
        "train_start" => ("train_start", :date),
        "train_end"   => ("train_end",   :date),
        "val_start"   => ("val_start",   :date),
        "val_end"     => ("val_end",     :date)),
    "prepare" => Dict(
        "rebuild_universe" => ("rebuild_universe", :bool),
        "skip_universe"    => ("skip_universe",    :bool),
        "skip_news"        => ("skip_news",        :bool),
        "skip_1min"        => ("skip_1min",        :bool)),
    "forecaster_cache" => Dict(
        "granularity"      => ("granularity",      :string),
        "exchange"         => ("exchange",         :string),
        "out"              => ("out",              :string),
        "from"             => ("from",             :date),
        "to"               => ("to",               :date),
        "min_day_fraction" => ("min_day_fraction", :float)),
    "forecaster" => Dict(
        "window"          => ("window",          :int),
        "d"               => ("d",               :int),
        "heads"           => ("heads",           :int),
        "blocks"          => ("blocks",          :int),
        "dropout"         => ("dropout",         :float),
        "epochs"          => ("epochs",          :int),
        "steps_per_epoch" => ("steps_per_epoch", :int),
        "batch"           => ("batch",           :int),
        "lr"              => ("lr",              :float),
        "weight_decay"    => ("weight_decay",    :float),
        "warmup_steps"    => ("warmup_steps",    :int),
        "val_windows"     => ("val_windows",     :int),
        "device"          => ("device",          :string),
        "dtype"           => ("dtype",           :string),
        "loss"            => ("loss",            :string),
        "nu"              => ("nu",              :float),
        "nll_beta"        => ("nll_beta",        :float),
        "seed"            => ("seed",            :int),
        "init_from"       => ("init_from",       :string),
        "resume"          => ("resume",          :bool)),
    "game" => Dict(
        "version"            => ("game_version",       :version),
        "initial_cash"       => ("initial_cash",       :float),
        "use_macro"          => ("use_macro",          :bool),
        "use_news"           => ("use_news",           :bool),
        "cash_penalty"       => ("cash_penalty",       :float),
        "hold_penalty"       => ("hold_penalty",       :float),
        "illegal_penalty"    => ("illegal_penalty",    :float),
        "history_encoder"    => ("history_encoder",    :string),
        "reward_mode"        => ("reward_mode",        :string),
        "reward_window_days" => ("reward_window_days", :int)),
    "train" => Dict(
        "iterations"      => ("iterations",      :int),
        "eval_every"      => ("eval_every",      :int),
        "lr"              => ("lr",              :float),
        "entropy"         => ("entropy_coef",    :float),
        "device"          => ("device",          :string),
        "minibatch"       => ("minibatch",       :int),
        "seed"            => ("seed",            :int),
        "val_steps_every" => ("val_steps_every", :int),
        "live"            => ("live",            :bool),
        "init_from"       => ("init_from",       :string),
        "resume"          => ("resume",          :bool),
        "news_db"         => ("news_db",         :string)),
)

"""Edit distance between two strings, for the did-you-mean hint."""
function _distance(a::AbstractString, b::AbstractString)::Int
    a, b = collect(a), collect(b)
    prev = collect(0:length(b))
    for i in 1:length(a)
        cur = [i]
        for j in 1:length(b)
            push!(cur, min(prev[j+1] + 1, cur[j] + 1, prev[j] + (a[i] == b[j] ? 0 : 1)))
        end
        prev = cur
    end
    return prev[end]
end

function _suggest(name::String, candidates)::String
    isempty(candidates) && return ""
    best = argmin(c -> _distance(name, c), collect(candidates))
    return _distance(name, best) <= max(2, length(name) ÷ 3) ? " — did you mean '$best'?" : ""
end

function _coerce(where_::String, v, kind::Symbol)
    bad(expected) = throw(ConfigError("$where_ must be $expected, got $(repr(v)) ($(typeof(v)))"))
    if kind === :int
        v isa Integer && !(v isa Bool) || bad("an integer")
        return Int(v)
    elseif kind === :float
        v isa Real && !(v isa Bool) || bad("a number")
        return Float64(v)
    elseif kind === :bool
        v isa Bool || bad("true or false")
        return v
    elseif kind === :string
        v isa AbstractString || bad("a string")
        return String(v)
    elseif kind === :date
        v isa Date || bad("a date written without quotes, e.g. 2022-01-03")
        return v
    elseif kind === :strings
        v isa AbstractVector && all(x -> x isa AbstractString, v) || bad("an array of strings")
        return String.(v)
    elseif kind === :version
        s = lowercase(strip(string(v)))
        s in ("1", "v1") && return 1
        s in ("2", "v2") && return 2
        s in ("3", "v3") && return 3
        bad("1, 2 or 3 (or \"v1\"…\"v3\")")
    end
    error("unknown kind $kind")
end

"""
    take_config_arg!(args) -> Union{String, Nothing}

Remove `--config PATH` from `args` (in place) and return the absolute path, or
`nothing` when it was not given.
"""
function take_config_arg!(args::Vector{String})::Union{String, Nothing}
    i = findfirst(==("--config"), args)
    i === nothing && return nothing
    i < length(args) || throw(ConfigError("--config needs a path"))
    path = abspath(args[i + 1])
    deleteat!(args, i:i + 1)
    return path
end

"""
    take_flag!(args, flag) -> Bool

Remove a bare `flag` (e.g. `--print-config`) from `args` and report whether it was there.
"""
function take_flag!(args::Vector{String}, flag::String)::Bool
    i = findfirst(==(flag), args)
    i === nothing && return false
    deleteat!(args, i)
    return true
end

"""
    load(path) -> Dict{String,Any}

Parse the TOML at `path` and check every section, key and value type against
`SCHEMA`. A typo is an error with a did-you-mean hint, not a silent no-op.
"""
function load(path::String)::Dict{String, Any}
    isfile(path) || throw(ConfigError("config file not found: $path"))
    cfg = try
        TOML.parsefile(path)
    catch e
        throw(ConfigError("$path is not valid TOML: $(sprint(showerror, e))"))
    end
    for (sec, body) in cfg
        haskey(SCHEMA, sec) || throw(ConfigError(
            "unknown section [$sec]$(_suggest(sec, keys(SCHEMA))); known sections: $(join(sort(collect(keys(SCHEMA))), ", "))"))
        body isa AbstractDict || throw(ConfigError("[$sec] must be a table"))
        for (key, v) in body
            haskey(SCHEMA[sec], key) || throw(ConfigError(
                "unknown key '$key' in [$sec]$(_suggest(key, keys(SCHEMA[sec]))); known keys: $(join(sort(collect(keys(SCHEMA[sec]))), ", "))"))
            _coerce("[$sec] $key", v, SCHEMA[sec][key][2])
        end
    end
    return cfg
end

"""
    load_or_exit(path) -> Dict{String,Any}

`load`, but a `ConfigError` prints its message alone and exits with status 2
instead of a stack trace — what the scripts call.
"""
function load_or_exit(path::String)::Dict{String, Any}
    try
        return load(path)
    catch e
        e isa ConfigError || rethrow()
        println(stderr, "ERROR: ", e.msg)
        exit(2)
    end
end

"""
    section_opts(cfg, section) -> Dict{String,Any}

The values `cfg` gives for `section`, keyed by each script's own option name
and converted to the right type. Empty when `cfg` is `nothing` or the section
is absent.
"""
function section_opts(cfg::Union{Nothing, AbstractDict}, section::String)::Dict{String, Any}
    out = Dict{String, Any}()
    (cfg === nothing || !haskey(cfg, section)) && return out
    for (key, v) in cfg[section]
        opt, kind = SCHEMA[section][key]
        out[opt] = _coerce("[$section] $key", v, kind)
    end
    return out
end

"""`LO:HI:QUOTA` (HI may be `inf`) → `(lo, hi) => quota`, for `BucketedRandom.quotas`."""
function parse_band(spec::AbstractString)
    parts = split(spec, ":")
    length(parts) == 3 || throw(ConfigError("a band must be LO:HI:QUOTA, got '$spec'"))
    lo_s, hi_s, q_s = parts
    hi = lowercase(hi_s) in ("inf", "infinity") ? Inf : parse(Float64, hi_s)
    return (parse(Float64, lo_s), hi) => parse(Int, q_s)
end

"""
    universe_strategy(u) -> UniverseStrategy

Build the `TradingGame` universe strategy from the options `u` (`strategy`,
`n`, `n_train`, `n_val`, `disjoint`, `seed`, `bands`). Needs `TradingGame`
loaded in `Main`, which both callers have.
"""
function universe_strategy(u::AbstractDict)
    name = u["strategy"]
    seed = get(u, "seed", nothing)
    if name == "shared-topcap"
        return Main.SharedTopMarketCap(n=u["n"])
    elseif name == "disjoint-topcap"
        return Main.DisjointTopMarketCap(n_train=u["n_train"], n_val=u["n_val"], seed=seed)
    elseif name == "random"
        return Main.RandomUniverse(n=u["n"], disjoint=u["disjoint"], seed=seed)
    elseif name == "random-bucketed"
        isempty(u["bands"]) && throw(ConfigError("strategy random-bucketed needs bands, e.g. bands = [\"0:5000:15\", \"5000:inf:15\"]"))
        return Main.BucketedRandom(quotas=[parse_band(b) for b in u["bands"]], disjoint=u["disjoint"], seed=seed)
    end
    throw(ConfigError("unknown universe strategy '$name'. Expected: shared-topcap, disjoint-topcap, random, random-bucketed"))
end

"""Print the fully resolved options (config file, then flags, then defaults) and nothing else."""
function print_resolved(io::IO, title::String, config_path, opts::AbstractDict)
    println(io, "Resolved settings for ", title)
    println(io, "  config file: ", config_path === nothing ? "(none — defaults and flags only)" : config_path)
    for k in sort!(collect(keys(opts)))
        println(io, "  ", rpad(k, 22), " ", repr(opts[k]))
    end
end

end # module
