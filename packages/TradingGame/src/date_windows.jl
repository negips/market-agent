"""
Train/val date-window resolution — pulled out as a package function (rather
than living inside `scripts/train_trading_policy.jl`) so any other script
that needs to know "what dates will this training run actually use" calls
the exact same code, not a second hand-copied implementation that can
silently drift from the real one. `scripts/prepare_training_data.jl` is the
other caller: it resolves these same windows to decide how to scope the
news backfill and 1-minute snapshot fetch, so what gets prepared and what
training actually trains on are guaranteed to match.
"""

using Dates, JSON3

"""
Resolve the train/val date windows for a run — see `--val-window` in
`train_trading_policy.jl`'s module docstring for the two modes:

- `"trailing"`: val is the last `val_days` of `[cache_start, cache_end]`,
  train is everything before it.
- `"same"`: train and val both span the full `[cache_start, cache_end]`
  range — only sound once train/val use different companies (see
  `UniverseStrategy`), since there's no date-based leakage guard left
  without it.

Explicit `explicit_train_start`/`explicit_train_end`/`explicit_val_start`/
`explicit_val_end` (any subset, `nothing` for "let `mode` decide") override
whichever bound `mode` would otherwise have picked for that specific bound.

# Returns
`(train_start, train_end, val_start, val_end)`, all `Date`. Throws if either
resulting window is empty or inverted.
"""
function resolve_date_windows(cache_start::Date, cache_end::Date, mode::String, val_days::Int,
                               explicit_train_start::Union{Date, Nothing}, explicit_train_end::Union{Date, Nothing},
                               explicit_val_start::Union{Date, Nothing}, explicit_val_end::Union{Date, Nothing})
    if mode == "trailing"
        val_start   = cache_end - Day(val_days)
        train_start = cache_start
        train_end   = val_start - Day(1)
        val_end     = cache_end
    elseif mode == "same"
        train_start = cache_start
        train_end   = cache_end
        val_start   = cache_start
        val_end     = cache_end
    else
        error("Unknown --val-window '$mode'. Expected: trailing, same")
    end

    train_start = something(explicit_train_start, train_start)
    train_end   = something(explicit_train_end, train_end)
    val_start   = something(explicit_val_start, val_start)
    val_end     = something(explicit_val_end, val_end)

    train_start < train_end || error(
        "Train window is empty or inverted: $train_start .. $train_end " *
        "(cache covers $cache_start .. $cache_end) — adjust --val-days/--val-window/--train-* flags")
    val_start < val_end || error(
        "Val window is empty or inverted: $val_start .. $val_end " *
        "(cache covers $cache_start .. $cache_end) — adjust --val-days/--val-window/--val-* flags")
    return train_start, train_end, val_start, val_end
end

"""
Persist a resolved train/val date window to `path` (JSON) — written by
`prepare_training_data.jl` right after it calls `resolve_date_windows`, so a
LATER, separately-invoked `train_trading_policy.jl` with no date flags of
its own can default to the exact same window instead of re-deriving one
from whatever the cache happens to cover (which, now that different
granularities can have different real floors, is not necessarily what you
actually meant). `val_window`/`val_days` are stored for information only —
what a reader actually restores is the four resolved dates themselves, not
the mode that produced them.
"""
function save_date_window(path::String, train_start::Date, train_end::Date, val_start::Date, val_end::Date;
                           val_window::String, val_days::Int)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON3.pretty(io, Dict(
            "train_start" => string(train_start),
            "train_end"   => string(train_end),
            "val_start"   => string(val_start),
            "val_end"     => string(val_end),
            "val_window"  => val_window,
            "val_days"    => val_days,
            "resolved_at" => string(now()),
        ))
    end
end

"""
Load a previously-saved date window (see `save_date_window`), as
`(train_start, train_end, val_start, val_end)::NTuple{4, Date}`. Returns
`nothing` if `path` doesn't exist — e.g. `prepare_training_data.jl` has
never been run — in which case the caller just falls back to its own
default resolution.
"""
function load_date_window(path::String)::Union{NTuple{4, Date}, Nothing}
    isfile(path) || return nothing
    saved = JSON3.read(read(path, String))
    return (Date(String(saved.train_start)), Date(String(saved.train_end)),
            Date(String(saved.val_start)),   Date(String(saved.val_end)))
end
