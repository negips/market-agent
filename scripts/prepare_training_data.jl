"""
prepare_training_data.jl

Single entry point for everything `train_trading_policy.jl` needs beyond
the inference cache itself: the candidate universe, the classified news
signals, and the 1-minute instant-price snapshots around those signals —
driven by ONE date-window resolution instead of the train/val dates being
separately re-derived (and risking drift) across three independent script
invocations.

The actual date math is `TradingGame.resolve_date_windows` — the exact same
function `train_trading_policy.jl` calls to decide its own train/val
windows (moved into the package for this reason, not duplicated here). So
whatever this script fetches for is, by construction, exactly what the
training run driven by the same `--val-window`/`--val-days`/`--train-*`/
`--val-*` flags will actually use — not a second implementation that could
quietly fall out of sync with the first.

Stages, run in order (each is a thin wrapper around an already-existing,
independently-tested tool — this script reuses their logic rather than
reimplementing it):

1. **Universe** (`TradingGame.eligible_candidates`/`build_universes`/
   `save_universe_snapshot` — the same package functions
   `build_market_universe_snapshot.jl` itself calls, run in-process here
   rather than as a subprocess since this script already has the
   TradingGame/Flux/CUDA stack loaded). Skipped if `website/data/
   trading_game/universe_latest.json` already exists, unless
   `--rebuild-universe` is given — an existing, possibly deliberately-built
   universe (a specific `--strategy`, a specific `--seed`) is never silently
   clobbered. Only `SharedTopMarketCap` is available here (`--n`); if you
   want `disjoint-topcap`/`random`/`random-bucketed`, run
   `build_market_universe_snapshot.jl` yourself first, then this script's
   `--skip-universe` picks up that file as-is.

2. **News backfill** (`scripts/backfill_news_signals.jl`, run as a
   subprocess under `--project=packages/NewsMonitor` — TradingGame
   deliberately has no package dependency on NewsMonitor, see CLAUDE.md's
   dependency table, so this can't be an in-process call). Scoped to
   `min(train_start, val_start) … max(train_end, val_end)` — one combined
   pass, since news classification is symbol-scoped only (no role split
   needed the way the 1-minute fetch has — see stage 3).

3. **1-minute snapshots** (`scripts/fetch_news_snapshot_ohlcv.jl`, run as
   two separate subprocesses — `--role train --from <train_start> --to
   <train_end>` and `--role val --from <val_start> --to <val_end>` — see
   that script's own docstring for why role-splitting, not symbol-
   narrowing, is the right lever here).

Each stage is a real subprocess (except stage 1): expect the TradingGame/
Flux/CUDA load time to repeat for stage 3's two calls (a minute or so each)
on top of this script's own startup — small next to the hours stage 2/3
themselves can take, not worth avoiding by refactoring those scripts into
package functions too.

`--skip-universe`/`--skip-news`/`--skip-1min` skip individual stages (e.g.
news already backfilled, only want a fresh 1-minute pass). `--dry-run`
prints the resolved windows and the exact commands this would run, with no
side effects at all — including no universe build, no Ollama calls, no
Kite calls.

Usage:
  julia --project=packages/TradingGame scripts/prepare_training_data.jl
  julia --project=packages/TradingGame scripts/prepare_training_data.jl --dry-run
  julia --project=packages/TradingGame scripts/prepare_training_data.jl --val-days 120
  julia --project=packages/TradingGame scripts/prepare_training_data.jl --val-window same
  julia --project=packages/TradingGame scripts/prepare_training_data.jl --train-start 2021-09-09 --val-start 2026-05-12
  julia --project=packages/TradingGame scripts/prepare_training_data.jl --skip-universe --skip-news
  julia --project=packages/TradingGame scripts/prepare_training_data.jl --n 100 --rebuild-universe

The resolved window is also saved to `website/data/trading_game/
date_window.json`. `train_trading_policy.jl` reads this back automatically
as a fallback default for any of `--train-start`/`--train-end`/`--val-
start`/`--val-end` NOT given explicitly on its own command line (an
explicit flag, or a `--resume`d run's saved `run_config.json`, both still
take priority) — so running this script once with your intended
`--val-window`/`--val-days`/`--train-*`/`--val-*` and then
`train_trading_policy.jl` with NO date flags at all reproduces the exact
same window, without having to repeat it. Passing matching flags to both
explicitly still works exactly as before, if you prefer to be explicit
every time.
"""

using StockSwingPredictor, TradingGame, Dates

const REPO_ROOT      = joinpath(@__DIR__, "..")
const CACHE_FILE     = normpath(joinpath(REPO_ROOT, "website", "data", "inference_cache.bson"))
const DATA_DIR       = normpath(joinpath(REPO_ROOT, "website", "data"))
const UNIVERSE_FILE  = normpath(joinpath(REPO_ROOT, "website", "data", "trading_game", "universe_latest.json"))
const DATE_WINDOW_FILE = normpath(joinpath(REPO_ROOT, "website", "data", "trading_game", "date_window.json"))
const SCRIPTS_DIR    = @__DIR__

function parse_args()
    args = Dict{String, Any}(
        "val_window"       => "trailing",
        "val_days"         => 60,
        "train_start"      => nothing,
        "train_end"        => nothing,
        "val_start"        => nothing,
        "val_end"          => nothing,
        "n"                => N_CANDIDATE_STOCKS,
        "rebuild_universe" => false,
        "skip_universe"    => false,
        "skip_news"        => false,
        "skip_1min"        => false,
        "dry_run"          => false,
    )
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a in ("-h", "--help")
            println("""
prepare_training_data.jl — universe + news backfill + 1-minute snapshots,
one consistent set of train/val dates shared across all three.

Options (date-window resolution — identical to train_trading_policy.jl):
  --val-window MODE     trailing (default) or same
  --val-days N          Held-out tail length in calendar days (default: 60)
  --train-start DATE    Explicit yyyy-mm-dd override, beats --val-window for
  --train-end DATE        this specific bound
  --val-start DATE
  --val-end DATE

Universe (stage 1):
  --n N                 SharedTopMarketCap size if a universe needs building
                        (default: $(args["n"]))
  --rebuild-universe    Rebuild even if universe_latest.json already exists

Stage control:
  --skip-universe       Use the existing universe_latest.json as-is
  --skip-news           Skip the news backfill stage
  --skip-1min           Skip the 1-minute snapshot stage
  --dry-run             Print resolved windows + planned commands, do nothing
  -h, --help            Show this message
""")
            exit(0)
        elseif a == "--val-window";       args["val_window"] = ARGS[i+1]; i += 2
        elseif a == "--val-days";         args["val_days"] = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--train-start";      args["train_start"] = Date(ARGS[i+1]); i += 2
        elseif a == "--train-end";        args["train_end"] = Date(ARGS[i+1]); i += 2
        elseif a == "--val-start";        args["val_start"] = Date(ARGS[i+1]); i += 2
        elseif a == "--val-end";          args["val_end"] = Date(ARGS[i+1]); i += 2
        elseif a == "--n";                args["n"] = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--rebuild-universe"; args["rebuild_universe"] = true; i += 1
        elseif a == "--skip-universe";    args["skip_universe"] = true; i += 1
        elseif a == "--skip-news";        args["skip_news"] = true; i += 1
        elseif a == "--skip-1min";        args["skip_1min"] = true; i += 1
        elseif a == "--dry-run";          args["dry_run"] = true; i += 1
        else
            @warn "Unknown argument: $a"; i += 1
        end
    end
    args["val_window"] in ("trailing", "same") ||
        error("Unknown --val-window '$(args["val_window"])'. Expected: trailing, same")
    return args
end

function main()
    args = parse_args()

    isfile(CACHE_FILE) || error(
        "Not found: $CACHE_FILE\nRun: julia --project=packages/StockSwingPredictor scripts/build_cache.jl")

    @info "Loading inference cache…"
    cache = load_inference_cache(CACHE_FILE)
    @info "  $(uppercase(cache.exchange)) cache, $(cache.bar_minutes)-minute bars"
    if cache.exchange != "nse"
        # The news pipeline is NSE-only: announcements come from NSE and the
        # 1-minute snapshots from ohlcv/nse/1min, keyed by NSE symbols.
        (args["skip_news"] && args["skip_1min"]) ||
            @warn "News backfill and 1-minute snapshots are NSE-only; skipping both for a $(uppercase(cache.exchange)) cache"
        args["skip_news"] = true
        args["skip_1min"] = true
    end
    # The intraday axis, not daily — episodes step through intraday bars, and
    # their real Kite floor is shallower than daily's (daily can go back to
    # 2010; hourly to ~2015, 15-minute to 2019 on BSE). Resolving against
    # cache.dates (daily) would default train_start to a date no intraday bar
    # can satisfy.
    cache_start = Date(first(cache.hourly_datetimes))
    cache_end   = Date(last(cache.hourly_datetimes))

    train_start, train_end, val_start, val_end = resolve_date_windows(
        cache_start, cache_end, args["val_window"], args["val_days"],
        args["train_start"], args["train_end"], args["val_start"], args["val_end"])

    @info "Train window: $train_start .. $train_end"
    @info "Val window:   $val_start .. $val_end"

    news_from = min(train_start, val_start)
    news_to   = max(train_end, val_end)

    backfill_script = joinpath(SCRIPTS_DIR, "backfill_news_signals.jl")
    fetch_script     = joinpath(SCRIPTS_DIR, "fetch_news_snapshot_ohlcv.jl")

    if args["dry_run"]
        println()
        println("--dry-run: no universe build, no Ollama calls, no Kite calls. Would run:")
        if args["skip_universe"]
            println("  (skip universe — --skip-universe)")
        elseif isfile(UNIVERSE_FILE) && !args["rebuild_universe"]
            println("  (universe_latest.json already exists — using as-is; pass --rebuild-universe to force)")
        else
            println("  [in-process] build universe: SharedTopMarketCap(n=$(args["n"])) → $UNIVERSE_FILE")
        end
        if args["skip_news"]
            println("  (skip news backfill — --skip-news)")
        else
            println("  julia --project=packages/NewsMonitor $backfill_script --from $news_from --to $news_to")
        end
        if args["skip_1min"]
            println("  (skip 1-minute snapshots — --skip-1min)")
        else
            println("  julia --project=packages/TradingGame $fetch_script --role train --from $train_start --to $train_end")
            println("  julia --project=packages/TradingGame $fetch_script --role val   --from $val_start --to $val_end")
        end
        return
    end

    save_date_window(DATE_WINDOW_FILE, train_start, train_end, val_start, val_end;
                      val_window=args["val_window"], val_days=args["val_days"])
    @info "  Saved resolved date window → $DATE_WINDOW_FILE (train_trading_policy.jl " *
          "picks this up automatically when run with no date flags of its own)"

    # ── Stage 1: universe ───────────────────────────────────────────────────
    if args["skip_universe"]
        @info "Stage 1/3: skipping universe (--skip-universe)"
    elseif isfile(UNIVERSE_FILE) && !args["rebuild_universe"]
        @info "Stage 1/3: universe_latest.json already exists, using as-is " *
              "(pass --rebuild-universe to regenerate): $UNIVERSE_FILE"
    else
        @info "Stage 1/3: building universe (SharedTopMarketCap, n=$(args["n"]))…"
        pool = eligible_candidates(cache, joinpath(DATA_DIR, "$(cache.exchange)_companies_latest.json"))
        strategy = SharedTopMarketCap(n=args["n"])
        train, val = build_universes(strategy, pool)
        save_universe_snapshot(strategy, train, val, UNIVERSE_FILE)
        @info "  Saved $(length(train)) train / $(length(val)) val candidates → $UNIVERSE_FILE"
    end

    # ── Stage 2: news backfill ──────────────────────────────────────────────
    if args["skip_news"]
        @info "Stage 2/3: skipping news backfill (--skip-news)"
    else
        @info "Stage 2/3: backfilling news signals ($news_from … $news_to)…"
        run(`julia --project=packages/NewsMonitor $backfill_script --from $(string(news_from)) --to $(string(news_to))`)
    end

    # ── Stage 3: 1-minute snapshots, role-split ─────────────────────────────
    if args["skip_1min"]
        @info "Stage 3/3: skipping 1-minute snapshots (--skip-1min)"
    else
        @info "Stage 3/3: fetching 1-minute snapshots — train ($train_start … $train_end)…"
        run(`julia --project=packages/TradingGame $fetch_script --role train --from $(string(train_start)) --to $(string(train_end))`)
        @info "Stage 3/3: fetching 1-minute snapshots — val ($val_start … $val_end)…"
        run(`julia --project=packages/TradingGame $fetch_script --role val --from $(string(val_start)) --to $(string(val_end))`)
    end

    @info "Done."
end

main()
