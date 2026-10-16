"""
backfill_news_signals.jl

Classifies historical NSE corporate announcements (`website/data/nse_announcements.db`,
raw text only — no sentiment/severity) into `NewsSignal`-shaped records using a
local Ollama model (`NewsMonitor.classify_item_ollama`), so `TradingGame`'s
news features (`packages/TradingGame/src/news_features.jl`) have real
historical data to train against instead of the neutral-zero placeholder.

Why local Ollama, not the Claude API: classifying every historical
announcement for even a modest candidate universe is tens of thousands of
calls — fine for a free local model running overnight, expensive for a paid
API. `classify_item_ollama` reuses the exact same tool schema/calibration
text as `classify_item` (the live daemon's Claude path), so a backfilled
signal and a live one are classified identically; only the model differs.

Scope: by default, every symbol in `train_candidates`/`val_candidates` of
`website/data/trading_game/universe_latest.json` (deduplicated) — NOT the
whole market. The confidence-passing pool alone is ~1960 symbols / ~1.08M
announcements; even at ~1s/call on a local GPU that's 12+ days. Restricting
to symbols actually in play keeps this a same-day background job (today's
20-symbol universe is ~38K rows total history, ~10-11 hours serial) and it
only grows as new symbols enter the universe — `--symbol` adds one or more
explicit symbols instead of reading the snapshot, for a case not covered by
today's universe yet. Also accepts the OLD flat universe-snapshot format
(`candidates`/`n_candidates`, pre train/val split) as well as the current
one, since `website/data/trading_game/universe_latest.json` may predate a
`build_market_universe_snapshot.jl` re-run.

Resumable: classified results are written keyed by `guid` into a new SQLite
table (`website/data/news_signals.db`), `INSERT OR IGNORE`'d — a re-run (e.g.
after adding a new symbol, or after an interrupted run) only classifies rows
not already present. A transient Ollama failure leaves that row unclassified
for the next run to retry, same policy as `fetch_nse_history.jl`'s
week-level retry.

Usage:
  julia --project=packages/NewsMonitor scripts/backfill_news_signals.jl
  julia --project=packages/NewsMonitor scripts/backfill_news_signals.jl --dry-run
  julia --project=packages/NewsMonitor scripts/backfill_news_signals.jl --symbol RELIANCE --symbol TCS
  julia --project=packages/NewsMonitor scripts/backfill_news_signals.jl --from 2021-01-01
  julia --project=packages/NewsMonitor scripts/backfill_news_signals.jl --ollama-model qwen3:latest

Exchange of the universe (`--exchange nse|bse`, default nse): the announcements are NSE's, keyed by NSE symbol. With
`--exchange bse` the universe's symbols are BSE symbols; each is mapped to its NSE symbol through
`website/data/bse_companies_latest.json` (`symbol` -> `nse_symbol`, set for dual-listed companies), the announcements are
read under the NSE symbol, and every signal is stored under the BSE symbol, which is what a BSE training cache looks up.
A BSE symbol with no NSE listing has no announcements here and is reported and skipped; its news has to come from a BSE
source. A guid already stored under the NSE symbol is re-keyed to the BSE symbol when the two differ.

Output: website/data/news_signals.db
Schema: news_signals(guid, symbol, event_type, sentiment, severity, summary,
                      source, headline, url, published_at, classified_at)
"""

using NewsMonitor, SQLite, JSON3, Dates

const REPO_ROOT = joinpath(@__DIR__, "..")

function parse_args()
    args = Dict{String,Any}(
        "from"            => nothing,
        "to"              => nothing,
        "symbols"         => String[],
        "universe"        => joinpath(REPO_ROOT, "website", "data", "trading_game", "universe_latest.json"),
        "announcements"   => joinpath(REPO_ROOT, "website", "data", "nse_announcements.db"),
        "out"             => joinpath(REPO_ROOT, "website", "data", "news_signals.db"),
        "ollama_host"     => NewsMonitor.OLLAMA_BASE,
        "ollama_model"    => NewsMonitor.OLLAMA_MODEL,
        "dry_run"         => false,
        "exchange"        => "nse",
        "companies"       => joinpath(REPO_ROOT, "website", "data", "bse_companies_latest.json"),
    )
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a in ("-h", "--help")
            println("""
backfill_news_signals.jl — classify historical NSE announcements via local Ollama

Flags:
  --symbol SYM         Add one symbol to classify (repeatable). Default:
                        every symbol in --universe's train+val candidates.
  --universe PATH      Universe snapshot JSON (default: $(args["universe"]))
  --announcements PATH Raw announcements SQLite DB (default: $(args["announcements"]))
  --out PATH            Output SQLite DB (default: $(args["out"]))
  --from DATE           Only classify announcements on/after this date
  --to DATE             Only classify announcements on/before this date
  --ollama-host URL    Ollama server base URL (default: $(args["ollama_host"]))
  --ollama-model NAME   Ollama model name (default: $(args["ollama_model"]))
  --exchange nse|bse    Exchange of the universe's symbols (default nse). bse maps each BSE symbol to its NSE symbol
                        for the lookup and stores the signals under the BSE symbol
  --companies PATH      BSE companies JSON with the nse_symbol mapping (default: $(args["companies"]))
  --dry-run             Print the row count that would be classified, no API/DB writes
  -h, --help            Show this message
""")
            exit(0)
        elseif a == "--symbol" && i + 1 <= length(ARGS)
            push!(args["symbols"], ARGS[i+1]); i += 2
        elseif a == "--universe" && i + 1 <= length(ARGS)
            args["universe"] = ARGS[i+1]; i += 2
        elseif a == "--announcements" && i + 1 <= length(ARGS)
            args["announcements"] = ARGS[i+1]; i += 2
        elseif a == "--out" && i + 1 <= length(ARGS)
            args["out"] = ARGS[i+1]; i += 2
        elseif a == "--from" && i + 1 <= length(ARGS)
            args["from"] = Date(ARGS[i+1]); i += 2
        elseif a == "--to" && i + 1 <= length(ARGS)
            args["to"] = Date(ARGS[i+1]); i += 2
        elseif a == "--ollama-host" && i + 1 <= length(ARGS)
            args["ollama_host"] = ARGS[i+1]; i += 2
        elseif a == "--ollama-model" && i + 1 <= length(ARGS)
            args["ollama_model"] = ARGS[i+1]; i += 2
        elseif a == "--exchange" && i + 1 <= length(ARGS)
            ARGS[i+1] in ("nse", "bse") || error("--exchange must be nse or bse, got $(ARGS[i+1])")
            args["exchange"] = ARGS[i+1]; i += 2
        elseif a == "--companies" && i + 1 <= length(ARGS)
            args["companies"] = ARGS[i+1]; i += 2
        elseif a == "--dry-run"
            args["dry_run"] = true; i += 1
        else
            error("Unknown argument: $a (see --help)")
        end
    end
    return args
end

# ── Universe symbol resolution ─────────────────────────────────────────────────

"""Symbols to classify: `--symbol` if given, else every symbol in `path`'s
train+val candidates (deduplicated), accepting both the current
(`train_candidates`/`val_candidates`) and the old flat (`candidates`)
snapshot format."""
function resolve_symbols(path::String, explicit::Vector{String})::Vector{String}
    isempty(explicit) || return sort(unique(explicit))

    isfile(path) || error(
        "backfill_news_signals.jl: no --symbol given and universe snapshot not found: $path\n" *
        "Run build_market_universe_snapshot.jl first, or pass --symbol explicitly.")

    raw = JSON3.read(read(path, String))
    syms = Set{String}()
    if haskey(raw, :train_candidates)
        for c in raw.train_candidates; push!(syms, String(c.symbol)); end
        for c in get(raw, :val_candidates, []); push!(syms, String(c.symbol)); end
    elseif haskey(raw, :candidates)
        for c in raw.candidates; push!(syms, String(c.symbol)); end
    else
        error("backfill_news_signals.jl: $path has neither train_candidates/val_candidates nor candidates — unrecognised format")
    end
    return sort(collect(syms))
end

"""Map each symbol to the NSE symbol its announcements are filed under: itself for `exchange == "nse"`, the BSE
companies file's `nse_symbol` for `"bse"`. Returns `(nse_to_target, unmapped)`: NSE symbol => the symbol signals are
stored under, and the symbols with no NSE listing (skipped)."""
function map_to_nse(symbols::Vector{String}, exchange::String, companies_path::String)
    exchange == "nse" && return Dict(s => s for s in symbols), String[]
    isfile(companies_path) || error("backfill_news_signals.jl: --exchange bse needs $companies_path (scripts/generate_bse_list.jl)")
    nse_of = Dict{String,String}()
    for c in JSON3.read(read(companies_path, String)).companies
        n = get(c, :nse_symbol, nothing)
        n === nothing || (nse_of[String(c.symbol)] = String(n))
    end
    nse_to_target = Dict{String,String}()
    unmapped = String[]
    for s in symbols
        n = get(nse_of, s, nothing)
        if n === nothing
            push!(unmapped, s)
        elseif haskey(nse_to_target, n)
            @warn "NSE symbol $n is the listing of both $(nse_to_target[n]) and $s; signals go to $(nse_to_target[n]) only"
        else
            nse_to_target[n] = s
        end
    end
    return nse_to_target, unmapped
end

# ── Historical announcement date parsing ("01-Apr-2011 09:06:00") ──────────────

const _MONTHS = Dict("Jan"=>1,"Feb"=>2,"Mar"=>3,"Apr"=>4,"May"=>5,"Jun"=>6,
                      "Jul"=>7,"Aug"=>8,"Sep"=>9,"Oct"=>10,"Nov"=>11,"Dec"=>12)

function _parse_an_dt(s::String)::DateTime
    m = match(r"^(\d{2})-([A-Za-z]{3})-(\d{4}) (\d{2}):(\d{2}):(\d{2})$", s)
    isnothing(m) && return DateTime(1970, 1, 1)
    day, mon, yr, h, mi, se = m.captures
    return DateTime(parse(Int, yr), _MONTHS[mon], parse(Int, day), parse(Int, h), parse(Int, mi), parse(Int, se))
end

# ── Output DB ─────────────────────────────────────────────────────────────────

function init_out_db(path::String)::SQLite.DB
    mkpath(dirname(path))
    db = SQLite.DB(path)
    DBInterface.execute(db, """
        CREATE TABLE IF NOT EXISTS news_signals (
            guid          TEXT PRIMARY KEY,
            symbol        TEXT NOT NULL,
            event_type    TEXT NOT NULL,
            sentiment     REAL NOT NULL,
            severity      REAL NOT NULL,
            summary       TEXT,
            source        TEXT,
            headline      TEXT,
            url           TEXT,
            published_at  TEXT NOT NULL,
            classified_at TEXT NOT NULL
        )
    """)
    DBInterface.execute(db, "CREATE INDEX IF NOT EXISTS idx_news_sig_sym_dt ON news_signals(symbol, published_at)")
    return db
end

"""The symbol a classified guid is stored under, or `nothing` when it has not been classified."""
function stored_symbol(db::SQLite.DB, guid::String)::Union{String,Nothing}
    rows = [NamedTuple(r) for r in DBInterface.execute(db, "SELECT symbol FROM news_signals WHERE guid = ?", [guid])]
    return isempty(rows) ? nothing : String(rows[1].symbol)
end

function insert_signal!(db::SQLite.DB, sig::NewsSignal)
    DBInterface.execute(db, """
        INSERT OR IGNORE INTO news_signals
            (guid, symbol, event_type, sentiment, severity, summary, source, headline, url, published_at, classified_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [sig.guid, sig.symbol, sig.event_type, Float64(sig.sentiment), Float64(sig.severity),
          sig.summary, sig.source, sig.headline, sig.url, string(sig.published_at), string(sig.classified_at)])
end

# ── Main ──────────────────────────────────────────────────────────────────────

function main()
    args = parse_args()
    universe_symbols = resolve_symbols(args["universe"], args["symbols"])
    isempty(universe_symbols) && error("backfill_news_signals.jl: no symbols to classify")
    nse_to_target, unmapped = map_to_nse(universe_symbols, args["exchange"], args["companies"])
    isempty(unmapped) || @warn "$(length(unmapped)) $(args["exchange"]) symbols have no NSE listing, so no NSE announcements (skipped): $(join(unmapped, ", "))"
    symbols = sort(collect(keys(nse_to_target)))
    isempty(symbols) && error("backfill_news_signals.jl: none of the symbols has NSE announcements")

    isfile(args["announcements"]) || error("Not found: $(args["announcements"])\nRun fetch_nse_history.jl first.")
    src_db = SQLite.DB(args["announcements"])

    placeholders = join(fill("?", length(symbols)), ",")
    # NamedTuple(r) materialises each row's values immediately — SQLite.jl's
    # Row is a forward-only cursor view that goes stale once iteration moves
    # on, so collect()ing raw Rows and reading them later (e.g. in filter!)
    # throws ArgumentError.
    rows = [NamedTuple(r) for r in DBInterface.execute(src_db,
        "SELECT guid, symbol, an_dt, desc, attchmnt_text, attchmnt_file FROM announcements " *
        "WHERE symbol IN ($placeholders)", symbols)]

    from_d, to_d = args["from"], args["to"]
    if from_d !== nothing || to_d !== nothing
        rows = filter(rows) do r
            d = Date(_parse_an_dt(String(r.an_dt)))
            (from_d === nothing || d >= from_d) && (to_d === nothing || d <= to_d)
        end
    end

    @info "Symbols: $(length(symbols))  Announcements matched: $(length(rows))"

    if args["dry_run"]
        @info "--dry-run: no classification or writes performed."
        return
    end

    out_db = init_out_db(args["out"])
    n_total = length(rows)
    n_skipped = 0
    n_classified = 0
    n_failed = 0
    n_rekeyed = 0
    t0 = time()

    for (i, r) in enumerate(rows)
        guid = String(r.guid)
        target = nse_to_target[String(r.symbol)]
        stored = stored_symbol(out_db, guid)
        if stored !== nothing
            if stored != target
                DBInterface.execute(out_db, "UPDATE news_signals SET symbol = ? WHERE guid = ?", [target, guid])
                n_rekeyed += 1
            end
            n_skipped += 1
            continue
        end

        item = NewsItem(
            guid         = guid,
            source       = "NSE",
            headline     = String(coalesce(r.desc, "")),
            body         = String(coalesce(r.attchmnt_text, "")),
            url          = String(coalesce(r.attchmnt_file, "")),
            published_at = _parse_an_dt(String(r.an_dt)),
            nse_symbol   = String(r.symbol),
        )

        sig = classify_item_ollama(item; model=args["ollama_model"], host=args["ollama_host"])
        if sig === nothing
            n_failed += 1
            @warn "Classification failed, will retry next run: $guid"
            continue
        end

        # `target` is the NSE symbol itself for an NSE universe, the mapped BSE symbol for a BSE one.
        # NSE's own `symbol` column is authoritative (unlike the live
        # RSS/BSE path, which often has no structured symbol at all and
        # genuinely needs the LLM's guess) — observed the model occasionally
        # return a near-miss variant (e.g. LICI -> LICIND/LICINDIA) or a
        # different company mentioned in the text. Always pin back to the
        # DB's symbol rather than trusting the model's guess here.
        sig = NewsSignal(guid=sig.guid, source=sig.source, headline=sig.headline, url=sig.url,
                          published_at=sig.published_at, classified_at=sig.classified_at,
                          symbol=target, event_type=sig.event_type,
                          sentiment=sig.sentiment, severity=sig.severity, summary=sig.summary)

        insert_signal!(out_db, sig)
        n_classified += 1

        if i % 100 == 0 || i == n_total
            elapsed = time() - t0
            rate = n_classified / max(elapsed, 1e-6)
            remaining = n_total - i
            eta_min = rate > 0 ? round(remaining / rate / 60, digits=1) : NaN
            @info "[$i/$n_total] classified=$n_classified skipped=$n_skipped failed=$n_failed  rate=$(round(rate, digits=2))/s  ETA≈$(eta_min)m"
        end
    end

    @info "Done. classified=$n_classified skipped(already done)=$n_skipped re-keyed=$n_rekeyed failed=$n_failed  → $(args["out"])"
end

main()
