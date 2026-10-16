"""
fetch_bse_announcements.jl

Archive BSE corporate announcements into SQLite, for companies that have no NSE listing (their news is not in
`nse_announcements.db`). Source: BSE's RSS feed (`NewsMonitor.fetch_bse_announcements`), which lists every filing since
midnight of the current day, about 10,000 items on a normal day. BSE's history API (`api.bseindia.com`) answers 403 to
scripted requests, so there is no way to backfill the past from here: history exists only from the first day this script
ran. Run it once a day after the close (15:30 IST or later, any time before midnight) and the archive grows by a day
each time. Re-runs are harmless (`INSERT OR IGNORE` on the guid).

Each item carries a scripcode; it is mapped to a BSE symbol through `website/data/bse_companies_latest.json`. Items
whose scripcode is not in that file (debt, preference and other non-equity series) are counted and dropped.

Usage:
  julia --project=packages/NewsMonitor scripts/news/fetch_bse_announcements.jl
  julia --project=packages/NewsMonitor scripts/news/fetch_bse_announcements.jl --scope bse-only
  julia --project=packages/NewsMonitor scripts/news/fetch_bse_announcements.jl --scope universe
  julia --project=packages/NewsMonitor scripts/news/fetch_bse_announcements.jl --dry-run
  julia --project=packages/NewsMonitor scripts/news/fetch_bse_announcements.jl --stats

Options:
  --scope S        which companies to keep (default `bse-only`): `all` (every scripcode in the companies file), `bse-only`
                   (no NSE listing: the ones the NSE archive cannot cover, default) or `universe` (the train and val
                   candidates of trading_game/universe_latest.json)
  --db PATH        output (default website/data/bse_announcements.db)
  --companies PATH BSE companies JSON (default website/data/bse_companies_latest.json)
  --universe PATH  universe snapshot for --scope universe
  --dry-run        fetch and count, write nothing
  --stats          print what the archive holds and exit (no fetch)
  -h, --help       this text

Output: announcements(guid, symbol, scripcode, an_dt, company, desc, url, fetched_at); `an_dt` is ISO
`yyyy-mm-ddTHH:MM:SS` (IST), so it sorts as text. The guid is `BSE:scripcode:filename` because one PDF is often filed
under several scripcodes of the same company.
"""

using NewsMonitor, SQLite, JSON3, Dates

const REPO_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const SCOPES = ("all", "bse-only", "universe")

function parse_args()
    args = Dict{String,Any}(
        "scope"     => "bse-only",
        "db"        => joinpath(REPO_ROOT, "website", "data", "bse_announcements.db"),
        "companies" => joinpath(REPO_ROOT, "website", "data", "bse_companies_latest.json"),
        "universe"  => joinpath(REPO_ROOT, "website", "data", "trading_game", "universe_latest.json"),
        "dry_run"   => false,
        "stats"     => false,
    )
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a in ("-h", "--help")
            println(match(r"\"\"\"\n(.*?)\"\"\""s, read(@__FILE__, String)).captures[1])
            exit(0)
        elseif a == "--scope" && i < length(ARGS)
            ARGS[i+1] in SCOPES || error("--scope must be one of $(join(SCOPES, ", ")), got $(ARGS[i+1])")
            args["scope"] = ARGS[i+1]; i += 2
        elseif a in ("--db", "--companies", "--universe") && i < length(ARGS)
            args[a[3:end]] = ARGS[i+1]; i += 2
        elseif a == "--dry-run"; args["dry_run"] = true; i += 1
        elseif a == "--stats"; args["stats"] = true; i += 1
        else error("Unknown argument: $a (see --help)") end
    end
    return args
end

"""
    scripcode_symbols(companies_path, scope, universe_path) -> Dict{String,String}

Scripcode => BSE symbol for the companies `scope` keeps.
"""
function scripcode_symbols(companies_path::String, scope::String, universe_path::String)
    isfile(companies_path) || error("Not found: $companies_path (scripts/companies/generate_bse_list.jl)")
    keep = nothing
    if scope == "universe"
        isfile(universe_path) || error("Not found: $universe_path (build_market_universe_snapshot.jl)")
        u = JSON3.read(read(universe_path, String))
        keep = Set(String(c.symbol) for key in (:train_candidates, :val_candidates) for c in get(u, key, []))
    end
    out = Dict{String,String}()
    for c in JSON3.read(read(companies_path, String)).companies
        get(c, :scripcode, nothing) === nothing && continue
        scope == "bse-only" && get(c, :nse_symbol, nothing) !== nothing && continue
        keep !== nothing && !(String(c.symbol) in keep) && continue
        out[string(Int(c.scripcode))] = String(c.symbol)
    end
    return out
end

function init_db(path::String)::SQLite.DB
    mkpath(dirname(path))
    db = SQLite.DB(path)
    DBInterface.execute(db, """
        CREATE TABLE IF NOT EXISTS announcements (
            guid       TEXT PRIMARY KEY,
            symbol     TEXT NOT NULL,
            scripcode  TEXT NOT NULL,
            an_dt      TEXT NOT NULL,
            company    TEXT,
            desc       TEXT,
            url        TEXT,
            fetched_at TEXT NOT NULL
        )""")
    DBInterface.execute(db, "CREATE INDEX IF NOT EXISTS idx_bse_ann_sym_dt ON announcements(symbol, an_dt)")
    return db
end

function print_stats(path::String)
    isfile(path) || (println("No archive yet: $path"); return)
    db = SQLite.DB(path)
    q(sql) = [NamedTuple(r) for r in DBInterface.execute(db, sql)]
    t = q("SELECT count(*) n, count(DISTINCT symbol) k, min(an_dt) a, max(an_dt) b, count(DISTINCT substr(an_dt,1,10)) d FROM announcements")[1]
    println("$path: $(t.n) announcements, $(t.k) symbols, $(t.d) days, $(t.a) .. $(t.b)")
    for r in q("SELECT substr(an_dt,1,10) day, count(*) n FROM announcements GROUP BY day ORDER BY day DESC LIMIT 7")
        println("  $(r.day)  $(r.n)")
    end
end

function main()
    args = parse_args()
    if args["stats"]
        print_stats(args["db"])
        return
    end
    sym_of = scripcode_symbols(args["companies"], args["scope"], args["universe"])
    @info "Scope $(args["scope"]): $(length(sym_of)) scripcodes"

    items = fetch_bse_announcements()
    isempty(items) && error("The BSE RSS feed returned nothing (network, or BSE changed the feed)")

    kept = NamedTuple[]
    outside = 0
    for it in items
        sym = get(sym_of, it.bse_code, nothing)
        if sym === nothing
            outside += 1
            continue
        end
        push!(kept, (guid = "BSE:$(it.bse_code):$(basename(it.url))", symbol = sym, scripcode = it.bse_code,
                     an_dt = Dates.format(it.published_at, dateformat"yyyy-mm-dd\THH:MM:SS"),
                     company = it.body, desc = it.headline, url = it.url))
    end
    @info "Feed: $(length(items)) items, $(length(kept)) in scope, $outside outside it"
    if isempty(kept)
        @warn "Nothing in scope"
        return
    end
    days = sort(unique(first(k.an_dt, 10) for k in kept))
    @info "In-scope items cover $(join(days, ", ")); symbols: $(length(unique(k.symbol for k in kept)))"
    args["dry_run"] && (@info "--dry-run: nothing written"; return)

    db = init_db(args["db"])
    before = only(NamedTuple(r).n for r in DBInterface.execute(db, "SELECT count(*) n FROM announcements"))
    stamp = string(now())
    DBInterface.execute(db, "BEGIN")
    for k in kept
        DBInterface.execute(db, "INSERT OR IGNORE INTO announcements (guid, symbol, scripcode, an_dt, company, desc, url, fetched_at) VALUES (?,?,?,?,?,?,?,?)",
                            [k.guid, k.symbol, k.scripcode, k.an_dt, k.company, k.desc, k.url, stamp])
    end
    DBInterface.execute(db, "COMMIT")
    after = only(NamedTuple(r).n for r in DBInterface.execute(db, "SELECT count(*) n FROM announcements"))
    @info "Added $(after - before) new announcements ($(length(kept) - (after - before)) already stored); archive now $after"
    print_stats(args["db"])
end

main()
