"""
Fetch all NSE corporate announcements from 2010-01-01 to yesterday and store
them in a SQLite database for model training.

Fetches one calendar week at a time (~830 requests for 2010–today). Resumable:
completed weeks are recorded in the database and skipped on re-run.

Usage:
  julia --project=packages/NewsMonitor scripts/fetch_nse_history.jl
  julia --project=packages/NewsMonitor scripts/fetch_nse_history.jl --from 2015-01-01
  julia --project=packages/NewsMonitor scripts/fetch_nse_history.jl --from 2010-01-01 --to 2020-12-31
  julia --project=packages/NewsMonitor scripts/fetch_nse_history.jl --delay 0.5
  julia --project=packages/NewsMonitor scripts/fetch_nse_history.jl --db /path/to/custom.db

Output: website/data/nse_announcements.db
Schema:
  announcements(guid, symbol, an_dt, desc, attchmnt_text, attchmnt_file, has_xbrl, raw_json)
  fetch_log(week_start, items_fetched, fetched_at)
"""

using HTTP, JSON3, SQLite, Dates

# ── CLI args ──────────────────────────────────────────────────────────────────

function parse_args()
    args = Dict{String,Any}(
        "from"  => Date(2010, 1, 1),
        "to"    => today() - Day(1),
        "delay" => 1.0,
        "db"    => joinpath(@__DIR__, "..", "website", "data", "nse_announcements.db"),
    )
    i = 1
    while i <= length(ARGS)
        if ARGS[i] in ("-h", "--help")
            println("""
fetch_nse_history.jl — build NSE corporate announcement history (SQLite)

Flags:
  --from  DATE    Start date, YYYY-MM-DD (default: 2010-01-01)
  --to    DATE    End date,   YYYY-MM-DD (default: yesterday)
  --delay SECS    Sleep between API calls in seconds (default: 1.0)
  --db    PATH    SQLite output path (default: website/data/nse_announcements.db)
  -h, --help      Show this message
""")
            exit(0)
        elseif ARGS[i] == "--from" && i+1 <= length(ARGS)
            args["from"] = Date(ARGS[i+1]); i += 2
        elseif ARGS[i] == "--to" && i+1 <= length(ARGS)
            args["to"] = Date(ARGS[i+1]); i += 2
        elseif ARGS[i] == "--delay" && i+1 <= length(ARGS)
            args["delay"] = parse(Float64, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--db" && i+1 <= length(ARGS)
            args["db"] = ARGS[i+1]; i += 2
        else
            @warn "Unknown argument: $(ARGS[i])"; i += 1
        end
    end
    return args
end

# ── NSE fetch ─────────────────────────────────────────────────────────────────

const NSE_ANN_URL = "https://www.nseindia.com/api/corporate-announcements"
const NSE_HEADERS = [
    "Referer"          => "https://www.nseindia.com/",
    "User-Agent"       => "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Accept"           => "application/json, text/plain, */*",
    "X-Requested-With" => "XMLHttpRequest",
]

function fetch_week(from_date::Date, to_date::Date)
    from_s = Dates.format(from_date, "dd-mm-yyyy")
    to_s   = Dates.format(to_date,   "dd-mm-yyyy")
    url    = "$NSE_ANN_URL?index=equities&from_date=$from_s&to_date=$to_s"

    resp = try
        HTTP.get(url; headers=NSE_HEADERS, request_timeout=30, status_exception=false)
    catch e
        @warn "HTTP error for $(from_s)–$(to_s): $(sprint(showerror, e))"
        return nothing
    end

    if resp.status != 200
        @warn "NSE API HTTP $(resp.status) for $(from_s)–$(to_s)"
        return nothing
    end

    rows = try JSON3.read(resp.body) catch
        @warn "JSON parse failed for $(from_s)–$(to_s)"
        return nothing
    end

    rows isa AbstractVector || return []
    return rows
end

# ── SQLite schema ─────────────────────────────────────────────────────────────

function init_db(path::String)::SQLite.DB
    mkpath(dirname(path))
    db = SQLite.DB(path)
    DBInterface.execute(db, """
        CREATE TABLE IF NOT EXISTS announcements (
            guid         TEXT PRIMARY KEY,
            symbol       TEXT NOT NULL,
            an_dt        TEXT NOT NULL,
            desc         TEXT,
            attchmnt_text TEXT,
            attchmnt_file TEXT,
            has_xbrl     INTEGER DEFAULT 0,
            raw_json     TEXT
        )
    """)
    DBInterface.execute(db, """
        CREATE INDEX IF NOT EXISTS idx_sym_dt ON announcements(symbol, an_dt)
    """)
    DBInterface.execute(db, """
        CREATE INDEX IF NOT EXISTS idx_dt ON announcements(an_dt)
    """)
    DBInterface.execute(db, """
        CREATE TABLE IF NOT EXISTS fetch_log (
            week_start   TEXT PRIMARY KEY,
            items_fetched INTEGER,
            fetched_at   TEXT
        )
    """)
    return db
end

function db_scalar(db::SQLite.DB, sql::String)::Int
    for row in DBInterface.execute(db, sql)
        v = row[1]
        return ismissing(v) ? 0 : Int(v)
    end
    return 0
end

function db_scalar(db::SQLite.DB, sql::String, params)::Int
    for row in DBInterface.execute(db, sql, params)
        v = row[1]
        return ismissing(v) ? 0 : Int(v)
    end
    return 0
end

function week_done(db::SQLite.DB, week_start::Date)::Bool
    return db_scalar(db,
        "SELECT COUNT(*) FROM fetch_log WHERE week_start = ?",
        [string(week_start)]) > 0
end

function insert_rows(db::SQLite.DB, rows, week_start::Date)::Int
    stmt = DBInterface.prepare(db, """
        INSERT OR IGNORE INTO announcements
            (guid, symbol, an_dt, desc, attchmnt_text, attchmnt_file, has_xbrl, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """)

    inserted = 0
    SQLite.transaction(db) do
        for row in rows
            symbol   = string(get(row, :symbol, ""))
            an_dt    = string(get(row, :an_dt, ""))
            file_url = string(get(row, :attchmntFile, ""))
            guid     = isempty(file_url) ?
                       "NSE:$(symbol):$(an_dt)" :
                       "NSE:" * basename(file_url)

            DBInterface.execute(stmt, [
                guid,
                symbol,
                an_dt,
                string(get(row, :desc, "")),
                string(get(row, :attchmntText, "")),
                file_url,
                get(row, :hasXbrl, false) ? 1 : 0,
                JSON3.write(row),
            ])
            inserted += 1
        end

        DBInterface.execute(db,
            "INSERT OR REPLACE INTO fetch_log VALUES (?, ?, ?)",
            [string(week_start), inserted, string(now())])
    end
    return inserted
end

# ── Week iterator ─────────────────────────────────────────────────────────────

function weeks_in_range(from::Date, to::Date)
    # Align start to the nearest Monday on or before from
    start = from - Day(dayofweek(from) - 1)
    weeks = Tuple{Date,Date}[]
    d = start
    while d <= to
        push!(weeks, (d, min(d + Day(6), to)))
        d += Week(1)
    end
    return weeks
end

# ── Main ──────────────────────────────────────────────────────────────────────

function main()
    args   = parse_args()
    db     = init_db(args["db"])
    delay  = args["delay"]
    weeks  = weeks_in_range(args["from"], args["to"])

    done_count = db_scalar(db, "SELECT COUNT(*) FROM fetch_log")

    @info "NSE history fetch: $(length(weeks)) weeks, $done_count already completed"
    @info "Output: $(args["db"])"

    total_items = 0
    fetched_weeks = 0
    skipped_weeks = 0

    for (i, (wstart, wend)) in enumerate(weeks)
        if week_done(db, wstart)
            skipped_weeks += 1
            continue
        end

        rows = fetch_week(wstart, wend)
        if isnothing(rows)
            # Transient error — don't mark done, will retry on next run
            @warn "Skipping $wstart (fetch failed)"
            sleep(delay * 2)
            continue
        end

        n = isempty(rows) ? 0 : insert_rows(db, rows, wstart)
        total_items  += n
        fetched_weeks += 1

        remaining = length(weeks) - skipped_weeks - fetched_weeks - done_count
        eta_min   = round(remaining * delay / 60, digits=1)
        @info "[$i/$(length(weeks))] $(Dates.format(wstart, "yyyy-mm-dd"))  " *
              "$n items  total=$total_items  ETA≈$(eta_min)m"

        sleep(delay)
    end

    total_in_db = db_scalar(db, "SELECT COUNT(*) FROM announcements")

    @info "Done. Fetched $fetched_weeks weeks, $total_items new items. " *
          "Database total: $total_in_db announcements."
end

main()
