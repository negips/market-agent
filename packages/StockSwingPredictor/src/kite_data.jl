"""
Kite Connect historical OHLCV data fetching.

Loads the NSE instrument list, resolves symbols to instrument tokens, and
fetches daily OHLCV candles via the Kite Connect historical data API.

Every `collect_ohlcv*`/`load_cached_ohlcv*` function here takes an `out_dir`
that's expected to already be a single granularity's directory (e.g.
`website/data/ohlcv/nse/hourly/`) and reads/writes `{SYMBOL}.csv` directly
inside it — not the exchange root with a `_hourly`/`_5min`/etc. filename
suffix. Callers (`collect_nse_ohlcv.jl`, `collect_bse_ohlcv.jl`,
`update_ohlcv.jl`) own the `{exchange}/{granularity}/` directory layout;
this file has no opinion about exchange or granularity beyond what `out_dir`
it's handed.

Rate limiting: Kite allows ~3 requests/second for historical data. All public
functions that make multiple requests add a 400 ms inter-request sleep.
"""

using HTTP, JSON3, CSV, DataFrames, Dates

const KITE_BASE             = "https://api.kite.trade"
const NSE_INSTRUMENTS_CACHE = joinpath(@__DIR__, "..", "..", "..", "website", "data",
                                       "ohlcv", "_instruments_nse_cache.csv")
const BSE_INSTRUMENTS_CACHE = joinpath(@__DIR__, "..", "..", "..", "website", "data",
                                       "ohlcv", "_instruments_bse_cache.csv")

# Known NSE index tradingsymbols → looked up from instruments list at runtime.
const NSE_INDICES = [
    "NIFTY 50",
    "NIFTY BANK",
    "NIFTY IT",
    "NIFTY AUTO",
    "NIFTY PHARMA",
    "NIFTY FMCG",
    "NIFTY METAL",
    "NIFTY ENERGY",
    "NIFTY REALTY",
    "NIFTY MIDCAP 100",
]

# Maps Tijori sector names (from nse_companies_latest.json) to NSE index names.
const SECTOR_TO_INDEX = Dict(
    "Information Technology" => "NIFTY IT",
    "Banking"                => "NIFTY BANK",
    "Financial Services"     => "NIFTY BANK",
    "Automobile"             => "NIFTY AUTO",
    "Auto Ancillaries"       => "NIFTY AUTO",
    "Pharmaceutical"         => "NIFTY PHARMA",
    "Healthcare"             => "NIFTY PHARMA",
    "FMCG"                   => "NIFTY FMCG",
    "Consumer Goods"         => "NIFTY FMCG",
    "Metal"                  => "NIFTY METAL",
    "Mining"                 => "NIFTY METAL",
    "Oil & Gas"              => "NIFTY ENERGY",
    "Energy"                 => "NIFTY ENERGY",
    "Realty"                 => "NIFTY REALTY",
    "Real Estate"            => "NIFTY REALTY",
)

"""
A Kite Connect session: `api_key`/`access_token` for the `Authorization`
header, plus `repo_root` so the session can re-authenticate itself (see
`relogin_kite!`) without every caller needing to separately track where the
repo lives. Mutable so a mid-run token refresh (`_kite_get`'s automatic
403-retry, or an explicit `relogin_kite!` call) is immediately visible to
every holder of the same session object — a long-running script loops
hundreds of calls over one `session` it loaded once at startup; after a
refresh, the very next call anywhere in that loop already sees the new
token, with nothing re-loaded or re-passed.
"""
mutable struct KiteSession
    api_key      :: String
    access_token :: String
    repo_root    :: String
    account      :: Int
end

KiteSession(api_key::String, access_token::String, repo_root::String) =
    KiteSession(api_key, access_token, repo_root, 1)

"""
Kite rate limits are per API key, so a second Kite Connect app (`KITE_HISTORICAL2_API_KEY`
/ `_SECRET` in `.env`) gives a second, independent request budget — two jobs
on two accounts never compete for it. `account` is the 1-based slot: 1 is
`KITE_HISTORICAL_*` with the session in `sidecar/kite_session.json`; N ≥ 2 is
`KITE_HISTORICAL{N}_*` with `sidecar/kite_session{N}.json`. Same trading
account (user id, password, TOTP) throughout — only the app key/secret differ.
"""
function kite_session_path(repo_root::String, account::Int=1)
    account >= 1 || error("Kite account must be >= 1, got $account")
    return joinpath(repo_root, "sidecar", account == 1 ? "kite_session.json" : "kite_session$(account).json")
end

"""
Read `--account N` from a script's command-line arguments (default 1). Shared
by every Kite-backed script so each one selects its API-key slot identically.

# Arguments
- `args`: argument vector, default `ARGS`

# Returns
- `Int`: the account slot

# Example
```julia
session = load_kite_session(REPO_ROOT; account=kite_account_from_args())
```
"""
function kite_account_from_args(args::AbstractVector{<:AbstractString}=ARGS)::Int
    i = findfirst(==("--account"), args)
    isnothing(i) && return 1
    i < length(args) || error("--account given with no number after it")
    n = tryparse(Int, args[i + 1])
    (isnothing(n) || n < 1) && error("--account expects an integer >= 1, got '$(args[i + 1])'")
    return n
end

"""
Read the Kite session for `account` (see `kite_session_path`; default 1,
`kite_session.json`). Throws if not found. Warns (does not throw) if the saved date doesn't match
today — the token itself may still be valid for a few hours past midnight;
`_kite_get`'s automatic 403-retry is what actually catches genuine expiry.
"""
function load_kite_session(repo_root::String; account::Int=1)::KiteSession
    path = kite_session_path(repo_root, account)
    login_cmd = account == 1 ? "node sidecar/kite_login.js" : "node sidecar/kite_login.js --account $account"
    isfile(path) || error("No Kite session found for account $account. Run: $login_cmd")
    s = JSON3.read(read(path, String))
    date_str = string(get(s, :date, ""))
    if date_str != string(today())
        @warn "Kite session (account $account) is from $date_str — token may be stale. " *
              "Run: $login_cmd"
    end
    return KiteSession(string(s.api_key), string(s.access_token), repo_root, account)
end

function _kite_headers(session)
    ["X-Kite-Version" => "3",
     "Authorization"  => "token $(session.api_key):$(session.access_token)"]
end

"""
Force a fresh Kite login by running `sidecar/kite_login.js` — Playwright-
driven, since Zerodha's login page has no plain REST endpoint to
re-authenticate against directly (see that script's own docstring for the
TOTP + OAuth redirect flow). Blocking: launches a real (visible) browser and
waits for the OAuth redirect, typically a few seconds, requires a display.
Reads `.env` itself (same as the script does standalone), so no credentials
need to be threaded through Julia.

Two methods: `relogin_kite!(repo_root)` bootstraps a brand new `KiteSession`
(no existing session file required — the first run of the day); `relogin_kite!
(session)` refreshes an existing session **in place**, so every other
holder of that same object picks up the new token too (see `KiteSession`'s
docstring). This is what `_kite_get` calls automatically on a 403; call it
directly to force a refresh proactively (e.g. right before an overnight job
you know will span the token's expiry).
"""
function relogin_kite!(repo_root::String; account::Int=1)::KiteSession
    login_script = joinpath(repo_root, "sidecar", "kite_login.js")
    isfile(login_script) || error("relogin_kite!: not found: $login_script")
    run(`node $login_script --account $account`)
    return load_kite_session(repo_root; account)
end

function relogin_kite!(session::KiteSession)::KiteSession
    fresh = relogin_kite!(session.repo_root; account=session.account)
    session.api_key      = fresh.api_key
    session.access_token = fresh.access_token
    return session
end

"""Minimum gap between relogin ATTEMPTS (successful or not), shared across
every `_kite_get` call regardless of which symbol/chunk/function triggered
it. Without this, a script with hundreds of chunks across many symbols
(`collect_nse_ohlcv.jl`, a multi-year backfill, …) would launch one browser
login per 403 it sees — if the underlying problem isn't a one-off (network
drops mid-relogin, Zerodha's login page changes, TOTP drifts), every one of
those chunks hits 403 independently and `_kite_get` had no memory between
calls, so it'd hammer out a fresh login attempt for every single one.
Module-level (not per-`KiteSession`) because this codebase only ever uses
one Kite account at a time — a global cooldown is the correct granularity,
not an added complication."""
const KITE_RELOGIN_COOLDOWN_SECONDS = 60.0
const _LAST_KITE_RELOGIN_ATTEMPT = Ref(-Inf)

"""Default per-request timeouts (seconds) applied to every Kite HTTP call
that doesn't explicitly override them. Without these, a connection that's
accepted at the TCP level but then never sends a response (or goes silent
mid-response) blocks `HTTP.get` indefinitely — observed live: a
`backfill_ohlcv.jl` hourly chunk stalled for 5+ hours on a single request,
zero CPU, zero error, zero progress, because nothing bounded it. Three
separate stages can each hang independently, so all three are set:
`connect_timeout` (TCP+TLS handshake), `request_timeout` (overall deadline
for the whole request/response), `read_idle_timeout` (max gap between
reads once a response starts streaming)."""
const KITE_CONNECT_TIMEOUT_SECONDS    = 15.0
const KITE_REQUEST_TIMEOUT_SECONDS    = 30.0
const KITE_READ_IDLE_TIMEOUT_SECONDS  = 30.0

"""
`HTTP.get` against a Kite endpoint, with ONE automatic re-authenticate-and-
retry if the response is a 403 (expired token) — every `fetch_ohlcv*`/
`load_instruments`/broker/macro function here calls this instead of raw
`HTTP.get(url; headers=_kite_headers(session), ...)`, so a token expiring
mid-run (an overnight job spanning the daily expiry boundary) is handled in
exactly one place rather than needing matching retry logic duplicated
across every call site. `kwargs` forward straight to `HTTP.get`; the three
timeout keywords above are applied as defaults here (not left to each call
site to remember) and a caller passing its own `connect_timeout`/
`request_timeout`/`read_idle_timeout` still overrides them as normal.

A connection failure (network down, DNS, a timeout firing) throws straight
out of the first `HTTP.get` call below, before `resp` even exists — this
function never gets a chance to inspect a status code, so relogin is never
considered for that case at all; it propagates to the caller's own
try/catch exactly as it always did, with no behaviour change from before
this function existed.

A 403 is a different case: Kite's server DID respond, meaning the network
is clearly up, and the token is genuinely invalid. Only here does a relogin
get attempted — at most once per `KITE_RELOGIN_COOLDOWN_SECONDS`, win or
lose, enforced by `_LAST_KITE_RELOGIN_ATTEMPT` — so a persistent problem
(not just an expired token, but a genuinely broken login) degrades to "log
a warning and let the caller's existing status-check handle the still-403
response" instead of retrying a losing battle on every single call. Either
way this returns a response exactly as `HTTP.get` would have, never an
exception from the relogin step itself, so every existing caller's status-
code handling downstream needs no change at all.

Every relogin outcome (skipped on cooldown, attempted, succeeded, failed)
is logged via `_log_summary(active_script_log(), ...)` so it lands in
whichever script's log file is currently open, not just the live terminal
— `relogin_kite!` shells out to a Playwright-driven browser login with no
timeout of its own, so this is often the ONLY record of why a run stalled
for an extended stretch (observed live: a single relogin took over an
hour) once the terminal scrolls past it.
"""
function _kite_get(url::String, session::KiteSession;
                    connect_timeout::Real=KITE_CONNECT_TIMEOUT_SECONDS,
                    request_timeout::Real=KITE_REQUEST_TIMEOUT_SECONDS,
                    read_idle_timeout::Real=KITE_READ_IDLE_TIMEOUT_SECONDS,
                    kwargs...)
    resp = HTTP.get(url; headers=_kite_headers(session), connect_timeout,
                     request_timeout, read_idle_timeout, kwargs...)
    resp.status != 403 && return resp

    elapsed = time() - _LAST_KITE_RELOGIN_ATTEMPT[]
    if elapsed < KITE_RELOGIN_COOLDOWN_SECONDS
        _log_summary(active_script_log(),
            "Kite token expired (403) — skipping relogin, last attempt " *
            "$(round(elapsed, digits=1))s ago (cooldown $(KITE_RELOGIN_COOLDOWN_SECONDS)s)";
            warn=true)
        return resp
    end

    _LAST_KITE_RELOGIN_ATTEMPT[] = time()
    _log_summary(active_script_log(), "Kite token expired (403) — re-authenticating…"; warn=true)
    relogin_start = time()
    try
        relogin_kite!(session)
        _log_summary(active_script_log(),
            "Kite relogin succeeded ($(round(time() - relogin_start, digits=1))s)")
        return HTTP.get(url; headers=_kite_headers(session), connect_timeout,
                         request_timeout, read_idle_timeout, kwargs...)
    catch e
        _log_summary(active_script_log(),
            "Kite relogin failed after $(round(time() - relogin_start, digits=1))s, " *
            "will not retry again for $(KITE_RELOGIN_COOLDOWN_SECONDS)s: $(sprint(showerror, e))";
            warn=true)
        return resp
    end
end

"""
Download the instrument list from Kite for a given exchange and cache it locally.
Returns a DataFrame with columns: instrument_token, tradingsymbol, name,
instrument_type, segment, exchange.

# Arguments
- `session`: `KiteSession` from `load_kite_session`
- `exchange`: `"NSE"` (default) or `"BSE"`
- `refresh`: force re-download even if cache exists (default false)
"""
function load_instruments(session; exchange::String="NSE", refresh::Bool=false)::DataFrame
    cache = exchange == "BSE" ? BSE_INSTRUMENTS_CACHE : NSE_INSTRUMENTS_CACHE
    if !refresh && isfile(cache)
        return CSV.read(cache, DataFrame)
    end

    resp = _kite_get("$KITE_BASE/instruments/$exchange", session;
                      request_timeout=30, status_exception=false)
    resp.status == 200 || error("Instruments endpoint returned HTTP $(resp.status)")

    df = CSV.read(IOBuffer(resp.body), DataFrame)

    mkpath(dirname(cache))
    CSV.write(cache, df)
    return df
end

"""
Build a symbol → instrument_token lookup dict for EQ stocks (and NSE INDICES).

# Arguments
- `instruments`: DataFrame from `load_instruments`
- `exchange`: `"NSE"` (default) or `"BSE"`

# Notes
Both exchanges' instrument dumps classify a large number of non-equity
instruments — sovereign/state government bonds, T-bills, Sovereign Gold
Bonds, corporate NCDs/debentures — as `instrument_type = "EQ"`, same as a
real company's shares. Verified live against the full NSE dump: roughly
60% of NSE's nominal "EQ" rows are actually debt instruments (State
Development Loans alone account for ~4,300), not company stock. Two
independent layers filter these out, applied to every `EQ` row regardless
of exchange:

1. Empty `name` field — NCDs and other unregistered debt series are listed
   with no descriptive name at all (e.g. tradingsymbol suffixes `-N0`
   through `-N9`/`-NA` through `-NZ` on NSE; `10IGG`, `FFTF16BGR` on BSE).
   Also catches BSE's MF fixed-maturity/closed-end debt scheme units.
2. `_DEBT_NAME_RE` — the `name` field reliably spells out what the
   remaining debt instruments are, even when the `tradingsymbol` alone
   wouldn't obviously say so (e.g. NSE's `66RJ30-SG` is named
   `"SDL RJ 6.6% 2030"`): State Development Loans (`SDL ...`), GOI loans
   and T-bills (`GOI ...`, `... TBILL ...`), Sovereign Gold Bonds
   (`...GOLDBONDS...`/`SOVEREIGN GOLD BOND...`), and any NCD/debenture/bond
   (`BOND` as a whole word — matches `BHARAT BOND ETF`, not a company name
   that merely contains the substring, e.g. `CHEMBOND CHEMICAL`,
   `BONDADA ENGINEERING`).
3. `_EXCLUDED_SUFFIXES` — `tradingsymbol` suffixes excluded by deliberate
   choice, not because the name regex misses them: `-RR` (REIT units,
   e.g. `EMBASSY-RR` → `"EMBASSY OFFICE PARKS REIT"`), `-IV` (InvIT units,
   e.g. `PGINVIT-IV` → `"POWERGRID INFRA INVESTMENT TRUST - INVIT"`) — both
   trust units, not a company's own shares — `-E1` (partly-paid-up shares,
   e.g. `ROCKPP-E1` → `"ROCKINGDCE RS.5 PPD UP"`), and `-BE`/`-BZ`/`-BL`/
   `-ST` (surveillance/trade-to-trade settlement series — still ordinary
   equity, just excluded anyway per explicit request). Only observed on
   NSE (`-BL` currently has zero matches but is kept in the list in case a
   future instrument dump adds one) — checked regardless of exchange
   anyway, since it's cheap and exchange-agnostic.
4. `_TRUST_NAME_RE` — whole-word `REIT`/`INVIT` in the `name` field.
   Redundant with `-RR`/`-IV` above on NSE, but BSE lists the exact same
   trusts (`EMBASSY`, `MINDSPACE`, `PGINVIT`, `IRBINVIT`, …) under a plain
   `tradingsymbol` with no suffix at all — BSE doesn't use NSE's
   hyphen-suffix scheme (verified: of 12,916 BSE `EQ` rows, only 18
   contain a hyphen, and none of those are a settlement/series marker —
   e.g. `BAJAJ-AUTO` is just how BSE spells the ticker). The name field is
   the only signal BSE gives for these.
5. `_FUND_NAME_RE`/`_ETF_NAME_RE` — whole-word `FUND`/`ETF` in the `name`
   field. `_FUND_NAME_RE` catches exchange-listed mutual fund scheme units
   (fixed-maturity plans, closed-end debt schemes — 21 distinct names
   across 17 AMCs on BSE, e.g. `"NIPPON INDIA MUTUAL FUND"`, plus the
   non-AMC closed-end fund `"FIRST CUSTODIAN FUND (INDIA) LTD"`) as well
   as catching `08`/`11`-series Nippon FMP codes whose symbol alone gives
   no hint (`11ADD`, `11ADR`, …; the AMC tags these inconsistently — some
   series have a blank `name`, caught by rule 1, this one doesn't).
   `_ETF_NAME_RE` catches exchange-traded funds: verified these were
   previously **completely unfiltered** — 300 of 3,648 NSE survivors and
   236 of 5,409 BSE survivors (`NIFTYBEES`, `GOLDBEES`, `BANKBEES`, and
   hundreds of sectoral/thematic/gold/silver/liquid ETFs from every major
   AMC) were nominal "EQ" rows with no debt/trust/fund keyword to catch
   them, since `ETF` was never checked for at all. Verified against the
   full dump that neither regex has a false-positive risk: every `FUND`-
   or `ETF`-named `EQ` row on both exchanges is a genuine fund vehicle,
   none is an operating company's own name.

BSE additionally needs two more checks, since many of its debt instruments
carry cryptic, non-descriptive names (e.g. `773CG2034` named just
`"773CG2034"`) that the name-based rules above can't catch:

6. `_BSE_DEBT_RE` — four regex patterns against the `tradingsymbol` itself:
   `^0` (government securities, `07ABB`), `^SGB` (Sovereign Gold Bonds,
   `SGBOCT26`), `^[\\d.]+[A-Za-z].*\\d\$` (coupon-prefixed NCDs/bonds,
   `001HCCL29`, `8.9JSWSL30`), `^\\d{3,}[A-Za-z].*\\d` (3-digit coupon
   prefix, `813CG2045A`, excludes `360ONE`), and `\\s` (whitespace —
   catches BSE index codes like `BSE CD`).
7. `tick_size == 0` — BSE index instruments (SENSEX, BANKEX, etc.).
8. `_BSE_DEBT_CODE_SYM_RE`/`_BSE_DEBT_CODE_NAME_RE` — a second, narrower
   SDL/G-Sec pattern that rule 6 misses: BSE spells these names with NO
   spaces (`"64GUJSDL30"`, `"69GS2065P"`), so `_DEBT_NAME_RE`'s `\\bSDL\\b`/
   `\\bGOI\\b` word-boundary check never fires (there's no boundary between
   two word characters), and rule 6 also misses most of them — they end
   in a disambiguating letter instead of a digit (`64GJ30A`), or the
   coupon prefix is only 2 digits where rule 6 requires 3+ (`69GS2065P`).
   `_BSE_DEBT_CODE_SYM_RE` (`^\\d{2,}[A-Za-z]+\\d+[A-Za-z]?\$`, 2+ digits,
   letters, 1+ digits, optional trailing letter) catches the tradingsymbol
   form directly (`64GJ30A`, `69GS2065P`, `70AP38A`, `73GS2053P`,
   `77MH33A`, `78GJ32A`, `78TN32A`); `_BSE_DEBT_CODE_NAME_RE` (the same
   digit-prefix/ends-in-digit shape as rule 6 but without its trailing
   `\\s` alternative, applied to `name` instead of `sym`) catches the one
   straggler whose symbol itself ends in letters with no trailing digit
   at all (`717MHSDL`, named `"717MHSDL29"`).

   Verified against the full instrument dump, not just the known junk:
   applying both new regexes changes NSE's kept count by zero (anchored
   on a leading digit, and NSE's digit-leading debt names all already
   have spaces, so rule 2 already catches them) and drops exactly these
   8 BSE rows, nothing else — every digit-leading real company (checked:
   `20MICRONS`, `21STCENMGM`, `360ONE`, `3BBLACKBIO`, `3BFILMS`, `3CIT`,
   `3IINFOLTD`, `3MINDIA`, `3PLAND`, `5PAISA`, `63MOONS`, `7NR`, `7SEASL`,
   `7TEC`) fails both patterns structurally: each either has only a
   single leading digit (`_BSE_DEBT_CODE_SYM_RE` requires 2+) or its name
   has a space immediately after the leading digit run, which neither
   pattern's `[A-Za-z]`/digit-run requirement can cross.

9. `_BSE_GSEC_SYM_RE` — G-Sec/strip/floating-rate-bond codes that start with
   letters, so every digit-prefix rule above misses them: `GS02JAN27C`,
   `GS12DEC2034`, `GS151226C` (government securities, `GS` + maturity date
   + optional cumulative `C`), `CS12DEC35` (coupon strips), `FRBGOI2035`
   (floating-rate bonds) and `GSEC190962`. BSE gives them `EQ` type, tick
   size 0.01 and a `name` identical to the symbol, so nothing else flags
   them. Verified against the full dump: 347 matches, none a company
   (real `GS…` tickers like `GSFC`/`GSPL` have no digit right after `GS`).

Neither of the two BSE-only checks before this one is needed for NSE:
verified NSE has zero `EQ` rows with `tick_size == 0`, and every NSE debt
instrument with a non-empty name already matches `_DEBT_NAME_RE`.
"""
const _BSE_DEBT_RE            = r"^0|^SGB|^[\d.]+[A-Za-z].*\d$|^\d{3,}[A-Za-z].*\d|\s"
const _BSE_DEBT_CODE_SYM_RE   = r"^\d{2,}[A-Za-z]+\d+[A-Za-z]?$"
const _BSE_DEBT_CODE_NAME_RE  = r"^[\d.]+[A-Za-z].*\d$"
const _BSE_GSEC_SYM_RE        = r"^(?:GS|CS)\d|^FRBGOI|^GSEC\d"
const _DEBT_NAME_RE      = r"\bSDL\b|\bGOI\b|TBILL|GOLD\s?BONDS?|\bNCD\b|DEBENTURE|\bBOND\b"i
const _EXCLUDED_SUFFIXES = ("-RR", "-IV", "-E1", "-BE", "-BZ", "-BL", "-ST")
const _TRUST_NAME_RE     = r"\bREIT\b|\bINVIT\b"i
const _FUND_NAME_RE      = r"\bFUND\b"i
const _ETF_NAME_RE       = r"\bETF\b"i

function build_token_map(instruments::DataFrame; exchange::String="NSE")::Dict{String, Int}
    map = Dict{String, Int}()
    for row in eachrow(instruments)
        type = string(get(row, :instrument_type, ""))
        exch = string(get(row, :exchange, ""))
        exch == exchange || continue
        if type == "EQ" || (exchange == "NSE" && type == "INDICES")
            sym = string(row.tradingsymbol)
            if type == "EQ"
                name = strip(string(coalesce(get(row, :name, ""), "")))
                isempty(name)                      && continue
                !isnothing(match(_DEBT_NAME_RE, name))  && continue
                !isnothing(match(_TRUST_NAME_RE, name)) && continue
                !isnothing(match(_FUND_NAME_RE, name))  && continue
                !isnothing(match(_ETF_NAME_RE, name))   && continue
                any(endswith(sym, s) for s in _EXCLUDED_SUFFIXES) && continue
                if exchange == "BSE"
                    !isnothing(match(_BSE_DEBT_RE, sym))           && continue
                    get(row, :tick_size, 1.0) == 0.0               && continue
                    !isnothing(match(_BSE_DEBT_CODE_SYM_RE, sym))  && continue
                    !isnothing(match(_BSE_DEBT_CODE_NAME_RE, name)) && continue
                    !isnothing(match(_BSE_GSEC_SYM_RE, sym))       && continue
                end
            end
            map[sym] = Int(row.instrument_token)
        end
    end
    return map
end

"""
Fetch daily OHLCV candles from Kite for one instrument.

Kite limits daily historical data to 2000 calendar days per request (~8 years).
This function automatically chunks the date range and concatenates the results.

# Arguments
- `token`: Kite instrument_token (integer)
- `from_date`, `to_date`: inclusive date range
- `session`: `KiteSession` (api_key, access_token, repo_root)

# Returns
DataFrame with columns: date, open, high, low, close, volume
Sorted ascending by date. Returns empty DataFrame on failure.
"""
function fetch_ohlcv(token::Int, from_date::Date, to_date::Date, session)::DataFrame
    chunk_days = 1999   # stay under the 2000-day Kite limit
    all_chunks = DataFrame[]

    chunk_start = from_date
    while chunk_start <= to_date
        chunk_end = min(chunk_start + Day(chunk_days), to_date)
        from_s = Dates.format(chunk_start, "yyyy-mm-dd") * "+09:15:00"
        to_s   = Dates.format(chunk_end,   "yyyy-mm-dd") * "+15:30:00"
        url = "$KITE_BASE/instruments/historical/$token/day?from=$from_s&to=$to_s&continuous=0&oi=0"

        resp = try
            _kite_get(url, session; request_timeout=20, status_exception=false)
        catch e
            @warn "Daily fetch error for token $token ($chunk_start…$chunk_end): $(sprint(showerror, e))"
            chunk_start = chunk_end + Day(1)
            sleep(0.35)
            continue
        end

        if resp.status == 200
            raw = try JSON3.read(resp.body) catch; nothing end
            if !isnothing(raw)
                candles_data = get(raw, :data, nothing)
                candles_arr  = isnothing(candles_data) ? nothing :
                               get(candles_data, :candles, nothing)
                if !isnothing(candles_arr) && !isempty(candles_arr)
                    rows = [(
                        date   = Date(string(c[1])[1:10]),
                        open   = Float64(c[2]),
                        high   = Float64(c[3]),
                        low    = Float64(c[4]),
                        close  = Float64(c[5]),
                        volume = Float64(c[6]),
                    ) for c in candles_arr]
                    push!(all_chunks, DataFrame(rows))
                end
            end
        else
            resp.status == 400 ?
                @debug("Daily HTTP 400 for token $token ($chunk_start…$chunk_end)") :
                @warn "Daily HTTP $(resp.status) for token $token ($chunk_start…$chunk_end)"
        end

        chunk_start = chunk_end + Day(1)
        sleep(0.35)
    end

    isempty(all_chunks) && return DataFrame()
    combined = vcat(all_chunks...)
    unique!(combined, :date)
    return sort!(combined, :date)
end

"""
Fetch and save OHLCV for a list of symbols. Skips symbols already cached
unless `refresh=true`. Saves each symbol to `out_dir/{SYMBOL}.csv` — `out_dir`
is expected to already be a granularity-specific directory (e.g.
`website/data/ohlcv/nse/daily/`), not the exchange root.

# Arguments
- `symbols`: NSE tradingsymbols
- `token_map`: from `build_token_map`
- `session`: Kite session
- `out_dir`: granularity-specific directory for CSV output
- `from_date`, `to_date`: date range
- `refresh`: re-fetch even if file exists
"""
function collect_ohlcv(symbols::Vector{String}, token_map::Dict{String,Int},
                       session, out_dir::String,
                       from_date::Date, to_date::Date;
                       refresh::Bool=false, slog::Union{ScriptLog, Nothing}=nothing)
    mkpath(out_dir)
    ok = skipped = failed = 0
    total = length(symbols)
    t_start = time()

    for (i, sym) in enumerate(symbols)
        path = joinpath(out_dir, "$(sym).csv")
        if !refresh && isfile(path)
            skipped += 1
        else
            token = get(token_map, sym, nothing)
            if isnothing(token)
                _log_summary(slog, "[$i/$total] No instrument token for $sym — skipping"; warn=true)
                failed += 1
            else
                df = fetch_ohlcv(token, from_date, to_date, session)
                if isempty(df)
                    failed += 1
                    _log_detail(slog, "[$i/$total] $sym — 0 bars")
                else
                    CSV.write(path, df)
                    ok += 1
                    _log_detail(slog, "[$i/$total] $sym — $(nrow(df)) days")
                end
                sleep(0.35)   # ~3 req/s rate limit
            end
        end

        if i % 100 == 0 || i == total
            elapsed = round(Int, time() - t_start)
            _log_summary(slog, "  ── [$i/$total] $ok fetched, $skipped skipped, $failed failed — $(elapsed)s elapsed")
        end
    end

    _log_summary(slog, "Daily OHLCV done: $ok fetched, $skipped skipped, $failed failed")
end

"""
Load a cached OHLCV CSV for a symbol. Returns empty DataFrame if not found.
"""
function load_cached_ohlcv(symbol::String, out_dir::String)::DataFrame
    path = joinpath(out_dir, "$(symbol).csv")
    isfile(path) || return DataFrame()
    return CSV.read(path, DataFrame; types=Dict(:date => Date))
end

"""
Map a Tijori sector string to the closest NSE sector index name.
Returns "NIFTY 50" as the default when no specific mapping exists.
"""
function sector_index_name(sector::String)::String
    get(SECTOR_TO_INDEX, sector, "NIFTY 50")
end

# ── Hourly (60-minute) OHLCV ─────────────────────────────────────────────────

"""
Fetch 60-minute OHLCV candles from Kite for one instrument.

Kite limits intraday historical data to 60-day windows per request. This
function automatically chunks the date range and concatenates the results.

# Returns
DataFrame with columns: datetime, open, high, low, close, volume.
Sorted ascending by datetime. Returns empty DataFrame on failure.
"""
function fetch_ohlcv_hourly(token::Int, from_date::Date, to_date::Date,
                             session)::DataFrame
    chunk_days = 59   # stay under the 60-day Kite limit
    all_chunks = DataFrame[]

    chunk_start = from_date
    while chunk_start <= to_date
        chunk_end = min(chunk_start + Day(chunk_days), to_date)
        from_s = Dates.format(chunk_start, "yyyy-mm-dd") * "+09:00:00"
        to_s   = Dates.format(chunk_end,   "yyyy-mm-dd") * "+15:30:00"
        url = "$KITE_BASE/instruments/historical/$token/60minute?from=$from_s&to=$to_s&continuous=0&oi=0"

        resp = try
            _kite_get(url, session; request_timeout=30, status_exception=false)
        catch e
            @warn "Hourly fetch error for token $token ($chunk_start…$chunk_end): $(sprint(showerror, e))"
            chunk_start = chunk_end + Day(1)
            sleep(0.35)
            continue
        end

        if resp.status == 200
            raw = try JSON3.read(resp.body) catch; nothing end
            if !isnothing(raw)
                candles_data = get(raw, :data, nothing)
                candles_arr  = isnothing(candles_data) ? nothing :
                               get(candles_data, :candles, nothing)
                if !isnothing(candles_arr) && !isempty(candles_arr)
                    rows = [(
                        datetime = DateTime(string(c[1])[1:19], "yyyy-mm-ddTHH:MM:SS"),
                        open     = Float64(c[2]),
                        high     = Float64(c[3]),
                        low      = Float64(c[4]),
                        close    = Float64(c[5]),
                        volume   = Float64(c[6]),
                    ) for c in candles_arr]
                    push!(all_chunks, DataFrame(rows))
                end
            end
        else
            resp.status == 400 ?
                @debug("Hourly HTTP 400 for token $token ($chunk_start…$chunk_end)") :
                @warn "Hourly HTTP $(resp.status) for token $token ($chunk_start…$chunk_end)"
        end

        chunk_start = chunk_end + Day(1)
        sleep(0.35)
    end

    isempty(all_chunks) && return DataFrame()
    return sort!(vcat(all_chunks...), :datetime)
end

"""
Fetch and cache 60-minute OHLCV for a list of symbols. Output:
`out_dir/{SYMBOL}.csv` — `out_dir` is expected to already be a
granularity-specific directory (e.g. `website/data/ohlcv/nse/hourly/`), not
the exchange root.
"""
function collect_ohlcv_hourly(symbols::Vector{String}, token_map::Dict{String,Int},
                               session, out_dir::String,
                               from_date::Date, to_date::Date;
                               refresh::Bool=false, slog::Union{ScriptLog, Nothing}=nothing)
    mkpath(out_dir)
    ok = skipped = failed = 0
    total = length(symbols)
    t_start = time()

    for (i, sym) in enumerate(symbols)
        path = joinpath(out_dir, "$(sym).csv")
        if !refresh && isfile(path)
            skipped += 1
        else
            token = get(token_map, sym, nothing)
            if isnothing(token)
                _log_summary(slog, "[$i/$total] No token for $sym — skipping hourly"; warn=true)
                failed += 1
            else
                df = fetch_ohlcv_hourly(token, from_date, to_date, session)
                if isempty(df)
                    failed += 1
                    _log_detail(slog, "[$i/$total] $sym hourly — 0 bars")
                else
                    CSV.write(path, df)
                    ok += 1
                    _log_detail(slog, "[$i/$total] $sym hourly — $(nrow(df)) bars")
                end
            end
        end

        if i % 100 == 0 || i == total
            elapsed = round(Int, time() - t_start)
            _log_summary(slog, "  ── [$i/$total] $ok fetched, $skipped skipped, $failed failed — $(elapsed)s elapsed")
        end
    end

    _log_summary(slog, "Hourly OHLCV done: $ok fetched, $skipped skipped, $failed failed")
end

"""
Load cached 60-minute OHLCV for a symbol. Returns empty DataFrame if not found.
"""
function load_cached_ohlcv_hourly(symbol::String, out_dir::String)::DataFrame
    path = joinpath(out_dir, "$(symbol).csv")
    isfile(path) || return DataFrame()
    return CSV.read(path, DataFrame; types=Dict(:datetime => DateTime))
end

# ── 5-minute OHLCV ───────────────────────────────────────────────────────────

"""
Fetch 5-minute OHLCV candles from Kite for one instrument.

Kite caps a single historical-data request to a 100-day span for this
interval — not a total retention limit (verified live: 5-minute candles from
2021 still return real data today). Requests are chunked into 90-day windows
so an arbitrarily old `from_date` is walked back correctly, the same way
`fetch_ohlcv`'s daily chunking already does for its own (2000-day) span cap.

# Returns
DataFrame with columns: datetime, open, high, low, close, volume.
Sorted ascending by datetime. Returns empty DataFrame on failure.
"""
function fetch_ohlcv_5min(token::Int, from_date::Date, to_date::Date,
                           session)::DataFrame
    chunk_days = 90
    all_chunks = DataFrame[]

    chunk_start = from_date
    while chunk_start <= to_date
        chunk_end = min(chunk_start + Day(chunk_days), to_date)
        from_s = Dates.format(chunk_start, "yyyy-mm-dd") * "+09:15:00"
        to_s   = Dates.format(chunk_end,   "yyyy-mm-dd") * "+15:30:00"
        url = "$KITE_BASE/instruments/historical/$token/5minute" *
              "?from=$from_s&to=$to_s&continuous=0&oi=0"

        resp = try
            _kite_get(url, session; request_timeout=30, status_exception=false)
        catch e
            @warn "5min fetch error for token $token ($chunk_start…$chunk_end): $(sprint(showerror, e))"
            chunk_start = chunk_end + Day(1); sleep(0.35); continue
        end

        if resp.status == 200
            raw = try JSON3.read(resp.body) catch; nothing end
            if !isnothing(raw)
                cd   = get(raw, :data,    nothing)
                carr = isnothing(cd) ? nothing : get(cd, :candles, nothing)
                if !isnothing(carr) && !isempty(carr)
                    rows = [(
                        datetime = DateTime(string(c[1])[1:19], "yyyy-mm-ddTHH:MM:SS"),
                        open     = Float64(c[2]),
                        high     = Float64(c[3]),
                        low      = Float64(c[4]),
                        close    = Float64(c[5]),
                        volume   = Float64(c[6]),
                    ) for c in carr]
                    push!(all_chunks, DataFrame(rows))
                end
            end
        else
            resp.status == 400 ?
                @debug("5min HTTP 400 for token $token ($chunk_start…$chunk_end)") :
                @warn "5min HTTP $(resp.status) for token $token ($chunk_start…$chunk_end)"
        end

        chunk_start = chunk_end + Day(1)
        sleep(0.35)
    end

    isempty(all_chunks) && return DataFrame()
    return sort!(vcat(all_chunks...), :datetime)
end

"""
Fetch and cache 5-minute OHLCV for a list of symbols. Output:
`out_dir/{SYMBOL}.csv` — `out_dir` is expected to already be a
granularity-specific directory (e.g. `website/data/ohlcv/nse/5min/`), not
the exchange root.
"""
function collect_ohlcv_5min(symbols::Vector{String}, token_map::Dict{String,Int},
                              session, out_dir::String,
                              from_date::Date, to_date::Date;
                              refresh::Bool=false, slog::Union{ScriptLog, Nothing}=nothing)
    mkpath(out_dir)
    ok = skipped = failed = 0
    total = length(symbols)
    t_start = time()

    for (i, sym) in enumerate(symbols)
        path = joinpath(out_dir, "$(sym).csv")
        if !refresh && isfile(path)
            skipped += 1
        else
            token = get(token_map, sym, nothing)
            if isnothing(token)
                _log_summary(slog, "[$i/$total] No token for $sym — skipping 5min"; warn=true)
                failed += 1
            else
                df = fetch_ohlcv_5min(token, from_date, to_date, session)
                if isempty(df)
                    failed += 1
                    _log_detail(slog, "[$i/$total] $sym 5min — 0 bars")
                else
                    CSV.write(path, df)
                    ok += 1
                    _log_detail(slog, "[$i/$total] $sym 5min — $(nrow(df)) bars")
                end
            end
        end

        if i % 100 == 0 || i == total
            elapsed = round(Int, time() - t_start)
            _log_summary(slog, "  ── [$i/$total] $ok fetched, $skipped skipped, $failed failed — $(elapsed)s elapsed")
        end
    end

    _log_summary(slog, "5min OHLCV done: $ok fetched, $skipped skipped, $failed failed")
end

"""
Load cached 5-minute OHLCV for a symbol. Returns empty DataFrame if not found.
"""
function load_cached_ohlcv_5min(symbol::String, out_dir::String)::DataFrame
    path = joinpath(out_dir, "$(symbol).csv")
    isfile(path) || return DataFrame()
    return CSV.read(path, DataFrame; types=Dict(:datetime => DateTime))
end

# ── 15-minute bars ────────────────────────────────────────────────────────────

"""
Fetch 15-minute OHLCV bars for a single NSE instrument token.

Kite caps a single historical-data request to a 200-day span for this
interval — not a total retention limit (verified live: 15-minute candles
from 2019 still return real data today). Requests are chunked into 175-day
windows so an arbitrarily old `from_date` is walked back correctly.

# Arguments
- `token`: Kite instrument token
- `from_date`, `to_date`: inclusive date range
- `session`: Kite session

# Returns
DataFrame with columns: datetime, open, high, low, close, volume.
Sorted ascending by datetime. Returns empty DataFrame on failure.
"""
function fetch_ohlcv_15min(token::Int, from_date::Date, to_date::Date,
                            session)::DataFrame
    chunk_days = 175
    all_chunks = DataFrame[]

    chunk_start = from_date
    while chunk_start <= to_date
        chunk_end = min(chunk_start + Day(chunk_days), to_date)
        from_s = Dates.format(chunk_start, "yyyy-mm-dd") * "+09:15:00"
        to_s   = Dates.format(chunk_end,   "yyyy-mm-dd") * "+15:30:00"
        url = "$KITE_BASE/instruments/historical/$token/15minute" *
              "?from=$from_s&to=$to_s&continuous=0&oi=0"

        resp = try
            _kite_get(url, session; request_timeout=30, status_exception=false)
        catch e
            @warn "15min fetch error for token $token ($chunk_start…$chunk_end): $(sprint(showerror, e))"
            chunk_start = chunk_end + Day(1); sleep(0.35); continue
        end

        if resp.status == 200
            raw = try JSON3.read(resp.body) catch; nothing end
            if !isnothing(raw)
                cd   = get(raw, :data,    nothing)
                carr = isnothing(cd) ? nothing : get(cd, :candles, nothing)
                if !isnothing(carr) && !isempty(carr)
                    rows = [(
                        datetime = DateTime(string(c[1])[1:19], "yyyy-mm-ddTHH:MM:SS"),
                        open     = Float64(c[2]),
                        high     = Float64(c[3]),
                        low      = Float64(c[4]),
                        close    = Float64(c[5]),
                        volume   = Float64(c[6]),
                    ) for c in carr]
                    push!(all_chunks, DataFrame(rows))
                end
            end
        else
            resp.status == 400 ?
                @debug("15min HTTP 400 for token $token ($chunk_start…$chunk_end)") :
                @warn "15min HTTP $(resp.status) for token $token ($chunk_start…$chunk_end)"
        end

        chunk_start = chunk_end + Day(1)
        sleep(0.35)
    end

    isempty(all_chunks) && return DataFrame()
    return sort!(vcat(all_chunks...), :datetime)
end

"""
Fetch and cache 15-minute OHLCV for a list of symbols. Output:
`out_dir/{SYMBOL}.csv` — `out_dir` is expected to already be a
granularity-specific directory (e.g. `website/data/ohlcv/nse/15min/`), not
the exchange root.
"""
function collect_ohlcv_15min(symbols::Vector{String}, token_map::Dict{String,Int},
                               session, out_dir::String,
                               from_date::Date, to_date::Date;
                               refresh::Bool=false, slog::Union{ScriptLog, Nothing}=nothing)
    mkpath(out_dir)
    ok = skipped = failed = 0
    total = length(symbols)
    t_start = time()

    for (i, sym) in enumerate(symbols)
        path = joinpath(out_dir, "$(sym).csv")
        if !refresh && isfile(path)
            skipped += 1
        else
            token = get(token_map, sym, nothing)
            if isnothing(token)
                _log_summary(slog, "[$i/$total] No token for $sym — skipping 15min"; warn=true)
                failed += 1
            else
                df = fetch_ohlcv_15min(token, from_date, to_date, session)
                if isempty(df)
                    failed += 1
                    _log_detail(slog, "[$i/$total] $sym 15min — 0 bars")
                else
                    CSV.write(path, df)
                    ok += 1
                    _log_detail(slog, "[$i/$total] $sym 15min — $(nrow(df)) bars")
                end
            end
        end

        if i % 100 == 0 || i == total
            elapsed = round(Int, time() - t_start)
            _log_summary(slog, "  ── [$i/$total] $ok fetched, $skipped skipped, $failed failed — $(elapsed)s elapsed")
        end
    end

    _log_summary(slog, "15min OHLCV done: $ok fetched, $skipped skipped, $failed failed")
end

"""
Load cached 15-minute OHLCV for a symbol. Returns empty DataFrame if not found.
"""
function load_cached_ohlcv_15min(symbol::String, out_dir::String)::DataFrame
    path = joinpath(out_dir, "$(symbol).csv")
    isfile(path) || return DataFrame()
    return CSV.read(path, DataFrame; types=Dict(:datetime => DateTime))
end

# ── 1-minute OHLCV ───────────────────────────────────────────────────────────

"""
Fetch 1-minute OHLCV for `token` over a short, EXACT datetime window
(`from_dt`/`to_dt`, to-the-minute — not whole trading days like
`fetch_ohlcv_1min`). One single Kite historical-data call, no chunking —
intended for a window of at most a couple of hours (e.g. `fetch_news_
snapshot_ohlcv.jl`'s "just the few minutes around a news timestamp" use
case), never a multi-day range.

# Returns
DataFrame with columns: datetime, open, high, low, close, volume. Sorted
ascending by datetime. Returns empty DataFrame on failure or if the window
contains no bars (e.g. outside market hours).
"""
function fetch_ohlcv_1min_window(token::Int, from_dt::DateTime, to_dt::DateTime, session)::DataFrame
    from_s = Dates.format(from_dt, "yyyy-mm-dd") * "+" * Dates.format(from_dt, "HH:MM:SS")
    to_s   = Dates.format(to_dt,   "yyyy-mm-dd") * "+" * Dates.format(to_dt,   "HH:MM:SS")
    url = "$KITE_BASE/instruments/historical/$token/minute" *
          "?from=$from_s&to=$to_s&continuous=0&oi=0"

    resp = try
        _kite_get(url, session; request_timeout=30, status_exception=false)
    catch e
        @warn "1min window fetch error for token $token ($from_dt…$to_dt): $(sprint(showerror, e))"
        return DataFrame()
    end

    if resp.status != 200
        resp.status == 400 ?
            @debug("1min window HTTP 400 for token $token ($from_dt…$to_dt)") :
            @warn "1min window HTTP $(resp.status) for token $token ($from_dt…$to_dt)"
        return DataFrame()
    end

    raw = try JSON3.read(resp.body) catch; nothing end
    isnothing(raw) && return DataFrame()
    cd   = get(raw, :data, nothing)
    carr = isnothing(cd) ? nothing : get(cd, :candles, nothing)
    (isnothing(carr) || isempty(carr)) && return DataFrame()

    rows = [(
        datetime = DateTime(string(c[1])[1:19], "yyyy-mm-ddTHH:MM:SS"),
        open     = Float64(c[2]),
        high     = Float64(c[3]),
        low      = Float64(c[4]),
        close    = Float64(c[5]),
        volume   = Float64(c[6]),
    ) for c in carr]
    return sort!(DataFrame(rows), :datetime)
end

"""
Fetch 1-minute OHLCV candles from Kite for one instrument.

Kite caps a single historical-data request to a 60-day span for this
interval — the shortest per-request cap of any interval here (5-minute: 100
days, 15-minute: 200 days, 60-minute: 400 days, day: 2000 days), but NOT a
total retention limit: verified live against the real API that 1-minute
candles from 2022 still return real data today. Requests are chunked into
55-day windows so an arbitrarily old `from_date` is walked back correctly,
same as every other interval here.

# Returns
DataFrame with columns: datetime, open, high, low, close, volume.
Sorted ascending by datetime. Returns empty DataFrame on failure.
"""
function fetch_ohlcv_1min(token::Int, from_date::Date, to_date::Date,
                           session)::DataFrame
    chunk_days = 55
    all_chunks = DataFrame[]

    chunk_start = from_date
    while chunk_start <= to_date
        chunk_end = min(chunk_start + Day(chunk_days), to_date)
        from_s = Dates.format(chunk_start, "yyyy-mm-dd") * "+09:15:00"
        to_s   = Dates.format(chunk_end,   "yyyy-mm-dd") * "+15:30:00"
        url = "$KITE_BASE/instruments/historical/$token/minute" *
              "?from=$from_s&to=$to_s&continuous=0&oi=0"

        resp = try
            _kite_get(url, session; request_timeout=30, status_exception=false)
        catch e
            @warn "1min fetch error for token $token ($chunk_start…$chunk_end): $(sprint(showerror, e))"
            chunk_start = chunk_end + Day(1); sleep(0.35); continue
        end

        if resp.status == 200
            raw = try JSON3.read(resp.body) catch; nothing end
            if !isnothing(raw)
                cd   = get(raw, :data,    nothing)
                carr = isnothing(cd) ? nothing : get(cd, :candles, nothing)
                if !isnothing(carr) && !isempty(carr)
                    rows = [(
                        datetime = DateTime(string(c[1])[1:19], "yyyy-mm-ddTHH:MM:SS"),
                        open     = Float64(c[2]),
                        high     = Float64(c[3]),
                        low      = Float64(c[4]),
                        close    = Float64(c[5]),
                        volume   = Float64(c[6]),
                    ) for c in carr]
                    push!(all_chunks, DataFrame(rows))
                end
            end
        else
            resp.status == 400 ?
                @debug("1min HTTP 400 for token $token ($chunk_start…$chunk_end)") :
                @warn "1min HTTP $(resp.status) for token $token ($chunk_start…$chunk_end)"
        end

        chunk_start = chunk_end + Day(1)
        sleep(0.35)
    end

    isempty(all_chunks) && return DataFrame()
    return sort!(vcat(all_chunks...), :datetime)
end

"""
Fetch and cache 1-minute OHLCV for a list of symbols. Output:
`out_dir/{SYMBOL}.csv` — `out_dir` is expected to already be a
granularity-specific directory (e.g. `website/data/ohlcv/nse/1min/`), not
the exchange root.
"""
function collect_ohlcv_1min(symbols::Vector{String}, token_map::Dict{String,Int},
                             session, out_dir::String,
                             from_date::Date, to_date::Date;
                             refresh::Bool=false, slog::Union{ScriptLog, Nothing}=nothing)
    mkpath(out_dir)
    ok = skipped = failed = 0
    total = length(symbols)
    t_start = time()

    for (i, sym) in enumerate(symbols)
        path = joinpath(out_dir, "$(sym).csv")
        if !refresh && isfile(path)
            skipped += 1
        else
            token = get(token_map, sym, nothing)
            if isnothing(token)
                _log_summary(slog, "[$i/$total] No token for $sym — skipping 1min"; warn=true)
                failed += 1
            else
                df = fetch_ohlcv_1min(token, from_date, to_date, session)
                if isempty(df)
                    failed += 1
                    _log_detail(slog, "[$i/$total] $sym 1min — 0 bars")
                else
                    CSV.write(path, df)
                    ok += 1
                    _log_detail(slog, "[$i/$total] $sym 1min — $(nrow(df)) bars")
                end
            end
        end

        if i % 100 == 0 || i == total
            elapsed = round(Int, time() - t_start)
            _log_summary(slog, "  ── [$i/$total] $ok fetched, $skipped skipped, $failed failed — $(elapsed)s elapsed")
        end
    end

    _log_summary(slog, "1min OHLCV done: $ok fetched, $skipped skipped, $failed failed")
end

"""
Load cached 1-minute OHLCV for a symbol. Returns empty DataFrame if not found.
"""
function load_cached_ohlcv_1min(symbol::String, out_dir::String)::DataFrame
    path = joinpath(out_dir, "$(symbol).csv")
    isfile(path) || return DataFrame()
    return CSV.read(path, DataFrame; types=Dict(:datetime => DateTime))
end
