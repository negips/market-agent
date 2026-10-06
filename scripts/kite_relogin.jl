"""
kite_relogin.jl

Force a fresh Kite Connect login from Julia — a thin wrapper around
`StockSwingPredictor.relogin_kite!`, which shells out to `sidecar/
kite_login.js` (Playwright-driven; Zerodha's login page has no plain REST
endpoint to re-authenticate against directly, see that script's own
docstring for the TOTP + OAuth redirect flow) and reloads the resulting
`kite_session.json`.

This is the SAME machinery every `fetch_ohlcv*`/`load_instruments`/broker/
macro function now calls automatically: a Kite historical-data/portfolio
call that gets a 403 (expired token) triggers exactly one re-authenticate-
and-retry, in place, via `KiteSession` being mutable (see `_kite_get` in
`kite_data.jl`) — so a long-running overnight job that spans the token's
daily expiry boundary recovers on its own mid-run, with no script-level
changes needed anywhere (`collect_nse_ohlcv.jl`, `update_ohlcv.jl`,
`backfill_ohlcv.jl`, `fetch_news_snapshot_ohlcv.jl`, etc. — they all just
pass the same `session` object through every call, same as before).

Run this script when you want to force a refresh PROACTIVELY instead —
e.g. right before kicking off a job you know will run long enough to span
the expiry boundary, so it starts the run already on a fresh token rather
than relying on hitting one 403 first.

Requires a display (launches a real, visible browser) and the credentials
in `.env` (KITE_HISTORICAL_API_KEY/SECRET, KITE_USER_ID, KITE_PASSWORD,
KITE_TOTP_SECRET) — same requirements as running `node sidecar/
kite_login.js` directly.

Usage:
  julia --project=packages/StockSwingPredictor scripts/kite_relogin.jl
"""

using StockSwingPredictor

const REPO_ROOT = joinpath(@__DIR__, "..")

function main()
    @info "Re-authenticating with Kite (launches a browser)…"
    session = relogin_kite!(REPO_ROOT)
    @info "Kite session refreshed — api_key=$(first(session.api_key, 6))… " *
          "token=$(first(session.access_token, 6))…"
end

main()
