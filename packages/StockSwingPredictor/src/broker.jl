"""
Kite Connect broker endpoints: portfolio, positions, funds, orders.

All functions accept a `session` named tuple from `load_kite_session` and return
DataFrames or plain Dicts. The same session token used for historical data works
for all Kite Connect REST endpoints.
"""

using HTTP, JSON3, DataFrames, Dates

# ── Holdings ──────────────────────────────────────────────────────────────────

"""
Fetch long-term demat holdings from Kite.

# Arguments
- `session`: named tuple from `load_kite_session`

# Returns
DataFrame with one row per holding:
  symbol, exchange, isin, quantity, average_price,
  last_price, close_price, pnl, day_change, day_change_pct
"""
function get_holdings(session)::DataFrame
    resp = HTTP.get("$KITE_BASE/portfolio/holdings",
                    _kite_headers(session); status_exception=false)
    resp.status == 200 || error("Holdings endpoint returned HTTP $(resp.status): $(String(resp.body))")
    data = JSON3.read(String(resp.body))[:data]
    isempty(data) && return DataFrame()

    DataFrame(
        symbol        = [String(h[:tradingsymbol])          for h in data],
        exchange      = [String(h[:exchange])               for h in data],
        isin          = [String(h[:isin])                   for h in data],
        quantity      = [Int(h[:quantity])                  for h in data],
        average_price = [Float64(h[:average_price])         for h in data],
        last_price    = [Float64(h[:last_price])            for h in data],
        close_price   = [Float64(h[:close_price])           for h in data],
        pnl           = [Float64(h[:pnl])                   for h in data],
        day_change    = [Float64(h[:day_change])            for h in data],
        day_change_pct= [Float64(h[:day_change_percentage]) for h in data],
    )
end

# ── Positions ─────────────────────────────────────────────────────────────────

"""
Fetch open intraday and overnight positions from Kite.

# Arguments
- `session`: named tuple from `load_kite_session`
- `kind`: `:net` (default) for combined view, `:day` for intraday only

# Returns
DataFrame with one row per open position:
  symbol, exchange, product, quantity, average_price,
  last_price, pnl, day_change, day_change_pct
Returns empty DataFrame when no positions are open.
"""
function get_positions(session; kind::Symbol=:net)::DataFrame
    resp = HTTP.get("$KITE_BASE/portfolio/positions",
                    _kite_headers(session); status_exception=false)
    resp.status == 200 || error("Positions endpoint returned HTTP $(resp.status): $(String(resp.body))")
    body = JSON3.read(String(resp.body))[:data]
    data = kind == :day ? body[:day] : body[:net]
    isempty(data) && return DataFrame()

    DataFrame(
        symbol        = [String(p[:tradingsymbol])          for p in data],
        exchange      = [String(p[:exchange])               for p in data],
        product       = [String(p[:product])                for p in data],
        quantity      = [Int(p[:quantity])                  for p in data],
        average_price = [Float64(p[:average_price])         for p in data],
        last_price    = [Float64(p[:last_price])            for p in data],
        pnl           = [Float64(p[:pnl])                  for p in data],
        day_change    = [Float64(p[:day_change])            for p in data],
        day_change_pct= [Float64(p[:day_change_percentage]) for p in data],
    )
end

# ── Margins / Funds ───────────────────────────────────────────────────────────

"""
Fetch available funds and margin utilisation from Kite.

# Arguments
- `session`: named tuple from `load_kite_session`
- `segment`: `:equity` (default) or `:commodity`

# Returns
NamedTuple with:
  net              — total available funds (₹)
  cash             — uninvested cash (₹)
  opening_balance  — balance at start of day (₹)
  live_balance     — current balance including intraday credits (₹)
  collateral       — collateral value (₹)
  debits           — total margin utilised (₹)
"""
function get_margins(session; segment::Symbol=:equity)
    seg = segment == :commodity ? "commodity" : "equity"
    resp = HTTP.get("$KITE_BASE/user/margins/$seg",
                    _kite_headers(session); status_exception=false)
    resp.status == 200 || error("Margins endpoint returned HTTP $(resp.status): $(String(resp.body))")
    d = JSON3.read(String(resp.body))[:data]
    av = d[:available]
    ut = d[:utilised]
    return (
        net             = Float64(d[:net]),
        cash            = Float64(av[:cash]),
        opening_balance = Float64(av[:opening_balance]),
        live_balance    = Float64(av[:live_balance]),
        collateral      = Float64(av[:collateral]),
        debits          = Float64(ut[:debits]),
    )
end

# ── Orders ────────────────────────────────────────────────────────────────────

"""
Fetch today's order book from Kite.

# Arguments
- `session`: named tuple from `load_kite_session`

# Returns
DataFrame with one row per order:
  order_id, symbol, exchange, transaction_type, product,
  quantity, price, status, filled_quantity, placed_at
Returns empty DataFrame when no orders have been placed today.
"""
function get_orders(session)::DataFrame
    resp = HTTP.get("$KITE_BASE/orders",
                    _kite_headers(session); status_exception=false)
    resp.status == 200 || error("Orders endpoint returned HTTP $(resp.status): $(String(resp.body))")
    data = JSON3.read(String(resp.body))[:data]
    isempty(data) && return DataFrame()

    DataFrame(
        order_id         = [String(o[:order_id])          for o in data],
        symbol           = [String(o[:tradingsymbol])     for o in data],
        exchange         = [String(o[:exchange])          for o in data],
        transaction_type = [String(o[:transaction_type])  for o in data],
        product          = [String(o[:product])           for o in data],
        quantity         = [Int(o[:quantity])              for o in data],
        price            = [Float64(o[:price])            for o in data],
        status           = [String(o[:status])            for o in data],
        filled_quantity  = [Int(o[:filled_quantity])      for o in data],
        placed_at        = [string(o[:order_timestamp])   for o in data],
    )
end
