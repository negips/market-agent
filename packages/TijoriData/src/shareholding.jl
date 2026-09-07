"""
Shareholding pattern data.

Returns 10 quarters of promoter, FII, DII, and public holding percentages.
The "Promoter Pledged" column (when present) is the key signal for the
company confidence module's promoter risk check.
"""

"""
    get_shareholding(slug) -> DataFrame

Fetch 10 quarters of shareholding pattern for `slug`.

Returns a DataFrame with:
- `period`           — quarter label (e.g. "Dec 24", "Sep 24")
- `Promoter`         — promoter + promoter group holding %
- `Promoter Pledged` — % of promoter shares pledged (key fraud signal)
- `FII`              — foreign institutional investor holding %
- `DII`              — domestic institutional investor holding %
- `Public`           — public / retail holding %
- Additional category columns if present in the Tijori table

Quarters are ordered oldest-first. Use `reverse(df)` for newest-first.

# Example
```julia
sh = get_shareholding("yes-bank-limited")
# Check promoter pledge trend
select(sh, :period, "Promoter", "Promoter Pledged")
```
"""
function get_shareholding(slug::String)::DataFrame
    raw = _get("/shareholding"; slug=slug)
    df = _shareholding_to_df(raw)
    isempty(df) && @warn "No shareholding data returned for $slug"
    return df
end
