"""`Base.show` overrides for TradingGame types."""

function Base.show(io::IO, p::Portfolio)
    reserved = isempty(p.reserved) ? 0.0 : sum(l.amount for l in p.reserved)
    print(io, "Portfolio(cash=", round(p.cash, digits=2),
              ", reserved=", round(reserved, digits=2),
              ", holdings=", length(p.holdings), ")")
end

function Base.show(io::IO, env::TradingGameEnv)
    if env.config === nothing
        print(io, "TradingGameEnv(not reset)")
        return
    end
    print(io, "TradingGameEnv(date=", env.current_date,
              ", value=", round(portfolio_value(env), digits=2),
              ", ", env.portfolio, ")")
end

function Base.show(io::IO, r::StepResult)
    print(io, "StepResult(reward=", round(r.reward, digits=6),
              ", done=", r.done,
              ", value=", round(get(r.info, "portfolio_value", NaN), digits=2), ")")
end
