# Build the TradingGame API + architecture docs.
#
#   julia --project=packages/TradingGame/docs packages/TradingGame/docs/make.jl
#
# Output: packages/TradingGame/docs/build/index.html (gitignored). The API
# reference is generated from the package's docstrings, so it can't drift from
# the code; docs/src/*.md only adds the narrative around it. The LaTeX write-ups
# in this folder (tradinggame_scaling.tex/.pdf) are separate and untouched.

using Documenter, TradingGame, StockSwingPredictor

DocMeta.setdocmeta!(TradingGame, :DocTestSetup, :(using TradingGame); recursive=true)

makedocs(;
    sitename = "TradingGame.jl",
    modules  = [TradingGame],
    authors  = "Trading Agent Project",
    format   = Documenter.HTML(; prettyurls = false, size_threshold = nothing,
                                 collapselevel = 1, assets = String[]),
    pages = [
        "Home"                    => "index.md",
        "Rules and constants"     => "rules.md",
        "A decision step"         => "decision_step.md",
        "API" => [
            "Simulator"           => "simulator.md",
            "Observation & policy" => "observation_policy.md",
            "Training"            => "training.md",
            "Universe & windows"  => "universe.md",
        ],
        "Related packages"        => "related.md",
    ],
    checkdocs = :exports,
    warnonly  = [:missing_docs],   # exported names without a docstring are listed as warnings, not build failures
)
