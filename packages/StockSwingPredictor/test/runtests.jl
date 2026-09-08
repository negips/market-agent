"""
StockSwingPredictor test suite.

Fast unit/shape tests run unconditionally.
Integration tests (real OHLCV data) run only when the env var is set:

  SSP_DATA_DIR=/path/to/website/data julia --project=. test/runtests.jl
"""

using Test, StockSwingPredictor, Flux, DataFrames, Dates, Statistics

include("test_features.jl")
include("test_model.jl")
include("test_dataset.jl")

# ── Integration tests (skipped when SSP_DATA_DIR is unset) ───────────────────

data_dir = get(ENV, "SSP_DATA_DIR", "")
if !isempty(data_dir)
    include("test_integration.jl")
else
    @info "Integration tests skipped — set SSP_DATA_DIR to enable."
end
