#!/usr/bin/env bash
# Serve website/ with the ops server (scripts/tools/ops_server.jl) and open a page in the browser.
# Besides the static pages it provides /api: the Stop and Rescan buttons and the Company series plot read from it.
# Run from the repo root.
#
# Usage:
#   ./serve.sh                              # opens website/index.html on port 8080
#   ./serve.sh website/setup.html           # open a specific page
#   ./serve.sh website/companies.html 9000  # custom port

FILE=${1:-website/index.html}
PORT=${2:-8080}

exec julia --project=packages/StockSwingPredictor scripts/tools/ops_server.jl --port "$PORT" --open "$FILE"
