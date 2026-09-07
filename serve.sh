#!/usr/bin/env bash
# Start a local HTTP server for nse_companies.html and open the browser.
# Run from the repo root: ./serve.sh

PORT=${1:-8080}
URL="http://localhost:$PORT/nse_companies.html"

echo "Serving on $URL  (Ctrl+C to stop)"
xdg-open "$URL" 2>/dev/null || open "$URL" 2>/dev/null || echo "Open $URL in your browser"

python3 -m http.server "$PORT"
