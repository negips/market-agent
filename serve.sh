#!/usr/bin/env bash
# Start a local HTTP server and open a page in the browser.
# Run from the repo root.
#
# Usage:
#   ./serve.sh                        # opens nse_companies.html on port 8080
#   ./serve.sh SETUP.html             # opens SETUP.html on port 8080
#   ./serve.sh nse_companies.html 9000  # custom port

FILE=${1:-nse_companies.html}
PORT=${2:-8080}
URL="http://localhost:$PORT/$FILE"

echo "Serving on $URL  (Ctrl+C to stop)"
xdg-open "$URL" 2>/dev/null || open "$URL" 2>/dev/null || echo "Open $URL in your browser"

python3 -m http.server "$PORT"
