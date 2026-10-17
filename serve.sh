#!/usr/bin/env bash
# Serve website/ with the ops server (scripts/tools/ops_server.jl) and open a page in the browser.
# Besides the static pages it provides /api: the Stop and Rescan buttons and the Company series plot read from it.
# By default the server is reachable from other machines (open http://THIS-MACHINE-IP:PORT/...); the Stop and Rescan
# buttons then work only on a page opened as localhost. There is no login, so use --local on an untrusted network.
# Run from the repo root.
#
# Usage:
#   ./serve.sh                              # opens website/index.html on port 8080
#   ./serve.sh website/setup.html           # open a specific page
#   ./serve.sh website/companies.html 9000  # custom port
#   ./serve.sh --local                      # this machine only (127.0.0.1)
#   ./serve.sh --allow-actions              # other machines may also use Stop / Rescan
#   (options can go anywhere among the arguments)

SERVER_ARGS=()
POSITIONAL=()
for a in "$@"; do
    case "$a" in
        --local) SERVER_ARGS+=(--local) ;;
        --allow-actions) SERVER_ARGS+=(--allow-remote-actions) ;;
        *) POSITIONAL+=("$a") ;;
    esac
done

FILE=${POSITIONAL[0]:-website/index.html}
PORT=${POSITIONAL[1]:-8080}

exec julia --project=packages/StockSwingPredictor scripts/tools/ops_server.jl --port "$PORT" --open "$FILE" "${SERVER_ARGS[@]}"
