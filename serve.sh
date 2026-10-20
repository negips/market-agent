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
#   ./serve.sh --restart                    # stop the server already running on the port first, then start a new one
#   (options can go anywhere among the arguments)

SERVER_ARGS=()
POSITIONAL=()
RESTART=0
for a in "$@"; do
    case "$a" in
        --local) SERVER_ARGS+=(--local) ;;
        --allow-actions) SERVER_ARGS+=(--allow-remote-actions) ;;
        --restart) RESTART=1 ;;
        *) POSITIONAL+=("$a") ;;
    esac
done

FILE=${POSITIONAL[0]:-website/index.html}
PORT=${POSITIONAL[1]:-8080}

STOP_WAIT_SECONDS=10
if [ "$RESTART" = 1 ]; then
    PATTERN="ops_server\.jl --port $PORT( |\$)"
    PIDS=$(pgrep -f "$PATTERN")
    if [ -n "$PIDS" ]; then
        echo "stopping the server on port $PORT (pid $(echo $PIDS))"
        kill $PIDS
        for _ in $(seq 1 $((STOP_WAIT_SECONDS * 10))); do
            pgrep -f "$PATTERN" > /dev/null || break
            sleep 0.1
        done
        pgrep -f "$PATTERN" > /dev/null && { echo "still running after ${STOP_WAIT_SECONDS}s, killing it"; pkill -9 -f "$PATTERN"; sleep 0.5; }
    else
        echo "no server on port $PORT; starting one"
    fi
fi

exec julia --project=packages/StockSwingPredictor scripts/tools/ops_server.jl --port "$PORT" --open "$FILE" "${SERVER_ARGS[@]}"
