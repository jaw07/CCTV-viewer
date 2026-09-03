#!/bin/bash
set -e

# The backend serves both the API and the static client (see the StaticFiles
# mount in backend/main.py), so a single port is all that is exposed. Serving
# from one origin is what lets the session cookie and the WebSocket work behind
# a single HTTPS hostname.
python3 backend/main.py &
BACKEND_PID=$!

cleanup() {
    echo "Stopping server..."
    kill $BACKEND_PID 2>/dev/null
    wait $BACKEND_PID 2>/dev/null
    echo "Server stopped."
    exit 0
}

trap cleanup SIGTERM SIGINT

wait $BACKEND_PID
EXIT_CODE=$?
cleanup
exit $EXIT_CODE
