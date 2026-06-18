#!/usr/bin/env bash
# Test script for terminal-capture-server.
# Run from the terminal-capture-server directory.

set -euo pipefail

# Deactivate conda if active so the project venv can be used.
if command -v conda &>/dev/null && [ "${CONDA_DEFAULT_ENV:-}" != "" ]; then
    conda deactivate
fi

# Use the project virtual environment if it exists, otherwise create one.
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

pip install -r requirements.txt

# Start the server in the background.
python main.py --host 127.0.0.1 --port 8080 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

# Wait for the server to be ready.
for i in {1..30}; do
    if curl -s -X POST http://127.0.0.1:8080/api/captures \
         -H "Content-Type: application/json" \
         -d '{"device_id":"test","timestamp":"2024-01-01T00:00:00Z","captures":[]}' >/dev/null 2>&1; then
        break
    fi
    sleep 0.5
done

# Send a test capture.
curl -s -X POST http://127.0.0.1:8080/api/captures \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "test-device",
    "timestamp": "2024-01-01T00:00:00Z",
    "captures": [{
      "session_id": "flashback-12345",
      "pane_id": "flashback-12345:%0",
      "target": "flashback-12345:%0",
      "ansi": "[31mhello[0m\nworld",
      "text": "hello\nworld",
      "hash": "a1b2c3d4e5f678901234567890123456",
      "cols": 0,
      "rows": 0,
      "timestamp": "2024-01-01T00:00:00Z"
    }]
  }'

echo
# The server output (printed above the curl response) shows the metadata summary.

# Give the server a moment to flush stdout before we tear it down.
sleep 1
