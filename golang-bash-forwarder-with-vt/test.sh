#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
./build.sh

SOCK="/tmp/bash-forwarder-vt-test.sock"
rm -f "$SOCK"

# Smoke test: run a non-interactive command and capture ANSI/text output.
./bin/forwarder -listen "$SOCK" --no-capture --headless \
    --capture /tmp/vt-capture.ansi --capture-text /tmp/vt-capture.txt \
    bash -c 'sleep 0.5; echo smoke-test-123' &
pid=$!

for i in {1..30}; do
    if [[ -S "$SOCK" ]]; then
        break
    fi
    sleep 0.1
done

output=$(./bin/receiver "$SOCK" 2>/dev/null | tr -d '\r')
if ! grep -q "smoke-test-123" <<<"$output"; then
    echo "smoke test FAILED: expected smoke-test-123, got: $output"
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    rm -f "$SOCK" /tmp/vt-capture.ansi /tmp/vt-capture.txt
    exit 1
fi
wait "$pid"

if ! grep -q "smoke-test-123" /tmp/vt-capture.txt; then
    echo "capture-text test FAILED"
    cat /tmp/vt-capture.txt
    rm -f "$SOCK" /tmp/vt-capture.ansi /tmp/vt-capture.txt
    exit 1
fi

rm -f "$SOCK" /tmp/vt-capture.ansi /tmp/vt-capture.txt

# Protocol round-trip test.
SOCK2="/tmp/bash-forwarder-vt-protocol.sock"
LOG="/tmp/bash-forwarder-vt-protocol.log"
rm -f "$SOCK2" "$LOG"
./bin/forwarder -listen "$SOCK2" --no-capture --headless --log "$LOG" bash -c 'sleep 0.3; printf hi' &
pid=$!
for i in {1..30}; do
    if [[ -S "$SOCK2" ]]; then
        break
    fi
    sleep 0.1
done
wait "$pid" 2>/dev/null || true

if ! grep -q '"method":"init"' "$LOG"; then
    echo "protocol test FAILED: missing init event"
    cat "$LOG"
    rm -f "$SOCK2" "$LOG"
    exit 1
fi
if ! grep -q '"method":"output"' "$LOG"; then
    echo "protocol test FAILED: missing output event"
    cat "$LOG"
    rm -f "$SOCK2" "$LOG"
    exit 1
fi
if ! grep -q '"method":"exit"' "$LOG"; then
    echo "protocol test FAILED: missing exit event"
    cat "$LOG"
    rm -f "$SOCK2" "$LOG"
    exit 1
fi

rm -f "$SOCK2" "$LOG"

# Clean exit test: forwarder exits with the child's status and does not hang.
SOCK_EXIT="/tmp/bash-forwarder-vt-exit.sock"
LOG_EXIT="/tmp/bash-forwarder-vt-exit.log"
rm -f "$SOCK_EXIT" "$LOG_EXIT"
./bin/forwarder -listen "$SOCK_EXIT" --no-capture --headless --log "$LOG_EXIT" bash -c 'echo hello; exit 42' &
pid=$!
for i in {1..30}; do
    if [[ -S "$SOCK_EXIT" ]]; then break; fi
    sleep 0.1
done
wait "$pid" || exit_code=$?
if [[ "${exit_code:-0}" != "42" ]]; then
    echo "exit test FAILED: expected forwarder exit code 42, got ${exit_code:-0}"
    rm -f "$SOCK_EXIT" "$LOG_EXIT"
    exit 1
fi
if ! grep -q '"method":"exit"' "$LOG_EXIT"; then
    echo "exit test FAILED: missing exit event"
    cat "$LOG_EXIT"
    rm -f "$SOCK_EXIT" "$LOG_EXIT"
    exit 1
fi
exit_status=$(grep '"method":"exit"' "$LOG_EXIT" | sed 's/.*"status":\([0-9]*\).*/\1/')
if [[ "$exit_status" != "42" ]]; then
    echo "exit test FAILED: expected logged exit status 42, got $exit_status"
    rm -f "$SOCK_EXIT" "$LOG_EXIT"
    exit 1
fi
rm -f "$SOCK_EXIT" "$LOG_EXIT"

# Stdout log destination test: -log - writes JSON events to stdout.
SOCK_STDOUT="/tmp/bash-forwarder-vt-stdout-log.sock"
rm -f "$SOCK_STDOUT"
./bin/forwarder -listen "$SOCK_STDOUT" --no-capture --headless -log - bash -c 'echo stdout-log' > /tmp/vt-stdout.log 2>/dev/null &
pid=$!
for i in {1..30}; do
    if [[ -S "$SOCK_STDOUT" ]]; then break; fi
    sleep 0.1
done
wait "$pid" 2>/dev/null || true
if ! grep -q '"method":"output"' /tmp/vt-stdout.log; then
    echo "stdout log test FAILED: missing output event on stdout"
    cat /tmp/vt-stdout.log
    rm -f "$SOCK_STDOUT" /tmp/vt-stdout.log
    exit 1
fi
if ! grep -q 'c3Rkb3V0LWxvZw' /tmp/vt-stdout.log; then
    echo "stdout log test FAILED: expected 'stdout-log' on stdout"
    cat /tmp/vt-stdout.log
    rm -f "$SOCK_STDOUT" /tmp/vt-stdout.log
    exit 1
fi
rm -f "$SOCK_STDOUT" /tmp/vt-stdout.log

echo "all tests passed"
