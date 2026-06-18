#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
./build.sh

SOCK="/tmp/pty-forward-test.sock"
rm -f "$SOCK"

# Smoke test: run a non-interactive command through the forwarder.
./bin/forwarder -listen "$SOCK" --no-capture --headless bash -c 'sleep 0.5; echo smoke-test-123' &
pid=$!

# Wait for the socket to appear.
for i in {1..30}; do
    if [[ -S "$SOCK" ]]; then
        break
    fi
    sleep 0.1
done

# Connect a receiver and look for the expected string.
output=$(./bin/receiver "$SOCK" 2>/dev/null | tr -d '\r')
if ! grep -q "smoke-test-123" <<<"$output"; then
    echo "smoke test FAILED: expected smoke-test-123, got:"
    echo "$output"
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    exit 1
fi

wait "$pid"
rm -f "$SOCK"

# Protocol round-trip test.
SOCK2="/tmp/pty-protocol-test.sock"
LOG="/tmp/pty-protocol.log"
rm -f "$SOCK2" "$LOG"
./bin/forwarder -listen "$SOCK2" --no-capture --headless --log "$LOG" bash -c 'sleep 0.3; printf hi' &
pid=$!
for i in {1..30}; do
    if [[ -S "$SOCK2" ]]; then
        break
    fi
    sleep 0.1
done

# Wait for the forwarder to finish and flush the log.
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

# Env propagation test: COLUMNS and LINES are forwarded to the child.
SOCK_ENV="/tmp/pty-env-test.sock"
LOG_ENV="/tmp/pty-env.log"
rm -f "$SOCK_ENV" "$LOG_ENV"
COLUMNS=80 LINES=24 ./bin/forwarder -listen "$SOCK_ENV" --no-capture --headless --log "$LOG_ENV" bash -c 'echo "$COLUMNS $LINES"' &
pid=$!
for i in {1..30}; do
    if [[ -S "$SOCK_ENV" ]]; then break; fi
    sleep 0.1
done
wait "$pid" 2>/dev/null || true
if ! grep -q '"method":"output"' "$LOG_ENV"; then
    echo "env test FAILED: missing output event"
    cat "$LOG_ENV"
    rm -f "$SOCK_ENV" "$LOG_ENV"
    exit 1
fi
if ! grep -q 'ODAgMjQ' "$LOG_ENV"; then
    echo "env test FAILED: expected '80 24' in output event log"
    cat "$LOG_ENV"
    rm -f "$SOCK_ENV" "$LOG_ENV"
    exit 1
fi
rm -f "$SOCK_ENV" "$LOG_ENV"

# Signal forwarding test: SIGTERM is forwarded to the child process group.
SOCK_SIG="/tmp/pty-sig-test.sock"
LOG_SIG="/tmp/pty-sig.log"
rm -f "$SOCK_SIG" "$LOG_SIG"
./bin/forwarder -listen "$SOCK_SIG" --no-capture --headless --log "$LOG_SIG" bash -c 'trap "echo trapped; exit 0" TERM; while true; do sleep 1; done' &
pid=$!
for i in {1..30}; do
    if [[ -S "$SOCK_SIG" ]]; then break; fi
    sleep 0.1
done
# Give the child time to install its trap.
sleep 0.2
kill -TERM "$pid"
for i in {1..50}; do
    if ! kill -0 "$pid" 2>/dev/null; then break; fi
    sleep 0.1
done
if kill -0 "$pid" 2>/dev/null; then
    echo "signal test FAILED: forwarder did not exit after SIGTERM"
    kill -9 "$pid" 2>/dev/null || true
    rm -f "$SOCK_SIG" "$LOG_SIG"
    exit 1
fi
if ! grep -q 'dHJhcHBlZA' "$LOG_SIG"; then
    echo "signal test FAILED: child did not receive SIGTERM (no trapped output)"
    cat "$LOG_SIG"
    rm -f "$SOCK_SIG" "$LOG_SIG"
    exit 1
fi
if ! grep -q '"method":"exit"' "$LOG_SIG"; then
    echo "signal test FAILED: missing exit event"
    cat "$LOG_SIG"
    rm -f "$SOCK_SIG" "$LOG_SIG"
    exit 1
fi
exit_status=$(grep '"method":"exit"' "$LOG_SIG" | sed 's/.*"status":\([0-9]*\).*/\1/')
if [[ "$exit_status" != "0" ]]; then
    echo "signal test FAILED: expected exit status 0, got $exit_status"
    rm -f "$SOCK_SIG" "$LOG_SIG"
    exit 1
fi
rm -f "$SOCK_SIG" "$LOG_SIG"

# Clean exit test: forwarder exits with the child's status and does not hang.
SOCK_EXIT="/tmp/pty-exit-test.sock"
LOG_EXIT="/tmp/pty-exit.log"
rm -f "$SOCK_EXIT" "$LOG_EXIT"
./bin/forwarder -listen "$SOCK_EXIT" --no-capture --headless --log "$LOG_EXIT" bash -c 'echo hello; exit 42' &
pid=$!
for i in {1..30}; do
    if [[ -S "$SOCK_EXIT" ]]; then break; fi
    sleep 0.1
done
wait "$pid" || exit_code=$?
if [[ "$exit_code" != "42" ]]; then
    echo "exit test FAILED: expected forwarder exit code 42, got $exit_code"
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
SOCK_STDOUT="/tmp/pty-stdout-log-test.sock"
rm -f "$SOCK_STDOUT"
./bin/forwarder -listen "$SOCK_STDOUT" --no-capture --headless -log - bash -c 'echo stdout-log' > /tmp/pty-stdout.log 2>/dev/null &
pid=$!
for i in {1..30}; do
    if [[ -S "$SOCK_STDOUT" ]]; then break; fi
    sleep 0.1
done
wait "$pid" 2>/dev/null || true
if ! grep -q '"method":"output"' /tmp/pty-stdout.log; then
    echo "stdout log test FAILED: missing output event on stdout"
    cat /tmp/pty-stdout.log
    rm -f "$SOCK_STDOUT" /tmp/pty-stdout.log
    exit 1
fi
if ! grep -q 'c3Rkb3V0LWxvZw' /tmp/pty-stdout.log; then
    echo "stdout log test FAILED: expected 'stdout-log' on stdout"
    cat /tmp/pty-stdout.log
    rm -f "$SOCK_STDOUT" /tmp/pty-stdout.log
    exit 1
fi
rm -f "$SOCK_STDOUT" /tmp/pty-stdout.log

echo "all tests passed"
