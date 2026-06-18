#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "Running tests..."
go test ./...

echo "Building..."
./build.sh

echo "Running check..."
./dist/flashback-shell-pty check

echo "Running new -c 'echo hello'..."
./dist/flashback-shell-pty new -c 'echo hello'

echo "Testing --capture and --capture-text..."
rm -f /tmp/pty-standalone-capture.ansi /tmp/pty-standalone-capture.txt
./dist/flashback-shell-pty new --capture /tmp/pty-standalone-capture.ansi --capture-text /tmp/pty-standalone-capture.txt -c 'echo capture-test-123'
if ! grep -q "capture-test-123" /tmp/pty-standalone-capture.txt; then
    echo "FAIL: --capture-text did not contain expected output"
    cat /tmp/pty-standalone-capture.txt
    exit 1
fi
rm -f /tmp/pty-standalone-capture.ansi /tmp/pty-standalone-capture.txt

echo "Testing --vt-log..."
rm -f /tmp/pty-standalone-vt.log
./dist/flashback-shell-pty new --vt-log /tmp/pty-standalone-vt.log --vt-log-interval 200ms -c 'echo line1; sleep 0.3; echo line2'
if ! grep -q "line1" /tmp/pty-standalone-vt.log; then
    echo "FAIL: --vt-log did not contain expected output"
    cat /tmp/pty-standalone-vt.log
    exit 1
fi
rm -f /tmp/pty-standalone-vt.log

echo "All tests passed."
