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

echo "All tests passed."
