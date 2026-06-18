#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "Building flashback-shell-pty..."

# Static, CGO-free binary.
CGO_ENABLED=0 go build -ldflags='-s -w' -o dist/flashback-shell-pty .

echo "Built: dist/flashback-shell-pty"
