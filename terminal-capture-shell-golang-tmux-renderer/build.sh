#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
go mod tidy
CGO_ENABLED=0 go build -o dist/flashback-shell .
echo "built dist/flashback-shell"
