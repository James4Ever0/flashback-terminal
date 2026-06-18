#!/usr/bin/env bash
# run-tests.sh — build, vet, test, and produce a release binary.
set -euo pipefail

cd "$(dirname "$0")"

echo "==> go build ./..."
go build ./...

echo "==> go vet ./..."
go vet ./...

echo "==> go test ./..."
go test ./...

echo "==> build release binary"
./build.sh

echo "==> all tests passed"
