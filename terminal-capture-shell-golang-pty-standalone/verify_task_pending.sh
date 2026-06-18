#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "Running go vet..."
go vet ./...

echo ""
echo "Running tests..."
./test.sh

echo ""
echo "Task verification complete."
