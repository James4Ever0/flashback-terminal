#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p bin
go mod tidy
go build -o bin/forwarder ./cmd/forwarder
go build -o bin/receiver  ./cmd/receiver
go build -o bin/logger    ./cmd/logger
echo "built bin/forwarder bin/receiver bin/logger"
