#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "Building terminal-capture-shell-rust-standalone..."

cargo build --release

echo "Built: target/release/terminal-capture-shell-rust-standalone"
