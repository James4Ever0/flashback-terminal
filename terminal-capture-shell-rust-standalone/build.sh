#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "Building terminal-capture-shell-rust-standalone..."

cargo build --release --target x86_64-unknown-linux-musl

echo "Built: target/x86_64-unknown-linux-musl/release/terminal-capture-shell-rust-standalone"
