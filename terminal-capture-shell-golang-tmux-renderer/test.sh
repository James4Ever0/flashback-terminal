#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

./build.sh

echo "==> smoke test: check command"
out=$(./dist/flashback-shell check)
echo "$out" | grep -q "CONFIG"
echo "check command passed"

echo "==> smoke test: list command (no sessions)"
tmpdir=$(mktemp -d)
cat > "$tmpdir/test-config.yaml" <<EOF
socket_dir: "$tmpdir/tmux"
EOF
out=$(./dist/flashback-shell -c "$tmpdir/test-config.yaml" list)
echo "$out" | grep -q "no flashback-shell renderer sessions found"
echo "list command passed"

echo "==> smoke test: capture command (no sessions)"
out=$(./dist/flashback-shell -c "$tmpdir/test-config.yaml" capture)
echo "$out" | grep -q "no changes detected"
echo "capture command passed"

rm -rf "$tmpdir"

echo "==> smoke test: protocol encode/decode"
go test ./internal/protocol/...

echo "all smoke tests passed"
