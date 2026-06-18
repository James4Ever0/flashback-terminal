# #!/bin/bash
# set -e

# # Install musl C toolchain (Debian/Ubuntu)
# sudo apt-get update
# sudo apt-get install -y musl-tools

# # Add Rust musl target
# rustup target add x86_64-unknown-linux-musl

# Build static binary
cargo build --release --target x86_64-unknown-linux-musl

# Verify static linkage
ldd target/x86_64-unknown-linux-musl/release/terminal-capture-shell-rust || true

# Verify no image crates in lockfile
grep -iE '^(name = "(png|image|jpeg|libpng)")' Cargo.lock && echo "ERROR: image crate found" || echo "OK: no image crates"
