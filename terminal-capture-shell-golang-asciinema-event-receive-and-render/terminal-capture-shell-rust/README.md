# terminal-capture-shell-rust

Pure-Rust single-binary terminal capture tool with internal VT rendering. It
mirrors the command-line interface of `terminal-capture-shell-golang` but does
not require tmux; instead it runs the shell inside a PTY, renders the screen
with a vendored copy of the `avt` VT emulator, and uploads periodic snapshots
to a remote server.

## Features

- Single statically-linkable binary (musl).
- No tmux dependency.
- No libpng / image-rendering dependencies.
- Internal VT emulator supporting plain-text and ANSI capture (equivalent to
  `tmux capture-pane -e`).
- Deduplication by MD5 hash and diff-only uploads (`suffix` and `index` modes).
- Local retry buffer for failed uploads.
- YAML configuration and `FLASHBACK_SHELL_RUST_*` environment variables.
- Nested session protection: refuses to start a session inside an existing
  `FLASHBACK_SHELL=1` environment unless `--allow-nested` is set.

## Commands

```
terminal-capture-shell-rust [global opts] <command>

Global opts:
  -c, --config <PATH>
  -v, -vv, -vvv
  -l, --log-file <PATH>
      --no-capture
      --allow-nested

Commands:
  new [shell args...]    Start a new shell session
  capture                Capture all sessions and upload changes
  list                   List running sessions
  kill <id>              Kill a specific session
  check                  Validate deps and show effective config
```

## Configuration

Default config path: `~/.config/terminal-capture-shell.yaml`

Example:

```yaml
server_url: "http://localhost:8080"
socket_dir: "~/.flashback-shell-rust/sockets"
shell: "/bin/bash"
device_id: "my-device"
capture_interval: 30
buffer_size: 100
diff_only: true
diff_mode: suffix
text_only: false
scrollback_lines: 1000
```

Environment variables override config values:

- `FLASHBACK_SHELL_RUST_SERVER_URL`
- `FLASHBACK_SHELL_RUST_SOCKET_DIR`
- `FLASHBACK_SHELL_RUST_SHELL`
- `FLASHBACK_SHELL_RUST_DEVICE_ID`
- `FLASHBACK_SHELL_RUST_CAPTURE_INTERVAL`
- `FLASHBACK_SHELL_RUST_BUFFER_SIZE`
- `FLASHBACK_SHELL_RUST_DIFF_ONLY`
- `FLASHBACK_SHELL_RUST_DIFF_MODE`
- `FLASHBACK_SHELL_RUST_TEXT_ONLY`
- `FLASHBACK_SHELL_RUST_SCROLLBACK_LINES`
- `FLASHBACK_SHELL_RUST_DISABLE_CAPTURE`
- `FLASHBACK_SHELL_RUST_ALLOW_NESTED`

## Building

### Native

```bash
cargo build --release
```

### Static musl

```bash
# install the musl C toolchain (Debian/Ubuntu)
sudo apt-get install musl-tools

# add the Rust target
rustup target add x86_64-unknown-linux-musl

# build
cargo build --release --target x86_64-unknown-linux-musl

# verify it is static
ldd target/x86_64-unknown-linux-musl/release/terminal-capture-shell-rust
# expected: "not a dynamic executable"
```

## Verification

Terminal 1:

```bash
./target/release/terminal-capture-shell-rust new --server-url http://localhost:8080
```

Terminal 2:

```bash
./target/release/terminal-capture-shell-rust capture
```

The server at `http://localhost:8080` should receive `POST /api/captures` with
`captures[0].ansi` and `captures[0].text`.

## Credits

The `src/vt/` module is derived from `avt` (asciinema virtual terminal) by
Marcin Kulik and contributors, licensed under the Apache License 2.0. See
`src/vt/LICENSE-avt` for the full license text.
