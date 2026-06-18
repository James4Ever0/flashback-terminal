# Plan: ALiS Unix-Socket Streamer + Go Receiver/Tmux Renderer

## Context

We want to stream a live terminal session from asciinema v3 into a headless tmux session so the rendered terminal view can be captured and uploaded to a remote server. The existing `terminal-capture-shell-golang` already captures tmux panes and uploads them; the missing piece is a way to feed asciinema's live ALiS stream into a tmux session. A Python proof-of-concept already does this end-to-end (`asciinema_v3_test/asciinema_stream_to_tmux_and_capture_headless.py`). This plan translates that pipeline into a production-style Go binary, adds a small asciinema patch to emit ALiS over a Unix domain socket, and documents the pure-Rust alternative.

## Decision Log

- **Rust binds the Unix socket; Go receiver connects.** asciinema already owns the session lifecycle, so having it publish events as a socket server is the cleanest fit with its existing `Output` trait.
- **Go binary includes render + capture + upload.** It will reuse packages from `terminal-capture-shell-golang` for tmux lifecycle, capture/diff, and HTTP upload.
- **Go binary name: `alis-receiver`**; module: `alis-receiver`.

## Architecture

```
[asciinema session] --ALiS binary--> [Unix socket A] <--(Go client)-- [alis-receiver]
                                                                            |
                                                                            v
                                                                    [parse ALiS]
                                                                            |
                                                                            v
                                                                    [write raw bytes]
                                                                            |
                                                                            v
                                                                    [Unix socket B] <--(tmux receiver)
                                                                            |
                                                                            v
                                                                    [tmux pane renders VT]
                                                                            |
                                                                            v
                                                                    [periodic capture-pane]
                                                                            |
                                                                            v
                                                                    [diff/dedup -> HTTP upload]
```

- **Socket A (ALiS)**: created by asciinema, accepted by `alis-receiver`.
- **Socket B (raw terminal bytes)**: created by `alis-receiver`, read by a small receiver running inside tmux.

## Part 1: Asciinema Rust Modifications

Add a new `--stream-unix-socket <PATH>` output mode to asciinema.

### New file: `reference/asciinema-develop/src/unix_socket_writer.rs`

Implement `session::Output` for a Unix domain socket writer:

- On first `event()`, bind a `tokio::net::UnixListener` to the given path and `accept()` exactly one client.
- Serialize each `session::Event` to ALiS binary using the existing `alis::EventSerializer` logic.
- Write `MAGIC_STRING` header before the first event.
- Remove the socket file on drop/cleanup.

Reuse:
- `session::Output` trait: `reference/asciinema-develop/src/session.rs:66`
- ALiS serializer: `reference/asciinema-develop/src/alis.rs:18` (make `EventSerializer` public or expose a `serialize_single_event` helper)
- Pattern to follow: `reference/asciinema-develop/src/file_writer.rs`

### Modify: `reference/asciinema-develop/src/alis.rs`

Make `EventSerializer` public and add a public `serialize_single_event(&mut self, event: Event) -> Vec<u8>` method. The `stream()` function already contains the serialization logic; refactor so the single-event path is reusable without a broadcast channel.

### Modify: `reference/asciinema-develop/src/cli.rs`

Add to the `Session` struct:

```rust
/// Stream ALiS-encoded terminal events to a Unix domain socket.
#[arg(long, value_name = "PATH", help = "Stream ALiS events to a Unix domain socket", long_help)]
pub stream_unix_socket: Option<PathBuf>,
```

Also add the same field to `Record` and `Stream` if the socket output should work with those commands; `Session` is the minimum.

### Modify: `reference/asciinema-develop/src/cmd/session.rs`

After `file_writer` and before `server`/`forwarder`/`stream`, instantiate the new output:

```rust
if let Some(path) = &self.stream_unix_socket {
    let writer = unix_socket_writer::UnixSocketWriter::new(path).await?;
    outputs.push(Box::new(writer));
}
```

### Modify: `reference/asciinema-develop/src/main.rs`

Add `mod unix_socket_writer;`.

### Test

```bash
cd reference/asciinema-develop
cargo build --release
./target/release/asciinema session --stream-unix-socket /tmp/alis.sock --headless --command "echo hello; sleep 2; echo world"
```

Verify `/tmp/alis.sock` is created and the client can read the ALiS header + events.

## Part 2: Go Binary `alis-receiver`

Create under `terminal-capture-shell-golang-asciinema-event-receive-and-render/`.

### Layout

```
alis-receiver/
├── go.mod
├── main.go                 # CLI entrypoint: flags, config, logger, subcommand dispatch
├── build.sh                # CGO-free static build (copy/adapt from flashback-shell)
├── README.md
├── cmd/
│   └── render.go           # "render" subcommand
├── pkg/
│   ├── alis/
│   │   ├── parser.go       # LEB128 + ALiS message parser
│   │   └── events.go       # Event type definitions
│   ├── socket/
│   │   └── client.go       # Connect to Rust ALiS Unix socket
│   ├── tmux/
│   │   └── renderer.go     # Headless tmux session + receiver lifecycle
│   ├── capture/
│   │   ├── capture.go      # Copied/adapted from flashback-shell pkg/capture
│   │   └── buffer.go       # Retry buffer
│   ├── server/
│   │   └── client.go       # Copied from flashback-shell pkg/server
│   ├── config/
│   │   └── config.go       # YAML + env vars (ALIS_RECEIVER_*)
│   └── log/
│       └── log.go          # Leveled logger (copy from flashback-shell)
```

### CLI

Global flags (same style as `flashback-shell`):

- `-c <path>` config file (`~/.config/alis-receiver.yaml`)
- `-v`, `-vv`, `-vvv` verbosity
- `-l <path>` log file

`render` subcommand:

```
alis-receiver render [flags] <alis-socket-path>
```

Flags:

- `--tmux-session <name>` (default: `alis-<pid>`)
- `--tmux-socket <path>` (default: `/tmp/alis-tmux-<pid>.sock`)
- `--capture-interval <seconds>` (default: 30, 0 disables)
- `--server-url <url>`
- `--no-capture`
- `--text-only`
- `--diff-only`
- `--diff-mode <suffix|index>`

### ALiS Parser (`pkg/alis/`)

Implement in Go the inverse of `reference/asciinema-develop/src/alis.rs`:

- `DecodeLEB128(data []byte, offset int) (uint64, int, error)`
- Message types: `Init` (0x01), `Output` ('o'), `Input` ('i'), `Resize` ('r'), `Marker` ('m'), `Exit` ('x'), `EOT` (0x04)
- `ParseMessage(buf []byte) (Event, int, error)` returns `(event, consumedBytes, err)`.
- Streaming parser state machine that buffers incomplete messages and resumes on the next read.

Handle `Init` theme payload:

- Theme flag byte `0` => no theme.
- Theme flag byte `16` => read `fg RGB (3) + bg RGB (3) + 16 palette colors * 3 = 54 bytes`.

### Socket Client (`pkg/socket/client.go`)

Connect to the Rust ALiS Unix socket path. Provide:

- `Connect(path string) (*Client, error)` using `net.Dial("unix", path)`
- `Read(buf []byte) (int, error)`
- `Close() error`

### Tmux Renderer (`pkg/tmux/renderer.go`)

Reuse patterns from `terminal-capture-shell-golang/pkg/shell/tmux.go`.

Responsibilities:

1. Write a kiosk-mode `tmux.conf` to a temp dir: `status off`, `mouse off`, `default-terminal xterm-256color`, unbind all keys.
2. Create a detached tmux session on a dedicated socket.
3. Launch a small receiver inside tmux that connects to the Go-created Unix socket B and copies bytes to stdout.
4. `Resize(cols, rows uint16)` => `tmux -S <socket> resize-window -t <session> -x <cols> -y <rows>`.
5. `CapturePane() (string, error)` => `tmux -S <socket> capture-pane -p -J -t <session>:0.0` (add `-e` for ANSI).
6. `Kill()` => kill session and remove tmux socket.
7. `WriteToTmux(data []byte)` => write to the accepted connection on socket B.

Receiver inside tmux: use `nc -U <socket-b-path>` or a tiny embedded Go helper. Prefer `nc` for the first version; detect availability in `check` command and fall back to a compiled helper if needed.

### Render Loop (`cmd/render.go`)

1. Load config/logger.
2. Parse `<alis-socket-path>` argument.
3. Connect to the Rust ALiS socket (socket A).
4. Start tmux renderer and create socket B.
5. Start the tmux receiver; it will connect to socket B.
6. Optionally start a background capture goroutine if `capture-interval > 0`.
7. Read ALiS bytes, parse messages, and dispatch:
   - `Init`: resize tmux; if init text present, write it to socket B.
   - `Output`: write raw `text` bytes to socket B.
   - `Resize`: resize tmux.
   - `EOT`/`Exit`: log and stop reading.
8. Clean up tmux and sockets on exit.

### Capture/Upload Integration

Copy and adapt from `terminal-capture-shell-golang`:

- `pkg/capture/capture.go`: `CaptureSession`, hash dedup, diff algorithms (`suffix`/`index`).
- `pkg/capture/buffer.go`: local retry buffer for failed uploads.
- `pkg/server/client.go`: POST JSON to `<server_url>/api/captures` with exponential backoff.

Background goroutine:

- `time.Ticker` every `capture-interval` seconds.
- Call `renderer.CapturePane()` for ANSI and plain text.
- Use capture engine to diff/dedup.
- Upload batch; buffer on failure.

Payload matches existing server API:

```json
{
  "device_id": "...",
  "timestamp": "...",
  "captures": [
    {
      "session_id": "alis-<pid>",
      "pane_id": "alis-<pid>:0.0",
      "target": "alis-<pid>:0.0",
      "ansi": "...",
      "text": "...",
      "hash": "...",
      "cols": 80,
      "rows": 24,
      "timestamp": "...",
      "metadata": {"ansi": "true"}
    }
  ]
}
```

### Config (`pkg/config/config.go`)

YAML + env vars with `ALIS_RECEIVER_` prefix:

```yaml
server_url: "http://localhost:8080"
socket_dir: "/tmp"
capture_interval: 30
device_id: ""
buffer_size: 100
diff_only: true
diff_mode: "suffix"
text_only: false
capture_scrollback: false
session_name: ""
tmux_socket_dir: ""
```

Use the same precedence as `flashback-shell`: CLI flags > env vars > config file > defaults.

## Part 3: Build & Test

### Rust

```bash
cd reference/asciinema-develop
cargo build --release
```

### Go

```bash
cd terminal-capture-shell-golang-asciinema-event-receive-and-render
go mod init alis-receiver
go mod tidy
./build.sh
```

### Integration Test

Terminal 1 (start asciinema with Unix socket output):

```bash
./reference/asciinema-develop/target/release/asciinema session \
  --stream-unix-socket /tmp/alis.sock \
  --headless \
  --command "bash -c 'echo hello; sleep 3; echo world'"
```

Terminal 2 (run receiver with capture):

```bash
./terminal-capture-shell-golang-asciinema-event-receive-and-render/dist/alis-receiver \
  render /tmp/alis.sock \
  --capture-interval 1 \
  --server-url http://localhost:8080
```

Verify:

1. `tmux -S /tmp/alis-tmux-*.sock ls` shows a session.
2. `tmux -S /tmp/alis-tmux-*.sock capture-pane -p` shows `hello` and `world`.
3. The server receives POST `/api/captures` payloads.

### Unit Tests

- LEB128 round-trip against known Rust-encoded samples.
- ALiS parser with hex fixtures for `Init`, `Output`, `Resize`, `EOT`.
- Tmux lifecycle: create, resize, capture, kill in a temp dir.

## Part 4: Pure-Rust Alternative Analysis

A pure-Rust implementation could skip tmux entirely by using asciinema's internal `avt` VT emulator (`reference/asciinema-develop/src/stream.rs` already uses `build_vt()` and `vt.feed_str()`). asciinema would run the session, feed output bytes directly into `avt::Vt`, periodically call `vt.dump()` or `vt.lines()`, and upload the result via HTTP.

### Pros

- Single binary, no tmux dependency.
- No extra processes or Unix sockets.
- Faster capture: in-memory VT state instead of `tmux capture-pane` subprocess.
- Lower overhead per session.

### Cons

- `avt` may not support every edge case that tmux handles (complex Unicode, bidirectional text, etc.).
- Need to reimplement capture, diff, hash dedup, retry buffer, config, and HTTP upload in Rust; none of that exists in the current asciinema codebase.
- Output format from `avt` may differ from `tmux capture-pane`; compatibility needs verification.

### Recommendation

Implement the **Go + tmux** version first. It reuses the proven Python pipeline, the existing `flashback-shell` capture/upload packages, and tmux's mature VT emulation. Add a small, clean Unix-socket output to asciinema. Document the pure-Rust path as a future optimization; once the Go version is stable, evaluate whether adding a `--capture-remote <URL>` mode directly to asciinema (using `avt` + periodic upload) is worth the extra Rust implementation effort.

## Critical Files

| Component | Path |
|-----------|------|
| Rust ALiS serializer | `reference/asciinema-develop/src/alis.rs` |
| Rust Unix socket output (new) | `reference/asciinema-develop/src/unix_socket_writer.rs` |
| Rust CLI | `reference/asciinema-develop/src/cli.rs` |
| Rust session wiring | `reference/asciinema-develop/src/cmd/session.rs` |
| Rust module list | `reference/asciinema-develop/src/main.rs` |
| Go entrypoint (new) | `terminal-capture-shell-golang-asciinema-event-receive-and-render/main.go` |
| Go render command (new) | `terminal-capture-shell-golang-asciinema-event-receive-and-render/cmd/render.go` |
| Go ALiS parser (new) | `terminal-capture-shell-golang-asciinema-event-receive-and-render/pkg/alis/parser.go` |
| Go tmux renderer (new) | `terminal-capture-shell-golang-asciinema-event-receive-and-render/pkg/tmux/renderer.go` |
| Reusable Go capture | `terminal-capture-shell-golang/pkg/capture/capture.go` |
| Reusable Go server client | `terminal-capture-shell-golang/pkg/server/client.go` |
| Reusable Go tmux manager | `terminal-capture-shell-golang/pkg/shell/tmux.go` |
