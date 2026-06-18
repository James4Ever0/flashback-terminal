# Plan: Minimal Pure-Rust Terminal Capture Tool

## Context

The original goal was to patch asciinema v3 to stream ALiS events over a Unix domain socket, then write a Go receiver that rendered the stream into a headless tmux session and uploaded periodic captures. That approach required:

- A patched asciinema binary.
- A separate Go binary (`alis-receiver`).
- tmux as an external renderer.
- Two Unix domain sockets and inter-process coordination.

The user now wants a **single self-contained binary written entirely in Rust**. The binary should mirror the command-line interface of `terminal-capture-shell-golang`, render the terminal view internally with a VT emulator, and upload captures to a remote server on a timer. It should also be easy to statically link (musl) and avoid image-rendering dependencies such as libpng.

This plan describes a **minimal pure-Rust rewrite** that:
- Borrows the server/client architecture from `terminal-capture-shell-golang-pty`.
- Vendors a **minimal subset** of the `avt` VT emulator directly into the crate instead of adding `avt` as an external dependency.
- Uses the PTY, TTY, and signal-handling patterns from `reference/asciinema-develop` to connect the user's shell to the program.
- Contains no image-rendering or platform-specific GUI crates.

## How Asciinema Connects the Terminal GUI, the User, and the Shell

Asciinema's architecture (`reference/asciinema-develop/src`) is the best reference for how a terminal recorder interacts with the real world:

```
[terminal emulator GUI, e.g. Konsole]
                |
                | /dev/tty (raw mode)
                v
    [asciinema session::run loop]
                |
                | PTY master FD
                v
            [shell / bash]
```

1. **The user's terminal is `/dev/tty`.** `src/tty/default.rs` opens `/dev/tty` read/write with `O_NONBLOCK`, saves the original `termios`, and calls `cfmakeraw` so keystrokes are sent byte-by-byte and output is written directly without local echo. `get_size()` uses `TIOCGWINSZ` to learn the terminal dimensions.
2. **The shell runs inside a PTY.** `src/pty.rs` calls `nix::pty::forkpty` to fork a child attached to a pseudo-terminal slave. The child sets any extra env vars and `execvp`s the shell command (default `/bin/sh -c "$SHELL"`). The parent keeps the PTY master as an `AsyncFd` for async read/write.
3. **The main loop bridges bytes.** `src/session.rs` runs a `tokio::select!` loop:
   - PTY output bytes are decoded from UTF-8, forwarded to each `Output`, and also written back to `/dev/tty` so the user sees them.
   - `/dev/tty` input bytes are forwarded to the PTY.
   - `SIGWINCH` triggers a PTY resize via `TIOCSWINSZ` and emits a `Resize` event.
   - `SIGCHLD` detects the shell exiting and emits an `Exit` event.
4. **Outputs receive typed events.** `session::Output` implementations (`FileWriter`, `LiveStream`) receive `Event::Output`, `Event::Input`, `Event::Resize`, `Event::Marker`, `Event::Exit`.
5. **`Stream` renders internally.** `src/stream.rs` is an `Output` that feeds every `Event::Output` into an `avt::Vt`. Subscribers receive `Init` (which includes `vt.dump()`, a full ANSI snapshot) plus a live stream of events.

**What Konsole/GNOME Terminal/alacritty do:** they are the GUI renderer. Asciinema does not render pixels; it passes PTY output back to the user's terminal emulator, which draws glyphs. For a headless capture tool we **skip** the GUI bridge (`/dev/tty`) and instead feed the PTY output into our own in-memory VT grid.

## Decision Log

- **Single binary, no tmux, no GUI.** The PTY server + VT renderer + capture/upload all live in one process.
- **Server/client split kept, both in Rust.** `new` starts a background server and attaches, exactly like `terminal-capture-shell-golang-pty`.
- **Minimal vendored VT.** Copy/adapt only the parts of `avt` we need (`parser`, `terminal`, `buffer`, `line`, `cell`, `pen`, `color`, `charset`, `tabs`) into `src/vt/`. This removes the external `avt` dependency and guarantees no surprise image crates.
- **ANSI capture is supported.** The vendored VT can reconstruct ANSI escape sequences from the cell grid, including SGR colors and attributes. This is done the same way `avt` does it: `Terminal::dump()` produces a `Vec<Function>` and `parser::dump()` serializes those functions back to ANSI. It is equivalent in spirit to `tmux capture-pane -e`.
- **No libpng / image crates.** Verified by grepping `reference/asciinema-develop` and `reference/avt-main`: neither contains `png`, `image`, `jpeg`, or `libpng`. Our crate will keep the same restriction.
- **Unix-only, but minimal platform-specific surface.** PTY/TIOCSWINSZ/termios are inherently Unix. We will use `nix` + `libc` + `tokio::io::unix::AsyncFd` exactly like asciinema, avoiding broad cross-platform abstractions such as `portable-pty` that pull in extra crates.
- **Async via `tokio`.** Matches asciinema's `session.rs` and simplifies cancellation, timers, and socket I/O.
- **Config: YAML + env vars.** Add `serde_yaml` so users can keep the same config files as the Go tool. Prefix: `FLASHBACK_SHELL_RUST_*`.

## Differences from Previous Plan

| Aspect | Previous Plan (`plan-go-tmux-deprecated.md`) | This Plan |
|--------|---------------------------|-----------|
| Languages | Rust patch + Go binary | Rust only |
| External renderer | tmux | In-process minimal VT |
| Number of binaries/processes | asciinema + Go receiver + tmux | One binary, one server process per session |
| Sockets | ALiS socket + raw-byte socket to tmux | One Unix socket per session for attach/control |
| VT source | `avt` crate dependency | Vendored minimal subset of `avt` in `src/vt/` |
| PTY source | `portable-pty` crate | Raw `nix::pty::forkpty` + `tokio::io::unix::AsyncFd` (asciinema style) |
| Capture source | `tmux capture-pane` | Internal VT grid snapshot |
| ANSI capture | `tmux capture-pane -e` | `vt.dump()` + `parser::dump()` (pen-diff SGR reconstruction) |
| Capture diff/hash | Reuse Go `pkg/capture` | Reimplement in Rust, porting the algorithm |
| Config format | YAML + env (`ALIS_RECEIVER_*`) | YAML + env (`FLASHBACK_SHELL_RUST_*`) |
| Binary name | `alis-receiver` | `terminal-capture-shell-rust` (working name) |
| Dependency risk | tmux must be installed | No tmux, no libpng, small dep tree, musl-static friendly |

## Architecture

```
user runs: terminal-capture-shell-rust new

                |
                v
    [foreground client] --Unix socket--> [background server]
                                                |
                                                v
                                        [PTY + shell process]
                                                |
                                                v
                                        [minimal VT feed_str]
                                                |
                                                v
                                        [periodic capture]
                                                |
                                                v
                                        [diff/hash/dedup]
                                                |
                                                v
                                        [HTTP POST /api/captures]
```

- **`new`** starts a background server (`__server`) that owns the PTY, shell, VT, and Unix socket, then attaches the user's terminal to it.
- **`capture`** connects to all discovered session sockets, asks each server for a screen snapshot, applies diff/dedup, and uploads.
- **`list`** scans the socket directory and prints running sessions.
- **`kill <id>`** sends a kill request to the matching server.
- **`check`** validates dependencies, shows effective config, and reports whether musl static builds are possible.

## ANSI Capture from the Vendored VT

The vendored VT can produce both plain text and ANSI-colored output, matching the behavior of `tmux capture-pane` and `tmux capture-pane -e`.

### Plain text

Each `Line` stores a vector of `Cell`s. `Line::text()` walks the cells, skips wide-character tails, and returns the visible characters as a `String`. This is equivalent to `tmux capture-pane` without flags.

### ANSI reconstruction

`avt` already implements full ANSI reconstruction:

1. `Terminal::dump()` (`reference/avt-main/src/terminal.rs:1327-1337`) iterates the visible buffer line by line.
2. For each line it walks cell chunks grouped by equal `Pen`. Whenever the pen changes it calls `to_sgr_diff()` (`reference/avt-main/src/terminal.rs:1625-1703`) to generate the minimal SGR sequence needed to move from the previous pen to the current pen.
3. `to_sgr_diff()` handles intensity, foreground/background colors (indexed 0-255 and RGB), italic, underline, blink, inverse, and strikethrough.
4. The resulting `Vec<Function>` is serialized to a string by `parser::dump()` (`reference/avt-main/src/parser.rs:946-954`), which converts each `Function::Sgr(...)` back to `\x1b[...m`.

Our vendored VT will expose the same two-step API:

```rust
impl Vt {
    /// Plain text, like `tmux capture-pane`.
    pub fn text(&self) -> Vec<String>;

    /// ANSI reconstruction, like `tmux capture-pane -e`.
    pub fn dump(&self) -> String;
}
```

This makes the Rust tool at least as capable as the Go `terminal-capture-shell-golang` tool, which calls `tmux capture-pane -p -e -J` to obtain ANSI output.

### Comparison with reference projects

| Project | ANSI capture approach | Completeness |
|---------|----------------------|--------------|
| `golang-bash-forwarder-with-vt` | `charmbracelet/x/vt` `emu.Render()` | Full for visible screen; scrollback lineToANSI is incomplete (no SGR) |
| `reference/tmux-master` | `grid_string_cells()` with `GRID_STRING_WITH_SEQUENCES` | Full, including 256/RGB colors, attributes, hyperlinks |
| `reference/avt-main` | `Terminal::dump()` → `parser::dump()` | Full for visible screen and attributes supported by `Pen` |

Our plan follows the `avt` approach because it is pure Rust, well-structured, and has no external dependencies. We will vendor only the parser/terminal code required to support the SGR features we need.

## Implementation

### 1. New Crate

Create `terminal-capture-shell-rust/` (or rename from `terminal-capture-shell-golang-asciinema-event-receive-and-render/`).

`Cargo.toml`:

```toml
[package]
name = "terminal-capture-shell-rust"
version = "0.1.0"
edition = "2021"

[dependencies]
# CLI / config
clap = { version = "4", features = ["derive"] }
config = "0.14"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
serde_yaml = "0.10"

# async runtime
tokio = { version = "1", features = ["full"] }
tokio-util = "0.7"
async-trait = "0.1"

# Unix PTY / TTY / signals
nix = { version = "0.29", features = ["term", "process", "pty", "signal"] }
libc = "0.2"

# HTTP upload
reqwest = { version = "0.12", features = ["json"] }

# hashing / misc
md5 = "0.7"
dirs = "5"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
anyhow = "1"
thiserror = "1"
unicode-width = "0.1"   # needed by the vendored VT for wide chars
rgb = "0.8"             # needed by the vendored VT for truecolor

[profile.release]
strip = true
lto = true
codegen-units = 1
```

No `png`, `image`, `jpeg`, `libpng`, or platform-specific GUI crates.

### 2. Vendored Minimal VT

Create `src/vt/` by copying and trimming the relevant files from `reference/avt-main/src/`:

| File | Purpose |
|------|---------|
| `src/vt/mod.rs` | Public API: `Vt::new`, `Vt::builder`, `feed_str`, `feed`, `resize`, `view`, `text`, `dump`, `size` |
| `src/vt/parser.rs` | ANSI/VT parser state machine **and** ANSI serialization (`dump()`) |
| `src/vt/terminal.rs` | Grid, cursor, scroll regions, mode handling, `Terminal::dump()`, `to_sgr_diff()` |
| `src/vt/buffer.rs` | Scrollback buffer and resize/reflow |
| `src/vt/line.rs` | Line of cells, text extraction, cell chunks by predicate |
| `src/vt/cell.rs` | Cell (char, width, pen) |
| `src/vt/pen.rs` | Foreground/background colors and attributes |
| `src/vt/color.rs` | `Color::Indexed` / `Color::RGB` |
| `src/vt/charset.rs` | DEC drawing charset |
| `src/vt/tabs.rs` | Tab stops |

Keep only what is needed for screen capture:
- All cursor movement and SGR attributes.
- Scrollback and resize.
- `text()` returning plain text.
- `dump()` returning ANSI reconstruction via `Terminal::dump()` + `parser::dump()`.
- No input handling, no pixel rendering, no image output.

This is the same technique asciinema uses in `src/stream.rs:211-216` and `src/stream.rs:96-99`, but with our own minimal copy so we control the dependency tree.

### 3. CLI

Mirror `terminal-capture-shell-golang`:

```
terminal-capture-shell-rust [global opts] <command>

Global opts:
  -c, --config <PATH>   # default ~/.config/terminal-capture-shell.yaml
  -v, -vv, -vvv         # verbosity
  -l, --log-file <PATH>
      --no-capture      # disable background capture for new sessions

Commands:
  new [shell args...]
  capture
  list
  kill <id>
  check
  __server <session-id> <socket-path> [internal]
```

Implemented with `clap` derive macros in `src/cli.rs`.

### 4. Config

File: `src/config.rs`

```rust
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub server_url: String,              // default http://localhost:8080
    pub socket_dir: PathBuf,             // default ~/.flashback-shell-rust/sockets
    pub shell: Option<String>,           // default $SHELL or /bin/bash
    pub device_id: Option<String>,
    pub capture_interval: u64,           // default 30, 0 disables
    pub buffer_size: usize,              // default 100
    pub diff_only: bool,                 // default true
    pub diff_mode: DiffMode,             // Suffix | Index
    pub text_only: bool,                 // default false
    pub scrollback_lines: usize,         // default 1000
}
```

Precedence: CLI flags > env vars (`FLASHBACK_SHELL_RUST_*`) > config file > defaults.

### 5. PTY

File: `src/pty.rs`

Copy/adapt `reference/asciinema-develop/src/pty.rs`:

```rust
pub struct Pty {
    child: Pid,
    master: AsyncFd<OwnedFd>,
}

impl Pty {
    pub async fn read(&self, buffer: &mut [u8]) -> io::Result<usize>;
    pub async fn write(&self, buffer: &[u8]) -> io::Result<usize>;
    pub fn resize(&self, winsize: Winsize);
    pub fn kill(&self);
    pub async fn wait(&self, options: Option<WaitPidFlag>) -> io::Result<WaitStatus>;
}

pub fn spawn<S: AsRef<str>>(
    command: &[S],
    winsize: Winsize,
    extra_env: &HashMap<String, String>,
) -> anyhow::Result<Pty>;
```

This is the exact same mechanism asciinema uses to run bash inside a PTY.

### 6. TTY / Raw Mode

File: `src/tty.rs`

Copy/adapt `reference/asciinema-develop/src/tty/default.rs` and `src/tty.rs`:

- `DevTty::open()` opens `/dev/tty`, makes it raw with `cfmakeraw`, and wraps it in `AsyncFd`.
- `get_size()` uses `TIOCGWINSZ`.
- `Drop` restores the original termios.
- Provide `NullTty` and `FixedSizeTty` for headless/capture-only operation.

Only needed for the `new` attach path; the background server uses `NullTty`.

### 7. Session Server

File: `src/session/server.rs`

Pattern copied from `terminal-capture-shell-golang-pty/pkg/session/server.go` and asciinema's `session.rs`:

1. Create PTY with `pty::spawn`.
2. Spawn the shell command as a child.
3. Create `Vt::builder().size(cols, rows).scrollback_limit(...).build()`.
4. Listen on a Unix socket for attach/capture/kill/resize requests.
5. Run a `tokio::select!` loop:
   - Read PTY bytes.
   - Forward bytes to attached client (if any).
   - Feed bytes to `vt.feed_str(...)`.
   - Read socket requests.
   - Handle periodic capture ticker.

Resize request updates both the PTY size and `vt.resize(cols, rows)`.

### 8. VT Capture Wrapper

File: `src/vt_capture.rs`

Thin wrapper around the vendored VT matching the Go `pkg/vtcapture/vtcapture.go` API:

```rust
pub struct VtCapture {
    vt: vt::Vt,
}

impl VtCapture {
    pub fn new(cols: usize, rows: usize, scrollback: usize) -> Self;
    pub fn feed(&mut self, text: &str);
    pub fn resize(&mut self, cols: usize, rows: usize);
    pub fn capture_text(&self) -> Vec<String>;   // plain text
    pub fn capture_ansi(&self) -> String;      // ANSI via vt.dump()
    pub fn size(&self) -> (usize, usize);
}
```

`capture_ansi()` calls `vt.dump()`, which internally:
1. Iterates visible lines.
2. Groups cells by equal `Pen`.
3. Emits minimal SGR diffs via `to_sgr_diff()`.
4. Serializes `Function`s to escape sequences via `parser::dump()`.

This produces output equivalent to `tmux capture-pane -e -J`.

### 9. Socket Protocol

File: `src/session/protocol.rs`

Same JSON line protocol as `terminal-capture-shell-golang-pty/pkg/session/protocol.go`:

```rust
#[derive(Serialize, Deserialize)]
#[serde(tag = "method", content = "params")]
enum Request {
    Attach,
    Resize { cols: u16, rows: u16 },
    Capture { text_only: bool },
    Kill,
    Status,
}

#[derive(Serialize, Deserialize)]
struct CaptureResponse {
    session_id: String,
    pane_id: String,
    target: String,
    ansi: String,
    text: String,
    hash: String,
    cols: usize,
    rows: usize,
    timestamp: DateTime<Utc>,
    metadata: HashMap<String, String>,
}
```

### 10. Capture Engine

File: `src/capture/engine.rs`

Port the logic from `terminal-capture-shell-golang/pkg/capture/capture.go`:

- Hash the full screen content (text or ANSI) with MD5.
- Skip upload if the hash matches the previous hash for this session.
- If `diff_only`:
  - `DiffMode::Suffix`: find the longest common suffix with the previous capture; upload only the changed prefix lines.
  - `DiffMode::Index`: build a map of line hash -> index; emit only lines whose hash changed or moved.
- Return a `Capture` struct that matches the existing server API.

File: `src/capture/buffer.rs`

On upload failure, append captures to a JSON-lines buffer under `~/.flashback-shell-rust/buffer/`. `capture` command retries buffered entries before capturing fresh state.

### 11. HTTP Upload Client

File: `src/server/client.rs`

Use `reqwest` to POST to `{server_url}/api/captures` with the same payload as `terminal-capture-shell-golang/pkg/server/client.go`:

```rust
#[derive(Serialize)]
struct UploadPayload {
    device_id: String,
    timestamp: DateTime<Utc>,
    captures: Vec<Capture>,
}
```

Retry 3× with exponential backoff; on final failure, push captures to the local buffer.

### 12. Foreground Client (`new`)

File: `src/cmd/new.rs`

1. Resolve config and generate `session_id`.
2. Ensure socket directory exists.
3. Spawn `__server` subcommand in the background, detached from the terminal.
4. Wait for the socket to appear.
5. Connect via `tokio::net::UnixStream`.
6. Put local TTY into raw mode.
7. Bidirectionally bridge stdin/stdout to the socket.
8. Monitor `SIGWINCH`; send `Resize` requests to the server.
9. On disconnect, restore TTY state and exit.

### 13. Background Capture Loop

File: `src/cmd/server.rs`

When starting a session server, spawn a `tokio::time::interval` task unless `--no-capture` or `capture_interval == 0`. Each tick:

1. Lock the VT (or clone the visible screen).
2. Build a `Capture` from `VtCapture`.
3. Run the capture engine (hash/diff).
4. Upload or buffer on failure.

### 14. Build & Distribution

Static musl build:

```bash
sudo apt-get install musl-tools            # debian/ubuntu
cargo build --release --target x86_64-unknown-linux-musl
./target/x86_64-unknown-linux-musl/release/terminal-capture-shell-rust --version
```

Verify no dynamic image dependencies:

```bash
ldd target/x86_64-unknown-linux-musl/release/terminal-capture-shell-rust
# should report "not a dynamic executable" or only ld-musl
```

Also verify Cargo.lock contains no `png`, `image`, or `libpng` entries:

```bash
grep -iE '^(name = "(png|image|jpeg|libpng)"|name = "avt")' Cargo.lock || echo "OK: no image/avt crates"
```

## Verification

### Unit Tests

- `vt/parser`: feed known ANSI sequences and assert parser output.
- `vt/terminal`: cursor movement, SGR, scrollback, resize.
- `vt/dump`: feed a colored prompt and assert the dumped string contains the expected `\x1b[...m` sequences.
- `vt_capture`: feed sequences and assert both `text()` and `ansi()` outputs.
- `capture/engine`: diff/hash round-trips for `suffix` and `index` modes.
- `session/protocol`: JSON line framing round-trip.
- `server/client`: mock HTTP server asserting payload shape.

### Integration Test

Terminal 1 (start server + attach):

```bash
./terminal-capture-shell-rust new --server-url http://localhost:8080
```

Inside the session, run:

```bash
echo -e "\x1b[31mred\x1b[0m normal"
```

Terminal 2 (force a capture):

```bash
./terminal-capture-shell-rust capture
```

Verify:

1. `terminal-capture-shell-rust list` shows the session.
2. The server at `http://localhost:8080` receives `POST /api/captures` with `captures[0].ansi` containing `\x1b[31m`.
3. `captures[0].text` contains `red normal` without escape sequences.
4. `ldd` confirms the musl binary is static.
5. `Cargo.lock` contains no image-rendering crates.

## Critical Files

| Component | Path |
|-----------|------|
| CLI | `terminal-capture-shell-rust/src/cli.rs` |
| Config | `terminal-capture-shell-rust/src/config.rs` |
| PTY | `terminal-capture-shell-rust/src/pty.rs` |
| TTY / raw mode | `terminal-capture-shell-rust/src/tty.rs` |
| Session server | `terminal-capture-shell-rust/src/session/server.rs` |
| Socket protocol | `terminal-capture-shell-rust/src/session/protocol.rs` |
| Vendored VT API | `terminal-capture-shell-rust/src/vt/mod.rs` |
| VT parser + ANSI dump | `terminal-capture-shell-rust/src/vt/parser.rs` |
| VT terminal grid + SGR diff | `terminal-capture-shell-rust/src/vt/terminal.rs` |
| VT buffer | `terminal-capture-shell-rust/src/vt/buffer.rs` |
| VT capture wrapper | `terminal-capture-shell-rust/src/vt_capture.rs` |
| Capture engine | `terminal-capture-shell-rust/src/capture/engine.rs` |
| Retry buffer | `terminal-capture-shell-rust/src/capture/buffer.rs` |
| HTTP upload | `terminal-capture-shell-rust/src/server/client.rs` |
| `new` command | `terminal-capture-shell-rust/src/cmd/new.rs` |
| `server` command | `terminal-capture-shell-rust/src/cmd/server.rs` |
| `capture` command | `terminal-capture-shell-rust/src/cmd/capture.rs` |
| Reference PTY server | `terminal-capture-shell-golang-pty/pkg/session/server.go` |
| Reference VT wrapper | `terminal-capture-shell-golang-pty/pkg/vtcapture/vtcapture.go` |
| Reference ANSI VT wrapper | `golang-bash-forwarder-with-vt/pkg/vtcapture/vtcapture.go` |
| Reference capture engine | `terminal-capture-shell-golang/pkg/capture/capture.go` |
| Reference HTTP client | `terminal-capture-shell-golang/pkg/server/client.go` |
| Reference tmux ANSI capture | `reference/tmux-master/cmd-capture-pane.c`, `reference/tmux-master/grid.c` |
| Asciinema PTY | `reference/asciinema-develop/src/pty.rs` |
| Asciinema TTY | `reference/asciinema-develop/src/tty.rs` |
| Asciinema session loop | `reference/asciinema-develop/src/session.rs` |
| Asciinema stream + VT usage | `reference/asciinema-develop/src/stream.rs` |
| `avt` source to vendor | `reference/avt-main/src/` |
