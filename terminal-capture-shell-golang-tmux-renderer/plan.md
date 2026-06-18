# Terminal Capture Shell — Socket-Based Forwarder + tmux Renderer Plan

## Objective

Replace the tmux-based session launcher in `terminal-capture-shell-golang` with a
single Go binary that:

1. Wraps an interactive shell (or any program) in a PTY.
2. Forwards all terminal IO to a local Unix socket as line-delimited JSON-RPC
   events.
3. Automatically launches a **tmux renderer** that connects to the socket and
   displays the mirrored terminal inside a tmux pane/session.
4. Lets `tmux capture-pane` capture the mirrored content, replacing the old
   `tmux capture-pane` based capture engine.
5. Keeps a familiar CLI surface (`new`, `capture`, `list`, `kill`, `check`) but
   drops flags/features that no longer make sense.

References:

- `terminal-capture-shell-golang` — CLI shape, config/logging, upload client.
- `golang-pty-forward-example` — PTY forwarder, JSON-RPC event protocol,
  receiver/render loop.

---

## High-level architecture

```
User terminal
      ^
      | raw bytes
      v
+------------------+
| flashback-shell  |
|  (single binary) |
|  - PTY master    |
|  - shell on PTY  |
|  - stdin -> PTY  |
|  - PTY  -> stdout|
|  - events -> sock|
+------------------+
      |
      | Unix socket (line-delimited JSON-RPC)
      v
+------------------+
| tmux renderer    |
|  (our receiver   |
|   inside tmux)   |
+------------------+
      |
      v
  tmux session/pane (capture target)
```

`flashback-shell new` does three things in order:

1. Start the **forwarder** (PTY + socket) in the current terminal.
2. Start a **tmux renderer** in a separate tmux session that connects to the
   same socket.
3. Attach the user to the forwarder's PTY. When the shell exits, both the
   forwarder and the tmux renderer shut down.

`flashback-shell capture` no longer calls `tmux capture-pane` on the user's
session directly. Instead it calls `tmux capture-pane` on the renderer session,
which already contains a faithful mirror of the terminal screen.

---

## CLI commands and flags (feasible subset)

Keep the same command names so existing scripts/docs stay recognizable. Drop
flags that are not meaningful for a socket-based architecture.

### Global flags

Global flags must appear **before** the subcommand.

| Flag | Kept? | Notes |
|------|-------|-------|
| `-c <path>` | yes | Config file path. |
| `-v`, `-vv`, `-vvv` | yes | Verbosity level. |
| `-l <path>` | yes | Log file. `-` means stderr. |
| `--no-capture` | yes | Disable background capture for `new`. |

Dropped global flags: none from the original set (the original only had these).

### `new [args...]`

Start a new shell wrapped in a PTY forwarder and launch a tmux renderer mirror.
Remaining args are passed to the shell.

Behavior:

- Pick shell binary: `cfg.Shell` → `$SHELL` → `/bin/bash`.
- Allocate a PTY and run the shell on it.
- Start a Unix socket listener on a path like
  `~/.flashback-shell/sockets/flashback-<pid>.sock`.
- Launch a detached tmux session whose single pane runs the built-in renderer
  connected to that socket.
- Put the user's terminal in raw mode and copy stdin↔PTY so the session feels
  like a normal shell.
- Handle `SIGWINCH` to resize the PTY and emit a `resize` event.
- On shell exit or signal, kill the tmux renderer session, remove the socket,
  restore the terminal, and exit.

### `capture`

Capture the current screen from the tmux renderer session(s) and upload.

Behavior:

- Discover renderer tmux sessions (scan `socket_dir` for active sockets, or tag
  sessions with a tmux session name prefix).
- For each renderer session, run `tmux capture-pane -p -e -t <target>`.
- Compute MD5 hash, deduplicate against `~/.flashback-shell/state/<id>.hash`.
- Upload changed captures to the configured server.
- Buffer on upload failure.

### `list`

List active forwarder sessions and their renderer tmux sessions.

### `kill <session-id>`

Kill the tmux renderer session and remove the socket for the given session.

### `check`

Validate dependencies (`tmux`, shell, writable socket/state/log dirs) and print
effective configuration.

---

## Component design

### 1. PTY forwarder (`internal/forwarder`)

Reuses the `golang-pty-forward-example` forwarder logic, packaged as a library.

Responsibilities:

- Start a command on a PTY via `github.com/creack/pty`.
- Put local terminal in raw mode (`golang.org/x/term`).
- Copy stdin → PTY and PTY → stdout.
- Emit JSON-RPC events: `output`, `input`, `resize`, `exit`, `eof`.
- Listen on a Unix socket and stream events to subscribers.
- Send an `init` event to new subscribers with current terminal size and a seed
  screen (optional; start with empty seed).
- Handle `SIGWINCH`: resize PTY, forward to child process group, publish
  `resize` event.
- Handle `SIGINT`/`SIGTERM`: forward to child, then clean up.

CLI entry for the forwarder is **not** a separate binary; it is invoked
internally by `flashback-shell new`.

### 2. Event bus + JSON-RPC protocol (`internal/event`, `internal/protocol`)

Reuses `golang-pty-forward-example/pkg/event` and `pkg/protocol`.

Event types:

```go
type Event struct { ID uint64; Time time.Duration }
type OutputEvent struct { Event; Data string } // base64
type InputEvent  struct { Event; Data string } // base64
type ResizeEvent struct { Event; Cols, Rows uint16 }
type ExitEvent   struct { Event; Status int }
type EofEvent    struct { Event }
type InitEvent   struct { Event; Cols, Rows uint16; Screen string }
```

Wire format: one JSON object per line, base64-encoded `data` fields.

### 3. tmux renderer (`internal/renderer`)

A Go program that runs **inside** tmux and replays events to its stdout. Built
into the same binary and invoked via a hidden `__render` subcommand or via
`flashback-shell --renderer <socket>`.

Responsibilities:

- Connect to the forwarder Unix socket.
- On `init`: write seed screen (if provided), emit ANSI resize sequence.
- On `output`: decode base64 and write bytes to stdout.
- On `resize`: emit `CSI 8 ; rows ; cols t` and/or `TIOCSWINSZ` on stdout fd.
- On `exit`/`eof`: exit cleanly.
- Drain stdin to avoid flow-control issues (read-only mirror).

Why run inside tmux instead of rendering directly? Because tmux gives us:

- A stable target for `tmux capture-pane`.
- Scrollback history.
- Session management (attach/detach).
- No need to rewrite a VT emulator/screen grid.

### 4. tmux launcher (`internal/tmux`)

Small wrapper around tmux for the renderer session:

- Session name: `flashback-<pid>` (matching the forwarder pid).
- Socket path: `~/.flashback-shell/tmux/<session>`.
- Start detached: `tmux -S <socket> new-session -d -s <name> -x <cols> -y <rows>
  "flashback-shell --renderer <forwarder-socket>"`.
- Kill session, check existence, list panes.

### 5. Capture engine (`internal/capture`)

Reuses the upload/hash/dedup/buffer logic from `terminal-capture-shell-golang`
but reads from renderer tmux sessions instead of the user's tmux session.

- Discover sessions by scanning `~/.flashback-shell/tmux/`.
- `tmux capture-pane -p -e -t <session>:0.0` for ANSI.
- `tmux capture-pane -p -t <session>:0.0` for plain text.
- MD5 hash, dedup, diff-only, text-only, scrollback support as before.

### 6. Server upload client (`internal/server`)

Reused from `terminal-capture-shell-golang`.

### 7. Config + logging (`internal/config`, `internal/log`)

Reused from `terminal-capture-shell-golang`. Keep env var names and config file
schema for backward compatibility, but remove options that no longer apply.

---

## Configuration

Keep the same config file path and env vars where possible.

### Kept config/env options

| Config key | Env var | Purpose |
|------------|---------|---------|
| `server_url` | `FLASHBACK_SHELL_SERVER_URL` | Upload endpoint. |
| `socket_dir` | `FLASHBACK_SHELL_SOCKET_DIR` | tmux socket / forwarder socket directory. |
| `shell` | `FLASHBACK_SHELL_SHELL` | Shell binary. |
| `buffer_size` | `FLASHBACK_SHELL_BUFFER_SIZE` | Max buffered upload batches. |
| `device_id` | `FLASHBACK_SHELL_DEVICE_ID` | Device identifier in uploads. |
| `capture_interval` | `FLASHBACK_SHELL_CAPTURE_INTERVAL` | Background capture interval for `new`. |
| `disable_capture` | `FLASHBACK_SHELL_DISABLE_CAPTURE` | Disable background capture. |
| `capture_scrollback` | `FLASHBACK_SHELL_CAPTURE_SCROLLBACK` | Capture full tmux scrollback. |
| `diff_only` | `FLASHBACK_SHELL_DIFF_ONLY` | Upload only newly appeared lines. |
| `diff_mode` | `FLASHBACK_SHELL_DIFF_MODE` | `suffix` or `index`. |
| `text_only` | `FLASHBACK_SHELL_TEXT_ONLY` | Plain-text captures only. |

### Dropped config options

- `allow_nested_tmux` — the forwarder runs the shell directly in a PTY, so the
  old nested-tmux problem disappears.

---

## Directory layout

```
terminal-capture-shell-golang-tmux-renderer/
├── go.mod
├── main.go                         // CLI entrypoint
├── build.sh                        // build the single binary
├── test.sh                         // smoke tests
├── README.md
├── internal/
│   ├── cmd/
│   │   ├── new.go                  // new command
│   │   ├── capture.go              // capture command
│   │   ├── list.go                 // list command
│   │   ├── kill.go                 // kill command
│   │   ├── check.go                // check command
│   │   └── renderer.go             // hidden renderer subcommand
│   ├── config/
│   │   └── config.go               // config + env + logger setup
│   ├── log/
│   │   └── log.go                  // leveled logger
│   ├── forwarder/
│   │   └── forwarder.go            // PTY + socket forwarder
│   ├── event/
│   │   └── event.go                // event bus
│   ├── protocol/
│   │   └── protocol.go             // JSON-RPC encode/decode
│   ├── renderer/
│   │   └── renderer.go             // tmux renderer loop
│   ├── tmux/
│   │   └── tmux.go                 // renderer tmux session mgmt
│   ├── capture/
│   │   ├── capture.go              // capture engine
│   │   └── buffer.go               // circular retry buffer
│   └── server/
│       └── client.go               // HTTP upload client
```

---

## Data flow

### `new` flow

```
flashback-shell new [-c cfg] [-v] [-l log] [--no-capture] [shell args...]
  │
  ├─ Load config, init logger.
  ├─ Pick shell binary and args.
  ├─ Choose socket path: ~/.flashback-shell/sockets/flashback-<pid>.sock
  ├─ Start forwarder: PTY + socket listener.
  │     ├─ Set terminal raw mode.
  │     ├─ Copy stdin↔PTY, PTY→stdout.
  │     ├─ Publish output/resize events to socket.
  │
  ├─ Start tmux renderer session:
  │     tmux new-session -d -s flashback-<pid> \
  │       "flashback-shell --renderer <socket-path>"
  │
  ├─ (optional) Start background capture goroutine.
  │
  ├─ Wait for shell exit / signal.
  │
  └─ Cleanup:
        ├─ Kill tmux renderer session.
        ├─ Close forwarder socket, remove socket file.
        ├─ Restore terminal.
        └─ Exit with shell status.
```

### `capture` flow

```
flashback-shell capture
  │
  ├─ Scan ~/.flashback-shell/tmux/ for renderer sessions.
  ├─ For each session:
  │     ├─ capture-pane -p -e  → ANSI
  │     ├─ capture-pane -p     → text
  │     ├─ MD5 hash
  │     ├─ dedup against state file
  │     └─ if changed → add to batch
  ├─ If batch not empty:
  │     ├─ upload to server_url
  │     └─ update hash files
  └─ Exit
```

---

## Implementation phases

### Phase 0: Bootstrap

1. `go mod init flashback-shell-tmux-renderer` (or keep module name consistent).
2. Copy/adapt `internal/config`, `internal/log` from
   `terminal-capture-shell-golang`.
3. Set up `main.go` with global flags and subcommand dispatch.

### Phase 1: PTY forwarder + JSON-RPC

1. Port `pkg/event`, `pkg/protocol`, `pkg/ptywrap`, `pkg/rawmode` from
   `golang-pty-forward-example` into `internal/event`, `internal/protocol`,
   etc.
2. Implement `internal/forwarder.Forwarder` that starts a PTY and publishes
   events to an event bus.
3. Add Unix socket listener that streams events to subscribers.
4. Add `SIGWINCH` and signal forwarding.
5. Write a small smoke test: run `bash -c 'echo hello'` through the forwarder
   and verify `output` events contain "hello".

### Phase 2: tmux renderer

1. Implement `internal/renderer.Renderer` that connects to a socket and writes
   output bytes to stdout.
2. Add `--renderer <socket>` hidden CLI path.
3. Manually launch it inside tmux and verify the screen mirrors the forwarder.
4. Add `TIOCSWINSZ` resize handling.

### Phase 3: `new` command integration

1. `cmd/new.go`:
   - Start forwarder in a goroutine.
   - Launch tmux renderer session.
   - Attach user terminal to forwarder PTY (stdin/stdout loop).
   - Clean up on exit.
2. Add `--no-capture` support.
3. Add background capture goroutine.

### Phase 4: `capture`, `list`, `kill`, `check`

1. `cmd/capture.go`: scan renderer tmux sessions, capture-pane, hash, dedup,
   upload.
2. `cmd/list.go`: list active sessions.
3. `cmd/kill.go`: kill renderer session + remove socket.
4. `cmd/check.go`: dependency/config validation.

### Phase 5: Polish

1. `build.sh` producing a single static binary.
2. `test.sh` with non-interactive smoke tests.
3. `README.md` documenting the new architecture and CLI.
4. Clean up socket/session leaks on crash (`list` should prune stale sockets).

---

## Build / test deliverables

### `build.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
go mod tidy
CGO_ENABLED=0 go build -o dist/flashback-shell .
echo "built dist/flashback-shell"
```

### `test.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
./build.sh

# Smoke test: non-interactive command through forwarder
rm -f /tmp/flashback-test.sock
./dist/flashback-shell --renderer /tmp/flashback-test.sock &
renderer_pid=$!
./dist/flashback-shell --forwarder /tmp/flashback-test.sock bash -c 'echo smoke-test' &
forwarder_pid=$!

sleep 1
# Verify renderer stdout contains the echoed text
test -f /tmp/flashback-test-renderer.out
wait $forwarder_pid
kill $renderer_pid 2>/dev/null || true

echo "smoke test passed"
```

(Adjust once the internal forwarder/renderer invocation shape is finalized.)

---

## Key design decisions

1. **Single binary**. The forwarder, renderer, and CLI all live in one
   executable invoked via subcommands/hidden flags. This simplifies deployment
   and matches the original static-binary goal.

2. **tmux is the capture target, not the session backend**. The user's shell
   runs directly on a PTY (no tmux wrapping). A separate tmux session runs the
   renderer and receives the event stream. `tmux capture-pane` is performed on
   the renderer, giving us a faithful, scrollback-capable mirror.

3. **Unix socket, not WebSocket/TCP by default**. The renderer runs locally,
   so a Unix socket is the simplest transport. TCP can be added later if
   needed.

4. **JSON-RPC line protocol**. Matches the reference implementation and is easy
   to debug with `nc -U <socket>`.

5. **Keep backward-compatible CLI and config**. Users can continue to run
   `flashback-shell new`, `flashback-shell capture`, etc., with the same env
   vars.

6. **Drop `allow_nested_tmux`**. Because the shell no longer runs inside tmux,
   nested-tmux guards are irrelevant.

---

## Risks and caveats

| Risk | Mitigation |
|------|------------|
| tmux must still be installed for capture/renderer. | `check` validates tmux; `new` can fall back to plain PTY shell without renderer if tmux is missing (capture disabled). |
| Terminal capability mismatch between user terminal and tmux renderer. | Force `TERM=xterm-256color` in both forwarder and renderer environments. |
| Renderer crashes / forwarder socket left behind. | Cleanup on signal; `list`/`kill` prune stale sockets/sessions. |
| Fast output overloads the event bus or renderer. | Non-blocking sends with dropped-event counting; consider bounded channels. |
| Late renderer joins see empty screen. | Phase 2 can seed an `init.Screen` from a minimal VT emulator; start without it. |
| Mouse input byte-transparent only. | Acceptable for first version; explicit mouse events are future work. |

---

## Open questions

1. Should `new` default to starting the renderer in a detached tmux session
   that the user can later attach to, or should it be a private session owned
   by the forwarder? **Proposal**: private session owned by the forwarder;
   `list`/`kill` manage it.

2. Should the forwarder socket path be derived from the session name or the
   process PID? **Proposal**: PID-based (`flashback-<pid>.sock`) for simplicity
   and uniqueness.

3. Should we keep a separate capture daemon or only capture from `new`'s
   background goroutine and explicit `capture` calls? **Proposal**: same model
   as today — `new` spawns a background capture goroutine, and `capture` is a
   manual trigger.

4. Should the renderer emit its own `capture-pane`-ready hooks, or should
   `capture` poll tmux directly? **Proposal**: `capture` polls tmux directly;
   no renderer protocol changes needed.
