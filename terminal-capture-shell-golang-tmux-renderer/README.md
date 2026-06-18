# flashback-shell — Socket-Based Forwarder + tmux Renderer

A Go CLI that wraps an interactive shell in a PTY, mirrors its IO to a local
Unix socket, and renders the screen inside a separate tmux session so it can be
captured with `tmux capture-pane`.

This replaces the previous tmux-based session launcher with a forwarder/renderer
architecture:

```
Your terminal
      |
      v
+---------------+
| flashback-shell new  |
|  - PTY master |
|  - shell      |
|  - socket bus |
+---------------+
      |
      | Unix socket (JSON-RPC events)
      v
+---------------+
| tmux renderer |
|  (read-only)  |
+---------------+
      |
      v
  tmux session/pane  <--  flashback-shell capture
```

## Requirements

- Go 1.22+ (to build from source)
- `tmux` installed and on `$PATH` (required for renderer and capture)

## Build

```bash
./build.sh
```

Produces a static, CGO-free binary at `dist/flashback-shell`.

## Quick start

```bash
# Start a new shell with background capture
./dist/flashback-shell new

# List renderer sessions
./dist/flashback-shell list

# Capture renderer sessions and upload changes
./dist/flashback-shell capture

# Validate dependencies and show effective config
./dist/flashback-shell check
```

## Commands

| Command | Description |
| --- | --- |
| `new [args...]` | Start a new shell wrapped in a PTY forwarder and launch a tmux renderer. Remaining args are passed to the shell. |
| `capture` | Capture all renderer tmux sessions and upload changed panes. |
| `list` | List managed renderer tmux sessions. |
| `kill <session-id>` | Kill a specific renderer session and remove its socket. |
| `check` | Validate required external binaries and print the effective configuration. |
| `__renderer <socket>` | Hidden subcommand used internally by tmux to run the renderer pane. |

## Global flags

Global flags must appear **before** the subcommand:

```bash
flashback-shell -vvv -l - new
```

| Flag | Description |
| --- | --- |
| `-c <path>` | Config file path. Default: `~/.config/terminal-capture-shell.yaml`. |
| `-v`, `-vv`, `-vvv` | Verbosity level. `-v` = warn, `-vv` = info, `-vvv` = debug. |
| `-l <path>` | Log output file. Default: `~/.flashback-shell/log/flashback-shell.log`. |
| `-l -` | Log to stderr. |
| `-l /dev/stdout` | Log to stdout. Not recommended with `new`. |
| `--no-capture` | Disable background capture for this session. Affects only `new`. |

## Configuration

The default config file is created automatically at
`~/.config/terminal-capture-shell.yaml` if it does not exist. You can also copy
the bundled example to that location and edit it:

```bash
mkdir -p ~/.config
cp terminal-capture-shell.example.yaml ~/.config/terminal-capture-shell.yaml
```

Use `-c <path>` to use a different config file for a single run:

```bash
flashback-shell -c /path/to/my-config.yaml new
```

A minimal example:

```yaml
# Remote server URL for uploads (empty = local only)
server_url: ""

# Directory where tmux renderer sockets and forwarder sockets are stored
socket_dir: "~/.flashback-shell/tmux"

# Shell binary (empty = $SHELL or /bin/bash)
shell: ""

# Maximum buffered capture batches kept for retry
buffer_size: 100

# Device identifier sent with uploads (empty = hostname)
device_id: ""

# Seconds between background captures for 'new' (0 = disable)
capture_interval: 30

# Disable background capture entirely for 'new'
disable_capture: false

# Capture the full tmux scrollback buffer instead of only the visible pane
capture_scrollback: false

# Only capture lines that newly appeared since the previous capture
diff_only: false

# Diff algorithm when diff_only is true: "suffix" (default) or "index"
diff_mode: suffix

# Send only plain-text captures; omit ANSI escape codes from uploads
text_only: false
```

See `terminal-capture-shell.example.yaml` in this directory for the full,
commented template.

Environment variables override config-file values. See `flashback-shell --help`
for the full list.

## Session lifecycle

When you run `flashback-shell new`:

1. A PTY is allocated and the configured shell is started.
2. A Unix socket is created at `~/.flashback-shell/tmux/flashback-<pid>.fwd.sock`.
3. A detached tmux session named `flashback-<pid>` is created whose single pane
   runs `flashback-shell __renderer <forwarder-socket>`.
4. Your terminal is put into raw mode and attached to the PTY.
5. A background goroutine captures the renderer pane every `capture_interval`
   seconds if capture is enabled.
6. On shell exit, signal, or panic, the renderer tmux session is killed and the
   socket files are removed.

## Capture behavior

`flashback-shell capture` reads from the renderer tmux sessions, not from your
interactive shell directly. This means:

- The renderer faithfully mirrors the terminal screen, including ANSI colors and
  escape sequences.
- `tmux capture-pane` is performed on the renderer, giving scrollback history
  when `capture_scrollback` is enabled.
- Hash-based deduplication avoids uploading unchanged panes.

## Protocol

The forwarder speaks a line-delimited JSON-RPC-like protocol over its Unix
socket. Events include:

```json
{"method":"init","id":0,"time":0,"params":{"cols":80,"rows":24,"screen":""}}
{"method":"output","id":1,"time":12345,"params":{"data":"base64(...)"}}
{"method":"input","id":2,"time":12350,"params":{"data":"base64(...)"}}
{"method":"resize","id":3,"time":23456,"params":{"cols":100,"rows":30}}
{"method":"exit","id":4,"time":34567,"params":{"status":0}}
{"method":"eof","id":5,"time":34567,"params":{}}
```

`data` fields are base64-encoded to safely carry binary terminal bytes.

## Limitations

- **No seed screen for late renderer joins**. A renderer that connects after the
  shell has already emitted output will only see new output from that point on.
  This is acceptable for interactive shells where output is continuous, but
  one-shot commands may not be fully mirrored. A future enhancement is to add a
  minimal VT emulator that feeds an `init.Screen` snapshot to new subscribers.
- **`new` should be run in a real terminal**. Running `flashback-shell new`
  inside a detached tmux/screen session or with stdin redirected from
  `/dev/null` is not supported, because the forwarder needs a terminal for raw
  mode and PTY sizing.

## Differences from the previous tmux-based implementation

- The user's shell runs directly on a PTY, not inside tmux.
- A separate tmux session is used only as a read-only mirror/renderer.
- `allow_nested_tmux` is removed because nested tmux is no longer relevant.
- The CLI and config file remain backward-compatible for the supported options.

## Development

```bash
# Build
./build.sh

# Run smoke tests
./test.sh

# Run all Go tests
go test ./...
```
