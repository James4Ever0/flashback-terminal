# flashback-shell-pty

A Go CLI that wraps an interactive terminal session inside a PTY, renders the
screen with an internal VT emulator, and captures/uploads screen snapshots.

This is a PTY/VT-based replacement for the tmux-backed `flashback-shell` in
`terminal-capture-shell-golang/`.

## Requirements

- Go 1.26+ (to build from source)
- A POSIX shell such as `bash` or `zsh`

## Build

```bash
./build.sh
```

The script produces a static, CGO-free binary at `dist/flashback-shell-pty`.

## Quick start

```bash
# Start a new shell
./dist/flashback-shell-pty new

# List running sessions
./dist/flashback-shell-pty list

# Capture all sessions and upload changes
./dist/flashback-shell-pty capture

# Kill a session
./dist/flashback-shell-pty kill <session-id>

# Validate dependencies and show effective configuration
./dist/flashback-shell-pty check
```

## Commands

| Command | Description |
| --- | --- |
| `new [args...]` | Start a new shell inside a PTY. Remaining args are passed to the shell. |
| `capture` | Capture all managed sessions and upload any changed screens. |
| `list` | List managed PTY sessions and their sizes. Dead/stale sessions are cleaned up. |
| `kill <session-id>` | Kill a specific managed session. |
| `check` | Validate required external binaries and print the effective configuration. |

## Global flags

Global flags must appear **before** the subcommand:

```bash
flashback-shell-pty -vvv -l - new
```

| Flag | Description |
| --- | --- |
| `-c <path>` | Config file path. Default: `~/.config/terminal-capture-shell-pty.yaml`. |
| `-v`, `-vv`, `-vvv` | Verbosity level. `-v` = warn, `-vv` = info, `-vvv` = debug. |
| `-l <path>` | Log output file. Default: `~/.flashback-shell-pty/log/flashback-shell-pty.log`. |
| `-l -` | Log to stderr. |
| `-l /dev/stdout` | Log to stdout. Not recommended with `new`. |
| `--no-capture` | Disable background capture for this session. |
| `--allow-nested` | Allow starting a session inside an existing flashback-shell session. |

## Nested sessions

The shell inside a `flashback-shell-pty new` session always has the
environment variable `FLASHBACK_SHELL=1`. By default, `new` refuses to start
when `FLASHBACK_SHELL` is already present in the environment, preventing
nested sessions. To override this for a single invocation, use the
`--allow-nested` global flag:

```bash
flashback-shell-pty --allow-nested new
```

To allow nested sessions by default, set `allow_nested: true` in the config
file or export `FLASHBACK_SHELL_PTY_ALLOW_NESTED=true`.

## Configuration

flashback-shell-pty looks for its config file at:

```
~/.config/terminal-capture-shell-pty.yaml
```

If the file does not exist, a template is created automatically at that path
the first time you run any command. You can also specify a different path with
the `-c` global flag.

A commented example is provided in [`config.example.yaml`](config.example.yaml).
Copy it to the default location and edit as needed:

```bash
cp config.example.yaml ~/.config/terminal-capture-shell-pty.yaml
```

Example contents:

```yaml
# Remote server URL for uploading captures (empty = local only)
server_url: ""

# Directory where session socket files are stored
socket_dir: "~/.flashback-shell-pty/sockets"

# Shell binary inside sessions (empty = $SHELL or /bin/bash)
shell: ""

# Maximum buffered capture batches kept for retry
buffer_size: 100

# Device identifier sent with uploads (empty = hostname)
device_id: ""

# Seconds between background captures for 'new' command (0 = disable)
capture_interval: 30

# Seconds to wait before the first background capture after 'new' starts.
first_capture_delay: 5

# Disable background capture entirely for 'new' command
disable_capture: false

# Only capture lines that newly appeared since the previous capture.
# First capture always returns the full screen.
diff_only: false

# Diff algorithm when diff_only is true: "suffix" (default) or "index".
diff_mode: suffix

# Send only plain-text captures; omit ANSI escape codes from uploads.
text_only: false

# Maximum scrollback lines kept by the VT emulator.
scrollback_lines: 1000

# Allow starting a session inside another flashback-shell session.
allow_nested: false
```

### Environment variables

Environment variables override config-file values:

| Variable | Maps to |
| --- | --- |
| `FLASHBACK_SHELL_PTY_SERVER_URL` | `server_url` |
| `FLASHBACK_SHELL_PTY_SOCKET_DIR` | `socket_dir` |
| `FLASHBACK_SHELL_PTY_SHELL` | `shell` |
| `FLASHBACK_SHELL_PTY_BUFFER_SIZE` | `buffer_size` |
| `FLASHBACK_SHELL_PTY_DEVICE_ID` | `device_id` |
| `FLASHBACK_SHELL_PTY_CAPTURE_INTERVAL` | `capture_interval` |
| `FLASHBACK_SHELL_PTY_FIRST_CAPTURE_DELAY` | `first_capture_delay` |
| `FLASHBACK_SHELL_PTY_DISABLE_CAPTURE` | `disable_capture` |
| `FLASHBACK_SHELL_PTY_DIFF_ONLY` | `diff_only` |
| `FLASHBACK_SHELL_PTY_DIFF_MODE` | `diff_mode` |
| `FLASHBACK_SHELL_PTY_TEXT_ONLY` | `text_only` |
| `FLASHBACK_SHELL_PTY_SCROLLBACK_LINES` | `scrollback_lines` |
| `FLASHBACK_SHELL_PTY_ALLOW_NESTED` | `allow_nested` |

### Precedence

```
cli flags > env vars > config file > defaults
```

## How it works

1. `new` starts a background **session server** that owns the PTY and an
   internal VT emulator (`charmbracelet/x/vt`).
2. The session server listens on a Unix socket in `socket_dir`.
3. The current CLI process becomes a client that attaches to the socket,
   enters raw mode, and bridges stdin/stdout.
4. A background ticker inside the server captures the VT screen and uploads
   changes to the configured `server_url`.
5. `capture` connects to every active session socket, requests a screen
   snapshot, applies deduplication/diff/text-only transforms, and uploads.
6. `list` and `kill` query the session server via the same socket protocol.

## Local directories

| Path | Purpose |
| --- | --- |
| `~/.config/terminal-capture-shell-pty.yaml` | Config file. |
| `~/.flashback-shell-pty/log/flashback-shell-pty.log` | Default log file. |
| `~/.flashback-shell-pty/sockets/` | Session server Unix sockets. |
| `~/.flashback-shell-pty/state/` | Per-session hashes and `.prev` snapshots. |
| `~/.flashback-shell-pty/buffer/` | Buffered capture batches pending upload retry. |

## Differences from flashback-shell (tmux)

- No tmux dependency.
- One shell per session (no panes/windows).
- Sessions are managed by internal Unix-socket servers instead of tmux sockets.
- Captures come from the VT emulator instead of `tmux capture-pane`.
- `allow_nested_tmux` is replaced by `allow_nested`, which guards against
  nested `flashback-shell-pty` sessions using the `FLASHBACK_SHELL`
  environment variable.
- `capture_scrollback` is replaced by `scrollback_lines`.

## Testing

```bash
./test.sh
```
