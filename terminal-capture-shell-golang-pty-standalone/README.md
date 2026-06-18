# flashback-shell-pty (standalone)

A Go CLI that wraps an interactive terminal session inside a PTY, renders the
screen with an internal VT emulator, and optionally captures/uploads screen
snapshots.

This is the standalone variant: a single process owns the PTY and VT emulator.
There are no session sockets, no session list, and no external programs can
inspect the terminal view.

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

# Run a non-interactive command
./dist/flashback-shell-pty new bash -c 'echo hello'

# Validate dependencies and show effective configuration
./dist/flashback-shell-pty check
```

## Commands

| Command | Description |
| --- | --- |
| `new [flags] [command [args...]]` | Start a new shell inside a PTY. Remaining args are passed to the shell. |
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

## `new` flags

| Flag | Description |
| --- | --- |
| `--capture <path>` | On exit, write an ANSI screen capture to this path. |
| `--capture-text <path>` | On exit, write a plain text screen capture to this path. |
| `--vt-log <path>` | Continuously dump the VT screen capture (ANSI) to this path. |
| `--vt-log-interval <duration>` | How often to write the VT capture log (default `2s`). |

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

# Shell binary inside sessions (empty = $SHELL or /bin/bash)
shell: ""

# Maximum buffered capture batches kept for retry
buffer_size: 100

# Retry buffer backend: "memory" (default) or "disk"
buffer_mode: memory

# Parent directory for the disk retry buffer (only used when buffer_mode: disk).
# Default is /tmp, which is cleared on reboot.
buffer_dir: "/tmp"

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

# Allow starting a flashback-shell session inside another flashback-shell
# session.
allow_nested: false
```

### Environment variables

Environment variables override config-file values:

| Variable | Maps to |
| --- | --- |
| `FLASHBACK_SHELL_PTY_SERVER_URL` | `server_url` |
| `FLASHBACK_SHELL_PTY_SHELL` | `shell` |
| `FLASHBACK_SHELL_PTY_BUFFER_SIZE` | `buffer_size` |
| `FLASHBACK_SHELL_PTY_BUFFER_MODE` | `buffer_mode` |
| `FLASHBACK_SHELL_PTY_BUFFER_DIR` | `buffer_dir` |
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

1. `new` starts a shell attached to a PTY in the current process.
2. It enters raw mode on the local terminal and bridges stdin/stdout to the
   PTY.
3. All PTY output is fed into an internal VT emulator
   (`charmbracelet/x/vt`).
4. A background goroutine periodically captures the VT screen, deduplicates
   it, optionally applies diff/text-only transforms, and uploads changes to
   the configured `server_url`.
5. Failed uploads are stored in a bounded retry buffer. By default the buffer
   is kept in a temporary subfolder under `/tmp` and removed when the process
   exits. Set `buffer_mode: disk` to spill the buffer to the configured
   `buffer_dir`.
6. On shell exit, optional `--capture` and `--capture-text` files are written.

## Local directories

| Path | Purpose |
| --- | --- |
| `~/.config/terminal-capture-shell-pty.yaml` | Config file. |
| `~/.flashback-shell-pty/log/flashback-shell-pty.log` | Default log file. |

No socket, state, or persistent buffer directories are created.

## Differences from flashback-shell (tmux)

- No tmux dependency.
- Single-process PTY wrapper; no session server or Unix sockets.
- No `capture`, `list`, or `kill` commands.
- Captures come from the VT emulator instead of `tmux capture-pane`.

## Testing

```bash
./test.sh
```
