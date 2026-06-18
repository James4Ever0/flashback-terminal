# flashback-shell

A small Go CLI that wraps `tmux` (or a plain shell fallback) to start an
interactive terminal session, periodically capture its pane contents, and upload
captures to a remote server.

## Requirements

- Go 1.22+ (to build from source)
- `tmux` installed and on `$PATH` (optional; a fallback shell is used if tmux is
  missing)

## Build

```bash
./build.sh
```

The script produces a static, CGO-free binary at `dist/flashback-shell`.

## Quick start

```bash
# Start a new shell
./dist/flashback-shell new

# List running sessions
./dist/flashback-shell list

# Manually capture all sessions and upload changes
./dist/flashback-shell capture

# Validate dependencies and show effective configuration
./dist/flashback-shell check
```

## Commands

| Command | Description |
| --- | --- |
| `new [args...]` | Start a new shell. Uses tmux if available; otherwise falls back to a plain shell. Remaining args are passed to the shell. |
| `capture` | Capture all managed tmux sessions and upload any changed panes. |
| `list` | List managed tmux sessions and their pane counts. Dead/stale sessions are hidden and cleaned up automatically. |
| `kill <session-id>` | Kill a specific managed session. |
| `check` | Validate required external binaries and print the effective configuration plus the source (`env`, `config`, `default`) of each value. |

## Global flags

Global flags must appear **before** the subcommand:

```bash
flashback-shell -vvv -l - new
```

| Flag | Description |
| --- | --- |
| `-c <path>` | Config file path. Default: `~/.config/terminal-capture-shell.yaml`. |
| `-v`, `-vv`, `-vvv` | Verbosity level. `-v` = warn, `-vv` = info, `-vvv` = debug. Without flags only errors are logged. |
| `-l <path>` | Log output file. Default: `~/.flashback-shell/log/flashback-shell.log`. |
| `-l -` | Log to stderr. |
| `-l /dev/stdout` | Log to stdout. Not recommended with `new` because it can corrupt the tmux attach output. |
| `--no-capture` | Disable background capture for this session. Affects only `new`. |

## Logging

`flashback-shell` uses a **single global log file** shared across every run and
every session. It is not created per session and is unrelated to the random
session name (`flashback-<pid>`).

- **Default path:** `~/.flashback-shell/log/flashback-shell.log`
- **Behavior:** every command appends to this file. It records config loading,
  session start/attach/kill, background capture activity, signal handling, and
  upload results.
- **Timestamps:** every log line includes the local timezone abbreviation (e.g.
  `2026-06-13 17:01:09 CST`).
- **Upload details:** when a capture is posted, the log records the HTTP method,
  full URL, number of captures, payload size in bytes, each retry attempt, the
  server response status, and the final result (success or failure). Network
  timeouts and refused connections are also logged.
- **No-server case:** if `server_url` is empty, the log explicitly states that
  the upload was declined and the reason why.
- **Rotation:** the application does not rotate or truncate the file. Manage it
  externally if it grows large.

### Capture upload details

Captures are uploaded with an HTTP **POST** to:

```
<server_url>/api/captures
```

The request body is JSON and includes:

- `device_id` — from config/env, or the hostname if unset
- `timestamp` — UTC RFC 3339 timestamp
- `captures` — array of changed pane contents

The HTTP client enforces a **30-second timeout** per attempt and retries up to
three times with exponential backoff. If all retries fail, the batch is saved to
`~/.flashback-shell/buffer/` for a later `capture` or `new` flush.

### Full scrollback capture

By default, only the **visible pane** is captured (`tmux capture-pane`). Set
`capture_scrollback: true` to capture the entire tmux history buffer for each
pane (`tmux capture-pane -S -`).

- Default: `false`
- Env override: `FLASHBACK_SHELL_CAPTURE_SCROLLBACK=true`

Use with care: if tmux `history-limit` is large, each capture can be many
megabytes and the MD5-based deduplication will almost always change, so every
interval uploads the full buffer.

### Nested tmux sessions

By default, flashback-shell does **not** unset `$TMUX`, so tmux will refuse to
start a session inside another tmux session with its standard guard message.
Set `allow_nested_tmux: true` to strip `TMUX`, `TMUX_PANE`, `TMUX_WINDOW`, and
`TMUX_SESSION` from the session environment, allowing flashback-shell to nest.

- Default: `false`
- Env override: `FLASHBACK_SHELL_ALLOW_NESTED_TMUX=true`

Only enable this if you intentionally want flashback-shell inside an existing
tmux session.

### Diff-only capture

Set `diff_only: true` to upload only the lines that **newly appeared** since the
previous capture, instead of the whole pane each time.

- Default: `false`
- Env override: `FLASHBACK_SHELL_DIFF_ONLY=true`

You can choose the diff algorithm with `diff_mode`:

- `"suffix"` (default) — finds the longest overlap between the end of the
  previous capture and the beginning of the current capture, then returns the
  trailing new lines. This models a ring-like scrollback where old lines scroll
  off the top and new lines append at the bottom.
- `"index"` — aligns the previous capture to the current capture length by
  trimming old lines from the top or padding with empty lines, then returns only
  the lines that changed at each vertical position. This works better for
  in-place screen updates such as progress bars or full-screen TUIs.

Behavior:

- The first capture for a pane always returns the full content (so the server
  has a baseline).
- Subsequent captures compare the current pane against the previous capture and
  return only the lines that changed or newly appeared, depending on the mode.
- If no lines differ, nothing is uploaded and the baseline is still updated.

Diff-only is most useful together with `capture_scrollback: true` so that newly
scrolled-in history lines are sent incrementally rather than as a full buffer
every interval.

Example:

```bash
# Default suffix diff for ring-like scrollback
FLASHBACK_SHELL_DIFF_ONLY=true FLASHBACK_SHELL_CAPTURE_SCROLLBACK=true flashback-shell new

# Index diff for in-place terminal updates
FLASHBACK_SHELL_DIFF_ONLY=true FLASHBACK_SHELL_DIFF_MODE=index flashback-shell new
```

### Text-only capture

Set `text_only: true` to send only the plain-text capture and omit ANSI escape
codes from the upload. The existing JSON structure is unchanged; the `ansi`
field is simply empty and each capture includes `metadata: { ansi: "false" }` so
the server knows the payload contains no formatting.

- Default: `false`
- Env override: `FLASHBACK_SHELL_TEXT_ONLY=true`

This is useful when the server only needs searchable text or when you want to
reduce payload size.

Example:

```bash
FLASHBACK_SHELL_TEXT_ONLY=true flashback-shell new
```

### Verbosity levels

The logger supports four levels:

| Level | Shown by default? | Enabled with |
| --- | --- | --- |
| ERROR | yes | always |
| WARN | no | `-v` |
| INFO | no | `-vv` |
| DEBUG | no | `-vvv` |

Examples:

```bash
# Default: errors only, written to the default log file
flashback-shell new

# WARN-level logging to the default log file
flashback-shell -v new

# INFO-level logging to the default log file
flashback-shell -vv new

# DEBUG-level logging to stderr for troubleshooting
flashback-shell -vvv -l - new

# Custom log file
flashback-shell -l /tmp/flashback.log new
```

### Why logs do not go to stdout/stderr by default

`new` runs `tmux attach` with inherited stdin/stdout/stderr. Writing log lines
to stdout or stderr would corrupt the terminal session output, so the default
log destination is a file. Use `-l -` or `-l /dev/stderr` only when debugging
and you accept the output mixing.

## Configuration

### Config file

The default config file is created automatically at
`~/.config/terminal-capture-shell.yaml` if it does not exist.

Example:

```yaml
# Remote server URL for uploading captures (empty = local only)
server_url: ""

# Directory where tmux socket files are stored
socket_dir: "~/.flashback-shell/tmux"

# Shell binary inside tmux sessions (empty = $SHELL or /bin/bash)
shell: ""

# Maximum buffered capture batches kept for retry
buffer_size: 100

# Device identifier sent with uploads (empty = hostname)
device_id: ""

# Seconds between background captures for 'new' command (0 = disable)
capture_interval: 30

# Disable background capture entirely for 'new' command
disable_capture: false

# Capture the full tmux scrollback buffer instead of only the visible pane.
# WARNING: with a large history-limit this can produce very large payloads.
capture_scrollback: false

# Allow flashback-shell to run inside an existing tmux session by unsetting
# TMUX variables. Default false; tmux will refuse to nest without this.
allow_nested_tmux: false

# Only capture lines that newly appeared since the previous capture.
# First capture always returns the full buffer/screen.
diff_only: false

# Diff algorithm when diff_only is true: "suffix" (default) or "index".
# See the "Diff-only capture" section above for details.
diff_mode: suffix

# Send only plain-text captures; omit ANSI escape codes from uploads.
# Each capture will include metadata.ansi=false when enabled.
text_only: false
```

### Environment variables

Environment variables override config-file values:

| Variable | Maps to |
| --- | --- |
| `FLASHBACK_SHELL_SERVER_URL` | `server_url` |
| `FLASHBACK_SHELL_SOCKET_DIR` | `socket_dir` |
| `FLASHBACK_SHELL_SHELL` | `shell` |
| `FLASHBACK_SHELL_BUFFER_SIZE` | `buffer_size` |
| `FLASHBACK_SHELL_DEVICE_ID` | `device_id` |
| `FLASHBACK_SHELL_CAPTURE_INTERVAL` | `capture_interval` |
| `FLASHBACK_SHELL_DISABLE_CAPTURE` | `disable_capture` |
| `FLASHBACK_SHELL_CAPTURE_SCROLLBACK` | `capture_scrollback` |
| `FLASHBACK_SHELL_ALLOW_NESTED_TMUX` | `allow_nested_tmux` |
| `FLASHBACK_SHELL_DIFF_ONLY` | `diff_only` |
| `FLASHBACK_SHELL_DIFF_MODE` | `diff_mode` |
| `FLASHBACK_SHELL_TEXT_ONLY` | `text_only` |

Boolean env vars accept `true`/`1`/`yes`/`on` for true and `false`/`0`/`no`/`off` for false, so a config-file true can be overridden back to false from the environment.

### Precedence

```
env vars > config file > defaults
```

The `--no-capture` CLI flag is applied after environment variables, so it always
wins for the current run regardless of config-file or env settings.

## Session lifecycle and cleanup

When you run `flashback-shell new`:

1. A detached tmux session is created on a dedicated Unix socket in
   `socket_dir` (`~/.flashback-shell/tmux/flashback-<pid>`).
2. A background goroutine captures pane contents every `capture_interval`
   seconds if capture is enabled.
3. `tmux attach` runs as a subprocess with inherited stdio.
4. On normal exit, signal (`SIGINT`/`SIGTERM`), or panic, the session is killed
   and the socket file is removed.
5. `flashback-shell list` also skips and deletes any stale sockets it finds, so
   dead sessions never appear.

## Local directories

| Path | Purpose |
| --- | --- |
| `~/.config/terminal-capture-shell.yaml` | Config file. |
| `~/.flashback-shell/log/flashback-shell.log` | Default log file. |
| `~/.flashback-shell/tmux/` | tmux socket files and temporary `tmux.conf`. |
| `~/.flashback-shell/state/` | Per-pane hashes used for deduplication, plus `.prev` snapshots when `diff_only` is enabled. |
| `~/.flashback-shell/buffer/` | Buffered capture batches pending upload retry. |

## Examples

```bash
# Start a new bash-like session with debug logs visible on stderr
flashback-shell -vvv -l - new

# Start a shell without background capture
flashback-shell --no-capture new

# Capture the full scrollback buffer instead of only the visible pane
FLASHBACK_SHELL_CAPTURE_SCROLLBACK=true flashback-shell new

# Capture only newly appeared lines (best with capture_scrollback=true)
FLASHBACK_SHELL_DIFF_ONLY=true FLASHBACK_SHELL_CAPTURE_SCROLLBACK=true flashback-shell new

# Run flashback-shell inside an existing tmux session
FLASHBACK_SHELL_ALLOW_NESTED_TMUX=true flashback-shell new

# Start zsh instead of the configured/default shell
FLASHBACK_SHELL_SHELL=/bin/zsh flashback-shell new

# Run a one-off command inside the wrapped shell
flashback-shell new -c 'echo hello && exit'

# Capture/upload now, independent of any running session
flashback-shell capture
```
