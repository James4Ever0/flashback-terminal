# Bash Forwarder with VT

This folder contains a variant of `golang-pty-forward-example` that integrates
[charmbracelet/x/vt](https://github.com/charmbracelet/x) (a Go VT terminal
emulator) directly into the forwarder.

The forwarder now internally renders the terminal output, knows the cursor
position, keeps a bounded scrollback, and can produce ANSI and plain-text
captures on demand.

## Why this exists

The original `golang-pty-forward-example` forwards raw bytes and relies on a
separate viewer (e.g., `tmux`) to render them. This version keeps an internal
virtual terminal so the forwarder itself knows what the screen looks like at any
time, enabling `tmux capture-pane`-style captures without a tmux dependency.

## Components

- `forwarder` — wraps a PTY program, mirrors the local terminal, emits
  JSON-RPC events, and maintains an internal VT state for capture.
- `receiver` — connects to a forwarder socket and replays output.
- `logger` — human-readable event viewer.

## Build

```bash
./build.sh
```

## Usage

### Interactive shell in Konsole with mirror

Terminal 1:

```bash
./bin/forwarder --listen /tmp/mirror.sock bash
```

Terminal 2 (inside tmux):

```bash
tmux new-session -s mirror './bin/receiver /tmp/mirror.sock'
```

### Non-interactive capture

```bash
./bin/forwarder --headless --no-capture \
    --capture out.ansi --capture-text out.txt \
    bash -c 'echo hello; tput setaf 1; echo red'

cat out.txt
```

### Flags

```text
  -capture string
        on exit, write an ANSI screen capture to this path
  -capture-text string
        on exit, write a plain text screen capture to this path
  -headless
        do not attach the local terminal (non-interactive)
  -listen string
        Unix socket or host:port to forward events to
  -log string
        path to write JSON event log
  -no-capture
        do not capture input events
  -scrollback int
        maximum scrollback lines kept by the VT emulator (default 1000)
  -vt-log string
        continuously write the VT screen capture (ANSI) to this path
  -vt-log-interval duration
        how often to write the VT capture log (default 2s)
```

## Capture API

The `pkg/vtcapture` package wraps `vt.Emulator` and exposes:

- `Write(p []byte)` — feed output bytes.
- `Resize(cols, rows)` — resize the virtual terminal.
- `SetScrollbackSize(n)` — limit history.
- `CaptureANSI() []byte` — current screen as ANSI-ish text.
- `CaptureText() []byte` — current screen as plain text.
- `CaptureScrollbackANSI(n)` / `CaptureScrollbackText(n)` — include last N
  scrollback lines.
- `Cursor()` — current cursor row/col.

## Implementation notes

- `charmbracelet/x/vt` is used as the VT emulator.
- The local `reference/x-main/vt` directory is used via Go `replace`
  directives, so the project builds without network access.
- The forwarder feeds every PTY output byte into the VT emulator.
- The `init` event sent to new subscribers contains the current screen dump.

## Testing

```bash
./test.sh
```

## Future improvements

- Add a control socket so captures can be requested while the forwarder is
  running, not just on exit.
- Improve ANSI reconstruction to preserve colors and attributes.
- Add a scrollback query endpoint.
- Improve receiver resize handling with `TIOCSWINSZ`.
