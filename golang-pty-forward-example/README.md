# PTY Forward Example

A self-contained Go example that wraps a terminal program (`bash`, `vim`, ...)
in a PTY, mirrors the user's terminal, and forwards a stream of typed events to
a consumer over a Unix socket or TCP.

It is inspired by [asciinema](https://github.com/asciinema/asciinema) but uses a
simple line-delimited JSON-RPC-like protocol and is designed to feed a read-only
`tmux` receiver.

## Components

- `forwarder` — wraps a program in a PTY, copies bytes locally, and emits JSON
  events to a socket and/or a log file.
- `receiver` — connects to a forwarder socket and replays terminal output to its
  own stdout (intended to run inside `tmux`).
- `logger` — connects to a forwarder socket and prints human-readable JSON
  events.

## Requirements

- Go 1.22 or later
- Linux or macOS
- `tmux` (only if you want a read-only mirror session)

## Build

```bash
./build.sh
```

This produces `bin/forwarder`, `bin/receiver`, and `bin/logger`.

## Quick start

In one terminal, start a forwarder with `bash`:

```bash
./bin/forwarder --listen /tmp/pty.sock bash
```

In another terminal, watch the event stream:

```bash
./bin/logger /tmp/pty.sock
```

Or replay the session into a read-only terminal:

```bash
./bin/receiver /tmp/pty.sock
```

## Running inside tmux

Create a tmux pane that continuously displays the mirrored session:

```bash
tmux new-session -s mirror './bin/receiver /tmp/pty.sock'
```

Then capture the mirrored content with tmux:

```bash
tmux capture-pane -t mirror -p
```

## Protocol

Events are encoded as one JSON object per line (`\n`). All `data` fields are
base64-encoded so binary terminal bytes survive intact.

```json
{"method":"init","id":0,"time":0,"params":{"cols":80,"rows":24,"screen":"..."}}
{"method":"output","id":1,"time":12345,"params":{"data":"base64(...)"}}
{"method":"input","id":2,"time":12350,"params":{"data":"base64(...)"}}
{"method":"resize","id":3,"time":23456,"params":{"cols":100,"rows":30}}
{"method":"marker","id":4,"time":34567,"params":{"label":"checkpoint"}}
{"method":"exit","id":5,"time":45678,"params":{"status":0}}
{"method":"eof","id":6,"time":45678,"params":{}}
```

- `id` is monotonically increasing so receivers can detect gaps.
- `time` is elapsed microseconds since the session started.
- `init` is sent to each new subscriber and contains the current terminal size
  and an optional base64 screen snapshot.

## Forwarder usage

```text
Usage: forwarder [flags] <command> [args...]

Flags:
  -headless
        do not attach the local terminal (non-interactive)
  -listen string
        Unix socket or host:port to forward events to
  -log string
        path to write JSON event log
  -no-capture
        do not capture input events
```

Examples:

```bash
# Interactive bash with local mirroring and event capture
./bin/forwarder --listen /tmp/pty.sock bash

# Non-interactive command that records a log file
./bin/forwarder --headless --no-capture --log session.json bash -c 'echo hello'

# Listen on TCP instead of a Unix socket
./bin/forwarder --listen :4000 --headless --no-capture vim file.go
```

## Receiver usage

```bash
./bin/receiver /tmp/pty.sock
```

Connects to the forwarder and writes decoded output bytes to stdout. Resize
events emit an ANSI `CSI 8 ; rows ; cols t` sequence.

## Logger usage

```bash
./bin/logger /tmp/pty.sock
```

Prints a human-readable summary of each JSON event. Useful for debugging.

## Testing

```bash
./test.sh
```

The test script builds everything, runs a non-interactive smoke test through the
forwarder/receiver pair, and verifies that `init`, `output`, and `exit` events
are produced.

## Architecture

```
User terminal (raw mode)
        ^
        | raw bytes
        v
+---------------------------+
| forwarder                 |
|  - PTY master             |
|  - shell on PTY slave     |
|  - stdin  -> PTY          |
|  - PTY    -> stdout       |
|  - events -> socket bus   |
+---------------------------+
        |
        | line-delimited JSON events
        v
+---------------------------+
| Unix/TCP socket           |
+---------------------------+
        |
        +--> file logger (optional)
        +--> tmux receiver (read-only)
        +--> debug logger
```

## Design decisions

1. **Unix socket by default** — the consumer is usually a local `tmux` receiver,
   so a plain Unix domain socket is simpler than WebSocket.
2. **JSON-RPC line protocol** — easy to inspect with `netcat` or `logger`.
3. **Typed event bus** — adding new outputs (file logger, remote forwarder,
   multiple receivers) is just another subscriber.
4. **Base64 data fields** — keeps the protocol text-safe while carrying arbitrary
   terminal bytes.
5. **Raw mode + SIGWINCH** — keystrokes pass through unchanged and resizes are
   forwarded as first-class events.
6. **Mouse stays byte-transparent** — mouse sequences are forwarded through the
   PTY as raw input/output bytes, matching how asciinema handles them.

## Caveats

- The terminal capability of the receiver (`tmux` pane) may differ from the
  user's terminal. Set `TERM=xterm-256color` for best results.
- Fast curses applications generate many output events; the VT snapshot is not
  implemented in this first version, so the `init` screen field is empty.
- If the forwarder crashes, the Unix socket file may be left behind. Remove it
  manually before restarting.
- Input capture is disabled by default in headless/non-interactive runs via the
  `--no-capture` flag; in interactive runs it records every keystroke.

## License

MIT (same as the surrounding project).
