#!/usr/bin/env python3
"""
Stream an asciinema v3 ALiS WebSocket into a headless tmux session and audit it.

Architecture (see ./architecture.txt):
1. Create a Unix domain socket.
2. Spawn a detached tmux session (no status bar, kiosk bindings) running a
   small Python receiver that reads from the socket and writes to the tmux pane.
3. Connect to an asciinema v3 ALiS WebSocket.
4. Forward Output events to the socket and apply Resize/Init sizes via tmux
   shell commands (asyncio subprocess).
5. Periodically capture the tmux pane content, output it to stdout, and log
   byte count plus whether it changed.
6. Exit when the WebSocket closes.

Usage:
    python asciinema_stream_to_tmux_and_capture_headless.py
    python asciinema_stream_to_tmux_and_capture_headless.py --uri ws://127.0.0.1:33601/ws
    python asciinema_stream_to_tmux_and_capture_headless.py --capture-interval 2.0
"""

import argparse
import asyncio
import hashlib
import os
import shutil
import sys
import tempfile
import traceback
from datetime import datetime
from typing import Optional, Tuple

import websockets

URI = "ws://127.0.0.1:46257/ws"
SUBPROTOCOL = "v1.alis"
ALIS_MAGIC = b"ALiS\x01"

TMUX_CONF = """\
# Headless/kiosk tmux configuration for ALiS stream mirror
set -g status off
set -g mouse off
set -g default-terminal "xterm-256color"
set -g default-command ""
"""

# Small receiver that runs *inside* tmux, reads from the Unix socket and
# echoes bytes to its stdout (i.e., the tmux pane).
RECEIVER_SCRIPT = """\
import socket
import sys
import time

sock_path = sys.argv[1]
sock = None
last_error = None

for _ in range(100):
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(sock_path)
        break
    except (FileNotFoundError, ConnectionRefusedError) as exc:
        last_error = exc
        sock.close()
        sock = None
        time.sleep(0.05)
else:
    print(f"receiver: could not connect to {sock_path}: {last_error}", file=sys.stderr)
    sys.exit(1)

try:
    while True:
        data = sock.recv(8192)
        if not data:
            break
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
finally:
    sock.close()
"""


class StreamBridge:
    """Bridges ALiS Output bytes from the WebSocket into the tmux receiver."""

    def __init__(self) -> None:
        self._writer: Optional[asyncio.StreamWriter] = None
        self.connected = asyncio.Event()
        self._lock = asyncio.Lock()

    async def handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        addr = writer.get_extra_info("sockname")
        print(f"[bridge] tmux receiver connected on {addr}")
        self._writer = writer
        self.connected.set()
        try:
            # Keep the connection open until the other side closes it.
            while True:
                data = await reader.read(4096)
                if not data:
                    break
        except ConnectionResetError:
            pass
        finally:
            print("[bridge] tmux receiver disconnected")
            self._writer = None
            self.connected.clear()

    async def write(self, data: bytes) -> None:
        """Write bytes to the tmux receiver (best-effort)."""
        async with self._lock:
            writer = self._writer
            if writer is None:
                print(f"[bridge] no receiver connected, dropping {len(data)} bytes")
                return
            writer.write(data)
            await writer.drain()

    async def close(self) -> None:
        async with self._lock:
            writer = self._writer
            if writer is not None:
                writer.close()
                await writer.wait_closed()
                self._writer = None
                self.connected.clear()


def decode_leb128(data: bytes, offset: int) -> Tuple[int, int]:
    """Decode an unsigned LEB128 value. Returns (value, next_offset)."""
    value = 0
    shift = 0
    while True:
        if offset >= len(data):
            raise ValueError("Truncated LEB128 value")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if (byte & 0x80) == 0:
            break
        shift += 7
    return value, offset


def read_leb128_u16(data: bytes, offset: int) -> Tuple[int, int]:
    """Read a LEB128 value and clamp it to a 16-bit unsigned integer."""
    value, offset = decode_leb128(data, offset)
    return value & 0xFFFF, offset


def parse_init_event(data: bytes) -> dict:
    """Parse an Init event (type byte 0x01) from the ALiS stream."""
    offset = 1
    event_id, offset = decode_leb128(data, offset)
    time_us, offset = decode_leb128(data, offset)
    cols, offset = read_leb128_u16(data, offset)
    rows, offset = read_leb128_u16(data, offset)

    if offset >= len(data):
        raise ValueError("Truncated theme flag")
    theme_flag = data[offset]
    offset += 1
    if theme_flag == 16:
        if offset + 6 + 16 * 3 > len(data):
            raise ValueError("Truncated theme colors")
        offset += 6 + 16 * 3
    elif theme_flag != 0:
        raise ValueError(f"Unsupported theme flag: {theme_flag}")

    init_len, offset = decode_leb128(data, offset)
    if offset + init_len > len(data):
        raise ValueError("Truncated init payload")
    init_text = data[offset : offset + init_len].decode("utf-8", errors="replace")

    return {
        "type": "Init",
        "id": event_id,
        "time_us": time_us,
        "cols": cols,
        "rows": rows,
        "init_chars": len(init_text),
    }


def parse_output_event(data: bytes) -> dict:
    """Parse an Output event (type byte 'o') from the ALiS stream."""
    offset = 1
    event_id, offset = decode_leb128(data, offset)
    rel_time_us, offset = decode_leb128(data, offset)
    text_len, offset = decode_leb128(data, offset)
    if offset + text_len > len(data):
        raise ValueError("Truncated Output payload")
    text = data[offset : offset + text_len]
    return {
        "type": "Output",
        "id": event_id,
        "rel_time_us": rel_time_us,
        "text": text,
    }


def parse_resize_event(data: bytes) -> dict:
    """Parse a Resize event (type byte 'r') from the ALiS stream."""
    offset = 1
    event_id, offset = decode_leb128(data, offset)
    rel_time_us, offset = decode_leb128(data, offset)
    cols, offset = read_leb128_u16(data, offset)
    rows, offset = read_leb128_u16(data, offset)
    return {
        "type": "Resize",
        "id": event_id,
        "rel_time_us": rel_time_us,
        "cols": cols,
        "rows": rows,
    }


def parse_eot_event(data: bytes) -> dict:
    """Parse an End-of-Transmission event (type byte 0x04)."""
    offset = 1
    event_id, offset = decode_leb128(data, offset)
    rel_time_us, offset = decode_leb128(data, offset)
    return {"type": "EOT", "id": event_id, "rel_time_us": rel_time_us}


def parse_alis_message(data: bytes) -> dict:
    """Parse a single ALiS binary message into a summary dict."""
    if not data:
        raise ValueError("Empty message")

    if data.startswith(ALIS_MAGIC):
        return {
            "type": "Header",
            "magic": data[:5].hex(),
            "version": data[5] if len(data) > 5 else None,
        }

    type_byte = data[0]
    if type_byte == 0x01:
        return parse_init_event(data)
    if type_byte == ord("o"):
        return parse_output_event(data)
    if type_byte == ord("r"):
        return parse_resize_event(data)
    if type_byte == 0x04:
        return parse_eot_event(data)

    raise ValueError(f"Unknown ALiS event type byte: {type_byte:#04x}")


def tmux_env() -> dict:
    """Build an environment for tmux commands that avoids nested-session issues."""
    env = {**os.environ}
    for var in (
        "TMUX",
        "TMUX_PANE",
        "TMUX_WINDOW",
        "TMUX_SESSION",
        "TERM_PROGRAM",
    ):
        env.pop(var, None)
    env["TERM"] = "xterm-256color"
    env["COLORTERM"] = "truecolor"
    return env


async def run_tmux(
    tmux_socket: str, args: list, capture_output: bool = False
) -> Tuple[int, Optional[bytes], Optional[bytes]]:
    """Run a tmux command against the custom socket and return (rc, stdout, stderr)."""
    cmd = ["tmux", "-S", tmux_socket] + args
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE if capture_output else None,
        stderr=asyncio.subprocess.PIPE if capture_output else None,
        env=tmux_env(),
    )
    stdout, stderr = await proc.communicate()
    return proc.returncode, stdout, stderr


async def start_tmux(
    tmux_socket: str,
    tmux_conf: str,
    session_name: str,
    receiver_path: str,
    socket_path: str,
) -> None:
    """Create a detached tmux session running the socket receiver."""
    cmd = [
        "tmux",
        "-S", tmux_socket,
        "-f", tmux_conf,
        "new-session", "-d",
        "-s", session_name,
        "-n", "main",
        "-e", "TERM=xterm-256color",
        "-e", "COLORTERM=truecolor",
        sys.executable,
        receiver_path,
        socket_path,
    ]
    proc = await asyncio.create_subprocess_exec(*cmd, env=tmux_env())
    await proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"tmux new-session failed with code {proc.returncode}")

    # Best-effort kiosk hardening: remove all key bindings.
    await run_tmux(tmux_socket, ["unbind-key", "-a"])


async def kill_tmux(tmux_socket: str, session_name: str) -> None:
    """Kill the tmux session and remove its socket file."""
    await run_tmux(tmux_socket, ["kill-session", "-t", session_name])
    try:
        os.unlink(tmux_socket)
    except FileNotFoundError:
        pass


async def resize_tmux(
    tmux_socket: str, session_name: str, cols: int, rows: int
) -> None:
    """Resize the tmux window externally via tmux shell command."""
    before = await get_tmux_info(tmux_socket, session_name)
    rc, _, stderr = await run_tmux(
        tmux_socket,
        ["resize-window", "-t", session_name, "-x", str(cols), "-y", str(rows)],
    )
    if rc != 0:
        err = (stderr or b"").decode("utf-8", errors="replace").strip()
        print(f"[tmux] resize-window failed: {err}", file=sys.stderr)
        return
    after = await get_tmux_info(tmux_socket, session_name)
    before_str = format_tmux_info(before) if before else "unknown"
    after_str = format_tmux_info(after) if after else "unknown"
    print(f"[tmux] resize {cols}x{rows}: before [{before_str}] -> after [{after_str}]")


async def capture_pane(tmux_socket: str, session_name: str) -> Optional[str]:
    """Capture the current tmux pane content as text."""
    target = f"{session_name}:0.0"
    rc, stdout, stderr = await run_tmux(
        tmux_socket,
        ["capture-pane", "-p", "-J", "-t", target],
        capture_output=True,
    )
    if rc != 0:
        err = (stderr or b"").decode("utf-8", errors="replace").strip()
        print(f"[tmux] capture-pane failed: {err}", file=sys.stderr)
        return None
    return (stdout or b"").decode("utf-8", errors="replace")


# Format string used to extract live session metadata from tmux.
# Tab-separated fields avoid ambiguity with spaces in values.
TMUX_INFO_FORMAT = (
    "#{session_name}\t#{window_width}\t#{window_height}\t"
    "#{cursor_x}\t#{cursor_y}\t#{cursor_flag}\t#{cursor_character}"
)


async def get_tmux_info(tmux_socket: str, session_name: str) -> Optional[dict]:
    """Return tmux session/pane metadata (size, cursor position, etc.)."""
    target = f"{session_name}:0.0"
    rc, stdout, stderr = await run_tmux(
        tmux_socket,
        ["display-message", "-t", target, "-p", TMUX_INFO_FORMAT],
        capture_output=True,
    )
    if rc != 0:
        err = (stderr or b"").decode("utf-8", errors="replace").strip()
        print(f"[tmux] display-message failed: {err}", file=sys.stderr)
        return None
    output = (stdout or b"").decode("utf-8", errors="replace").strip()
    if not output:
        return None
    parts = output.split("\t")
    try:
        return {
            "session_name": parts[0],
            "window_width": int(parts[1]),
            "window_height": int(parts[2]),
            "cursor_x": int(parts[3]),
            "cursor_y": int(parts[4]),
            "cursor_flag": int(parts[5]),
            "cursor_character": parts[6] if len(parts) > 6 else "",
        }
    except (IndexError, ValueError) as exc:
        print(f"[tmux] failed to parse info line {output!r}: {exc}", file=sys.stderr)
        return None


def format_tmux_info(info: dict) -> str:
    """Render tmux metadata as a concise log line."""
    return (
        f"size={info['window_width']}x{info['window_height']} "
        f"cursor=({info['cursor_x']},{info['cursor_y']}) "
        f"cursor_flag={info['cursor_flag']} "
        f"cursor_char={info['cursor_character']!r}"
    )


async def capture_loop(
    tmux_socket: str, session_name: str, interval: float
) -> None:
    """Periodically capture tmux pane content and log/print it."""
    last_hash: Optional[str] = None
    while True:
        await asyncio.sleep(interval)
        info = await get_tmux_info(tmux_socket, session_name)
        info_str = format_tmux_info(info) if info else "info unavailable"
        text = await capture_pane(tmux_socket, session_name)
        if text is None:
            continue
        encoded = text.encode("utf-8", errors="replace")
        current_hash = hashlib.sha256(encoded).hexdigest()[:16]
        changed = current_hash != last_hash
        if changed:
            last_hash = current_hash
        ts = datetime.now().isoformat(timespec="milliseconds")
        print(f"[{ts}] capture bytes={len(encoded)} changed={changed} {info_str}")
        if changed:
            print("===== tmux pane =====")
            print(text, end="" if text.endswith("\n") else "\n")
            print("=====================")


async def websocket_loop(uri: str, bridge: StreamBridge, tmux_socket: str, session_name: str) -> None:
    """Read ALiS events from the WebSocket and mirror them into tmux."""
    print(f"[ws] connecting to {uri} (subprotocol {SUBPROTOCOL})")
    async with websockets.connect(uri, subprotocols=[SUBPROTOCOL]) as websocket:
        print("[ws] connected")
        async for message in websocket:
            if not isinstance(message, bytes):
                print(f"[ws] ignoring non-binary message: {type(message)}")
                continue
            try:
                event = parse_alis_message(message)
            except Exception as exc:
                print(f"[ws] parse error: {exc}", file=sys.stderr)
                traceback.print_exc()
                continue

            etype = event["type"]
            if etype == "Header":
                print(f"[ws] Header magic={event['magic']} version={event['version']}")
            elif etype == "Init":
                print(
                    f"[ws] Init id={event['id']} time_us={event['time_us']} "
                    f"cols={event['cols']} rows={event['rows']} "
                    f"init_chars={event['init_chars']}"
                )
                await resize_tmux(tmux_socket, session_name, event["cols"], event["rows"])
            elif etype == "Output":
                print(
                    f"[ws] Output id={event['id']} rel_time_us={event['rel_time_us']} "
                    f"text_bytes={len(event['text'])}"
                )
                await bridge.write(event["text"])
            elif etype == "Resize":
                print(
                    f"[ws] Resize id={event['id']} rel_time_us={event['rel_time_us']} "
                    f"cols={event['cols']} rows={event['rows']}"
                )
                await resize_tmux(tmux_socket, session_name, event["cols"], event["rows"])
            elif etype == "EOT":
                print(f"[ws] EOT id={event['id']} rel_time_us={event['rel_time_us']}")
            else:
                print(f"[ws] unhandled event type: {etype}")


async def async_main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream an asciinema v3 ALiS WebSocket into a headless tmux session."
    )
    parser.add_argument("--uri", default=URI, help=f"WebSocket endpoint (default: {URI}).")
    parser.add_argument(
        "--capture-interval",
        type=float,
        default=1.0,
        help="Seconds between tmux pane captures (default: 1.0).",
    )
    parser.add_argument(
        "--work-dir",
        help="Directory for the Unix socket, tmux socket and receiver script.",
    )
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="Do not delete the work directory on exit.",
    )
    args = parser.parse_args()

    if shutil.which("tmux") is None:
        print("error: tmux is required but not found in PATH", file=sys.stderr)
        sys.exit(1)

    work_dir = args.work_dir or tempfile.mkdtemp(prefix="alis-tmux-")
    os.makedirs(work_dir, exist_ok=True)

    socket_path = os.path.join(work_dir, "stream.sock")
    tmux_socket = os.path.join(work_dir, "tmux.sock")
    tmux_conf_path = os.path.join(work_dir, "tmux.conf")
    receiver_path = os.path.join(work_dir, "receiver.py")
    session_name = f"alis-{os.getpid()}"

    # Persist helper files.
    with open(tmux_conf_path, "w", encoding="utf-8") as fh:
        fh.write(TMUX_CONF)
    with open(receiver_path, "w", encoding="utf-8") as fh:
        fh.write(RECEIVER_SCRIPT)

    bridge = StreamBridge()
    server = await asyncio.start_unix_server(bridge.handle_client, socket_path)

    try:
        print(f"[setup] work_dir={work_dir}")
        print(f"[setup] stream socket={socket_path}")
        print(f"[setup] tmux socket={tmux_socket}")
        print(f"[setup] session_name={session_name}")

        await start_tmux(
            tmux_socket, tmux_conf_path, session_name, receiver_path, socket_path
        )
        print(f"[setup] tmux session '{session_name}' started")

        # Wait for the receiver inside tmux to connect to our Unix socket.
        try:
            await asyncio.wait_for(bridge.connected.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            print("[setup] timed out waiting for tmux receiver to connect", file=sys.stderr)
            sys.exit(1)

        info = await get_tmux_info(tmux_socket, session_name)
        if info:
            print(f"[setup] tmux info: {format_tmux_info(info)}")
        else:
            print("[setup] tmux info unavailable", file=sys.stderr)

        capture_task = asyncio.create_task(
            capture_loop(tmux_socket, session_name, args.capture_interval)
        )

        try:
            await websocket_loop(args.uri, bridge, tmux_socket, session_name)
        except websockets.ConnectionClosed as exc:
            print(f"[ws] connection closed: {exc}")
        finally:
            capture_task.cancel()
            try:
                await capture_task
            except asyncio.CancelledError:
                pass
    finally:
        print("[cleanup] shutting down")
        await bridge.close()
        server.close()
        await server.wait_closed()
        try:
            os.unlink(socket_path)
        except FileNotFoundError:
            pass
        await kill_tmux(tmux_socket, session_name)
        if not args.keep_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)
        else:
            print(f"[cleanup] kept work_dir={work_dir}")


if __name__ == "__main__":
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nClient stopped by user.")
        sys.exit(0)
