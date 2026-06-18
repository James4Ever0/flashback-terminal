#!/usr/bin/env python3
"""
WebSocket client that only reads messages from an asciinema v3 ALiS stream.

Connects to ws://127.0.0.1:46257/ws (subprotocol v1.alis) by default, parses the
binary ALiS protocol, and prints a human-readable breakdown of every received
event.

Usage:
    python connect_to_websocket.py
    python connect_to_websocket.py --no-traceback
    python connect_to_websocket.py --uri ws://127.0.0.1:33601/ws

Implements:
- LEB128 decoding for event fields.
- Parsing of Init, Output, Input, Resize, Marker, Exit and EOT events.
- Automatic reconnect with exponential backoff when the connection is reset.
- Per-event metadata: type, content summary, byte length and timestamps.
- Wall-clock receive time displayed in CST (UTC-6).
- Server stream time displayed as a duration relative to stream start.
- Detailed tracebacks by default; use --no-traceback for brief errors.
"""

import argparse
import asyncio
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import websockets

URI = "ws://127.0.0.1:46257/ws"
SUBPROTOCOL = "v1.alis"
ALIS_MAGIC = b"ALiS\x01"

# Central Standard Time (UTC-6), as requested. If daylight-time handling is
# required, swap this for ZoneInfo("America/Chicago").
CST = timezone(timedelta(hours=-6), name="CST")

# Runtime configuration (populated by main()).
CONFIG = {"traceback": True}

# Reconnection tuning
RECONNECT_MIN_DELAY = 1.0
RECONNECT_MAX_DELAY = 30.0
RECONNECT_BACKOFF_FACTOR = 2.0


def log_error(message: str, exc: Optional[Exception] = None) -> None:
    """Print an error message, plus a full traceback when traceback mode is on."""
    print(message, file=sys.stderr)
    if exc is not None and CONFIG["traceback"]:
        traceback.print_exception(type(exc), exc, exc.__traceback__)


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


def format_bytes(data: bytes) -> str:
    """Return a compact hex dump of the first few bytes of a payload."""
    preview = data[:32]
    hex_part = " ".join(f"{b:02x}" for b in preview)
    if len(data) > 32:
        hex_part += f" ... ({len(data)} bytes total)"
    return hex_part


def format_wall_clock(dt: datetime) -> str:
    """Format a datetime as an ISO-8601 timestamp in the configured timezone."""
    return dt.astimezone(CST).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + f" {dt.astimezone(CST).strftime('%Z')}"


def format_stream_time(micros: int) -> str:
    """Format stream-relative microseconds as a duration string."""
    delta = timedelta(microseconds=micros)
    return f"{delta.total_seconds():.6f}s"


def format_delta(micros: int) -> str:
    """Format a microsecond delta as a human-readable duration with + prefix."""
    delta = timedelta(microseconds=micros)
    return f"+{delta.total_seconds():.6f}s"


def make_printable(text: str, max_len: int = 200) -> str:
    """Return a printable, optionally truncated preview of a string."""
    if len(text) > max_len:
        text = text[:max_len] + "..."
    # Replace control chars except common whitespace with their escape form.
    out = []
    for ch in text:
        if ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\x{ord(ch):02x}")
        else:
            out.append(ch)
    return "".join(out)


@dataclass
class ParserState:
    """Tracks stream timing across reconnects and events."""

    base_time_us: Optional[int] = None
    last_event_time_us: int = 0
    event_count: int = 0

    def record_absolute(self, time_us: int) -> int:
        """Record an absolute timestamp and return it."""
        self.last_event_time_us = time_us
        if self.base_time_us is None:
            self.base_time_us = time_us
        return time_us

    def record_relative(self, rel_us: int) -> int:
        """Record a relative timestamp and return the absolute time."""
        self.last_event_time_us += rel_us
        if self.base_time_us is None:
            self.base_time_us = self.last_event_time_us
        return self.last_event_time_us


@dataclass
class ParsedEvent:
    """A decoded ALiS event ready for display."""

    event_type: str
    event_id: Optional[int]
    absolute_time_us: int
    metadata: dict = field(default_factory=dict)
    content_preview: str = ""
    raw_bytes: bytes = field(default=b"", repr=False)

    def print_summary(self, index: int, received_at: datetime) -> None:
        wall_clock = format_wall_clock(received_at)
        stream_time = format_stream_time(self.absolute_time_us)
        rel_us = self.absolute_time_us - (state.base_time_us or 0)
        print(f"[{index:>5}] {self.event_type:12} id={self.event_id} len={len(self.raw_bytes):>5} received={wall_clock} stream_time={stream_time} ({format_delta(rel_us)} from start)")
        for key, value in self.metadata.items():
            print(f"        {key}: {value}")
        if self.content_preview:
            print(f"        content: {self.content_preview}")
        print(f"        hex: {format_bytes(self.raw_bytes)}")
        print()


# Module-level parser state so reconnects preserve timing.
state = ParserState()


def parse_theme(data: bytes, offset: int) -> Tuple[Optional[dict], int]:
    """Parse the optional theme block in an Init event."""
    if offset >= len(data):
        raise ValueError("Truncated theme flag")
    theme_flag = data[offset]
    offset += 1

    if theme_flag == 0:
        return None, offset

    if theme_flag != 16:
        raise ValueError(f"Unsupported theme flag: {theme_flag}")

    if offset + 6 + 16 * 3 > len(data):
        raise ValueError("Truncated theme colors")

    fg = tuple(data[offset : offset + 3])
    bg = tuple(data[offset + 3 : offset + 6])
    offset += 6

    palette = []
    for _ in range(16):
        palette.append(tuple(data[offset : offset + 3]))
        offset += 3

    return {"fg": fg, "bg": bg, "palette": palette}, offset


def parse_init_event(data: bytes, offset: int, parser_state: ParserState) -> ParsedEvent:
    """Parse an Init event (type byte 0x01)."""
    event_id, offset = decode_leb128(data, offset)
    abs_time_us, offset = decode_leb128(data, offset)
    cols, offset = read_leb128_u16(data, offset)
    rows, offset = read_leb128_u16(data, offset)
    theme, offset = parse_theme(data, offset)
    init_len, offset = decode_leb128(data, offset)

    if offset + init_len > len(data):
        raise ValueError("Truncated init payload")

    init_text = data[offset : offset + init_len].decode("utf-8", errors="replace")
    offset += init_len

    abs_time = parser_state.record_absolute(abs_time_us)

    return ParsedEvent(
        event_type="Init",
        event_id=event_id,
        absolute_time_us=abs_time,
        metadata={
            "cols": cols,
            "rows": rows,
            "theme": f"{theme['fg']} fg, {theme['bg']} bg" if theme else "none",
            "init_chars": len(init_text),
        },
        content_preview=make_printable(init_text, max_len=120),
        raw_bytes=data,
    )


def parse_text_event(
    data: bytes,
    offset: int,
    event_type: str,
    type_byte: int,
    parser_state: ParserState,
) -> ParsedEvent:
    """Parse events with the shape: id, rel_time, text_len, text."""
    event_id, offset = decode_leb128(data, offset)
    rel_time_us, offset = decode_leb128(data, offset)
    text_len, offset = decode_leb128(data, offset)

    if offset + text_len > len(data):
        raise ValueError(f"Truncated {event_type} payload")

    text = data[offset : offset + text_len].decode("utf-8", errors="replace")
    offset += text_len

    abs_time = parser_state.record_relative(rel_time_us)

    return ParsedEvent(
        event_type=event_type,
        event_id=event_id,
        absolute_time_us=abs_time,
        metadata={"text_bytes": text_len, "text_chars": len(text)},
        content_preview=make_printable(text, max_len=200),
        raw_bytes=data,
    )


def parse_resize_event(data: bytes, offset: int, parser_state: ParserState) -> ParsedEvent:
    """Parse a Resize event (type byte 'r')."""
    event_id, offset = decode_leb128(data, offset)
    rel_time_us, offset = decode_leb128(data, offset)
    cols, offset = read_leb128_u16(data, offset)
    rows, offset = read_leb128_u16(data, offset)

    abs_time = parser_state.record_relative(rel_time_us)

    return ParsedEvent(
        event_type="Resize",
        event_id=event_id,
        absolute_time_us=abs_time,
        metadata={"cols": cols, "rows": rows},
        raw_bytes=data,
    )


def parse_exit_event(data: bytes, offset: int, parser_state: ParserState) -> ParsedEvent:
    """Parse an Exit event (type byte 'x')."""
    event_id, offset = decode_leb128(data, offset)
    rel_time_us, offset = decode_leb128(data, offset)
    status, offset = decode_leb128(data, offset)

    abs_time = parser_state.record_relative(rel_time_us)

    return ParsedEvent(
        event_type="Exit",
        event_id=event_id,
        absolute_time_us=abs_time,
        metadata={"exit_status": status},
        raw_bytes=data,
    )


def parse_eot_event(data: bytes, offset: int, parser_state: ParserState) -> ParsedEvent:
    """Parse an End-of-Transmission event (type byte 0x04)."""
    event_id, offset = decode_leb128(data, offset)
    rel_time_us, offset = decode_leb128(data, offset)

    abs_time = parser_state.record_relative(rel_time_us)

    return ParsedEvent(
        event_type="EOT",
        event_id=event_id,
        absolute_time_us=abs_time,
        metadata={"note": "stream ended"},
        raw_bytes=data,
    )


def parse_alis_message(data: bytes, parser_state: ParserState) -> ParsedEvent:
    """Parse a single ALiS binary message into a ParsedEvent."""
    if not data:
        raise ValueError("Empty message")

    if data.startswith(ALIS_MAGIC):
        # Header is informational only; the player treats subsequent binary
        # frames as events. We log it as a protocol header.
        return ParsedEvent(
            event_type="Header",
            event_id=None,
            absolute_time_us=parser_state.last_event_time_us,
            metadata={"magic": data[:5].hex(), "version": data[5] if len(data) > 5 else None},
            content_preview=data.decode("latin-1", errors="replace"),
            raw_bytes=data,
        )

    type_byte = data[0]
    offset = 1

    if type_byte == 0x01:
        return parse_init_event(data, offset, parser_state)
    if type_byte == ord("o"):
        return parse_text_event(data, offset, "Output", type_byte, parser_state)
    if type_byte == ord("i"):
        return parse_text_event(data, offset, "Input", type_byte, parser_state)
    if type_byte == ord("r"):
        return parse_resize_event(data, offset, parser_state)
    if type_byte == ord("m"):
        return parse_text_event(data, offset, "Marker", type_byte, parser_state)
    if type_byte == ord("x"):
        return parse_exit_event(data, offset, parser_state)
    if type_byte == 0x04:
        return parse_eot_event(data, offset, parser_state)

    raise ValueError(f"Unknown ALiS event type byte: {type_byte:#04x}")


def print_banner(uri: str) -> None:
    print(f"Connected to {uri} (subprotocol {SUBPROTOCOL})")
    print("Listening for ALiS events (press Ctrl+C to stop)...")
    print()


async def read_only_client(uri: str) -> None:
    delay = RECONNECT_MIN_DELAY

    while True:
        try:
            async with websockets.connect(
                uri,
                subprotocols=[SUBPROTOCOL],
                ping_interval=20,
                ping_timeout=10,
            ) as websocket:
                print_banner(uri)
                print("[reconnect delay reset]")
                delay = RECONNECT_MIN_DELAY

                async for message in websocket:
                    received_at = datetime.now(tz=timezone.utc)
                    state.event_count += 1

                    if isinstance(message, bytes):
                        try:
                            event = parse_alis_message(message, state)
                            event.print_summary(state.event_count, received_at)
                        except Exception as exc:
                            log_error(
                                f"[{state.event_count:>5}] ParseError: {exc}",
                                exc,
                            )
                            print(f"        hex: {format_bytes(message)}")
                            print()
                    elif isinstance(message, str):
                        print(
                            f"[{state.event_count:>5}] TextMessage: {make_printable(message)}"
                        )
                    else:
                        print(
                            f"[{state.event_count:>5}] UnknownMessageType: {type(message)}"
                        )

        except websockets.ConnectionClosed as exc:
            log_error(f"Connection closed by server: {exc}", exc)
        except websockets.InvalidHandshake as exc:
            log_error(
                f"Handshake failed (is the server running and speaking v1.alis?): {exc}",
                exc,
            )
            # Handshake errors are usually configuration problems; back off quickly.
        except OSError as exc:
            log_error(f"Network error: {exc}", exc)
        except Exception as exc:
            log_error(f"Unexpected error: {type(exc).__name__}: {exc}", exc)

        print(f"Reconnecting in {delay:.1f}s...")
        await asyncio.sleep(delay)
        delay = min(delay * RECONNECT_BACKOFF_FACTOR, RECONNECT_MAX_DELAY)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only ALiS WebSocket client for asciinema v3 streams."
    )
    parser.add_argument(
        "--no-traceback",
        dest="traceback",
        action="store_false",
        default=True,
        help="Suppress detailed tracebacks and only show brief error messages.",
    )
    parser.add_argument(
        "--uri",
        default=URI,
        help=f"WebSocket endpoint to connect to (default: {URI}).",
    )
    return parser


async def async_main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    CONFIG["traceback"] = args.traceback
    CONFIG["uri"] = args.uri
    await read_only_client(args.uri)


if __name__ == "__main__":
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nClient stopped by user.")
        sys.exit(0)
