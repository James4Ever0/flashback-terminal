"""Simple terminal capture receiver server.

Receives POST /api/captures from the flashback-shell Go client, computes
metadata for each capture, prints a concise summary to the terminal, and
discards the payload without persistence.
"""

import re
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel

# Comprehensive ANSI escape sequence matcher (CSI and single-byte sequences).
ANSI_RE = re.compile(r"\x1b(?:[@-Z\-_]|\[[0-?]*[ -/]*[@-~])")

app = FastAPI(title="terminal-capture-server")


@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    """Log every incoming request when enabled, including 404s."""
    if getattr(app.state, "log_requests", False):
        method = request.method
        url = str(request.url)
        query_params = dict(request.query_params)
        body_bytes = await request.body()
        try:
            body_text = body_bytes.decode("utf-8", errors="replace")
        except Exception:
            body_text = "<binary>"

        ts = datetime.now().isoformat(sep=" ", timespec="seconds")
        print(f"[{ts}] REQUEST method={method} url={url} query={query_params} body={body_text}")

    response = await call_next(request)
    return response


class Capture(BaseModel):
    """Single pane capture. Mirrors the Go capture.Capture struct."""

    session_id: str
    pane_id: str
    target: str
    ansi: str
    text: str
    hash: str
    cols: int
    rows: int
    timestamp: datetime
    metadata: Optional[Dict[str, str]] = None


class UploadPayload(BaseModel):
    """Upload payload. Mirrors the Go server.uploadPayload struct."""

    device_id: str
    timestamp: datetime
    captures: List[Capture]


def strip_ansi(s: str) -> str:
    """Remove ANSI escape sequences from a string."""
    return ANSI_RE.sub("", s)


def compute_metadata(capture: Capture) -> dict:
    """Compute byte count, row count, and visible column count."""
    byte_count = len(capture.ansi.encode("utf-8"))

    lines = capture.text.splitlines()
    if not lines:
        lines = [""]

    row_count = len(lines)
    col_count = max(len(strip_ansi(line)) for line in lines)

    return {
        "hash": capture.hash,
        "bytes": byte_count,
        "rows": row_count,
        "cols": col_count,
    }


def display_capture(device_id: str, capture: Capture, meta: dict) -> None:
    """Print a concise metadata summary for a single capture."""
    ts = datetime.now().isoformat(sep=" ", timespec="seconds")
    print(f"[{ts}] Capture received")
    print(f"  Client:  {device_id}")
    print(f"  Session: {capture.session_id}")
    print(f"  Pane:    {capture.pane_id}")
    print(f"  Hash:    {meta['hash']}")
    print(f"  Bytes:   {meta['bytes']}")
    print(f"  Rows:    {meta['rows']}")
    print(f"  Cols:    {meta['cols']}")
    print(f"  Metadata:    {capture.metadata}")
    print()


@app.post("/api/captures")
async def receive_captures(payload: UploadPayload):
    """Receive terminal captures and print their metadata."""
    for capture in payload.captures:
        meta = compute_metadata(capture)
        display_capture(payload.device_id, capture, meta)
    return {"status": "ok", "received": len(payload.captures)}
