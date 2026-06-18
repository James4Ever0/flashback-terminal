"""Entry point for terminal-capture-server.

Parses host/port from CLI arguments and environment variables, then launches
uvicorn with the FastAPI app defined in server.py.
"""

import argparse
import os
import sys

import uvicorn

from server import app

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080


def parse_args() -> argparse.Namespace:
    """Parse CLI args, falling back to environment variables."""
    parser = argparse.ArgumentParser(
        description="Receive terminal captures and display metadata."
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("CAPTURE_SERVER_HOST", DEFAULT_HOST),
        help=f"Bind host (default: {DEFAULT_HOST}, env: CAPTURE_SERVER_HOST)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("CAPTURE_SERVER_PORT", DEFAULT_PORT)),
        help=f"Bind port (default: {DEFAULT_PORT}, env: CAPTURE_SERVER_PORT)",
    )
    parser.add_argument(
        "--log-requests",
        action="store_true",
        default=os.environ.get("CAPTURE_SERVER_LOG_REQUESTS", "").lower() in ("1", "true", "yes"),
        help="Log every incoming request method, URL, query params, and body (env: CAPTURE_SERVER_LOG_REQUESTS)",
    )
    return parser.parse_args()


def main() -> None:
    """Launch the server."""
    cfg = parse_args()
    app.state.log_requests = cfg.log_requests
    print(
        f"Starting terminal-capture-server on {cfg.host}:{cfg.port}",
        file=sys.stderr,
    )
    if cfg.log_requests:
        print("Request logging enabled", file=sys.stderr)
    uvicorn.run(
        "server:app",
        host=cfg.host,
        port=cfg.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
