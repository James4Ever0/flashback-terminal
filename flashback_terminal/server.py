"""FastAPI server for flashback-terminal."""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from flashback_terminal.api.websocket import TerminalWebSocketHandler
from flashback_terminal.config import get_config
from flashback_terminal.database import Database
from flashback_terminal.logger import Logger, log_function, logger
from flashback_terminal.retention import RetentionManager
from flashback_terminal.search import SearchEngine
from flashback_terminal.terminal import TerminalManager

# Global instances
db: Optional[Database] = None
terminal_manager: Optional[TerminalManager] = None
ws_handler: Optional[TerminalWebSocketHandler] = None
search_engine: Optional[SearchEngine] = None
retention_manager: Optional[RetentionManager] = None


class SearchRequest(BaseModel):
    query: str
    mode: str = "text"
    scope: str = "all"
    session_ids: Optional[List[str]] = None
    limit: int = 50


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global db, terminal_manager, ws_handler, search_engine, retention_manager

    logger.info("Starting flashback-terminal lifespan manager")

    config = get_config()
    logger.info(f"Configuration loaded: verbosity={config.verbosity}")
    logger.debug(f"Data directory: {config.data_dir}")

    logger.info("Initializing database...")
    db = Database(config.db_path)
    logger.info(f"Database initialized: {config.db_path}")

    logger.info("Initializing terminal manager...")
    terminal_manager = TerminalManager(db)
    logger.info("Terminal manager initialized")

    logger.info("Initializing WebSocket handler...")
    ws_handler = TerminalWebSocketHandler(terminal_manager, db)
    logger.info("WebSocket handler initialized")

    if config.is_module_enabled("history_keeper"):
        logger.info("Initializing search engine (history_keeper enabled)...")
        search_engine = SearchEngine(db)
        logger.info("Search engine initialized")
    else:
        logger.warning("History keeper disabled - search functionality unavailable")

    if config.is_worker_enabled("retention"):
        logger.info("Initializing retention manager...")
        retention_manager = RetentionManager(db)
        asyncio.create_task(retention_scheduler())
        logger.info("Retention manager initialized and scheduler started")
    else:
        logger.warning("Retention worker disabled")

    logger.info("flashback-terminal startup complete")
    yield

    logger.info("Shutting down flashback-terminal...")
    logger.info("Shutdown complete")


async def retention_scheduler():
    """Background task for retention management."""
    config = get_config()
    interval = config.get("workers.retention.check_interval_seconds", 3600)

    while True:
        await asyncio.sleep(interval)
        if retention_manager:
            retention_manager.run_cleanup()


app = FastAPI(title="flashback-terminal", lifespan=lifespan)

# Static files and templates
static_dir = os.path.join(os.path.dirname(__file__), "static")
templates_dir = os.path.join(os.path.dirname(__file__), "templates")

if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

templates = Jinja2Templates(directory=templates_dir if os.path.exists(templates_dir) else static_dir)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main terminal UI with Jinja2 templating for verbosity."""
    config = get_config()
    verbosity = config.verbosity

    logger.debug(f"Serving index page with verbosity={verbosity}")

    # Create template context
    context = {
        "request": request,  # Pass the actual request object
        "verbosity_level": verbosity,
    }

    # Try to use Jinja2 template
    template_path = os.path.join(templates_dir, "index.html")
    if os.path.exists(template_path):
        return templates.TemplateResponse(request=request, name="index.html", context=context)

    # Fallback: read static file and inject verbosity
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as f:
            content = f.read()
        # Inject verbosity before closing </head>
        verbosity_script = f"<script>window.VERBOSITY_LEVEL = {verbosity};</script>"
        content = content.replace("</head>", f"{verbosity_script}</head>")
        return content

    return "<h1>flashback-terminal</h1><p>Static files not found</p>"


@app.websocket("/ws/terminal/{session_uuid}")
async def terminal_websocket(websocket: WebSocket, session_uuid: str):
    """WebSocket endpoint for terminal sessions."""
    if ws_handler:
        await ws_handler.handle(websocket, session_uuid)


@app.get("/api/sessions")
@log_function(Logger.DEBUG)
async def list_sessions(
    status: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """List terminal sessions."""
    logger.debug(f"list_sessions called: status={status}, limit={limit}, offset={offset}")

    sessions = db.list_sessions(status=status, limit=limit, offset=offset)

    logger.info(f"Listed {len(sessions)} sessions (status={status}, limit={limit})")

    return {
        "sessions": [
            {
                "id": s.id,
                "uuid": s.uuid,
                "name": s.name,
                "profile_name": s.profile_name,
                "created_at": s.created_at.isoformat(),
                "status": s.status,
                "last_cwd": s.last_cwd,
            }
            for s in sessions
        ]
    }


@app.post("/api/sessions")
@log_function(Logger.DEBUG)
async def create_session(profile: str = "default", name: Optional[str] = None):
    """Create a new terminal session."""
    logger.info(f"Creating new session: profile={profile}, name={name}")

    session = terminal_manager.create_session(profile_name=profile, name=name)
    if not session:
        logger.error(f"Failed to create session: profile={profile}, name={name}")
        raise HTTPException(status_code=500, detail="Failed to create session")

    logger.info(f"Session created: id={session.session_id}, uuid={session.uuid}")

    return {
        "session_id": session.session_id,
        "uuid": session.uuid,
        "name": name or f"Terminal {session.session_id}",
    }


@app.get("/api/sessions/{session_uuid}")
async def get_session(session_uuid: str):
    """Get session details."""
    session = db.get_session_by_uuid(session_uuid)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "id": session.id,
        "uuid": session.uuid,
        "name": session.name,
        "profile_name": session.profile_name,
        "created_at": session.created_at.isoformat(),
        "status": session.status,
        "last_cwd": session.last_cwd,
    }


@app.put("/api/sessions/{session_uuid}")
async def update_session(session_uuid: str, name: str):
    """Update session (rename)."""
    session = db.get_session_by_uuid(session_uuid)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    db.update_session(session.id, name=name)
    return {"success": True}


@app.delete("/api/sessions/{session_uuid}")
async def delete_session(session_uuid: str):
    """Delete a session."""
    session = db.get_session_by_uuid(session_uuid)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    terminal_manager.close_session(session_uuid)
    db.delete_session(session.id)
    return {"success": True}


@app.post("/api/sessions/{session_uuid}/restore")
async def restore_session(session_uuid: str):
    """Restore an archived session."""
    raise HTTPException(status_code=501, detail="Archive restoration not yet implemented")


@app.get("/api/profiles")
async def list_profiles():
    """List available terminal profiles."""
    config = get_config()
    profiles = config.get("profiles", [])
    return {"profiles": profiles}


@app.post("/api/search")
@log_function(Logger.DEBUG)
async def search(request: SearchRequest):
    """Search terminal history."""
    logger.info(f"Search request: query={request.query[:50]}..., mode={request.mode}, scope={request.scope}")

    if not search_engine:
        logger.error("Search not available - search_engine is None")
        raise HTTPException(status_code=503, detail="Search not available")

    target_session_ids = None
    if request.scope == "current" and request.session_ids:
        target_session_ids = []
        for uuid in request.session_ids:
            session = db.get_session_by_uuid(uuid)
            if session:
                target_session_ids.append(session.id)
        logger.debug(f"Search limited to session_ids: {target_session_ids}")

    logger.debug("Executing search...")
    results = search_engine.search(
        query=request.query,
        mode=request.mode,
        scope=request.scope,
        session_ids=target_session_ids,
        limit=request.limit,
    )

    logger.info(f"Search completed: found {len(results)} results")

    return {"query": request.query, "mode": request.mode, "results": results}


@app.get("/api/history/{session_uuid}")
async def get_history(
    session_uuid: str,
    from_seq: Optional[int] = None,
    to_seq: Optional[int] = None,
):
    """Get terminal history for a session."""
    session = db.get_session_by_uuid(session_uuid)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    outputs = db.get_terminal_output(session.id, from_seq, to_seq)
    return {
        "session_uuid": session_uuid,
        "outputs": [
            {
                "sequence_num": o.sequence_num,
                "timestamp": o.timestamp.isoformat(),
                "content": o.content,
                "content_type": o.content_type,
            }
            for o in outputs
        ],
    }


@app.get("/api/screenshots/{session_uuid}")
async def list_screenshots(session_uuid: str, limit: int = 100):
    """List screenshots for a session."""
    session = db.get_session_by_uuid(session_uuid)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    screenshots = db.get_screenshots(session.id, limit)
    return {
        "session_uuid": session_uuid,
        "screenshots": [
            {
                "id": s.id,
                "timestamp": s.timestamp.isoformat(),
                "file_path": s.file_path,
                "width": s.width,
                "height": s.height,
            }
            for s in screenshots
        ],
    }


@app.post("/api/retention/run")
async def run_retention():
    """Manually trigger retention cleanup."""
    if not retention_manager:
        raise HTTPException(status_code=503, detail="Retention not enabled")

    retention_manager.run_cleanup()
    return {"success": True}
