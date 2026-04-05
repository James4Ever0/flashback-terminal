"""Terminal session management for flashback-terminal.

Uses GNU Screen or Tmux for session management (no local PTY mode).
This enables backend screenshot capture and text extraction.
"""

import uuid as uuid_mod
from datetime import datetime
from typing import Any, Callable, Dict, Optional
import traceback
import os
import asyncio
import atexit

from flashback_terminal.config import get_config
from flashback_terminal.database import Database
from flashback_terminal.logger import Logger, log_function, logger
from flashback_terminal.session_manager import (
    BaseSession,
    SessionCapture,
    get_session_manager,
)


class TerminalSession:
    """Manages a single terminal session (wrapper around BaseSession)."""

    def __init__(
        self,
        session_id: int,
        uuid: str,
        db: Database,
        profile: Dict[str, Any],
        session_type: Optional[str] = None,
        on_output: Optional[Callable[[str], None]] = None,
        on_clear: Optional[Callable[[], None]] = None,
        on_cursor: Optional[Callable[[int, int], None]] = None,
    ):
        config = get_config()
        self.session_id = session_id
        self.uuid = uuid
        self.db = db
        self.on_output = on_output
        self.on_clear = on_clear
        self.on_cursor = on_cursor
        self.profile = profile

        if not session_type:
            session_type = config.session_manager_mode

        self.session_type = session_type

        self._terminal_size :Dict[str, int] = dict(rows=-1,cols=-1)

        self._session: Optional[BaseSession] = None
        self.sequence_num = 0
        self._cwd: Optional[str] = None
        self._running = False
        self.is_running_buffered=False

    def _on_session_clear(self) -> None:
        """Handle clear event from underlying session."""
        if self.on_clear:
            self.on_clear()
    
    def _on_session_cursor(self, col:int, row:int) -> None:
        if self.on_cursor:
            self.on_cursor(col, row)

    async def _on_session_output(self, content: str) -> None:
        """Handle output from underlying session."""
        config = get_config()
        logger.debug(f"[TerminalSession] Received output (len={len(content)}): {content[:50]}...")

        if config.is_module_enabled("history_keeper"):
            self.sequence_num += 1
            logger.debug(f"[TerminalSession] Storing output in database (session_id={self.session_id}, seq={self.sequence_num})")
            await self.db.insert_terminal_output(
                self.session_id, self.sequence_num, content, "output"
            )

        if self.on_output:
            self.on_output(content)

    @log_function(Logger.DEBUG)
    async def start(self) -> bool:
        """Start the terminal session."""
        logger.info(f"Starting terminal session: uuid={self.uuid}, profile={self.profile.get('name', 'default')}")

        config = get_config()
        session_manager = get_session_manager()

        self._session = await session_manager.create_session(
            session_id=self.uuid,
            name=f"Terminal-{self.session_id}",
            profile=self.profile,
            on_output=self._on_session_output,
            on_clear=self._on_session_clear,
            on_cursor = self._on_session_cursor,
        )

        if self._session:
            self._running = True
            logger.info(f"Terminal session started: uuid={self.uuid}")
            return True
        else:
            logger.error(f"Failed to start terminal session: uuid={self.uuid}")
            return False

    async def resize(self, rows: int, cols: int) -> None:
        """Resize the terminal."""
        logger.debug(f"[TerminalSession] Resize request: rows={rows}, cols={cols}")
        if self._session:
            logger.debug(f"[TerminalSession] Forwarding resize to session {self._session.__class__.__name__}")
            await self._session.resize(rows, cols)
            self._terminal_size = dict(rows=rows,cols=cols)
        else:
            logger.warning(f"[TerminalSession] Cannot resize - no active session")

    async def write(self, data: str) -> None:
        """Write data to the terminal."""
        if self._session and self._running:
            await self._session.write(data)

    async def read(self, timeout: float = 0.1) -> Optional[str]:
        """Read data from the terminal."""
        if self._session is None or not self._running:
            return None

        data = await self._session.read(timeout)
        if data is None and not await self._session.is_running():
            self._running = False
        return data

    async def update_cwd(self, cwd: str) -> None:
        """Update the current working directory."""
        self._cwd = cwd
        if self._session:
            self._session.update_cwd(cwd)
        await self.db.update_session(self.session_id, last_cwd=cwd)

    def get_cwd(self) -> Optional[str]:
        """Get the current working directory."""
        if self._session:
            return self._session.get_cwd()
        return self._cwd

    async def is_running(self) -> bool:
        """Check if the session is still running."""
        ret = False
        if self._session and self._running:
            ret = await self._session.is_running()
        self.is_running_buffered = ret
        return ret

    async def capture(self, full_scrollback: bool = False) -> Optional[SessionCapture]:
        """Capture session content (for backend screenshots)."""
        if self._session:
            return await self._session.capture(full_scrollback)
        return None

    async def stop(self) -> None:
        """Stop the terminal session."""
        self._running = False
        if self._session:
            await self._session.stop()



def check_socket_present(session_uuid:str, session_type:str) -> bool:
    config = get_config()
    socket_accessible=False

    if session_type == "tmux":
        socket_dir = config.get("session_manager.tmux.socket_dir", "~/.flashback-terminal/tmux")

        socket_dir = os.path.expanduser(socket_dir)

        if not os.path.isdir(socket_dir):
            logger.warning(f"[Terminal] Socket directory {socket_dir} does not exist, cannot verify presense of tmux session")
        else:
            # check if in this dir there is a file with session_uuid in it.
            files: list[str] = os.listdir(socket_dir)

            # filter out those config files, also socket file is of zero file length.
            files = [it for it in files if not it.endswith(".conf")]

            if "flashback-" + session_uuid in files:
                socket_accessible = True
            else:
                logger.warning(f"[Terminal] Tmux socket file for session {session_uuid} not found in {socket_dir}")
    elif session_type == "screen":
        socket_dir = config.get("session_manager.screen.socket_dir", "~/.flashback-terminal/screen")

        socket_dir = os.path.expanduser(socket_dir)
        socket_accessible=False

        if not os.path.isdir(socket_dir):
            logger.warning(f"[Terminal] Socket directory {socket_dir} does not exist, cannot verify presense of screen session")
        else:
            # check if in this dir there is a file with session_uuid in it.
            files = os.listdir(socket_dir)

            files = [it for it in files if not it.endswith(".rc")]
            files = [it for it in files if it.count(".") == 1]

            socket_lookups = [it.split(".", 1)[1] for it in files] # remove pid?

            if "flashback-" + session_uuid in socket_lookups:
                socket_accessible = True
            else:
                logger.warning(f"[Terminal] Screen socket file for session {session_uuid} not found in {socket_dir}")

    else:
        raise RuntimeError("[Terminal] Unsupported session type lookup for socket presence: %s" % session_type)
                
    return socket_accessible


class TerminalManager:
    """Manages multiple terminal sessions."""

    @log_function(Logger.DEBUG)
    def __init__(self, db: Database):
        self.db = db
        self.sessions: Dict[str, TerminalSession] = {}
        self.config = get_config()
        logger.debug("TerminalManager initialized")
        # TODO: cancel the watch dog task on exit?
        self._watchdog_interval = 1
        self._watchdog_running = True
        # TODO: watch dog is bad. it would kill normal background sessions after server exit. how about inject some "KeyboardInterrupt" listener, set a flag, and protecting from watchdog task taking effect?
        self._watchdog_task = asyncio.create_task(self.watchdog())
        atexit.register(self._close_sync)
        self._closing = False

    @log_function(Logger.DEBUG)
    async def revive_session(self, session_uuid:str) -> Optional[TerminalSession]:
        # assume socket is not present, just recreate it from db data.
        logger.info(f"Attempting to revive session: {session_uuid}")
        
        # Get session info from database
        db_session = await self.db.get_session_by_uuid(session_uuid)
        if not db_session:
            logger.warning(f"[TerminalManager] Session {session_uuid} not found in database")
            return None

        term_session = self.get_session(session_uuid)
        if term_session:
            logger.info(f"[TerminalManager] Session {session_uuid} already exists")
            return term_session
        
        # assert socket not to present.
        socket_present = check_socket_present(session_uuid, db_session.session_type)

        if socket_present:
            logger.warning("Attempting to revive session with socket present: %s" % session_uuid)
            return await self.restore_session(session_uuid) 

        profile = self.config.get_profile(db_session.profile_name) or {"name": "default"}

        session = TerminalSession(
            session_id=db_session.id,
            uuid=db_session.uuid,
            db=self.db,
            profile=profile,
        )

        if await session.start():
            self.sessions[session_uuid] = session
            return session
        else:
            logger.error(f"Failed to revive session: {db_session.uuid}")
            await self.db.delete_session(db_session.id)
            return None
    
    @log_function(Logger.DEBUG)
    async def restore_session(self, session_uuid: str) -> Optional[TerminalSession]:
        """Restore a terminal session by checking socket existence first."""
        logger.info(f"Attempting to restore session: {session_uuid}")
        
        # Get session info from database
        db_session = await self.db.get_session_by_uuid(session_uuid)
        if not db_session:
            logger.warning(f"[TerminalManager] Session {session_uuid} not found in database")
            return None
        
        # Get session manager and create a temporary session to check socket existence
        config = get_config()

        term_session = self.get_session(session_uuid)
        if term_session:
            logger.info(f"[TerminalManager] Session {session_uuid} already exists")
            return term_session
        
        # Create a temporary session object just to check if the underlying socket/session exists
        try:
            profile = self.config.get_profile(db_session.profile_name) or {"name": "default"}
            
            if db_session.session_type == "tmux":
                socket_dir = config.get("session_manager.tmux.socket_dir", "~/.flashback-terminal/tmux")

                socket_dir = os.path.expanduser(socket_dir)
                socket_accessible=False

                if not os.path.isdir(socket_dir):
                    logger.warning(f"[TerminalManager] Socket directory {socket_dir} does not exist, cannot verify presense of tmux session")
                else:
                    # check if in this dir there is a file with session_uuid in it.
                    files: list[str] = os.listdir(socket_dir)

                    # filter out those config files, also socket file is of zero file length.
                    files = [it for it in files if not it.endswith(".conf")]

                    if "flashback-" + session_uuid in files:
                        socket_accessible = True
                    else:
                        logger.warning(f"[TerminalManager] Tmux socket file for session {session_uuid} not found in {socket_dir}")
                        
                if socket_accessible:
                    # Create proper TerminalSession wrapper
                    terminal_session = TerminalSession(
                        session_id=db_session.id,
                        uuid=session_uuid,
                        db=self.db,
                        profile=profile,
                    )

                    if not await terminal_session.start():
                        logger.error(f"[TerminalManager] Failed to start terminal session {session_uuid}")
                        return None
                        
                    # Update database to reflect actual status - only after confirming socket exists
                    await self.db.update_session(db_session.id, status="active")
                        
                    # Add to active sessions
                    self.sessions[session_uuid] = terminal_session
                            
                    logger.info(f"[TerminalManager] Successfully restored session {session_uuid}")
                    return terminal_session
                else:
                    logger.warning(f"[TerminalManager] Tmux session {session_uuid} socket not accessible")
                    # Update database to reflect actual status
                    await self.db.update_session(db_session.id, status="inactive")
                    return None
            elif db_session.session_type == "screen":
                socket_dir = config.get("session_manager.screen.socket_dir", "~/.flashback-terminal/screen")

                socket_dir = os.path.expanduser(socket_dir)
                socket_accessible = False

                if not os.path.isdir(socket_dir):
                    logger.warning(f"[TerminalManager] Socket directory {socket_dir} does not exist, cannot verify presense of screen session")
                else:
                    # check if in this dir there is a file with session_uuid in it.
                    files = os.listdir(socket_dir)

                    files = [it for it in files if not it.endswith(".rc")]
                    files = [it for it in files if it.count(".") == 1]

                    socket_lookups = [it.split(".", 1)[1] for it in files] # remove pid?

                    if "flashback-" + session_uuid in socket_lookups:
                        socket_accessible = True
                    else:
                        logger.warning(f"[TerminalManager] Screen socket file for session {session_uuid} not found in {socket_dir}")

                if socket_accessible:
                    # Create proper TerminalSession wrapper
                    terminal_session = TerminalSession(
                        session_id=db_session.id,
                        uuid=session_uuid,
                        db=self.db,
                        profile=profile,
                        on_output=None,
                        on_clear=None,
                        on_cursor=None
                    )

                    if not await terminal_session.start():
                        logger.error(f"[TerminalManager] Failed to start terminal session {session_uuid}")
                        return None
                    
                    # Update database to reflect actual status - only after confirming socket exists
                    await self.db.update_session(db_session.id, status="active")

                    # Add to active sessions
                    self.sessions[session_uuid] = terminal_session
                    
                    logger.info(f"[TerminalManager] Successfully restored session {session_uuid}")
                    return terminal_session
                else:
                    logger.warning(f"[TerminalManager] Screen session {session_uuid} socket not accessible")
                    # Update database to reflect actual status
                    await self.db.update_session(db_session.id, status="inactive")
                    return None
            else:
                logger.error(f"[TerminalManager] Unsupported session type: {db_session.session_type}")
                return None
        except Exception as e:
            logger.error(f"[TerminalManager] Error restoring session {session_uuid}: {e}")
            # Update database to reflect actual status
            if db_session:
                await self.db.update_session(db_session.id, status="inactive")
            return None

    @log_function(Logger.DEBUG)
    async def create_session(
        self, profile_name: str = "default", name: Optional[str] = None, session_type: Optional[str]=None
    ) -> Optional[TerminalSession]:
        """Create a new terminal session."""
        logger.info(f"Creating session: profile={profile_name}, name={name}")
        profile = self.config.get_profile(profile_name)

        if not session_type:
            session_type = self.config.session_manager_mode

        if not profile:
            logger.error(f"Profile not found: {profile_name}")
            return None

        uuid_str = str(uuid_mod.uuid4())
        session_name = name or f"Terminal {len(self.sessions) + 1}"

        session_id = await self.db.create_session(
            uuid=uuid_str, name=session_name, profile_name=profile_name, session_type=session_type,
        )

        session = TerminalSession(
            session_id=session_id,
            uuid=uuid_str,
            db=self.db,
            profile=profile,
        )

        if await session.start():
            self.sessions[uuid_str] = session
            return session
        else:
            await self.db.delete_session(session_id)
            return None

    def get_session(self, uuid: str) -> Optional[TerminalSession]:
        """Get a session by UUID."""
        return self.sessions.get(uuid)

    async def close_session(self, uuid: str) -> None:
        """Close a session."""
        if uuid in self.sessions:
            session = self.sessions[uuid]
            await session.stop()
            await self.db.update_session(
                session.session_id,
                status="inactive",
                ended_at=datetime.now().isoformat(),
            )
            del self.sessions[uuid]
    
    async def _watchdog_loop(self) -> tuple[int, int]:
        """check all terminal sessions in 1 second period."""
        # use asyncio.wait to wait for tasks.
        check_tasks: list[asyncio.Task] = []
        close_tasks: list[asyncio.Task] = []
        # seriously, how do you know if one task is running

        session_ids_to_close = set()
        

        for uuid, term_sess in self.sessions.items():
            term_sess_sess = term_sess._session
            if not term_sess_sess:
                # remove this entry already. no need to exist.
                session_ids_to_close.add(uuid)
                continue
            if not term_sess.is_running_buffered:
                # since the buffered is_running result shows it is dead, we will purge it.
                session_ids_to_close.add(uuid)
                continue
            check_task_coro = term_sess_sess._is_running()
            _check_task = asyncio.create_task(check_task_coro, name=uuid)
            check_tasks.append(_check_task)

        done, pending = await asyncio.wait(check_tasks, timeout=1)

        for it in done:
            task_name = it.get_name()
            running = False
            try:
                running = it.result()
            except Exception as e:
                tb = traceback.format_exc()
                logger.error("[TerminalManager] Error getting running check result from session %s: \n%s" % (task_name, tb))
            if not running:
                session_ids_to_close.add(task_name)

        
        for it in pending:
            task_name = it.get_name()
            # assume to be dead.
            logger.info("[TerminalManager] Assume session %s to be dead because of timeout." % task_name)
            it.cancel("[TerminalManager] Watchdog timeout for checking session %s running status" % task_name)
            session_ids_to_close.add(task_name)

        # for those pending, i suspect they are blocked. shall we terminate them as well?

        if not self._watchdog_running:
            return 0, len(session_ids_to_close)

        if self._closing:
            return 0, len(session_ids_to_close)

        for uuid in session_ids_to_close:
            close_task_coro = self.close_session(uuid)
            _close_task = asyncio.create_task(close_task_coro, name=uuid)
            close_tasks.append(_close_task)
        
        # TODO: add timeout for closing task?
        done_close, pending_close = await asyncio.wait(close_tasks, timeout=2)

        done_close_count = 0
        fail_close_count = 0

        for it in done_close:
            task_name = it.get_name()
            try:
                it.result()
                done_close_count += 1
            except Exception as e:
                fail_close_count += 1
                tb = traceback.format_exc()
                logger.error("[TerminalManager] Error closing session %s: \n%s" % (task_name, tb))
        
        for it in pending_close:
            task_name = it.get_name()
            fail_close_count += 1
            logger.info("[TerminalManager] Failed to stop session %s because of timeout." % task_name)
            it.cancel("[TerminalManager] Watchdog timeout for stopping session %s" % task_name)

        return done_close_count, fail_close_count
    
    async def watchdog(self):
        while self._watchdog_running:
            if self._closing: break
            time_start = asyncio.get_running_loop().time()
            # TODO: report status?
            success_count, fail_count = await self._watchdog_loop()
            logger.info("[TerminalManager] Watchdog closed (success=%s, fail=%s, total=%s) inactive sessions" % (success_count, fail_count, success_count+fail_count))
            time_end = asyncio.get_running_loop().time()
            duration = time_end-time_start
            if duration > self._watchdog_interval:
                # skip sleep
                continue
            else:
                sleep_time = self._watchdog_interval - duration
                await asyncio.sleep(sleep_time)



    async def capture_session(
        self,
        uuid: str,
        full_scrollback: bool = False,
    ) -> Optional[SessionCapture]:
        """Capture a session's content."""
        session = self.sessions.get(uuid)
        if session:
            return await session.capture(full_scrollback)
        return None
    
    @log_function(Logger.DEBUG)
    def _close_sync(self):
        asyncio.run(self.close())

    @log_function(Logger.DEBUG)
    async def close(self):
        if self._closing: return # already closing.
        self._closing=True
        # first close all sessions? bad idea, since that will terminate the underlying session too. we want them be running in background even if server is down.

        # close_tasks = []
        # for uuid in self.sessions:
        #     _close_coro = self.close_session(uuid)
        #     _close_task = asyncio.create_task(_close_coro, name=uuid)
        #     close_tasks.append(_close_task)
        # await asyncio.gather(*close_tasks)

        del self.sessions
        del self.db
        del self.config

        # close watchdog task.
        self._watchdog_running = False
        if self._watchdog_task:
            self._watchdog_task.cancel("[TerminalManager] Closing watchdog task on exit")
            del self._watchdog_task