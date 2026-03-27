"""Session management abstraction for flashback-terminal.

Supports multiple session backends:
- local: Direct PTY fork (default)
- screen: GNU Screen session management
- tmux: Tmux session management
"""

import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from flashback_terminal.config import get_config
from flashback_terminal.logger import Logger, log_function, logger


class SessionManagerError(Exception):
    """Error raised by session manager."""
    pass


class BinaryNotFoundError(SessionManagerError):
    """Error raised when required binary is not found."""

    def __init__(self, binary: str, install_cmd: str):
        self.binary = binary
        self.install_cmd = install_cmd
        super().__init__(f"Binary '{binary}' not found in PATH")

    def __str__(self) -> str:
        return f"""
{'='*70}
BINARY NOT FOUND: {self.binary}
{'='*70}

This feature requires '{self.binary}' to be installed and in your PATH.

To install:

    {self.install_cmd}

Then ensure it's in your PATH and try again.

{'='*70}
"""


class SessionCapture:
    """Captured session content (text and/or ANSI)."""

    def __init__(
        self,
        text: Optional[str] = None,
        ansi: Optional[str] = None,
        timestamp: Optional[float] = None,
        session_name: Optional[str] = None,
    ):
        self.text = text
        self.ansi = ansi
        self.timestamp = timestamp or time.time()
        self.session_name = session_name


class SessionInfo:
    """Information about a managed session."""

    def __init__(
        self,
        session_id: str,
        name: str,
        created_at: float,
        pid: Optional[int] = None,
        socket_path: Optional[str] = None,
        is_running: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.session_id = session_id
        self.name = name
        self.created_at = created_at
        self.pid = pid
        self.socket_path = socket_path
        self.is_running = is_running
        self.metadata = metadata or {}


class BaseSession(ABC):
    """Abstract base class for terminal sessions."""

    def __init__(
        self,
        session_id: str,
        name: str,
        profile: Dict[str, Any],
        on_output: Optional[Callable[[str], None]] = None,
    ):
        self.session_id = session_id
        self.name = name
        self.profile = profile
        self.on_output = on_output
        self._sequence_num = 0
        self._cwd: Optional[str] = None
        self._created_at = time.time()

    @abstractmethod
    def start(self) -> bool:
        """Start the session."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the session."""
        pass

    @abstractmethod
    def write(self, data: str) -> None:
        """Write data to the session."""
        pass

    @abstractmethod
    def read(self, timeout: float = 0.1) -> Optional[str]:
        """Read data from the session."""
        pass

    @abstractmethod
    def resize(self, rows: int, cols: int) -> None:
        """Resize the terminal."""
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """Check if session is running."""
        pass

    @abstractmethod
    def capture(self, full_scrollback: bool = False) -> Optional[SessionCapture]:
        """Capture session content (for backend screenshots)."""
        pass

    def update_cwd(self, cwd: str) -> None:
        """Update current working directory."""
        self._cwd = cwd

    def get_cwd(self) -> Optional[str]:
        """Get current working directory."""
        return self._cwd

    def _log_output(self, content: str) -> None:
        """Log output for history keeper."""
        self._sequence_num += 1
        if self.on_output:
            self.on_output(content)


class LocalSession(BaseSession):
    """Local PTY-based session (original implementation)."""

    def __init__(
        self,
        session_id: str,
        name: str,
        profile: Dict[str, Any],
        on_output: Optional[Callable[[str], None]] = None,
    ):
        super().__init__(session_id, name, profile, on_output)
        self._pid: Optional[int] = None
        self._fd: Optional[int] = None
        self._running = False
        import fcntl
        import pty
        import select
        import struct
        import termios
        self._fcntl = fcntl
        self._pty = pty
        self._select = select
        self._struct = struct
        self._termios = termios

    @log_function(Logger.DEBUG)
    def start(self) -> bool:
        """Start local PTY session."""
        logger.info(f"Starting local PTY session: {self.session_id}")

        config = get_config()
        shell = self.profile.get("shell") or os.environ.get("SHELL", "/bin/bash")
        args = self.profile.get("args", [])
        cwd = Path(self.profile.get("cwd", "~")).expanduser()
        env = {**os.environ, **self.profile.get("env", {})}

        if self.profile.get("login_shell", True):
            shell_name = os.path.basename(shell)
            args = [f"-{shell_name}"] + args

        try:
            self._pid, self._fd = self._pty.fork()

            if self._pid == 0:
                os.chdir(cwd)
                os.execvpe(shell, [shell] + args, env)
            else:
                self._running = True
                self.resize(
                    config.get("terminal.rows", 24),
                    config.get("terminal.cols", 80)
                )
                logger.info(f"Local PTY session started: pid={self._pid}")
                return True
        except Exception as e:
            logger.error(f"Failed to start local PTY session: {e}")
            return False

    def stop(self) -> None:
        """Stop local PTY session."""
        self._running = False
        if self._pid:
            try:
                os.kill(self._pid, 15)  # SIGTERM
                logger.debug(f"Sent SIGTERM to PID {self._pid}")
            except Exception as e:
                logger.debug(f"Error killing PID {self._pid}: {e}")
        if self._fd:
            try:
                os.close(self._fd)
            except Exception:
                pass

    def write(self, data: str) -> None:
        """Write to local PTY."""
        if self._fd is not None and self._running:
            try:
                os.write(self._fd, data.encode())
            except Exception as e:
                logger.debug(f"Write error: {e}")

    def read(self, timeout: float = 0.1) -> Optional[str]:
        """Read from local PTY."""
        if self._fd is None or not self._running:
            return None

        try:
            ready, _, _ = self._select.select([self._fd], [], [], timeout)
            if ready:
                data = os.read(self._fd, 4096)
                if data:
                    text = data.decode("utf-8", errors="replace")
                    self._log_output(text)
                    return text
                else:
                    self._running = False
        except Exception as e:
            logger.debug(f"Read error: {e}")
            self._running = False

        return None

    def resize(self, rows: int, cols: int) -> None:
        """Resize local PTY."""
        if self._fd is not None:
            try:
                size = self._struct.pack("HHHH", rows, cols, 0, 0)
                self._fcntl.ioctl(self._fd, self._termios.TIOCSWINSZ, size)
            except Exception as e:
                logger.debug(f"Resize error: {e}")

    def is_running(self) -> bool:
        """Check if local PTY is running."""
        if self._pid and self._running:
            try:
                pid, _ = os.waitpid(self._pid, os.WNOHANG)
                if pid == 0:
                    return True
            except Exception:
                pass
        return False

    def capture(self, full_scrollback: bool = False) -> Optional[SessionCapture]:
        """Capture not supported for local PTY (requires frontend)."""
        return None


class TmuxSession(BaseSession):
    """Tmux-based session management."""

    def __init__(
        self,
        session_id: str,
        name: str,
        profile: Dict[str, Any],
        socket_dir: str,
        on_output: Optional[Callable[[str], None]] = None,
    ):
        super().__init__(session_id, name, profile, on_output)
        self._socket_dir = Path(socket_dir).expanduser()
        self._socket_name = f"flashback-{self.session_id}"
        self._socket_path = self._socket_dir / self._socket_name
        self._tmux_binary = "tmux"
        self._target = f"{self._socket_name}:0.0"
        self._config_file: Optional[str] = None
        self._running = False

    def _get_env(self) -> Dict[str, str]:
        """Get environment for tmux commands (unsets TMUX for nested sessions)."""
        env = {**os.environ}
        # Unset tmux-related environment variables for nested session support
        tmux_vars = [
            "TMUX", "TMUX_PANE", "TMUX_WINDOW", "TMUX_SESSION",
            "TMUXinator_CONFIG", "TMUXINATOR_CONFIG"
        ]
        for var in tmux_vars:
            env.pop(var, None)
        # Set custom socket
        env["TMUX_TMPDIR"] = str(self._socket_dir)
        return env

    def _run_tmux(self, args: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run tmux command with custom socket."""
        cmd = [
            self._tmux_binary,
            "-S", str(self._socket_path),
        ]
        if self._config_file:
            cmd.extend(["-f", self._config_file])
        cmd.extend(args)

        env = self._get_env()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )

        if check and result.returncode != 0:
            raise SessionManagerError(
                f"tmux command failed: {' '.join(args)}\n{result.stderr}"
            )
        return result

    @log_function(Logger.DEBUG)
    def start(self) -> bool:
        """Start tmux session."""
        logger.info(f"Starting tmux session: {self.session_id}")

        self._socket_dir.mkdir(parents=True, exist_ok=True)

        shell = self.profile.get("shell") or os.environ.get("SHELL", "/bin/bash")
        args = self.profile.get("args", [])
        cwd = Path(self.profile.get("cwd", "~")).expanduser()
        profile_env = self.profile.get("env", {})

        if self.profile.get("login_shell", True):
            shell_name = os.path.basename(shell)
            args = [f"-{shell_name}"] + args

        # Build command to start shell
        start_command = f"cd {cwd} && exec {' '.join([shell] + args)}"

        try:
            # Create new session detached
            self._run_tmux([
                "new-session",
                "-d",
                "-s", self._socket_name,
                "-n", "main",
                start_command,
            ])

            # Set environment variables
            for key, value in profile_env.items():
                self._run_tmux([
                    "set-environment",
                    "-t", self._socket_name,
                    key, value,
                ], check=False)

            self._running = True
            logger.info(f"Tmux session started: {self._socket_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to start tmux session: {e}")
            return False

    def stop(self) -> None:
        """Stop tmux session."""
        try:
            self._run_tmux([
                "kill-session",
                "-t", self._socket_name,
            ], check=False)
        except Exception as e:
            logger.debug(f"Error stopping tmux session: {e}")
        self._running = False

    def write(self, data: str) -> None:
        """Send keys to tmux session."""
        if not self._running:
            return

        try:
            # Escape special characters for tmux send-keys
            escaped = data.replace("'", "'\"'\"'")
            self._run_tmux([
                "send-keys",
                "-t", self._target,
                escaped,
            ], check=False)
        except Exception as e:
            logger.debug(f"Write error: {e}")

    def read(self, timeout: float = 0.1) -> Optional[str]:
        """Read from tmux session (via capture-pane)."""
        if not self._running:
            return None

        try:
            result = self._run_tmux([
                "capture-pane",
                "-p",  # Print to stdout
                "-t", self._target,
            ], check=False)

            if result.returncode == 0 and result.stdout:
                # Only return new content since last read would require state tracking
                # For now, we rely on capture for content
                text = result.stdout
                self._log_output(text)
                return text
        except Exception as e:
            logger.debug(f"Read error: {e}")

        return None

    def resize(self, rows: int, cols: int) -> None:
        """Resize tmux window."""
        try:
            self._run_tmux([
                "resize-window",
                "-t", self._target,
                "-x", str(cols),
                "-y", str(rows),
            ], check=False)
        except Exception as e:
            logger.debug(f"Resize error: {e}")

    def is_running(self) -> bool:
        """Check if tmux session is running."""
        try:
            result = self._run_tmux([
                "has-session",
                "-t", self._socket_name,
            ], check=False)
            return result.returncode == 0
        except Exception:
            return False

    def capture(self, full_scrollback: bool = False) -> Optional[SessionCapture]:
        """Capture tmux pane content."""
        try:
            # Capture with escape sequences (ANSI)
            ansi_args = ["capture-pane", "-p", "-e", "-t", self._target]
            if full_scrollback:
                ansi_args.extend(["-S", "-", "-E", "-"])
            ansi_result = self._run_tmux(ansi_args, check=False)

            # Capture plain text
            text_args = ["capture-pane", "-p", "-t", self._target]
            if full_scrollback:
                text_args.extend(["-S", "-", "-E", "-"])
            text_result = self._run_tmux(text_args, check=False)

            return SessionCapture(
                text=text_result.stdout if text_result.returncode == 0 else None,
                ansi=ansi_result.stdout if ansi_result.returncode == 0 else None,
                session_name=self._socket_name,
            )
        except Exception as e:
            logger.error(f"Capture error: {e}")
            return None


class ScreenSession(BaseSession):
    """GNU Screen-based session management."""

    def __init__(
        self,
        session_id: str,
        name: str,
        profile: Dict[str, Any],
        socket_dir: str,
        on_output: Optional[Callable[[str], None]] = None,
    ):
        super().__init__(session_id, name, profile, on_output)
        self._socket_dir = Path(socket_dir).expanduser()
        self._session_name = f"flashback-{self.session_id}"
        self._socket_path = self._socket_dir / self._session_name
        self._screen_binary = "screen"
        self._running = False

    def _run_screen(self, args: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run screen command with custom socket."""
        cmd = [
            self._screen_binary,
            "-S", str(self._socket_path),
        ]
        cmd.extend(args)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )

        if check and result.returncode != 0:
            raise SessionManagerError(
                f"screen command failed: {' '.join(args)}\n{result.stderr}"
            )
        return result

    @log_function(Logger.DEBUG)
    def start(self) -> bool:
        """Start screen session."""
        logger.info(f"Starting screen session: {self.session_id}")

        self._socket_dir.mkdir(parents=True, exist_ok=True)

        shell = self.profile.get("shell") or os.environ.get("SHELL", "/bin/bash")
        args = self.profile.get("args", [])
        cwd = Path(self.profile.get("cwd", "~")).expanduser()
        profile_env = self.profile.get("env", {})

        if self.profile.get("login_shell", True):
            shell_name = os.path.basename(shell)
            args = [f"-{shell_name}"] + args

        # Build environment setup
        env_setup = ""
        for key, value in profile_env.items():
            env_setup += f'export {key}="{value}"; '

        start_command = f"cd {cwd} && {env_setup}exec {' '.join([shell] + args)}"

        try:
            # Create new detached session
            self._run_screen([
                "-d", "-m",  # Detached, auto-exit on process end
                "-s", shell,
                "bash", "-c", start_command,
            ])

            self._running = True
            logger.info(f"Screen session started: {self._session_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to start screen session: {e}")
            return False

    def stop(self) -> None:
        """Stop screen session."""
        try:
            self._run_screen([
                "-X", "quit",
            ], check=False)
        except Exception as e:
            logger.debug(f"Error stopping screen session: {e}")
        self._running = False

    def write(self, data: str) -> None:
        """Send input to screen session."""
        if not self._running:
            return

        try:
            # Use screen's stuff command to send input
            escaped = data.replace("'", "'\"'\"'")
            self._run_screen([
                "-X", "stuff", escaped,
            ], check=False)
        except Exception as e:
            logger.debug(f"Write error: {e}")

    def read(self, timeout: float = 0.1) -> Optional[str]:
        """Read from screen session (not directly supported, use capture)."""
        # Screen doesn't have a direct read mechanism like PTY
        # We use hardcopy for capture instead
        return None

    def resize(self, rows: int, cols: int) -> None:
        """Resize screen window."""
        try:
            self._run_screen([
                "-X", "stty", f"{rows}", f"{cols}",
            ], check=False)
        except Exception as e:
            logger.debug(f"Resize error: {e}")

    def is_running(self) -> bool:
        """Check if screen session is running."""
        try:
            result = self._run_screen([
                "-ls",
            ], check=False)
            return self._session_name in result.stdout
        except Exception:
            return False

    def capture(self, full_scrollback: bool = False) -> Optional[SessionCapture]:
        """Capture screen session content using hardcopy."""
        try:
            # Create temp file for hardcopy
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as f:
                temp_path = f.name

            # Use hardcopy to dump screen content
            self._run_screen([
                "-X", "hardcopy", "-h" if full_scrollback else "", temp_path,
            ], check=False)

            # Read the captured content
            time.sleep(0.1)  # Give screen time to write
            with open(temp_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

            # Clean up
            os.unlink(temp_path)

            return SessionCapture(
                text=content,
                ansi=None,  # Screen hardcopy doesn't preserve ANSI
                session_name=self._session_name,
            )
        except Exception as e:
            logger.error(f"Capture error: {e}")
            return None


class SessionManager:
    """Factory and manager for terminal sessions."""

    def __init__(self):
        self.config = get_config()
        self._sessions: Dict[str, BaseSession] = {}
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Check if required binaries are in PATH."""
        mode = self.config.session_manager_mode

        if mode == "tmux":
            binary = self.config.get("session_manager.tmux.binary", "tmux")
            if not shutil.which(binary):
                raise BinaryNotFoundError(
                    binary,
                    "sudo apt-get install tmux  # Debian/Ubuntu\n"
                    "sudo yum install tmux      # RHEL/CentOS\n"
                    "brew install tmux          # macOS"
                )

        elif mode == "screen":
            binary = self.config.get("session_manager.screen.binary", "screen")
            if not shutil.which(binary):
                raise BinaryNotFoundError(
                    binary,
                    "sudo apt-get install screen  # Debian/Ubuntu\n"
                    "sudo yum install screen      # RHEL/CentOS\n"
                    "brew install screen          # macOS"
                )

    @log_function(Logger.DEBUG)
    def create_session(
        self,
        session_id: str,
        name: str,
        profile: Dict[str, Any],
        on_output: Optional[Callable[[str], None]] = None,
    ) -> Optional[BaseSession]:
        """Create a new session based on configured mode."""
        mode = self.config.session_manager_mode
        logger.info(f"Creating session with mode '{mode}': {session_id}")

        if mode == "local":
            session = LocalSession(session_id, name, profile, on_output)
        elif mode == "tmux":
            socket_dir = self.config.get("session_manager.tmux.socket_dir", "~/.flashback-terminal/tmux")
            config_file = self.config.get("session_manager.tmux.config_file")
            session = TmuxSession(session_id, name, profile, socket_dir, on_output)
            session._config_file = config_file
        elif mode == "screen":
            socket_dir = self.config.get("session_manager.screen.socket_dir", "~/.flashback-terminal/screen")
            session = ScreenSession(session_id, name, profile, socket_dir, on_output)
        else:
            logger.error(f"Unknown session manager mode: {mode}")
            return None

        if session.start():
            self._sessions[session_id] = session
            return session
        return None

    def get_session(self, session_id: str) -> Optional[BaseSession]:
        """Get session by ID."""
        return self._sessions.get(session_id)

    def close_session(self, session_id: str) -> None:
        """Close a session."""
        if session_id in self._sessions:
            session = self._sessions[session_id]
            session.stop()
            del self._sessions[session_id]

    def list_sessions(self) -> List[SessionInfo]:
        """List all managed sessions."""
        sessions = []
        for session_id, session in self._sessions.items():
            sessions.append(SessionInfo(
                session_id=session_id,
                name=session.name,
                created_at=session._created_at,
                is_running=session.is_running(),
            ))
        return sessions

    def capture_session(
        self,
        session_id: str,
        full_scrollback: bool = False,
    ) -> Optional[SessionCapture]:
        """Capture session content."""
        session = self._sessions.get(session_id)
        if session:
            return session.capture(full_scrollback)
        return None


def get_session_manager() -> SessionManager:
    """Get or create the global session manager instance."""
    return SessionManager()
