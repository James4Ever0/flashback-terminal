"""Logging system for flashback-terminal with verbosity control."""

import logging
import sys
from functools import wraps
from typing import Any, Callable, Optional


class Logger:
    """Global logger with verbosity control."""

    _instance: Optional["Logger"] = None
    _verbosity: int = 0  # 0=ERROR, 1=WARNING, 2=INFO, 3=DEBUG, 4=TRACE

    # Verbosity levels
    ERROR = 0
    WARNING = 1
    INFO = 2
    DEBUG = 3
    TRACE = 4

    # Level names for display
    LEVEL_NAMES = {
        0: "ERROR",
        1: "WARN",
        2: "INFO",
        3: "DEBUG",
        4: "TRACE",
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._setup_logger()
        return cls._instance

    def _setup_logger(self):
        """Setup the underlying logger."""
        self._logger = logging.getLogger("flashback-terminal")
        self._logger.setLevel(logging.DEBUG)

        # Console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)

        # Formatter with verbosity level
        formatter = logging.Formatter(
            "[%(levelname)s] [%(name)s] %(message)s"
        )
        handler.setFormatter(formatter)

        # Remove existing handlers to avoid duplicates
        self._logger.handlers.clear()
        self._logger.addHandler(handler)

    @classmethod
    def set_verbosity(cls, level: int) -> None:
        """Set global verbosity level (0-4)."""
        cls._verbosity = max(0, min(4, level))
        instance = cls()
        instance.log(cls.INFO, f"Verbosity level set to {cls._verbosity} ({cls.LEVEL_NAMES[cls._verbosity]})")

    @classmethod
    def get_verbosity(cls) -> int:
        """Get current verbosity level."""
        return cls._verbosity

    @classmethod
    def should_log(cls, level: int) -> bool:
        """Check if message at given level should be logged."""
        return cls._verbosity >= level

    def log(self, level: int, message: str, *args, **kwargs) -> None:
        """Log a message if verbosity permits."""
        if not self.should_log(level):
            return

        # Map our levels to logging levels
        logging_levels = {
            self.ERROR: logging.ERROR,
            self.WARNING: logging.WARNING,
            self.INFO: logging.INFO,
            self.DEBUG: logging.DEBUG,
            self.TRACE: logging.DEBUG,  # TRACE maps to DEBUG for underlying logger
        }

        log_level = logging_levels.get(level, logging.INFO)

        # Add verbosity indicator for TRACE
        if level == self.TRACE:
            message = f"[TRACE] {message}"

        self._logger.log(log_level, message, *args, **kwargs)

    @classmethod
    def error(cls, message: str, *args, **kwargs) -> None:
        """Log error message."""
        cls().log(cls.ERROR, message, *args, **kwargs)

    @classmethod
    def warning(cls, message: str, *args, **kwargs) -> None:
        """Log warning message."""
        cls().log(cls.WARNING, message, *args, **kwargs)

    @classmethod
    def info(cls, message: str, *args, **kwargs) -> None:
        """Log info message."""
        cls().log(cls.INFO, message, *args, **kwargs)

    @classmethod
    def debug(cls, message: str, *args, **kwargs) -> None:
        """Log debug message."""
        cls().log(cls.DEBUG, message, *args, **kwargs)

    @classmethod
    def trace(cls, message: str, *args, **kwargs) -> None:
        """Log trace message (highest verbosity)."""
        cls().log(cls.TRACE, message, *args, **kwargs)


def log_function(level: int = Logger.DEBUG):
    """Decorator to log function entry/exit with parameters."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = Logger()
            func_name = func.__name__
            module_name = func.__module__

            # Log function entry
            if logger.should_log(level):
                # Format arguments
                args_str = ""
                if args:
                    args_str += f"args={_truncate_args(args)}"
                if kwargs:
                    if args_str:
                        args_str += ", "
                    args_str += f"kwargs={_truncate_args(kwargs)}"

                logger.log(level, f"[ENTER] {module_name}.{func_name}({args_str})")

            try:
                # Execute function
                result = func(*args, **kwargs)

                # Log function exit
                if logger.should_log(level):
                    result_str = _truncate_value(result)
                    logger.log(level, f"[EXIT] {module_name}.{func_name} -> {result_str}")

                return result

            except Exception as e:
                # Log exception
                logger.error(f"[EXCEPTION] {module_name}.{func_name}: {type(e).__name__}: {e}")
                raise

        return wrapper
    return decorator


def log_progress(operation: str, level: int = Logger.INFO):
    """Decorator to log progress of long-running operations."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = Logger()
            func_name = func.__name__

            logger.log(level, f"[PROGRESS] {operation}: Starting {func_name}...")

            try:
                result = func(*args, **kwargs)
                logger.log(level, f"[PROGRESS] {operation}: Completed {func_name}")
                return result
            except Exception as e:
                logger.error(f"[PROGRESS] {operation}: Failed - {e}")
                raise

        return wrapper
    return decorator


def _truncate_args(args, max_len: int = 100) -> str:
    """Truncate arguments for logging."""
    args_str = str(args)
    if len(args_str) > max_len:
        return args_str[:max_len] + "..."
    return args_str


def _truncate_value(value, max_len: int = 100) -> str:
    """Truncate value for logging."""
    value_str = repr(value)
    if len(value_str) > max_len:
        return value_str[:max_len] + "..."
    return value_str


# Convenience instance
logger = Logger()
