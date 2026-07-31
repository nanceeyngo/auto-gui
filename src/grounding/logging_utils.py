"""
Centralized structured logging configuration.

This module is the single source of truth for logging setup across both
the `grounding` and `agent` packages. It replaces ad-hoc `print()` calls
with a configurable `logging`-based setup that supports:

    - Configurable log levels via environment variables.
    - Formatted stdout output.
    - Optional file-based log sink for post-execution debugging.
    - Structured contextual metadata via the `extra=` mechanism.

Environment variables
----------------------
GUI_AGENT_LOG_LEVEL
    Log level applied to the shared "auto_gui" logger hierarchy
    (e.g. "DEBUG", "INFO", "WARNING", "ERROR"). Defaults to "INFO".

GUI_AGENT_LOG_FILE
    Optional path to a file that log records should also be written to.
    If unset, only stdout logging is configured.

GUI_AGENT_LOG_FORMAT
    Optional override for the log line format string.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Final

_ROOT_LOGGER_NAME: Final[str] = "auto_gui"

_DEFAULT_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

_configured = False


def configure_logging(
    *,
    level: str | int | None = None,
    log_file: str | None = None,
    fmt: str | None = None,
    force: bool = False,
) -> logging.Logger:
    """
    Configure the shared "auto_gui" logger hierarchy.

    Safe to call multiple times; configuration is applied once unless
    ``force=True`` is passed (useful for tests that need to reconfigure
    handlers/levels).

    Returns the configured root logger for the package hierarchy.
    """
    global _configured

    root = logging.getLogger(_ROOT_LOGGER_NAME)

    if _configured and not force:
        return root

    resolved_level = level or os.environ.get("GUI_AGENT_LOG_LEVEL", "INFO")
    resolved_format = fmt or os.environ.get("GUI_AGENT_LOG_FORMAT", _DEFAULT_FORMAT)
    resolved_log_file = log_file or os.environ.get("GUI_AGENT_LOG_FILE")

    root.setLevel(resolved_level)
    root.propagate = False

    for handler in tuple(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(resolved_format)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if resolved_log_file:
        file_handler = logging.FileHandler(resolved_log_file)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _configured = True

    return root


def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger under the shared "auto_gui" hierarchy.

    Ensures `configure_logging()` has run at least once so that any
    logger obtained this way is immediately usable.
    """
    configure_logging()

    if name.startswith(f"{_ROOT_LOGGER_NAME}."):
        return logging.getLogger(name)

    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")


def log_event(
    logger: logging.Logger,
    level: int,
    message: str,
    **context: Any,
) -> None:
    """
    Emit a log record with structured contextual metadata.

    Context keyword arguments are attached via the stdlib `extra`
    mechanism (as `record.context`) and also appended to the rendered
    message so they remain visible with the default formatter.
    """
    if context:
        rendered_context = " ".join(
            f"{key}={value!r}" for key, value in context.items()
        )
        message = f"{message} | {rendered_context}"

    logger.log(level, message, extra={"context": context})


__all__ = [
    "configure_logging",
    "get_logger",
    "log_event",
]
