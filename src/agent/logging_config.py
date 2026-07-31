"""
Logging configuration for the GUI automation agent.

The agent package reuses the shared logging setup defined in
`grounding.logging_utils` so that both packages log through the same
"auto_gui" logger hierarchy, with a single set of environment variables
controlling verbosity and output sinks (see that module's docstring).
"""

from grounding.logging_utils import configure_logging, get_logger, log_event

__all__ = [
    "configure_logging",
    "get_logger",
    "log_event",
]
