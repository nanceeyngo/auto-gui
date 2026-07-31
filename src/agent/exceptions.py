"""
Exceptions raised by the GUI automation agent's execution loop.
"""


class AgentError(Exception):
    """
    Base exception for all agent-level errors.
    """


class AgentExecutionError(AgentError):
    """
    Raised when a task fails to complete due to an unrecoverable error
    in the underlying reasoning/tool-use loop (e.g. the language model
    backend errors out, or an unhandled exception escapes tool
    execution).

    The original exception is available via `__cause__`.
    """

    def __init__(self, message: str, *, goal: str | None = None) -> None:
        super().__init__(message)
        self.goal = goal


__all__ = [
    "AgentError",
    "AgentExecutionError",
]
