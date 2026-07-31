"""
Window management tools for the GUI automation agent.
"""

import time

import pywinctl
from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict

from ..logging_config import get_logger

logger = get_logger("agent.tools.windows")


class WindowInfo(BaseModel):
    """
    Serializable representation of an application window.
    """

    model_config = ConfigDict(
        frozen=True,
    )

    title: str

    left: int
    top: int

    width: int
    height: int

    is_active: bool
    is_minimized: bool
    is_maximized: bool
    is_visible: bool


def _to_window_info(
    window: pywinctl.Window,
) -> WindowInfo:
    """
    Convert a pywinctl window into a serializable model.
    """
    return WindowInfo(
        title=window.title,
        left=window.left,
        top=window.top,
        width=window.width,
        height=window.height,
        is_active=window.isActive,
        is_minimized=window.isMinimized,
        is_maximized=window.isMaximized,
        is_visible=window.isVisible,
    )


class WindowManager:
    """
    High-level wrapper around pywinctl.

    The manager exposes strongly-typed operations while hiding the
    underlying pywinctl API from the rest of the agent.
    """

    __slots__ = ()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    @staticmethod
    def list_windows() -> tuple[WindowInfo, ...]:
        """
        Return all visible titled windows.
        """
        windows: list[WindowInfo] = []

        for window in pywinctl.getAllWindows():
            title = window.title.strip()

            if not title:
                continue

            if not window.isVisible:
                continue

            windows.append(_to_window_info(window))

        return tuple(windows)

    @staticmethod
    def find_windows(
        title: str,
        *,
        exact: bool = False,
    ) -> tuple[pywinctl.Window, ...]:
        """
        Find windows whose titles match.

        Parameters
        ----------
        title
            Title (or substring) to search for.

        exact
            Whether an exact title match is required.
        """
        windows = tuple(
            window
            for window in pywinctl.getWindowsWithTitle(title)
            if window.title.strip()
        )

        if exact:
            windows = tuple(window for window in windows if window.title == title)

        return windows

    def get_window(
        self,
        title: str,
        *,
        exact: bool = False,
    ) -> pywinctl.Window:
        """
        Return a single matching window.

        Raises
        ------
        RuntimeError
            If zero or multiple windows match.
        """
        windows = self.find_windows(
            title,
            exact=exact,
        )

        if not windows:
            raise RuntimeError(f"No window found with title '{title}'.")

        if len(windows) > 1:
            raise RuntimeError(f"Multiple windows match title '{title}'.")

        assert len(windows) == 1

        return windows[0]

    @staticmethod
    def get_active_window() -> WindowInfo:
        """
        Return the currently active window.
        """
        window = pywinctl.getActiveWindow()

        if window is None:
            raise RuntimeError("No active window.")

        return _to_window_info(window)

    # ------------------------------------------------------------------
    # Waiting
    # ------------------------------------------------------------------

    def wait_for_window(
        self,
        title: str,
        *,
        timeout: float = 10.0,
        poll_interval: float = 0.25,
        exact: bool = False,
    ) -> pywinctl.Window:
        """
        Wait until a matching window appears.

        Raises
        ------
        TimeoutError
            If no matching window appears before timeout.
        """
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            windows = self.find_windows(
                title,
                exact=exact,
            )

            if len(windows) == 1:
                return windows[0]

            if len(windows) > 1:
                raise RuntimeError(f"Multiple windows match title '{title}'.")

            time.sleep(poll_interval)

        raise TimeoutError(f"Timed out waiting for window '{title}'.")

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def activate_window(
        self,
        title: str,
        *,
        exact: bool = False,
    ) -> WindowInfo:
        """
        Activate a window.
        """
        window = self.get_window(
            title,
            exact=exact,
        )

        window.activate()

        try:
            window.restore()
        except Exception as exc:
            logger.warning(
                "Failed to restore window after activation",
                extra={
                    "context": {
                        "title": title,
                        "exception_type": type(exc).__name__,
                        "exception": str(exc),
                    }
                },
            )

        return _to_window_info(window)

    def minimize_window(
        self,
        title: str,
        *,
        exact: bool = False,
    ) -> WindowInfo:
        """
        Minimize a window.
        """
        window = self.get_window(
            title,
            exact=exact,
        )

        window.minimize()

        return _to_window_info(window)

    def maximize_window(
        self,
        title: str,
        *,
        exact: bool = False,
    ) -> WindowInfo:
        """
        Maximize a window.
        """
        window = self.get_window(
            title,
            exact=exact,
        )

        window.maximize()

        return _to_window_info(window)

    def restore_window(
        self,
        title: str,
        *,
        exact: bool = False,
    ) -> WindowInfo:
        """
        Restore a minimized/maximized window.
        """
        window = self.get_window(
            title,
            exact=exact,
        )

        window.restore()

        return _to_window_info(window)

    def close_window(
        self,
        title: str,
        *,
        exact: bool = False,
    ) -> None:
        """
        Close a window.
        """
        window = self.get_window(
            title,
            exact=exact,
        )

        window.close()


_manager = WindowManager()

# ----------------------------------------------------------------------
# LangChain tools
# ----------------------------------------------------------------------


@tool
def list_windows() -> list[str]:
    """
    List all visible application windows.
    """
    return [window.title for window in _manager.list_windows()]


@tool
def get_active_window() -> str:
    """
    Return the title of the currently active window.
    """
    return _manager.get_active_window().title


@tool
def activate_window(
    title: str,
) -> str:
    """
    Activate an application window.

    Parameters
    ----------
    title
        Window title.
    """
    return _manager.activate_window(title).title


@tool
def minimize_window(
    title: str,
) -> str:
    """
    Minimize an application window.
    """
    return _manager.minimize_window(title).title


@tool
def maximize_window(
    title: str,
) -> str:
    """
    Maximize an application window.
    """
    return _manager.maximize_window(title).title


@tool
def restore_window(
    title: str,
) -> str:
    """
    Restore an application window.
    """
    return _manager.restore_window(title).title


@tool
def close_window(
    title: str,
) -> str:
    """
    Close an application window.
    """
    _manager.close_window(title)

    return title


__all__ = [
    "WindowInfo",
    "WindowManager",
    "activate_window",
    "close_window",
    "get_active_window",
    "list_windows",
    "maximize_window",
    "minimize_window",
    "restore_window",
]
