"""
Test doubles for the OS automation layer (`pyautogui` / `pywinctl`).

The real `pyautogui`/`pywinctl` packages require a live display (X11,
Windows desktop, or macOS session) just to *import*, which makes the
`agent` package untestable in headless CI environments. These fakes
stand in for the real modules so that `agent.tools.action.actions`,
`agent.tools.screenshot.*`, and `agent.tools.windows` can be imported
and exercised deterministically, with no GUI/display dependency.

Usage: `tests/conftest.py` installs these into `sys.modules` before any
test module is collected.
"""

from __future__ import annotations

import types
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MouseState:
    x: int = 0
    y: int = 0


@dataclass
class CallRecorder:
    """
    Records every call made against the fake pyautogui module, so
    tests can assert on exactly what was dispatched to "the OS".
    """

    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = field(
        default_factory=list
    )

    def record(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append((name, args, kwargs))

    def calls_named(self, name: str) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
        return [
            (args, kwargs)
            for (call_name, args, kwargs) in self.calls
            if call_name == name
        ]


class _FakeImage:
    """
    Minimal PIL.Image.Image stand-in exposing only `.size`, plus
    `.save()` so ScreenshotResult's persistence path works untouched.
    """

    def __init__(self, width: int, height: int) -> None:
        self.size = (width, height)
        self.width = width
        self.height = height

    def save(self, path: Any) -> None:
        from pathlib import Path

        from PIL import Image

        Image.new("RGB", self.size, color=(0, 0, 0)).save(Path(path))

    def close(self) -> None:
        return None

    def copy(self) -> _FakeImage:
        return _FakeImage(*self.size)

    def load(self) -> None:
        return None


def make_fake_pyautogui(
    *,
    logical_size: tuple[int, int] = (1440, 900),
    physical_size: tuple[int, int] = (1440, 900),
) -> tuple[types.ModuleType, CallRecorder, MouseState]:
    """
    Build a fake `pyautogui` module.

    Parameters
    ----------
    logical_size
        Value returned by `pyautogui.size()` -- the OS logical input
        coordinate space.

    physical_size
        Size of the image returned by `pyautogui.screenshot()`. Set
        this different from `logical_size` to simulate a High-DPI /
        Retina display for coordinate-mapping tests.
    """
    recorder = CallRecorder()
    mouse = MouseState()

    module = types.ModuleType("pyautogui")
    module.FAILSAFE = True  # type: ignore[attr-defined]

    def moveTo(x: int, y: int, duration: float = 0.0, **kwargs: Any) -> None:
        recorder.record("moveTo", x=x, y=y, duration=duration)
        mouse.x, mouse.y = x, y

    def click(
        x: int | None = None,
        y: int | None = None,
        button: str = "left",
        clicks: int = 1,
        interval: float = 0.0,
        **kwargs: Any,
    ) -> None:
        recorder.record(
            "click",
            x=x,
            y=y,
            button=button,
            clicks=clicks,
            interval=interval,
        )
        if x is not None and y is not None:
            mouse.x, mouse.y = x, y

    def dragTo(
        x: int, y: int, duration: float = 0.0, button: str = "left", **kw: Any
    ) -> None:
        recorder.record("dragTo", x=x, y=y, duration=duration, button=button)
        mouse.x, mouse.y = x, y

    def scroll(clicks: int, **kwargs: Any) -> None:
        recorder.record("scroll", clicks=clicks)

    def hscroll(clicks: int, **kwargs: Any) -> None:
        recorder.record("hscroll", clicks=clicks)

    def write(text: str, interval: float = 0.0, **kwargs: Any) -> None:
        recorder.record("write", text=text, interval=interval)

    def press(key: str, presses: int = 1, interval: float = 0.0, **kwargs: Any) -> None:
        recorder.record("press", key=key, presses=presses, interval=interval)

    def hotkey(*keys: str, **kwargs: Any) -> None:
        recorder.record("hotkey", keys=keys)

    def keyDown(key: str) -> None:
        recorder.record("keyDown", key=key)

    def keyUp(key: str) -> None:
        recorder.record("keyUp", key=key)

    def position() -> tuple[int, int]:
        return mouse.x, mouse.y

    def size() -> tuple[int, int]:
        return logical_size

    def screenshot(region: tuple[int, int, int, int] | None = None) -> Any:
        if region is not None:
            _, _, width, height = region
            return _FakeImage(width, height)

        return _FakeImage(*physical_size)

    module.moveTo = moveTo  # type: ignore[attr-defined]
    module.click = click  # type: ignore[attr-defined]
    module.dragTo = dragTo  # type: ignore[attr-defined]
    module.scroll = scroll  # type: ignore[attr-defined]
    module.hscroll = hscroll  # type: ignore[attr-defined]
    module.write = write  # type: ignore[attr-defined]
    module.press = press  # type: ignore[attr-defined]
    module.hotkey = hotkey  # type: ignore[attr-defined]
    module.keyDown = keyDown  # type: ignore[attr-defined]
    module.keyUp = keyUp  # type: ignore[attr-defined]
    module.position = position  # type: ignore[attr-defined]
    module.size = size  # type: ignore[attr-defined]
    module.screenshot = screenshot  # type: ignore[attr-defined]

    return module, recorder, mouse


def make_fake_pywinctl() -> types.ModuleType:
    """
    Build a minimal fake `pywinctl` module sufficient for
    `agent.tools.windows` and the screenshot manager's
    window-capture helpers.
    """
    module = types.ModuleType("pywinctl")

    class Window:
        def __init__(
            self,
            title: str,
            *,
            left: int = 0,
            top: int = 0,
            width: int = 800,
            height: int = 600,
        ) -> None:
            self.title = title
            self.left = left
            self.top = top
            self.width = width
            self.height = height
            self.isActive = True
            self.isMinimized = False
            self.isMaximized = False
            self.isVisible = True

        def activate(self) -> None:
            self.isActive = True

        def restore(self) -> None:
            self.isMinimized = False
            self.isMaximized = False

        def minimize(self) -> None:
            self.isMinimized = True

        def maximize(self) -> None:
            self.isMaximized = True

        def close(self) -> None:
            self.isVisible = False

    module.Window = Window  # type: ignore[attr-defined]
    module._windows: list[Window] = []  # type: ignore[attr-defined]

    def getAllWindows() -> list[Window]:
        return list(module._windows)

    def getWindowsWithTitle(title: str) -> list[Window]:
        return [w for w in module._windows if title in w.title]

    def getActiveWindow() -> Window | None:
        for w in module._windows:
            if w.isActive:
                return w
        return None

    module.getAllWindows = getAllWindows  # type: ignore[attr-defined]
    module.getWindowsWithTitle = getWindowsWithTitle  # type: ignore[attr-defined]
    module.getActiveWindow = getActiveWindow  # type: ignore[attr-defined]

    return module


__all__ = [
    "CallRecorder",
    "MouseState",
    "make_fake_pyautogui",
    "make_fake_pywinctl",
]
