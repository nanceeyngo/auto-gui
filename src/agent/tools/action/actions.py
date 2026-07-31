"""
GUI automation action helpers.

This module provides high-level mouse and keyboard actions used by the
GUI automation agent.
"""

import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any

import pyautogui

from grounding.models import (
    BoundingBox,
    GroundingDetection,
)

from ...config import settings
from ...logging_config import get_logger
from ..coordinates import CoordinateMapper
from .action_models import (
    Action,
    ActionResult,
    ActionTarget,
    KeyName,
    MouseButton,
    MousePosition,
)

logger = get_logger("agent.tools.action")

# ---------------------------------------------------------------------------
# Manager Class
# ---------------------------------------------------------------------------


class ActionManager:
    """
    Performs low-level GUI automation using PyAutoGUI.
    """

    __slots__ = (
        "_coordinate_mapper",
        "_failsafe",
        "_post_action_delay",
    )

    def __init__(
        self,
        post_action_delay: float = 0.5,
        failsafe: bool = True,
        *,
        coordinate_mapper: CoordinateMapper | None = None,
    ) -> None:
        self._post_action_delay = post_action_delay
        self._failsafe = failsafe

        pyautogui.FAILSAFE = failsafe

        self._coordinate_mapper = coordinate_mapper or CoordinateMapper(
            physical_size_fn=lambda: pyautogui.screenshot().size,
            logical_size_fn=pyautogui.size,
        )

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _center_of_bbox(bounding_box: BoundingBox) -> MousePosition:
        """
        Return the center point of a bounding box.
        """
        return MousePosition(
            x=(bounding_box.x1 + bounding_box.x2) // 2,
            y=(bounding_box.y1 + bounding_box.y2) // 2,
        )

    def _resolve_position(
        self,
        target: ActionTarget,
    ) -> MousePosition:
        """
        Resolve an action target into an absolute *logical* screen
        position suitable for OS input APIs.

        All incoming coordinates (raw tuples, bounding boxes, and
        grounding detections) are treated as screenshot-space pixel
        coordinates, since that is the coordinate system every
        grounding provider operates in. They are converted to logical
        OS coordinates via the coordinate mapper to remain correct on
        High-DPI/Retina displays where screenshot pixel resolution
        differs from the OS logical coordinate space.
        """
        if isinstance(target, tuple):
            if len(target) != 2:
                raise ValueError("Coordinate tuple must contain exactly two integers.")

            raw_x, raw_y = target
            x, y = self._coordinate_mapper.to_logical(int(raw_x), int(raw_y))

            return MousePosition(
                x=x,
                y=y,
            )

        if isinstance(target, BoundingBox):
            bbox_position = self._center_of_bbox(target)
        else:
            try:
                bbox_position = self._center_of_bbox(target.bbox)
            except (AttributeError, ValueError) as exc:
                raise TypeError(
                    f"Unsupported action target type: {type(target)!r}"
                ) from exc

        logical_x, logical_y = self._coordinate_mapper.to_logical(
            bbox_position.x, bbox_position.y
        )

        return MousePosition(x=logical_x, y=logical_y)

    def _move_to(
        self,
        target: ActionTarget,
        *,
        duration: float = 0.0,
    ) -> MousePosition:
        """
        Move the mouse to a target.
        """
        position = self._resolve_position(target)

        pyautogui.moveTo(
            x=position.x,
            y=position.y,
            duration=duration,
        )

        return position

    def _make_result(
        self,
        *,
        action: Action,
        position: MousePosition | None = None,
        message: str | None = None,
        started: float | None = None,
    ) -> ActionResult:
        """
        Construct a successful ActionResult.

        When ``started`` (a ``time.perf_counter()`` timestamp taken at
        the start of the action) is provided, the elapsed execution
        time is recorded on the result and emitted as a structured log
        event, giving visibility into per-action latency.
        """
        latency_ms = (
            (time.perf_counter() - started) * 1000.0 if started is not None else None
        )

        result = ActionResult(
            success=True,
            action=action,
            position=position,
            message=message,
            latency_ms=latency_ms,
        )

        logger.info(
            "Executed GUI action",
            extra={
                "context": {
                    "action": action,
                    "position": (
                        (position.x, position.y) if position is not None else None
                    ),
                    "latency_ms": (
                        round(latency_ms, 2) if latency_ms is not None else None
                    ),
                }
            },
        )

        return result

    def _wait_for_gui(self) -> None:
        """
        Allow the GUI to settle after an action.
        """
        if self._post_action_delay > 0:
            time.sleep(self._post_action_delay)

    # -------------------------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------------------------

    @staticmethod
    def mouse_position() -> MousePosition:
        """
        Return the current mouse cursor position.
        """
        x, y = pyautogui.position()

        return MousePosition(
            x=x,
            y=y,
        )

    @staticmethod
    def screen_size() -> tuple[int, int]:
        """
        Return the current screen resolution.
        """
        width, height = pyautogui.size()
        return width, height

    def close(self) -> None:
        """
        Release any resources owned by this manager.
        """
        return

    # -------------------------------------------------------------------------
    # Mouse actions
    # -------------------------------------------------------------------------

    def move_mouse(
        self,
        target: ActionTarget,
        *,
        duration: float = 0.0,
    ) -> ActionResult:
        """
        Move the mouse to a target.
        """
        started = time.perf_counter()
        start = self.mouse_position()
        position = self._move_to(
            target,
            duration=duration,
        )

        return self._make_result(
            action="move_mouse",
            position=position,
            message=(
                f"Moved cursor from ({start.x}, {start.y}) "
                f"to ({position.x}, {position.y})."
            ),
            started=started,
        )

    def click(
        self,
        target: ActionTarget,
        *,
        button: MouseButton = "left",
        clicks: int = 1,
        interval: float = 0.0,
        duration: float = 0.0,
    ) -> ActionResult:
        """
        Click a target.
        """
        started = time.perf_counter()
        position = self._move_to(
            target,
            duration=duration,
        )

        pyautogui.click(
            x=position.x,
            y=position.y,
            button=button,
            clicks=clicks,
            interval=interval,
        )

        self._wait_for_gui()

        return self._make_result(
            action="click",
            position=position,
            message=(
                f"{button[0].upper()}{button[1:]}-clicked "
                f"{str(clicks) + ' times' if clicks > 1 else 'once'} "
                f"at ({position.x}, {position.y})."
            ),
            started=started,
        )

    def double_click(
        self,
        target: ActionTarget,
        *,
        button: MouseButton = "left",
        interval: float = 0.1,
        duration: float = 0.0,
    ) -> ActionResult:
        """
        Double-click a target.
        """
        return self.click(
            target,
            button=button,
            clicks=2,
            interval=interval,
            duration=duration,
        )

    def right_click(
        self,
        target: ActionTarget,
        *,
        duration: float = 0.0,
    ) -> ActionResult:
        """
        Right-click a target.
        """
        return self.click(
            target,
            button="right",
            duration=duration,
        )

    def middle_click(
        self,
        target: ActionTarget,
        *,
        duration: float = 0.0,
    ) -> ActionResult:
        """
        Middle-click a target.
        """
        return self.click(
            target,
            button="middle",
            duration=duration,
        )

    def drag(
        self,
        *,
        start: ActionTarget | None = None,
        end: ActionTarget,
        button: MouseButton = "left",
        duration: float = 0.5,
    ) -> ActionResult:
        """
        Drag from one target to another.
        """
        started = time.perf_counter()

        if start is not None:
            start_position = self._move_to(start)
        else:
            start_position = self.mouse_position()

        end_position = self._resolve_position(end)

        pyautogui.dragTo(
            x=end_position.x,
            y=end_position.y,
            duration=duration,
            button=button,
        )

        self._wait_for_gui()

        return self._make_result(
            action="drag",
            position=end_position,
            message=(
                f"Dragged from "
                f"({start_position.x}, {start_position.y}) "
                f"to "
                f"({end_position.x}, {end_position.y})."
            ),
            started=started,
        )

    def drag_to(
        self,
        target: ActionTarget,
        *,
        button: MouseButton = "left",
        duration: float = 0.5,
    ) -> ActionResult:
        """
        Drag from the current mouse position to a target.
        """
        return self.drag(
            end=target,
            button=button,
            duration=duration,
        )

    def drag_detection_to_detection(
        self,
        *,
        source: GroundingDetection,
        target: GroundingDetection,
        button: MouseButton = "left",
        duration: float = 0.5,
    ) -> ActionResult:
        """
        Drag one detected UI element onto another.
        """
        return self.drag(
            start=source,
            end=target,
            button=button,
            duration=duration,
        )

    def scroll(
        self,
        clicks: int,
        *,
        target: ActionTarget | None = None,
    ) -> ActionResult:
        """
        Scroll vertically.
        """
        position: MousePosition | None = None

        if target is not None:
            position = self._move_to(target)

        pyautogui.scroll(clicks)

        self._wait_for_gui()

        if position is None:
            message = f"Scrolled vertically by {clicks} clicks."
        else:
            message = (
                f"Scrolled vertically at "
                f"({position.x}, {position.y}) "
                f"by {clicks} clicks."
            )

        return self._make_result(action="scroll", position=position, message=message)

    def hscroll(
        self,
        clicks: int,
        *,
        target: ActionTarget | None = None,
    ) -> ActionResult:
        """
        Scroll horizontally.
        """
        position: MousePosition | None = None

        if target is not None:
            position = self._move_to(target)

        pyautogui.hscroll(clicks)

        self._wait_for_gui()

        if position is None:
            message = f"Scrolled horizontally by {clicks} clicks."
        else:
            message = (
                f"Scrolled horizontally at "
                f"({position.x}, {position.y}) "
                f"by {clicks} clicks."
            )

        return self._make_result(action="scroll", position=position, message=message)

    # -------------------------------------------------------------------------
    # Keyboard actions
    # -------------------------------------------------------------------------

    def type_text(
        self,
        text: str,
        *,
        interval: float = 0.0,
    ) -> ActionResult:
        """
        Type text.
        """
        pyautogui.write(
            text,
            interval=interval,
        )

        self._wait_for_gui()

        return self._make_result(
            action="type_text",
            message=text,
        )

    def press_key(
        self,
        key: KeyName,
        *,
        presses: int = 1,
        interval: float = 0.0,
    ) -> ActionResult:
        """
        Press a keyboard key.
        """
        pyautogui.press(
            key,
            presses=presses,
            interval=interval,
        )

        self._wait_for_gui()

        return self._make_result(
            action="press_key",
            message=key,
        )

    def hotkey(
        self,
        *keys: KeyName,
    ) -> ActionResult:
        """
        Execute a keyboard shortcut.
        """
        pyautogui.hotkey(*keys)

        self._wait_for_gui()

        return self._make_result(
            action="hotkey",
            message=" + ".join(keys),
        )

    @staticmethod
    @contextmanager
    def hold_key(
        key: KeyName,
    ) -> Generator[None, Any, None]:
        """
        Hold a key for the duration of a context.
        """
        pyautogui.keyDown(key)

        try:
            yield
        finally:
            pyautogui.keyUp(key)

    # -------------------------------------------------------------------------
    # Other actions
    # -------------------------------------------------------------------------

    def delay(
        self,
        seconds: float,
    ) -> ActionResult:
        """
        Pause execution.
        """
        if seconds < 0:
            raise ValueError("seconds must be non-negative.")

        time.sleep(seconds)

        return self._make_result(
            action="wait",
            message=f"{seconds:.3f}s",
        )

    def click_and_type(
        self,
        target: ActionTarget,
        text: str,
        *,
        click_duration: float = 0.0,
        typing_interval: float = 0.0,
    ) -> ActionResult:
        """
        Click a target and immediately type text.
        """
        _ = self.click(
            target,
            duration=click_duration,
        )

        _ = self.type_text(
            text,
            interval=typing_interval,
        )

        target_position = self._resolve_position(target)

        return self._make_result(
            action="type_text",
            message=(
                f"Clicked target at ({target_position.x}, {target_position.y})"
                f" and typed {text!r}."
            ),
        )

    def with_modifier(
        self,
        key: KeyName,
        action: Callable[[], ActionResult],
    ) -> ActionResult:
        with self.hold_key(key):
            return action()


_default_action_manager = ActionManager(
    post_action_delay=settings.post_action_delay,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def move_mouse(
    target: ActionTarget,
    *,
    duration: float = 0.0,
) -> ActionResult:
    return _default_action_manager.move_mouse(
        target=target,
        duration=duration,
    )


def click(
    target: ActionTarget,
    *,
    button: MouseButton = "left",
    clicks: int = 1,
    interval: float = 0.0,
    duration: float = 0.0,
) -> ActionResult:
    return _default_action_manager.click(
        target,
        button=button,
        clicks=clicks,
        interval=interval,
        duration=duration,
    )


def double_click(
    target: ActionTarget,
    *,
    button: MouseButton = "left",
    interval: float = 0.1,
    duration: float = 0.0,
) -> ActionResult:
    return _default_action_manager.double_click(
        target=target,
        button=button,
        interval=interval,
        duration=duration,
    )


def right_click(
    target: ActionTarget,
    *,
    duration: float = 0.0,
) -> ActionResult:
    return _default_action_manager.right_click(
        target=target,
        duration=duration,
    )


def middle_click(
    target: ActionTarget,
    *,
    duration: float = 0.0,
) -> ActionResult:
    return _default_action_manager.middle_click(
        target=target,
        duration=duration,
    )


def drag(
    *,
    start: ActionTarget | None = None,
    end: ActionTarget,
    button: MouseButton = "left",
    duration: float = 0.5,
) -> ActionResult:
    return _default_action_manager.drag(
        start=start,
        end=end,
        button=button,
        duration=duration,
    )


def drag_to(
    target: ActionTarget,
    *,
    button: MouseButton = "left",
    duration: float = 0.5,
) -> ActionResult:
    return _default_action_manager.drag_to(
        target=target,
        button=button,
        duration=duration,
    )


def drag_detection_to_detection(
    *,
    source: GroundingDetection,
    target: GroundingDetection,
    button: MouseButton = "left",
    duration: float = 0.5,
) -> ActionResult:
    return _default_action_manager.drag_detection_to_detection(
        source=source,
        target=target,
        button=button,
        duration=duration,
    )


def scroll(
    clicks: int,
    *,
    target: ActionTarget | None = None,
) -> ActionResult:
    return _default_action_manager.scroll(
        clicks=clicks,
        target=target,
    )


def hscroll(
    clicks: int,
    *,
    target: ActionTarget | None = None,
) -> ActionResult:
    return _default_action_manager.hscroll(
        clicks=clicks,
        target=target,
    )


def type_text(
    text: str,
    *,
    interval: float = 0.0,
) -> ActionResult:
    return _default_action_manager.type_text(
        text=text,
        interval=interval,
    )


def press_key(
    key: KeyName,
    *,
    presses: int = 1,
    interval: float = 0.0,
) -> ActionResult:
    return _default_action_manager.press_key(
        key=key,
        presses=presses,
        interval=interval,
    )


def hotkey(
    *keys: KeyName,
) -> ActionResult:
    return _default_action_manager.hotkey(
        *keys,
    )


def click_and_type(
    target: ActionTarget,
    text: str,
    *,
    click_duration: float = 0.0,
    typing_interval: float = 0.0,
) -> ActionResult:
    return _default_action_manager.click_and_type(
        target=target,
        text=text,
        click_duration=click_duration,
        typing_interval=typing_interval,
    )


def delay(
    seconds: float,
) -> ActionResult:
    return _default_action_manager.delay(
        seconds=seconds,
    )


def current_mouse_position() -> MousePosition:
    return _default_action_manager.mouse_position()


def screen_size() -> tuple[int, int]:
    return _default_action_manager.screen_size()


def close() -> None:
    _default_action_manager.close()


__all__ = [
    "ActionManager",
    "ActionResult",
    "MousePosition",
    "_default_action_manager",
    "click",
    "click_and_type",
    "close",
    "current_mouse_position",
    "delay",
    "double_click",
    "drag",
    "drag_detection_to_detection",
    "drag_to",
    "hotkey",
    "hscroll",
    "middle_click",
    "move_mouse",
    "press_key",
    "right_click",
    "screen_size",
    "scroll",
    "type_text",
]
