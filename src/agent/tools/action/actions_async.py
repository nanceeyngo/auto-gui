"""
Asynchronous wrappers around GUI automation actions.
"""

import asyncio

from grounding.models import (
    GroundingDetection,
)

from .action_models import (
    ActionResult,
    ActionTarget,
    KeyName,
    MouseButton,
    MousePosition,
)
from .actions import _default_action_manager


async def amove_mouse(
    target: ActionTarget,
    *,
    duration: float = 0.0,
) -> ActionResult:
    return await asyncio.to_thread(
        _default_action_manager.move_mouse,
        target,
        duration=duration,
    )


async def aclick(
    target: ActionTarget,
    *,
    button: MouseButton = "left",
    clicks: int = 1,
    interval: float = 0.0,
    duration: float = 0.0,
) -> ActionResult:
    return await asyncio.to_thread(
        _default_action_manager.click,
        target,
        button=button,
        clicks=clicks,
        interval=interval,
        duration=duration,
    )


async def adouble_click(
    target: ActionTarget,
    *,
    button: MouseButton = "left",
    interval: float = 0.1,
    duration: float = 0.0,
) -> ActionResult:
    return await asyncio.to_thread(
        _default_action_manager.double_click,
        target,
        button=button,
        interval=interval,
        duration=duration,
    )


async def aright_click(
    target: ActionTarget,
    *,
    duration: float = 0.0,
) -> ActionResult:
    return await asyncio.to_thread(
        _default_action_manager.right_click,
        target,
        duration=duration,
    )


async def amiddle_click(
    target: ActionTarget,
    *,
    duration: float = 0.0,
) -> ActionResult:
    return await asyncio.to_thread(
        _default_action_manager.middle_click,
        target,
        duration=duration,
    )


async def adrag(
    *,
    start: ActionTarget | None,
    end: ActionTarget,
    button: MouseButton = "left",
    duration: float = 0.5,
) -> ActionResult:
    return await asyncio.to_thread(
        _default_action_manager.drag,
        start=start,
        end=end,
        button=button,
        duration=duration,
    )


async def adrag_to(
    target: ActionTarget,
    *,
    button: MouseButton = "left",
    duration: float = 0.5,
) -> ActionResult:
    return await asyncio.to_thread(
        _default_action_manager.drag_to,
        target,
        button=button,
        duration=duration,
    )


async def adrag_detection_to_detection(
    source: GroundingDetection,
    destination: GroundingDetection,
    *,
    button: MouseButton = "left",
    duration: float = 0.5,
) -> ActionResult:
    return await asyncio.to_thread(
        _default_action_manager.drag_detection_to_detection,
        source=source,
        target=destination,
        button=button,
        duration=duration,
    )


async def ascroll(
    clicks: int,
    *,
    target: ActionTarget | None = None,
) -> ActionResult:
    return await asyncio.to_thread(
        _default_action_manager.scroll,
        clicks,
        target=target,
    )


async def ahscroll(
    clicks: int,
    *,
    target: ActionTarget | None = None,
) -> ActionResult:
    return await asyncio.to_thread(
        _default_action_manager.hscroll,
        clicks,
        target=target,
    )


async def atype_text(
    text: str,
    *,
    interval: float = 0.0,
) -> ActionResult:
    return await asyncio.to_thread(
        _default_action_manager.type_text,
        text,
        interval=interval,
    )


async def aclick_and_type(
    target: ActionTarget,
    text: str,
    *,
    click_duration: float = 0.0,
    typing_interval: float = 0.0,
) -> ActionResult:
    return await asyncio.to_thread(
        _default_action_manager.click_and_type,
        target,
        text,
        click_duration=click_duration,
        typing_interval=typing_interval,
    )


async def apress_key(
    key: KeyName,
    *,
    presses: int = 1,
    interval: float = 0.0,
) -> ActionResult:
    return await asyncio.to_thread(
        _default_action_manager.press_key,
        key,
        presses=presses,
        interval=interval,
    )


async def ahotkey(
    *keys: KeyName,
) -> ActionResult:
    return await asyncio.to_thread(
        _default_action_manager.hotkey,
        *keys,
    )


async def adelay(
    seconds: float,
) -> ActionResult:
    return await asyncio.to_thread(
        _default_action_manager.delay,
        seconds,
    )


async def amouse_position() -> MousePosition:
    return await asyncio.to_thread(
        _default_action_manager.mouse_position,
    )


async def ascreen_size() -> tuple[int, int]:
    return await asyncio.to_thread(
        _default_action_manager.screen_size,
    )


async def aclose() -> None:
    await asyncio.to_thread(
        _default_action_manager.close,
    )


__all__ = [
    "aclick",
    "aclick_and_type",
    "aclose",
    "adelay",
    "adouble_click",
    "adrag",
    "adrag_detection_to_detection",
    "adrag_to",
    "ahotkey",
    "ahscroll",
    "amiddle_click",
    "amouse_position",
    "amove_mouse",
    "apress_key",
    "aright_click",
    "ascreen_size",
    "ascroll",
    "atype_text",
]
