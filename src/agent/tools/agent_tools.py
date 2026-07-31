"""
LangChain tool wrappers for the GUI automation agent.

These tools expose a simplified, JSON-serializable interface over the
automation library.
"""

import math
from collections.abc import Sequence

from langchain_core.tools import BaseTool, tool
from langgraph.runtime import Runtime

from grounding.exceptions import GroundingProviderError
from grounding.models import GroundingResponse

from ..context import AgentContext
from ..logging_config import get_logger
from .action.action_models import ActionResult
from .tool_models import (
    ClickTargetToolInput,
    ClickToolInput,
    DragToolInput,
    HotkeyToolInput,
    LocateToolInput,
    PressKeyToolInput,
    ScreenshotToolResult,
    TypeTextToolInput,
    WaitToolInput,
)

logger = get_logger("agent.tools.agent_tools")

# Two detections are treated as "the same still-present element" (and
# therefore evidence that a click did not register) when their centers
# are within this many pixels of one another.
_SAME_ELEMENT_PIXEL_TOLERANCE = 12.0

# ---------------------------------------------------------------------------
# Screenshots
# ---------------------------------------------------------------------------


@tool(
    description=(
        "Capture the current screen and return the path to the screenshot "
        "with the screenshot's width and height."
    ),
)
def capture_screen(runtime: Runtime[AgentContext]) -> ScreenshotToolResult:
    """
    Capture the current desktop.
    """
    context = runtime.context
    result = context.services.screenshots.capture()
    context.last_screenshot = result

    assert result.path is not None
    assert result.width is not None
    assert result.height is not None

    return ScreenshotToolResult(
        path=result.path,
        width=result.width,
        height=result.height,
    )


# ---------------------------------------------------------------------------
# Grounding
# ---------------------------------------------------------------------------


@tool(
    args_schema=LocateToolInput,
    description="Locate a GUI element inside a screenshot.",
)
def locate(
    locate_input: LocateToolInput,
    runtime: Runtime[AgentContext],
) -> GroundingResponse:
    """
    Locate a GUI element inside a screenshot.
    """
    context = runtime.context
    result = context.services.grounding.locate_image(
        image=locate_input.image,
        query=locate_input.query,
        provider=locate_input.provider,
    )
    context.grounding_response = result

    return result


# ---------------------------------------------------------------------------
# Mouse
# ---------------------------------------------------------------------------


@tool(
    args_schema=ClickToolInput,
    description="Click the center of a detected GUI element.",
)
def click(
    click_input: ClickToolInput,
    runtime: Runtime[AgentContext],
) -> ActionResult:
    """
    Click the center of a detected GUI element.
    """
    context = runtime.context
    result: ActionResult = context.services.actions.click(
        target=click_input.detection.center,
        button=click_input.button,
        clicks=click_input.clicks,
    )
    context.last_action = result

    return result


@tool(
    args_schema=ClickToolInput,
    description="Double-click a detected GUI element.",
)
def double_click(
    click_input: ClickToolInput,
    runtime: Runtime[AgentContext],
) -> ActionResult:
    """
    Double-click the center of a detected GUI element.
    """
    context = runtime.context
    result: ActionResult = context.services.actions.double_click(
        target=click_input.detection.center,
        button=click_input.button,
    )
    context.last_action = result

    return result


@tool(
    args_schema=ClickToolInput,
    description="Right-click a detected GUI element.",
)
def right_click(
    click_input: ClickToolInput,
    runtime: Runtime[AgentContext],
) -> ActionResult:
    """
    Right-click the center of a detected GUI element.
    """
    context = runtime.context
    result: ActionResult = context.services.actions.right_click(
        target=click_input.detection.center
    )
    context.last_action = result

    return result


def _detections_at_same_spot(
    a: tuple[int, int],
    b: tuple[int, int],
    *,
    tolerance: float = _SAME_ELEMENT_PIXEL_TOLERANCE,
) -> bool:
    return math.dist(a, b) <= tolerance


@tool(
    args_schema=ClickTargetToolInput,
    description=(
        "Locate a UI element by natural-language description and click "
        "it. Automatically re-captures the screen and retries (up to "
        "max_attempts times) if the element cannot be located, or if "
        "the same element is still visible in the same place "
        "immediately after clicking (a strong signal the click did not "
        "register). Prefer this over calling `locate` followed by "
        "`click` separately, since it is self-correcting."
    ),
)
def click_target(
    click_target_input: ClickTargetToolInput,
    runtime: Runtime[AgentContext],
) -> ActionResult:
    """
    Locate-and-click a UI element with automatic retry and dynamic
    re-capture of the screen on apparent click failure.
    """
    context = runtime.context
    services = context.services

    last_result: ActionResult | None = None

    for attempt in range(1, click_target_input.max_attempts + 1):
        screenshot = services.screenshots.capture()
        context.last_screenshot = screenshot

        try:
            response = services.grounding.locate(
                screenshot=screenshot,
                query=click_target_input.query,
                provider=click_target_input.provider,
            )
        except GroundingProviderError as exc:
            logger.warning(
                "click_target: grounding failed, retrying",
                extra={
                    "context": {
                        "query": click_target_input.query,
                        "attempt": attempt,
                        "exception_type": type(exc).__name__,
                    }
                },
            )
            continue

        context.grounding_response = response

        if not response.success or response.best_detection is None:
            logger.info(
                "click_target: no detection, retrying",
                extra={
                    "context": {
                        "query": click_target_input.query,
                        "attempt": attempt,
                        "status": response.status,
                    }
                },
            )
            continue

        detection = response.best_detection
        context.selected_detection = detection

        result = services.actions.click(
            target=detection.center,
            button=click_target_input.button,
        )
        context.last_action = result
        last_result = result

        # Dynamically re-capture and re-locate to verify the click
        # actually had an effect on the UI.
        verification_screenshot = services.screenshots.capture()

        try:
            verification = services.grounding.locate(
                screenshot=verification_screenshot,
                query=click_target_input.query,
                provider=click_target_input.provider,
            )
        except GroundingProviderError:
            # Cannot verify; assume the click succeeded rather than
            # looping forever on a grounding outage.
            return result

        still_present = verification.best_detection

        click_likely_failed = (
            verification.success
            and still_present is not None
            and _detections_at_same_spot(detection.center, still_present.center)
        )

        if not click_likely_failed:
            return result

        logger.info(
            "click_target: element still present after click, retrying",
            extra={
                "context": {
                    "query": click_target_input.query,
                    "attempt": attempt,
                }
            },
        )

    if last_result is not None:
        return last_result

    raise GroundingProviderError(
        f"Unable to locate a clickable match for "
        f"{click_target_input.query!r} after "
        f"{click_target_input.max_attempts} attempts."
    )


@tool(
    args_schema=DragToolInput,
    description="Drag from one detected UI element to another.",
)
def drag(
    drag_input: DragToolInput,
    runtime: Runtime[AgentContext],
) -> ActionResult:
    """
    Drag from one detected UI element to another.
    """
    context = runtime.context
    result: ActionResult = context.services.actions.drag(
        start=drag_input.source.center,
        end=drag_input.destination.center,
        button=drag_input.button,
    )
    context.last_action = result

    return result


# ---------------------------------------------------------------------------
# Keyboard
# ---------------------------------------------------------------------------


@tool(
    args_schema=TypeTextToolInput,
    description="Type text into the currently focused input.",
)
def type_text(
    typing_input: TypeTextToolInput,
    runtime: Runtime[AgentContext],
) -> ActionResult:
    """
    Type text using the keyboard into the currently focused input.
    """
    context = runtime.context
    result: ActionResult = context.services.actions.type_text(typing_input.text)
    context.last_action = result

    return result


@tool(
    args_schema=PressKeyToolInput,
    description="Press a given non-text key.",
)
def press_key(
    press_key_input: PressKeyToolInput,
    runtime: Runtime[AgentContext],
) -> ActionResult:
    """
    Press a given non-text key.
    """
    context = runtime.context
    result: ActionResult = context.services.actions.press_key(
        key=press_key_input.key,
        presses=press_key_input.presses,
    )
    context.last_action = result

    return result


@tool(
    args_schema=HotkeyToolInput,
    description="Press a given hotkey combination.",
)
def hotkey(
    hotkey_input: HotkeyToolInput,
    runtime: Runtime[AgentContext],
) -> ActionResult:
    """
    Press a hotkey combination.
    """
    context = runtime.context
    result: ActionResult = context.services.actions.hotkey(*hotkey_input.keys)
    context.last_action = result

    return result


@tool(
    args_schema=WaitToolInput,
    description=("Wait for a given number of seconds before taking next action."),
)
def wait(
    delay: WaitToolInput,
    runtime: Runtime[AgentContext],
) -> ActionResult:
    """
    Wait for a given number of seconds before taking next action.
    """
    context = runtime.context
    result: ActionResult = context.services.actions.delay(delay.seconds)
    context.last_action = result

    return result


GUI_AGENT_TOOLS: Sequence[BaseTool] = (
    capture_screen,
    locate,
    click,
    click_target,
    double_click,
    right_click,
    drag,
    type_text,
    press_key,
    hotkey,
    wait,
)


__all__ = [
    "GUI_AGENT_TOOLS",
    "capture_screen",
    "click",
    "click_target",
    "double_click",
    "drag",
    "hotkey",
    "locate",
    "press_key",
    "right_click",
    "type_text",
    "wait",
]
