"""
LangChain tool wrappers for the GUI automation agent.

These tools expose a simplified, JSON-serializable interface over the
automation library.
"""

from collections.abc import Sequence

from langchain_core.tools import BaseTool, tool
from langgraph.runtime import Runtime

from grounding.models import GroundingResponse

from ..context import AgentContext
from .action.action_models import ActionResult
from .tool_models import (
    ClickToolInput,
    DragToolInput,
    HotkeyToolInput,
    LocateToolInput,
    PressKeyToolInput,
    ScreenshotToolResult,
    TypeTextToolInput,
    WaitToolInput,
)

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
    result: ActionResult = context.services.actions.type_text(
        typing_input.text
    )
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
    description=(
        "Wait for a given number of seconds before taking next action."
    ),
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
    "double_click",
    "drag",
    "hotkey",
    "locate",
    "press_key",
    "right_click",
    "type_text",
    "wait",
]
