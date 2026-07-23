"""
Adapter Pydantic models for the LangChain tool wrappers.
"""

from pathlib import Path

from pydantic import BaseModel, Field

from grounding.models import GroundingDetection

from .action.action_models import KeyName, MouseButton


class ClickToolInput(BaseModel):
    """
    Input for mouse click tools.
    """

    detection: GroundingDetection

    button: MouseButton = "left"

    clicks: int = Field(
        default=1,
        ge=1,
    )


class DragToolInput(BaseModel):
    """
    Input for the drag tool.
    """

    source: GroundingDetection

    destination: GroundingDetection

    button: MouseButton = "left"


class TypeTextToolInput(BaseModel):
    """
    Input for the type_text tool.
    """

    text: str


class PressKeyToolInput(BaseModel):
    """
    Input for the press_key tool.
    """

    key: KeyName

    presses: int = Field(
        default=1,
        ge=1,
    )


class HotkeyToolInput(BaseModel):
    """
    Input for the hotkey tool.
    """

    keys: list[KeyName] = Field(
        min_length=2,
    )


class LocateToolInput(BaseModel):
    """
    Input for the locate tool.
    """

    image: str = Field(
        ...,
        description="The path to the image in which to locate a UI element.",
    )
    query: str = Field(
        ...,
        description=(
            "Natural language description of the UI element to locate."
        ),
    )
    provider: str | None = Field(
        ...,
        description=(
            "Optional; name of grounding provider for UI element location."
        ),
    )


class WaitToolInput(BaseModel):
    """
    Input for the wait tool.
    """

    seconds: float


class ScreenshotToolResult(BaseModel):
    path: Path
    width: int
    height: int
