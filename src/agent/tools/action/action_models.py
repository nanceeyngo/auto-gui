"""
Literals, type aliases, and result models for `actions.py`.

This module provides high-level mouse and keyboard actions used by the
GUI automation agent.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from grounding.models import (
    BoundingBox,
    GroundingDetection,
)

# ---------------------------------------------------------------------------
# Literals and Type aliases
# ---------------------------------------------------------------------------

Action = Literal[
    "click",
    "drag",
    "hotkey",
    "move_mouse",
    "press_key",
    "type_text",
    "scroll",
    "wait",
]

MouseButton = Literal[
    "left",
    "middle",
    "right",
]

type Coordinate = tuple[int, int]
type KeyName = str
type ActionTarget = Coordinate | BoundingBox | GroundingDetection


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class MousePosition(BaseModel):
    """
    Absolute screen coordinates.
    """

    model_config = ConfigDict(
        frozen=True,
    )

    x: int

    y: int


class ActionResult(BaseModel):
    """
    Result returned by an action.
    """

    model_config = ConfigDict(
        frozen=True,
    )

    success: bool = True

    action: Action

    position: MousePosition | None = None

    message: str | None = None
