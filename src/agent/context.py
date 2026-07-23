"""
Execution context for a single GUI agent run.
"""

from pydantic import BaseModel, ConfigDict, Field

from grounding.models import (
    GroundingDetection,
    GroundingResponse,
)

from .config import AgentSettings
from .tools.action.actions import ActionManager, ActionResult
from .tools.grounding import GroundingManager
from .tools.screenshot.screenshot import ScreenshotManager, ScreenshotResult


class AgentServices(BaseModel):
    """
    Immutable long-lived services shared across multiple agent executions.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    grounding: GroundingManager
    screenshots: ScreenshotManager
    actions: ActionManager
    config: AgentSettings


class AgentContext(BaseModel):
    """
    Mutable state associated with one agent execution.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    # ------------------------------------------------------------------
    # Goal
    # ------------------------------------------------------------------

    goal: str

    iteration: int = Field(
        default=0,
        ge=0,
    )

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    last_screenshot: ScreenshotResult | None = None

    grounding_response: GroundingResponse | None = None

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    selected_detection: GroundingDetection | None = None

    last_action: ActionResult | None = None

    last_tool: str | None = None

    # ------------------------------------------------------------------
    # Conversation
    # ------------------------------------------------------------------

    messages: list[object] = Field(
        default_factory=list,
    )

    # ------------------------------------------------------------------
    # Shared services
    # ------------------------------------------------------------------

    services: AgentServices

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def has_screenshot(self) -> bool:
        return self.last_screenshot is not None

    @property
    def has_grounding(self) -> bool:
        return self.grounding_response is not None

    @property
    def detections(self) -> tuple[GroundingDetection, ...]:
        if self.grounding_response is None:
            return ()

        return tuple(self.grounding_response.detections)

    def next_iteration(self) -> int:
        """
        Advance the iteration counter.
        """
        self.iteration += 1
        return self.iteration

    def reset_observation(self) -> None:
        """
        Clear the latest observation.
        """
        self.last_screenshot = None
        self.grounding_response = None
        self.selected_detection = None

    def reset_execution(self) -> None:
        self.reset_observation()
        self.last_action = None
        self.last_tool = None

    def __str__(self) -> str:
        screenshot = (
            self.last_screenshot.path
            if self.last_screenshot is not None
            else None
        )

        return (
            f"Goal: {self.goal}\n"
            f"Iteration: {self.iteration}\n"
            f"Last tool: {self.last_tool}\n"
            f"Last action: {self.last_action}\n"
            f"Screenshot: {screenshot}\n"
            f"Detection: {self.selected_detection}\n"
            f"Grounding: {self.grounding_response}"
        )


__all__ = ["AgentContext", "AgentServices"]
