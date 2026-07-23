"""
Shared data models for the grounding framework.

These models define the common interface between the GroundingClient
and all grounding providers (VLM Run, OmniParser, GroundingDINO, etc.).
"""

from enum import StrEnum
from pathlib import Path
from typing import Any

from PIL.Image import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

type ImageLike = str | Path | Image


# ---------------------------------------------------------------------------
# Detection status
# ---------------------------------------------------------------------------


class DetectionStatus(StrEnum):
    """
    Outcome of a grounding request.
    """

    SUCCESS = "success"
    NO_MATCH = "no_match"
    AMBIGUOUS = "ambiguous"
    ERROR = "error"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Bounding box
# ---------------------------------------------------------------------------


class BoundingBox(BaseModel):
    """
    Bounding box in image pixel coordinates.

        (x1, y1)
            ┌────────────────────────┐
            │                        │
            │                        │
            │                        │
            └────────────────────────┘
                            (x2, y2)
    """

    model_config = ConfigDict(frozen=True)

    x1: int = Field(..., ge=0, description="Left coordinate.")
    y1: int = Field(..., ge=0, description="Top coordinate.")
    x2: int = Field(..., ge=0, description="Right coordinate.")
    y2: int = Field(..., ge=0, description="Bottom coordinate.")

    @model_validator(mode="after")
    def validate_coordinates(self) -> "BoundingBox":
        if self.x2 <= self.x1:
            raise ValueError("x2 must be greater than x1.")

        if self.y2 <= self.y1:
            raise ValueError("y2 must be greater than y1.")

        return self

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[int, int]:
        return (
            (self.x1 + self.x2) // 2,
            (self.y1 + self.y2) // 2,
        )

    def as_tuple(self) -> tuple[int, int, int, int]:
        """
        Returns (x1, y1, x2, y2)
        """
        return self.x1, self.y1, self.x2, self.y2

    def normalized(
        self,
        image_width: int,
        image_height: int,
    ) -> tuple[float, float, float, float]:
        """
        Returns (x1, y1, x2, y2) coordinates normalized to the screen/image
        dimensions, in the range [0, 1].
        """

        return (
            self.x1 / image_width,
            self.y1 / image_height,
            self.x2 / image_width,
            self.y2 / image_height,
        )

    @classmethod
    def from_center(
        cls,
        center_x: int,
        center_y: int,
        width: int,
        height: int,
    ) -> "BoundingBox":
        """
        Factory method for creating a BoundingBox from center coordinates,
        width and height.
        """
        half_width = width // 2
        half_height = height // 2

        return cls(
            x1=center_x - half_width,
            y1=center_y - half_height,
            x2=center_x + half_width,
            y2=center_y + half_height,
        )


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class GroundingRequest(BaseModel):
    """
    Input passed to a grounding provider.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: ImageLike

    query: str = Field(
        ...,
        min_length=1,
        description="Natural-language description of the target UI element.",
    )

    top_k: int = Field(
        default=1,
        ge=1,
        description="Maximum number of detections requested.",
    )

    confidence_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum acceptable confidence level of returned candidate "
            "detections."
        ),
    )

    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Individual detection
# ---------------------------------------------------------------------------


class GroundingDetection(BaseModel):
    """
    Represents one candidate detection.
    """

    model_config = ConfigDict(frozen=True)

    bbox: BoundingBox

    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    label: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def center(self) -> tuple[int, int]:
        return self.bbox.center

    @property
    def width(self) -> int:
        return self.bbox.width

    @property
    def height(self) -> int:
        return self.bbox.height

    @property
    def area(self) -> int:
        return self.bbox.area

    @property
    def has_label(self) -> bool:
        return self.label is not None

    @property
    def has_confidence(self) -> bool:
        return self.confidence is not None


# ---------------------------------------------------------------------------
# Provider response
# ---------------------------------------------------------------------------


class GroundingResponse(BaseModel):
    """
    Response returned by every grounding provider.

    Even when only a single detection exists, providers should return
    a GroundingResponse containing one GroundingDetection.
    """

    model_config = ConfigDict(frozen=True)

    request_query: str = Field(
        ...,
        min_length=1,
        description=(
            "Natural-language description of the target UI element, from the "
            "originating GroundingRequest."
        ),
    )
    provider: str = Field(
        ...,
        description="Grounding backend that produced this result.",
    )

    provider_version: str = Field(
        ...,
        description="Version of grounding backend that produced this result.",
    )

    status: DetectionStatus = DetectionStatus.UNKNOWN

    detections: list[GroundingDetection] = Field(default_factory=list)

    latency_ms: float | None = Field(
        default=None,
        ge=0.0,
    )

    image_width: int | None = Field(
        default=None,
        ge=1,
        description="Width of the image/screenshot in pixels.",
    )

    image_height: int | None = Field(
        default=None,
        ge=1,
        description="Height of the image/screenshot in pixels.",
    )

    provider_request_id: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def best_detection(self) -> GroundingDetection | None:
        """
        Returns the highest-confidence detection.
        """

        if not self.detections:
            return None

        if all(d.confidence is None for d in self.detections):
            return self.detections[0]

        return max(
            self.detections,
            key=lambda d: d.confidence or 0.0,
        )

    @property
    def best_bbox(self) -> BoundingBox | None:
        detection = self.best_detection
        return detection.bbox if detection else None

    @property
    def best_center(self) -> tuple[int, int] | None:
        detection = self.best_detection
        return detection.center if detection else None

    @property
    def success(self) -> bool:
        return (
            self.status == DetectionStatus.SUCCESS and len(self.detections) > 0
        )

    def sorted_by_confidence(
        self,
        descending: bool = True,
    ) -> list[GroundingDetection]:
        """
        Returns detections sorted by confidence.
        """

        return sorted(
            self.detections,
            key=lambda d: d.confidence or 0.0,
            reverse=descending,
        )


# ---------------------------------------------------------------------------
# Structured failure
# ---------------------------------------------------------------------------


class GroundingFailure(BaseModel):
    """
    Optional structured error model.

    Mainly useful for benchmarking multiple providers.
    """

    provider: str

    message: str

    exception_type: str | None = None

    latency_ms: float | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)
