"""
Tests for the shared grounding data models.
"""

import pytest

from grounding.models import (
    BoundingBox,
    DetectionStatus,
    GroundingDetection,
    GroundingRequest,
    GroundingResponse,
)

# ---------------------------------------------------------------------------
# BoundingBox
# ---------------------------------------------------------------------------


def test_bounding_box_properties() -> None:
    bbox = BoundingBox(
        x1=10,
        y1=20,
        x2=110,
        y2=220,
    )

    assert bbox.width == 100
    assert bbox.height == 200
    assert bbox.area == 20_000
    assert bbox.center == (60, 120)


def test_bounding_box_as_tuple() -> None:
    bbox = BoundingBox(
        x1=1,
        y1=2,
        x2=3,
        y2=4,
    )

    assert bbox.as_tuple() == (1, 2, 3, 4)


def test_bounding_box_normalized() -> None:
    bbox = BoundingBox(
        x1=20,
        y1=40,
        x2=120,
        y2=240,
    )

    assert bbox.normalized(
        image_width=200,
        image_height=400,
    ) == (
        0.1,
        0.1,
        0.6,
        0.6,
    )


def test_bounding_box_from_center() -> None:
    bbox = BoundingBox.from_center(
        center_x=100,
        center_y=200,
        width=40,
        height=60,
    )

    assert bbox == BoundingBox(
        x1=80,
        y1=170,
        x2=120,
        y2=230,
    )


@pytest.mark.parametrize(
    ("x1", "y1", "x2", "y2"),
    [
        (10, 0, 10, 20),
        (10, 0, 5, 20),
        (0, 10, 20, 10),
        (0, 10, 20, 5),
        (-2, 3, 20, 15),
    ],
)
def test_bounding_box_invalid_coordinates(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> None:
    with pytest.raises(ValueError):
        BoundingBox(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        )


# ---------------------------------------------------------------------------
# GroundingDetection
# ---------------------------------------------------------------------------


def test_grounding_detection_properties() -> None:
    detection = GroundingDetection(
        bbox=BoundingBox(
            x1=10,
            y1=20,
            x2=110,
            y2=220,
        ),
        confidence=0.9,
        label="button",
    )

    assert detection.center == (60, 120)
    assert detection.width == 100
    assert detection.height == 200
    assert detection.area == 20_000


# ---------------------------------------------------------------------------
# GroundingRequest
# ---------------------------------------------------------------------------


def test_grounding_request_defaults() -> None:
    request = GroundingRequest(
        image="image.png",
        query="Submit button",
    )

    assert request.top_k == 1
    assert request.confidence_threshold is None
    assert request.metadata == {}


# ---------------------------------------------------------------------------
# GroundingResponse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("is_detections_empty", "status"),
    [
        (False, "success"),
        (False, "no_match"),
        (False, "ambiguous"),
        (False, "error"),
        (False, "unknown"),
        (True, "success"),
        (True, "no_match"),
        (True, "ambiguous"),
        (True, "error"),
        (True, "unknown"),
    ],
)
def test_grounding_response_success(
    is_detections_empty: bool,
    status: DetectionStatus,
) -> None:
    if is_detections_empty:
        detections = []
    else:
        detections = [
            GroundingDetection(
                bbox=BoundingBox(
                    x1=10,
                    y1=20,
                    x2=110,
                    y2=220,
                ),
                confidence=0.9,
            )
        ]
    response = GroundingResponse(
        request_query="button",
        provider="fake",
        provider_version="1.0.0",
        detections=detections,
        status=status,
    )

    assert response.success == (
        not is_detections_empty and status == DetectionStatus.SUCCESS
    )


def test_grounding_response_failure_when_no_detections() -> None:
    response = GroundingResponse(
        request_query="button",
        provider="fake",
        provider_version="1.0.0",
        detections=[],
    )

    assert response.success is False


def test_grounding_response_failure_when_status_not_success() -> None:
    response = GroundingResponse(
        request_query="button",
        provider="fake",
        provider_version="1.0.0",
        status=DetectionStatus.ERROR,
        detections=[
            GroundingDetection(
                bbox=BoundingBox(
                    x1=1,
                    y1=2,
                    x2=11,
                    y2=12,
                ),
                confidence=0.9,
            )
        ],
    )

    assert response.success is False


def test_best_detection_returns_highest_confidence() -> None:
    low = GroundingDetection(
        bbox=BoundingBox(
            x1=0,
            y1=0,
            x2=10,
            y2=10,
        ),
        confidence=0.25,
    )

    high = GroundingDetection(
        bbox=BoundingBox(
            x1=20,
            y1=20,
            x2=40,
            y2=40,
        ),
        confidence=0.95,
    )

    response = GroundingResponse(
        request_query="button",
        provider="fake",
        provider_version="1.0.0",
        detections=[
            low,
            high,
        ],
    )

    assert response.best_detection is high
    assert response.best_bbox == high.bbox
    assert response.best_center == high.center


def test_best_detection_with_no_confidence_returns_first_detection() -> None:
    first = GroundingDetection(
        bbox=BoundingBox(
            x1=0,
            y1=0,
            x2=10,
            y2=10,
        ),
    )

    second = GroundingDetection(
        bbox=BoundingBox(
            x1=20,
            y1=20,
            x2=40,
            y2=40,
        ),
    )

    response = GroundingResponse(
        request_query="button",
        provider="fake",
        provider_version="1.0.0",
        detections=[
            first,
            second,
        ],
    )

    assert response.best_detection is first


def test_best_detection_returns_none_when_empty() -> None:
    response = GroundingResponse(
        request_query="button",
        provider="fake",
        provider_version="1.0.0",
    )

    assert response.best_detection is None
    assert response.best_bbox is None
    assert response.best_center is None


def test_sorted_by_confidence_descending() -> None:
    low = GroundingDetection(
        bbox=BoundingBox(
            x1=0,
            y1=0,
            x2=10,
            y2=10,
        ),
        confidence=0.2,
    )

    high = GroundingDetection(
        bbox=BoundingBox(
            x1=20,
            y1=20,
            x2=40,
            y2=40,
        ),
        confidence=0.8,
    )

    response = GroundingResponse(
        request_query="button",
        provider="fake",
        provider_version="1.0.0",
        detections=[
            low,
            high,
        ],
    )

    assert response.sorted_by_confidence() == [
        high,
        low,
    ]


def test_sorted_by_confidence_ascending() -> None:
    low = GroundingDetection(
        bbox=BoundingBox(
            x1=0,
            y1=0,
            x2=10,
            y2=10,
        ),
        confidence=0.2,
    )

    high = GroundingDetection(
        bbox=BoundingBox(
            x1=20,
            y1=20,
            x2=40,
            y2=40,
        ),
        confidence=0.8,
    )

    response = GroundingResponse(
        request_query="button",
        provider="fake",
        provider_version="1.0.0",
        detections=[
            low,
            high,
        ],
    )

    assert response.sorted_by_confidence(
        descending=False,
    ) == [
        low,
        high,
    ]
