"""
Integration tests for VLMRunGroundingProvider response parsing.
"""

import datetime
import math
from collections.abc import Mapping
from typing import Any

import pytest
from PIL import Image
from vlmrun.client.types import CreditUsage, PredictionResponse

from grounding.exceptions import InvalidGroundingResultError
from grounding.models import (
    BoundingBox,
    GroundingDetection,
    GroundingRequest,
)
from grounding.providers.vlmrun import (
    VLMRunGroundingProvider,
    VLMRunSettings,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def provider() -> VLMRunGroundingProvider:
    provider = VLMRunGroundingProvider()
    provider._settings = VLMRunSettings()

    return provider


@pytest.fixture
def grounding_request() -> GroundingRequest:
    return GroundingRequest(
        image=Image.new("RGB", (1000, 500)),
        query="Login button",
    )


def make_prediction(
    response: Mapping,
) -> PredictionResponse:
    """
    Minimal stand-in for a PredictionResponse.
    """
    prediction = PredictionResponse(
        id="dummy_prediction_id",
        status="completed",
        created_at=datetime.datetime.now(datetime.UTC),
        usage=CreditUsage(),
        response=response,
    )
    return prediction


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_single_detection(
    provider: VLMRunGroundingProvider,
    grounding_request: GroundingRequest,
) -> None:
    prediction = make_prediction(
        {
            "detections": [
                {
                    "bbox": [0.1, 0.2, 0.3, 0.4],
                    "confidence": 0.95,
                    "label": "button",
                }
            ]
        }
    )

    detections = provider._parse_prediction(
        request=grounding_request,
        prediction=prediction,
    )

    assert len(detections) == 1

    detection = detections[0]

    assert isinstance(
        detection,
        GroundingDetection,
    )

    assert detection.label == "button"
    assert detection.confidence == 0.95

    assert detection.bbox == BoundingBox(
        x1=100,
        y1=100,
        x2=400,
        y2=300,
    )
    assert detection.center == (250, 200)
    assert detection.width == 300
    assert detection.height == 200


def test_multiple_detections(
    provider: VLMRunGroundingProvider,
    grounding_request: GroundingRequest,
) -> None:
    prediction = make_prediction(
        {
            "detections": [
                {
                    "bbox": [0.0, 0.0, 0.2, 0.2],
                    "confidence": 0.90,
                    "label": "first",
                },
                {
                    "bbox": [0.5, 0.5, 0.2, 0.2],
                    "confidence": 0.80,
                    "label": "second",
                },
            ]
        }
    )

    detections = provider._parse_prediction(
        request=grounding_request,
        prediction=prediction,
    )

    assert len(detections) == 2


def test_empty_detection_list(
    provider: VLMRunGroundingProvider,
    grounding_request: GroundingRequest,
) -> None:
    prediction = make_prediction({"detections": []})

    detections = provider._parse_prediction(
        request=grounding_request,
        prediction=prediction,
    )

    assert detections == []


def test_detection_order_is_preserved(
    provider: VLMRunGroundingProvider,
    grounding_request: GroundingRequest,
):
    prediction = make_prediction(
        {
            "detections": [
                {
                    "bbox": [0.6, 0.6, 0.1, 0.1],
                    "label": "third",
                },
                {
                    "bbox": [0.1, 0.1, 0.1, 0.1],
                    "label": "first",
                },
                {
                    "bbox": [0.3, 0.3, 0.1, 0.1],
                    "label": "second",
                },
            ]
        }
    )

    detections = provider._parse_prediction(
        request=grounding_request,
        prediction=prediction,
    )

    assert [d.label for d in detections] == [
        "third",
        "first",
        "second",
    ]


# ---------------------------------------------------------------------------
# Invalid top-level response
# ---------------------------------------------------------------------------


def test_response_must_be_mapping(
    provider: VLMRunGroundingProvider,
    grounding_request: GroundingRequest,
) -> None:
    prediction = make_prediction(
        ["not", "a", "mapping"]  # type: ignore[arg-type]
    )

    with pytest.raises(
        InvalidGroundingResultError,
        match="mapping",
    ):
        provider._parse_prediction(
            request=grounding_request,
            prediction=prediction,
        )


def test_missing_detections_field(
    provider: VLMRunGroundingProvider,
    grounding_request: GroundingRequest,
) -> None:
    prediction = make_prediction({})

    with pytest.raises(
        InvalidGroundingResultError,
        match="detections",
    ):
        provider._parse_prediction(
            request=grounding_request,
            prediction=prediction,
        )


def test_detections_must_be_list(
    provider: VLMRunGroundingProvider,
    grounding_request: GroundingRequest,
) -> None:
    prediction = make_prediction({"detections": {}})

    with pytest.raises(
        InvalidGroundingResultError,
    ):
        provider._parse_prediction(
            request=grounding_request,
            prediction=prediction,
        )


# ---------------------------------------------------------------------------
# Invalid individual detections
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bbox",
    [
        None,
        {},
        [],
        [1],
        [1, 2, 3],
        [1, 2, 3, 4, 5],
    ],
)
def test_invalid_bbox_shapes_are_ignored(
    provider: VLMRunGroundingProvider,
    grounding_request: GroundingRequest,
    bbox: Any,
) -> None:
    prediction = make_prediction(
        {
            "detections": [
                {
                    "bbox": bbox,
                    "confidence": 0.9,
                }
            ]
        }
    )

    detections = provider._parse_prediction(
        request=grounding_request,
        prediction=prediction,
    )

    assert detections == []


@pytest.mark.parametrize(
    "bbox",
    [
        [-0.1, 0.2, 0.3, 0.4],
        [0.1, -0.2, 0.3, 0.4],
        [1.2, 0.2, 0.3, 0.4],
        [0.1, 0.2, 2.0, 0.4],
    ],
)
def test_out_of_range_bbox_is_ignored(
    provider: VLMRunGroundingProvider,
    grounding_request: GroundingRequest,
    bbox: Any,
) -> None:
    prediction = make_prediction(
        {
            "detections": [
                {
                    "bbox": bbox,
                    "confidence": 0.9,
                }
            ]
        }
    )

    detections = provider._parse_prediction(
        request=grounding_request,
        prediction=prediction,
    )

    assert detections == []


@pytest.mark.parametrize(
    "bbox",
    [
        ["0.1", 0.2, 0.3, 0.4],
        [0.1, 0.2, "0.3", 0.4],
        [0.2, "0.2", 0.3, 0.4],
        [0.1, 0.2, 0.1, "0.4"],
    ],
)
def test_bbox_with_non_numeric_members_is_ignored(
    provider: VLMRunGroundingProvider,
    grounding_request: GroundingRequest,
    bbox: Any,
) -> None:
    prediction = make_prediction(
        {
            "detections": [
                {
                    "bbox": bbox,
                    "confidence": 0.95,
                }
            ]
        }
    )

    detections = provider._parse_prediction(
        request=grounding_request,
        prediction=prediction,
    )

    assert detections == []


@pytest.mark.parametrize(
    "bbox",
    [
        [0.3, 0.2, math.inf, 0.4],
        [math.nan, 0.2, 0.3, 0.4],
        [0.2, 0.6, 0.3, math.inf],
        [0.1, math.nan, 0.1, 0.4],
    ],
)
def test_bbox_with_nan_or_inf_members_is_ignored(
    provider: VLMRunGroundingProvider,
    grounding_request: GroundingRequest,
    bbox: Any,
) -> None:
    prediction = make_prediction(
        {
            "detections": [
                {
                    "bbox": bbox,
                    "confidence": 0.95,
                }
            ]
        }
    )

    detections = provider._parse_prediction(
        request=grounding_request,
        prediction=prediction,
    )

    assert detections == []


def test_invalid_confidence_becomes_none(
    provider: VLMRunGroundingProvider,
    grounding_request: GroundingRequest,
) -> None:
    prediction = make_prediction(
        {
            "detections": [
                {
                    "bbox": [0.1, 0.2, 0.3, 0.4],
                    "confidence": "high",
                }
            ]
        }
    )

    detections = provider._parse_prediction(
        request=grounding_request,
        prediction=prediction,
    )

    assert len(detections) == 1
    assert detections[0].confidence is None


def test_valid_integer_confidence_becomes_float(
    provider: VLMRunGroundingProvider,
    grounding_request: GroundingRequest,
) -> None:
    prediction = make_prediction(
        {
            "detections": [
                {
                    "bbox": [0.1, 0.2, 0.3, 0.4],
                    "confidence": 1,
                }
            ]
        }
    )

    detections = provider._parse_prediction(
        request=grounding_request,
        prediction=prediction,
    )

    assert len(detections) == 1
    assert isinstance(detections[0].confidence, float)
    assert detections[0].confidence == 1.0


def test_missing_confidence_becomes_none(
    provider: VLMRunGroundingProvider,
    grounding_request: GroundingRequest,
) -> None:
    prediction = make_prediction(
        {
            "detections": [
                {
                    "bbox": [0.1, 0.2, 0.3, 0.4],
                }
            ]
        }
    )

    detections = provider._parse_prediction(
        request=grounding_request,
        prediction=prediction,
    )

    assert len(detections) == 1
    assert detections[0].confidence is None


def test_missing_label_is_allowed(
    provider: VLMRunGroundingProvider,
    grounding_request: GroundingRequest,
) -> None:
    prediction = make_prediction(
        {
            "detections": [
                {
                    "bbox": [0.1, 0.2, 0.3, 0.4],
                    "confidence": 0.9,
                }
            ]
        }
    )

    detections = provider._parse_prediction(
        request=grounding_request,
        prediction=prediction,
    )

    assert detections[0].label is None


def test_non_mapping_detection_is_ignored(
    provider: VLMRunGroundingProvider,
    grounding_request: GroundingRequest,
) -> None:
    prediction = make_prediction(
        {
            "detections": [
                "not-a-detection",
                123,
                None,
            ]
        }
    )

    detections = provider._parse_prediction(
        request=grounding_request,
        prediction=prediction,
    )

    assert detections == []


# ---------------------------------------------------------------------------
# Mixed detection issues
# ---------------------------------------------------------------------------


def test_partially_invalid_detections(
    provider: VLMRunGroundingProvider,
    grounding_request: GroundingRequest,
):
    prediction = make_prediction(
        {
            "detections": [
                # valid
                {
                    "bbox": [0.10, 0.20, 0.30, 0.40],
                    "confidence": 0.95,
                    "label": "button",
                },
                # invalid bbox (wrong length)
                {
                    "bbox": [0.1, 0.2, 0.3],
                    "confidence": 0.80,
                    "label": "invalid-1",
                },
                # not even a mapping
                "not-a-detection",
                # bbox values outside [0, 1]
                {
                    "bbox": [1.2, 0.2, 0.3, 0.4],
                    "confidence": 0.60,
                    "label": "invalid-2",
                },
                # valid (confidence cannot be parsed)
                {
                    "bbox": [0.60, 0.10, 0.20, 0.20],
                    "confidence": "high",
                    "label": "textbox",
                },
            ]
        }
    )

    detections = provider._parse_prediction(
        request=grounding_request,
        prediction=prediction,
    )

    # Only the two valid detections should survive.
    assert len(detections) == 2

    first, second = detections

    assert first.label == "button"
    assert first.confidence == 0.95
    assert first.bbox == BoundingBox(
        x1=100,
        y1=100,
        x2=400,
        y2=300,
    )

    assert second.label == "textbox"
    assert second.confidence is None
    assert second.bbox == BoundingBox(
        x1=600,
        y1=50,
        x2=800,
        y2=150,
    )
