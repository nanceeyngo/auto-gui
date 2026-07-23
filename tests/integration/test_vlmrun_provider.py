"""
Integration tests for the VLM Run grounding provider.

These tests verify that the provider correctly orchestrates the
request lifecycle while mocking the VLM Run SDK.
"""

from unittest.mock import Mock

import pytest

from grounding.models import (
    BoundingBox,
    DetectionStatus,
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
    from PIL import Image

    return GroundingRequest(
        image=Image.new("RGB", (800, 600)),
        query="Login button",
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def test_initialize_creates_client(
    provider: VLMRunGroundingProvider,
) -> None:
    provider._initialize()

    assert provider._client is not None


def test_close_clears_client(
    provider: VLMRunGroundingProvider,
) -> None:
    provider._client = Mock()

    provider._close()

    assert provider._client is None


# ---------------------------------------------------------------------------
# _locate orchestration
# ---------------------------------------------------------------------------


def test_locate_calls_pipeline_in_order(
    provider: VLMRunGroundingProvider,
    grounding_request: GroundingRequest,
    monkeypatch,
) -> None:
    messages = [{"role": "user"}]

    prediction = Mock()

    detections = [
        provider.make_detection(
            bounding_box=BoundingBox(
                x1=10,
                y1=20,
                x2=100,
                y2=120,
            ),
            confidence=0.95,
            label="button",
        )
    ]

    build_messages = Mock(return_value=messages)

    submit_prediction = Mock(return_value=prediction)

    wait_prediction = Mock(return_value=prediction)

    parse_prediction = Mock(return_value=detections)

    monkeypatch.setattr(
        provider,
        "_build_messages",
        build_messages,
    )

    monkeypatch.setattr(
        provider,
        "_submit_prediction",
        submit_prediction,
    )

    monkeypatch.setattr(
        provider,
        "_wait_for_prediction",
        wait_prediction,
    )

    monkeypatch.setattr(
        provider,
        "_parse_prediction",
        parse_prediction,
    )

    response = provider._locate(grounding_request)

    build_messages.assert_called_once_with(grounding_request)

    submit_prediction.assert_called_once_with(
        messages,
    )

    wait_prediction.assert_called_once_with(
        prediction,
    )

    parse_prediction.assert_called_once_with(
        request=grounding_request,
        prediction=prediction,
    )

    assert response.success
    assert len(response.detections) == 1
    assert response.best_detection == detections[0]


def test_locate_returns_empty_response(
    provider: VLMRunGroundingProvider,
    grounding_request: GroundingRequest,
    monkeypatch,
):
    monkeypatch.setattr(
        provider,
        "_build_messages",
        Mock(return_value=[]),
    )

    prediction = Mock()

    monkeypatch.setattr(
        provider,
        "_submit_prediction",
        Mock(return_value=prediction),
    )

    monkeypatch.setattr(
        provider,
        "_wait_for_prediction",
        Mock(return_value=prediction),
    )

    monkeypatch.setattr(
        provider,
        "_parse_prediction",
        Mock(return_value=[]),
    )

    response = provider._locate(grounding_request)

    assert response.status is DetectionStatus.NO_MATCH
    assert response.detections == []


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method",
    [
        "_build_messages",
        "_submit_prediction",
        "_wait_for_prediction",
        "_parse_prediction",
    ],
)
def test_pipeline_errors_are_not_swallowed(
    provider: VLMRunGroundingProvider,
    grounding_request: GroundingRequest,
    monkeypatch,
    method: str,
):
    prediction = Mock()

    monkeypatch.setattr(
        provider,
        "_build_messages",
        Mock(return_value=[]),
    )

    monkeypatch.setattr(
        provider,
        "_submit_prediction",
        Mock(return_value=prediction),
    )

    monkeypatch.setattr(
        provider,
        "_wait_for_prediction",
        Mock(return_value=prediction),
    )

    monkeypatch.setattr(
        provider,
        "_parse_prediction",
        Mock(return_value=[]),
    )

    monkeypatch.setattr(
        provider,
        method,
        Mock(
            side_effect=RuntimeError(
                "boom",
            )
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="boom",
    ):
        provider._locate(grounding_request)


# ---------------------------------------------------------------------------
# Response construction
# ---------------------------------------------------------------------------


def test_response_contains_request_metadata(
    provider: VLMRunGroundingProvider,
    grounding_request: GroundingRequest,
    monkeypatch,
):
    detection = provider.make_detection(
        bounding_box=BoundingBox(
            x1=5,
            y1=5,
            x2=50,
            y2=50,
        ),
        confidence=0.8,
    )

    prediction = Mock()

    monkeypatch.setattr(
        provider,
        "_build_messages",
        Mock(return_value=[]),
    )

    monkeypatch.setattr(
        provider,
        "_submit_prediction",
        Mock(return_value=prediction),
    )

    monkeypatch.setattr(
        provider,
        "_wait_for_prediction",
        Mock(return_value=prediction),
    )

    monkeypatch.setattr(
        provider,
        "_parse_prediction",
        Mock(return_value=[detection]),
    )

    response = provider._locate(grounding_request)

    assert response.request_query == grounding_request.query
    assert response.provider == provider.provider
    assert response.provider_version == provider.version
    assert response.best_detection == detection
