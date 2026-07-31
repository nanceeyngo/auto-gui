"""
Integration tests for the VLM Run grounding provider.

These tests verify that the provider correctly orchestrates the
request lifecycle while mocking the VLM Run SDK.
"""

from unittest.mock import Mock, patch

import pytest
from pydantic import SecretStr

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
    # A dummy key is supplied so `_initialize()` can construct the SDK
    # client without requiring a real VLMRUN_API_KEY in the ambient
    # environment. Without this, `test_initialize_creates_client`
    # would only pass on machines that happen to have real
    # credentials configured, breaking fully-offline test execution.
    provider._settings = VLMRunSettings(api_key=SecretStr("test-dummy-api-key"))
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
    # The VLMRun SDK constructor performs a live network call (a
    # health check) when instantiated, which previously made this
    # test silently depend on internet access and real credentials.
    # Patch the SDK class so `_initialize()` is fully offline.
    with patch("grounding.providers.vlmrun.VLMRun", autospec=True) as mock_client_cls:
        provider._initialize()

        mock_client_cls.assert_called_once()
        assert provider._client is mock_client_cls.return_value


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
    monkeypatch: pytest.MonkeyPatch,
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
