"""
Tests for BaseGroundingProvider.
"""

import pytest

from grounding.exceptions import (
    GroundingProviderError,
    GroundingRequestError,
)
from grounding.models import (
    DetectionStatus,
    GroundingRequest,
)

from .fakes import (
    EmptyGroundingProvider,
    FailingGroundingProvider,
    FakeGroundingProvider,
    InitializableGroundingProvider,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def grounding_request() -> GroundingRequest:
    return GroundingRequest(
        image="dummy.png",
        query="Login button",
    )


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


def test_empty_query_is_rejected() -> None:
    provider = FakeGroundingProvider()

    grounding_request = GroundingRequest(
        image="dummy.png",
        query="   ",
    )

    with pytest.raises(GroundingRequestError):
        provider.locate(grounding_request)


# ---------------------------------------------------------------------------
# Successful locate()
# ---------------------------------------------------------------------------


def test_locate_returns_response(
    grounding_request: GroundingRequest,
) -> None:
    provider = FakeGroundingProvider()

    response = provider.locate(grounding_request)

    assert response.success
    assert response.status is DetectionStatus.SUCCESS
    assert len(response.detections) == 1
    assert response.best_detection is not None


def test_elapsed_time_is_recorded(
    grounding_request: GroundingRequest,
) -> None:
    provider = FakeGroundingProvider()

    response = provider.locate(grounding_request)

    assert "elapsed_ms" in response.metadata
    assert response.metadata["elapsed_ms"] >= 0.0


# ---------------------------------------------------------------------------
# Empty responses
# ---------------------------------------------------------------------------


def test_empty_provider_returns_no_detections(
    grounding_request: GroundingRequest,
) -> None:
    provider = EmptyGroundingProvider()

    response = provider.locate(grounding_request)

    assert response.detections == []
    assert not response.success


# ---------------------------------------------------------------------------
# Exception translation
# ---------------------------------------------------------------------------


def test_unexpected_exception_is_translated(
    grounding_request: GroundingRequest,
) -> None:
    provider = FailingGroundingProvider()

    with pytest.raises(GroundingProviderError):
        provider.locate(grounding_request)


# ---------------------------------------------------------------------------
# Automatic initialization
# ---------------------------------------------------------------------------


def test_provider_is_initialized_on_first_request(
    grounding_request: GroundingRequest,
) -> None:
    provider = InitializableGroundingProvider()

    assert not provider.initialized

    provider.locate(grounding_request)

    assert provider.initialized
    assert provider.initialize_calls == 1


def test_provider_is_initialized_only_once(
    grounding_request: GroundingRequest,
) -> None:
    provider = InitializableGroundingProvider()

    provider.locate(grounding_request)
    provider.locate(grounding_request)

    assert provider.initialize_calls == 1


# ---------------------------------------------------------------------------
# Manual lifecycle
# ---------------------------------------------------------------------------


def test_initialize_is_idempotent() -> None:
    provider = InitializableGroundingProvider()

    provider.initialize()
    provider.initialize()

    assert provider.initialize_calls == 1


def test_close_is_idempotent() -> None:
    provider = InitializableGroundingProvider()

    provider.initialize()

    provider.close()
    provider.close()

    assert provider.close_calls == 1
    assert not provider.initialized


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


def test_make_detection() -> None:
    detection = FakeGroundingProvider.make_detection(
        bounding_box=FakeGroundingProvider.make_bounding_box(
            (0.1, 0.2, 0.3, 0.4),
            image_width=1000,
            image_height=500,
        ),
        confidence=0.8,
        label="button",
    )

    assert detection.label == "button"
    assert detection.confidence == 0.8


def test_empty_response_helper(
    grounding_request: GroundingRequest,
) -> None:
    provider = FakeGroundingProvider()

    response = provider.empty_response(grounding_request)

    assert response.detections == []
    assert response.request_query == grounding_request.query
    assert response.provider == provider.provider
    assert response.provider_version == provider.version


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


def test_default_health_check() -> None:
    provider = FakeGroundingProvider()

    assert provider.health_check()
