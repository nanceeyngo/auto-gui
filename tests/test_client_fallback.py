"""
Tests for `GroundingClient.locate_with_fallback` / `.alocate_with_fallback`.

Covers the pluggable fallback strategy: try a fast/cheap provider
first, and fall back to a higher-capacity provider when the primary
fails, returns nothing, or its confidence is below a threshold.
"""

from typing import cast

import pytest

from grounding.client import GroundingClient
from grounding.config import GroundingSettings
from grounding.exceptions import GroundingFallbackError
from grounding.models import GroundingRequest
from grounding.registry import GroundingRegistry

from .fakes import (
    ConfigurableGroundingProvider,
    EmptyGroundingProvider,
    FailingGroundingProvider,
    FakeGroundingProvider,
)


@pytest.fixture
def registry() -> GroundingRegistry:
    registry = GroundingRegistry()
    registry.register(FailingGroundingProvider)
    registry.register(FakeGroundingProvider)
    registry.register(EmptyGroundingProvider)
    registry.register(ConfigurableGroundingProvider)
    return registry


@pytest.fixture
def client(registry: GroundingRegistry) -> GroundingClient:
    return GroundingClient(
        registry=registry,
        settings=GroundingSettings(default_provider="fake"),
    )


@pytest.fixture
def grounding_request() -> GroundingRequest:
    return GroundingRequest(image="dummy.png", query="Login button")


class TestLocateWithFallbackSync:
    def test_first_provider_succeeds_no_fallback_needed(
        self, client: GroundingClient, grounding_request: GroundingRequest
    ) -> None:
        response = client.locate_with_fallback(
            providers=["fake", "empty"],
            request=grounding_request,
        )

        assert response.success

    def test_falls_back_when_first_provider_raises(
        self, client: GroundingClient, grounding_request: GroundingRequest
    ) -> None:
        response = client.locate_with_fallback(
            providers=["failing", "fake"],
            request=grounding_request,
        )

        assert response.success

    def test_falls_back_when_first_provider_returns_no_detections(
        self, client: GroundingClient, grounding_request: GroundingRequest
    ) -> None:
        response = client.locate_with_fallback(
            providers=["empty", "fake"],
            request=grounding_request,
        )

        assert response.success

    def test_falls_back_when_confidence_below_threshold(
        self,
        client: GroundingClient,
        registry: GroundingRegistry,
        grounding_request: GroundingRequest,
    ) -> None:
        low_confidence_provider = cast(
            ConfigurableGroundingProvider,
            registry.create("configurable"),
        )
        low_confidence_provider.configure_response(
            low_confidence_provider.make_response(
                request=grounding_request,
                detections=[
                    low_confidence_provider.make_detection(
                        bounding_box=__import__(
                            "grounding.models", fromlist=["BoundingBox"]
                        ).BoundingBox(x1=0, y1=0, x2=10, y2=10),
                        confidence=0.2,
                        label="low-confidence match",
                    )
                ],
            )
        )
        client._providers["configurable"] = low_confidence_provider

        response = client.locate_with_fallback(
            providers=["configurable", "fake"],
            request=grounding_request,
            min_confidence=0.5,
        )

        # falls through to "fake", which is high-confidence
        assert response.success
        assert response.best_detection is not None
        assert response.best_detection.confidence == pytest.approx(0.95)

    def test_raises_fallback_error_when_all_providers_fail(
        self, client: GroundingClient, grounding_request: GroundingRequest
    ) -> None:
        with pytest.raises(GroundingFallbackError) as exc_info:
            client.locate_with_fallback(
                providers=["failing", "empty"],
                request=grounding_request,
            )

        assert "failing" in exc_info.value.failures
        assert "empty" in exc_info.value.failures

    def test_empty_providers_list_raises_value_error(
        self, client: GroundingClient, grounding_request: GroundingRequest
    ) -> None:
        with pytest.raises(ValueError):
            client.locate_with_fallback(
                providers=[],
                request=grounding_request,
            )

    def test_can_build_request_from_image_and_query(
        self, client: GroundingClient
    ) -> None:
        response = client.locate_with_fallback(
            providers=["fake"],
            image="dummy.png",
            query="Submit",
        )

        assert response.success


class TestLocateWithFallbackAsync:
    @pytest.mark.asyncio
    async def test_falls_back_on_failure(
        self, client: GroundingClient, grounding_request: GroundingRequest
    ) -> None:
        response = await client.alocate_with_fallback(
            providers=["failing", "fake"],
            request=grounding_request,
        )

        assert response.success

    @pytest.mark.asyncio
    async def test_raises_when_all_fail(
        self, client: GroundingClient, grounding_request: GroundingRequest
    ) -> None:
        with pytest.raises(GroundingFallbackError):
            await client.alocate_with_fallback(
                providers=["failing", "empty"],
                request=grounding_request,
            )
