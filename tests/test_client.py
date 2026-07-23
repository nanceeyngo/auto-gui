"""
Tests for GroundingClient.
"""

import pytest

from grounding.client import GroundingClient
from grounding.config import GroundingSettings
from grounding.exceptions import (
    GroundingProviderError,
    NoDefaultGroundingProviderError,
    UnknownGroundingProviderError,
)
from grounding.models import (
    GroundingRequest,
    GroundingResponse,
)
from grounding.registry import GroundingRegistry

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
def registry() -> GroundingRegistry:
    registry = GroundingRegistry()
    registry.register(FailingGroundingProvider)
    registry.register(FakeGroundingProvider)
    registry.register(EmptyGroundingProvider)
    registry.register(InitializableGroundingProvider)
    return registry


@pytest.fixture
def client(
    registry: GroundingRegistry,
) -> GroundingClient:
    return GroundingClient(
        registry=registry,
        settings=GroundingSettings(
            default_provider="fake",
        ),
    )


@pytest.fixture
def grounding_request() -> GroundingRequest:
    return GroundingRequest(
        image="dummy.png",
        query="Login button",
    )


# ---------------------------------------------------------------------------
# Provider lookup
# ---------------------------------------------------------------------------


def test_get_provider_caches_instances(
    client: GroundingClient,
) -> None:
    first = client.get_provider("fake")
    second = client.get_provider("fake")

    assert first is second


def test_unknown_provider_raises(
    client: GroundingClient,
) -> None:
    with pytest.raises(
        UnknownGroundingProviderError,
    ):
        client.get_provider("unknown")


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


def test_locate_from_request(
    client: GroundingClient,
    grounding_request: GroundingRequest,
) -> None:
    response = client.locate(request=grounding_request)

    assert isinstance(response, GroundingResponse)
    assert response.success


def test_locate_from_arguments(
    client: GroundingClient,
) -> None:
    response = client.locate(
        image="dummy.png",
        query="Login button",
    )

    assert response.success


def test_explicit_provider(
    client: GroundingClient,
    grounding_request: GroundingRequest,
) -> None:
    response = client.locate(
        request=grounding_request,
        provider="empty",
    )

    assert not response.success
    assert response.detections == []


def test_build_request_returns_existing_request(
    grounding_request: GroundingRequest,
) -> None:
    assert (
        GroundingClient._build_request(
            request=grounding_request,
        )
        is grounding_request
    )


def test_build_request_requires_image() -> None:
    with pytest.raises(ValueError):
        GroundingClient._build_request(
            query="Button",
        )


def test_build_request_requires_query() -> None:
    with pytest.raises(ValueError):
        GroundingClient._build_request(
            image="dummy.png",
        )


# ---------------------------------------------------------------------------
# Default provider
# ---------------------------------------------------------------------------


def test_missing_default_provider() -> None:
    client = GroundingClient(
        registry=GroundingRegistry(),
        settings=GroundingSettings(),
    )

    with pytest.raises(
        NoDefaultGroundingProviderError,
    ):
        client.get_provider()


# ---------------------------------------------------------------------------
# Available providers
# ---------------------------------------------------------------------------


def test_available_providers(
    client: GroundingClient,
) -> None:
    assert client.available_providers == (
        "empty",
        "failing",
        "fake",
        "initializable",
    )


def test_client_unknown_provider_raises(
    grounding_request: GroundingRequest,
    client: GroundingClient,
):
    with pytest.raises(UnknownGroundingProviderError):
        client.locate(
            request=grounding_request,
            provider="unknown",
        )


def test_client_provider_error_propagates(
    grounding_request: GroundingRequest,
    client: GroundingClient,
):
    with pytest.raises(GroundingProviderError):
        client.locate(
            request=grounding_request,
            provider="failing",
        )


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


def test_clear_cache(
    client: GroundingClient,
) -> None:
    provider = client.get_provider("fake")

    client.clear_cache()

    new_provider = client.get_provider("fake")

    assert provider is not new_provider


# ---------------------------------------------------------------------------
# Close
# ---------------------------------------------------------------------------


def test_close_clears_provider_cache(
    client: GroundingClient,
) -> None:
    provider = client.get_provider(
        "initializable",
    )

    provider.initialize()

    client.close()

    assert len(client._providers) == 0


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


def test_context_manager(
    client: GroundingClient,
) -> None:
    with client as c:
        assert c is client


# ---------------------------------------------------------------------------
# Collection protocol
# ---------------------------------------------------------------------------


def test_contains(
    client: GroundingClient,
) -> None:
    assert "fake" in client
    assert "unknown" not in client


def test_len(
    client: GroundingClient,
) -> None:
    assert len(client) == 4


def test_iter(
    client: GroundingClient,
) -> None:
    assert tuple(client) == (
        "empty",
        "failing",
        "fake",
        "initializable",
    )


def test_repr(
    client: GroundingClient,
) -> None:
    representation = repr(client)

    assert "GroundingClient" in representation
    assert "fake" in representation
