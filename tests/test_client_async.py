"""
Tests for GroundingClient asynchronous behavior.
"""

import asyncio

import pytest

from grounding.client import GroundingClient
from grounding.config import GroundingSettings
from grounding.exceptions import (
    GroundingProviderError,
    UnknownGroundingProviderError,
)
from grounding.models import GroundingRequest
from grounding.registry import GroundingRegistry

from .fakes import (
    FailingGroundingProvider,
    FakeGroundingProvider,
    InitializableGroundingProvider,
    SlowGroundingProvider,
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


@pytest.fixture
def registry() -> GroundingRegistry:
    registry = GroundingRegistry()

    registry.register(FakeGroundingProvider)
    registry.register(SlowGroundingProvider)
    registry.register(FailingGroundingProvider)
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


# ---------------------------------------------------------------------------
# locate()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_client_alocate(
    grounding_request: GroundingRequest,
    client: GroundingClient,
):
    response = await client.alocate(
        request=grounding_request,
        provider="fake",
    )

    assert response.success
    assert len(response.detections) == 1


@pytest.mark.asyncio
async def test_client_unknown_provider(
    grounding_request: GroundingRequest, client: GroundingClient
):
    with pytest.raises(UnknownGroundingProviderError):
        await client.alocate(
            request=grounding_request,
            provider="unknown",
        )


@pytest.mark.asyncio
async def test_client_provider_error_propagates(
    grounding_request: GroundingRequest,
    client: GroundingClient,
):
    with pytest.raises(GroundingProviderError):
        await client.alocate(
            request=grounding_request,
            provider="failing",
        )


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_context_manager(
    grounding_request: GroundingRequest,
    client: GroundingClient,
):
    async with client as c:
        assert c is client

        response = await c.alocate(
            request=grounding_request,
            provider="fake",
        )

        assert response.success


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ahealth_check(
    client: GroundingClient,
):
    provider_ids = client.available_providers

    providers = [client.get_provider(pid) for pid in provider_ids]

    health = await client.ahealth_check()

    for provider in providers:
        assert health[provider.provider_id] == await provider.ahealth_check()


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aclose_clears_cache(
    registry: GroundingRegistry,
):
    client = GroundingClient(registry=registry)

    client.get_provider("fake")
    client.get_provider("slow")

    assert len(client._providers) == 2

    await client.aclose()

    assert client._providers == {}


# ---------------------------------------------------------------------------
# Concurrent client requests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiple_client_requests(
    registry: GroundingRegistry,
    grounding_request: GroundingRequest,
):
    client = GroundingClient(registry=registry)

    results = await asyncio.gather(
        client.alocate(
            request=grounding_request,
            provider="fake",
        ),
        client.alocate(
            request=grounding_request,
            provider="fake",
        ),
        client.alocate(
            request=grounding_request,
            provider="fake",
        ),
    )

    assert all(r.success for r in results)
