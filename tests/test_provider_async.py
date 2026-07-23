"""
Tests for BaseGroundingProvider asynchronous behavior.
"""

import asyncio

import pytest

from grounding.exceptions import GroundingProviderError
from grounding.models import GroundingRequest

from .fakes import (
    FailingGroundingProvider,
    FakeGroundingProvider,
    FlakyGroundingProvider,
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


# ---------------------------------------------------------------------------
# Success
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_alocate_success(
    grounding_request: GroundingRequest,
) -> None:
    provider = FakeGroundingProvider()

    response = await provider.alocate(grounding_request)

    assert response.success


@pytest.mark.asyncio
async def test_alocate_matches_locate(
    grounding_request: GroundingRequest,
):
    provider = FakeGroundingProvider()

    sync = provider.locate(grounding_request)
    async_result = await provider.alocate(grounding_request)

    assert async_result.request_query == sync.request_query
    assert async_result.provider == sync.provider
    assert async_result.provider_version == sync.provider_version
    assert async_result.status == sync.status
    assert async_result.detections == sync.detections
    assert async_result.provider_request_id == sync.provider_request_id


@pytest.mark.asyncio
async def test_async_initialization_success(
    grounding_request: GroundingRequest,
) -> None:
    provider = InitializableGroundingProvider()

    await provider.alocate(grounding_request)

    assert provider.initialize_calls == 1
    assert provider.initialized


# ---------------------------------------------------------------------------
# Exception wrapping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_runtime_error_is_wrapped(
    grounding_request: GroundingRequest,
):
    provider = FailingGroundingProvider()

    with pytest.raises(
        GroundingProviderError,
        match=r"Simulated provider failure.",
    ):
        await provider.alocate(grounding_request)


@pytest.mark.asyncio
async def test_async_grounding_error_is_preserved(
    grounding_request: GroundingRequest,
):
    provider = FlakyGroundingProvider()

    provider.configure_provider_max_retries(0)
    provider.configure_failures_before_success(1)
    provider.configure_exception(
        GroundingProviderError("temporary backend failure")
    )

    with pytest.raises(
        GroundingProviderError,
        match="temporary backend failure",
    ):
        await provider.alocate(grounding_request)


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_requests_are_supported(
    grounding_request: GroundingRequest,
):
    provider = SlowGroundingProvider()

    provider.configure_timeout_duration(1.0)
    provider.configure_should_return_empty_response(False)

    results = await asyncio.gather(
        provider.alocate(grounding_request),
        provider.alocate(grounding_request),
        provider.alocate(grounding_request),
    )

    assert all(r.success for r in results)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_async_health_check() -> None:
    provider = FakeGroundingProvider()

    assert await provider.ahealth_check()
