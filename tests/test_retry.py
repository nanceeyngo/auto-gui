"""
Tests for BaseGroundingProvider retry behavior.
"""

import pytest

from grounding.exceptions import GroundingProviderError
from grounding.models import GroundingRequest

from .fakes import (
    CountingGroundingProvider,
    FlakyGroundingProvider,
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
# Successful execution
# ---------------------------------------------------------------------------


def test_success_without_retry(
    grounding_request: GroundingRequest,
):
    provider = CountingGroundingProvider()

    response = provider.locate(grounding_request)

    assert response.success
    assert provider.calls == 1


# ---------------------------------------------------------------------------
# Retry succeeds
# ---------------------------------------------------------------------------


def test_retry_then_success(
    grounding_request: GroundingRequest,
):
    provider = FlakyGroundingProvider()

    provider.configure_failures_before_success(2)
    # For successful execution, make max allowed retries >= number of failures
    provider.configure_provider_max_retries(3)

    response = provider.locate(grounding_request)

    assert response.success
    assert provider.calls == 3


def test_success_on_last_allowed_retry(
    grounding_request: GroundingRequest,
):
    provider = FlakyGroundingProvider()

    provider.configure_failures_before_success(2)
    # For successful execution, make max allowed retries >= number of failures
    provider.configure_provider_max_retries(2)

    response = provider.locate(grounding_request)

    assert response.success
    assert provider.calls == 3


# ---------------------------------------------------------------------------
# Retry exhaustion
# ---------------------------------------------------------------------------


def test_retry_exhaustion_raises(
    grounding_request: GroundingRequest,
):
    provider = FlakyGroundingProvider()

    provider.configure_failures_before_success(3)
    provider.configure_provider_max_retries(2)

    with pytest.raises(GroundingProviderError):
        provider.locate(grounding_request)

    assert provider.calls == 3


# ---------------------------------------------------------------------------
# Retry state
# ---------------------------------------------------------------------------


def test_retry_counter_resets_between_requests(
    grounding_request: GroundingRequest,
):
    provider = FlakyGroundingProvider()

    provider.configure_failures_before_success(1)

    provider.locate(grounding_request)

    assert provider.calls == 2

    provider.reset()

    provider.configure_failures_before_success(0)

    provider.locate(grounding_request)

    assert provider.calls == 1


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


def test_original_exception_is_preserved(
    grounding_request: GroundingRequest,
):
    provider = FlakyGroundingProvider()

    provider.configure_failures_before_success(2)
    provider.configure_provider_max_retries(1)

    provider.configure_exception(
        GroundingProviderError("Temporary backend failure")
    )

    with pytest.raises(
        GroundingProviderError,
        match="Temporary backend failure",
    ):
        provider.locate(grounding_request)


# ---------------------------------------------------------------------------
# Async execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_success_without_retry(
    grounding_request: GroundingRequest,
):
    provider = CountingGroundingProvider()

    response = await provider.alocate(grounding_request)

    assert response.success
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_async_retry_then_success(
    grounding_request: GroundingRequest,
):
    provider = FlakyGroundingProvider()

    provider.configure_failures_before_success(2)
    # For successful execution, make max allowed retries >= number of failures
    provider.configure_provider_max_retries(3)

    response = await provider.alocate(grounding_request)

    assert response.success
    assert provider.calls == 3


@pytest.mark.asyncio
async def test_async_retry_exhaustion(
    grounding_request: GroundingRequest,
):
    provider = FlakyGroundingProvider()

    provider.configure_failures_before_success(2)
    provider.configure_provider_max_retries(1)

    with pytest.raises(GroundingProviderError):
        await provider.alocate(grounding_request)

    assert provider.calls == 2


@pytest.mark.asyncio
async def test_async_retry_counter_resets_between_requests(
    grounding_request: GroundingRequest,
):
    provider = FlakyGroundingProvider()

    provider.configure_failures_before_success(1)

    await provider.alocate(grounding_request)

    assert provider.calls == 2

    provider.reset()

    provider.configure_failures_before_success(0)

    await provider.alocate(grounding_request)

    assert provider.calls == 1
