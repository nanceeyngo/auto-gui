"""
Tests for provider initialization and shutdown behavior.
"""

import pytest

from grounding.exceptions import GroundingInitializationError
from grounding.models import GroundingRequest

from .fakes import (
    FailingInitializationProvider,
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
# Initialization
# ---------------------------------------------------------------------------


def test_provider_is_not_initialized_at_construction():
    provider = InitializableGroundingProvider()

    assert provider.initialize_calls == 0


def test_first_request_initializes_provider(
    grounding_request: GroundingRequest,
):
    provider = InitializableGroundingProvider()

    provider.locate(grounding_request)

    assert provider.initialize_calls == 1


def test_initialization_occurs_only_once(
    grounding_request: GroundingRequest,
):
    provider = InitializableGroundingProvider()

    provider.locate(grounding_request)
    provider.locate(grounding_request)
    provider.locate(grounding_request)

    assert provider.initialize_calls == 1


@pytest.mark.asyncio
async def test_async_initialization_occurs_only_once(
    grounding_request: GroundingRequest,
):
    provider = InitializableGroundingProvider()

    await provider.alocate(grounding_request)
    await provider.alocate(grounding_request)

    assert provider.initialize_calls == 1


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


def test_close_calls_provider_close(
    grounding_request: GroundingRequest,
):
    provider = InitializableGroundingProvider()

    provider.locate(grounding_request)

    provider.close()

    assert provider.close_calls == 1


def test_close_without_initialization_is_safe():
    provider = InitializableGroundingProvider()

    provider.close()

    assert provider.close_calls == 0


def test_close_is_idempotent(
    grounding_request: GroundingRequest,
):
    provider = InitializableGroundingProvider()

    provider.locate(grounding_request)

    provider.close()
    provider.close()

    assert provider.close_calls == 1


@pytest.mark.asyncio
async def test_async_close_calls_provider_close(
    grounding_request: GroundingRequest,
):
    provider = InitializableGroundingProvider()

    await provider.alocate(grounding_request)

    await provider.aclose()

    assert provider.close_calls == 1


# ---------------------------------------------------------------------------
# Context managers
# ---------------------------------------------------------------------------


def test_context_manager_closes_provider(
    grounding_request: GroundingRequest,
):
    provider = InitializableGroundingProvider()

    with provider:
        provider.locate(grounding_request)

    assert provider.initialize_calls == 1
    assert provider.close_calls == 1


@pytest.mark.asyncio
async def test_async_context_manager_closes_provider(
    grounding_request: GroundingRequest,
):
    provider = InitializableGroundingProvider()

    async with provider:
        await provider.alocate(grounding_request)

    assert provider.initialize_calls == 1
    assert provider.close_calls == 1


# ---------------------------------------------------------------------------
# Initialization failures
# ---------------------------------------------------------------------------


def test_initialization_failure_is_wrapped(
    grounding_request: GroundingRequest,
):
    provider = FailingInitializationProvider()

    with pytest.raises(
        GroundingInitializationError,
        match=r"Boom! Initialization failed.",
    ):
        provider.locate(grounding_request)


@pytest.mark.asyncio
async def test_async_initialization_failure_is_wrapped(
    grounding_request: GroundingRequest,
):
    provider = FailingInitializationProvider()

    with pytest.raises(
        GroundingInitializationError,
        match=r"Boom! Initialization failed.",
    ):
        await provider.alocate(grounding_request)
