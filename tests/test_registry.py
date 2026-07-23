"""
Tests for the grounding provider registry.
"""

import pytest

from grounding.exceptions import UnknownGroundingProviderError
from grounding.registry import GroundingRegistry

from .fakes import (
    EmptyGroundingProvider,
    FakeGroundingProvider,
    InitializableGroundingProvider,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry() -> GroundingRegistry:
    return GroundingRegistry()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_register_provider(
    registry: GroundingRegistry,
) -> None:
    registry.register(FakeGroundingProvider)

    assert "fake" in registry
    assert len(registry) == 1


def test_register_multiple_providers(
    registry: GroundingRegistry,
) -> None:
    registry.register(FakeGroundingProvider)
    registry.register(EmptyGroundingProvider)

    assert len(registry) == 2

    assert set(registry.registered_ids()) == {
        "fake",
        "empty",
    }


def test_duplicate_registration_raises(
    registry: GroundingRegistry,
) -> None:
    registry.register(FakeGroundingProvider)

    with pytest.raises(
        ValueError,
    ):
        registry.register(FakeGroundingProvider)


def test_duplicate_registration_with_overwrite(
    registry: GroundingRegistry,
) -> None:
    registry.register(FakeGroundingProvider)

    registry.register(
        FakeGroundingProvider,
        overwrite=True,
    )

    assert len(registry) == 1


# ---------------------------------------------------------------------------
# Unregistration
# ---------------------------------------------------------------------------


def test_unregister_provider(
    registry: GroundingRegistry,
) -> None:
    registry.register(FakeGroundingProvider)

    registry.unregister("fake")

    assert "fake" not in registry
    assert len(registry) == 0


def test_unregister_unknown_provider_raises(
    registry: GroundingRegistry,
) -> None:
    with pytest.raises(
        UnknownGroundingProviderError,
    ):
        registry.unregister("does-not-exist")


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------


def test_create_provider(
    registry: GroundingRegistry,
) -> None:
    registry.register(FakeGroundingProvider)

    provider = registry.create("fake")

    assert isinstance(
        provider,
        FakeGroundingProvider,
    )


def test_create_returns_new_instance_each_time(
    registry: GroundingRegistry,
) -> None:
    registry.register(FakeGroundingProvider)

    first = registry.create("fake")
    second = registry.create("fake")

    assert first is not second


def test_create_unknown_provider_raises(
    registry: GroundingRegistry,
) -> None:
    with pytest.raises(
        UnknownGroundingProviderError,
    ):
        registry.create("missing")


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


def test_registered_ids(
    registry: GroundingRegistry,
) -> None:
    registry.register(FakeGroundingProvider)
    registry.register(EmptyGroundingProvider)
    registry.register(InitializableGroundingProvider)

    assert registry.registered_ids() == (
        "empty",
        "fake",
        "initializable",
    )


def test_contains(
    registry: GroundingRegistry,
) -> None:
    registry.register(FakeGroundingProvider)

    assert "fake" in registry
    assert "missing" not in registry


def test_len(
    registry: GroundingRegistry,
) -> None:
    assert len(registry) == 0

    registry.register(FakeGroundingProvider)

    assert len(registry) == 1

    registry.register(EmptyGroundingProvider)

    assert len(registry) == 2


def test_iter_returns_provider_ids(
    registry: GroundingRegistry,
) -> None:
    registry.register(FakeGroundingProvider)
    registry.register(EmptyGroundingProvider)

    assert tuple(registry) == (
        "empty",
        "fake",
    )


# ---------------------------------------------------------------------------
# Clearing
# ---------------------------------------------------------------------------


def test_clear(
    registry: GroundingRegistry,
) -> None:
    registry.register(FakeGroundingProvider)
    registry.register(EmptyGroundingProvider)

    registry.clear()

    assert len(registry) == 0
    assert registry.registered_ids() == ()
