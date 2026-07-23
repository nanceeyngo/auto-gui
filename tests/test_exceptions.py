"""
Tests for the grounding exception hierarchy.
"""

import pytest

from grounding.exceptions import (
    GroundingCancelledError,
    GroundingConnectionError,
    GroundingError,
    GroundingFallbackError,
    GroundingInitializationError,
    GroundingProviderError,
    GroundingRequestError,
    GroundingResponseError,
    GroundingShutdownError,
    GroundingTimeoutError,
    InvalidGroundingResultError,
    NoDefaultGroundingProviderError,
    NoGroundingResultError,
    UnknownGroundingProviderError,
    UnsupportedCapabilityError,
)

# ---------------------------------------------------------------------------
# Base exception
# ---------------------------------------------------------------------------


def test_grounding_error_is_exception():
    assert issubclass(GroundingError, Exception)


def test_grounding_error_preserves_message():
    exc = GroundingError("something went wrong")

    assert str(exc) == "something went wrong"


# ---------------------------------------------------------------------------
# Inheritance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exception_type", "base_type"),
    [
        (GroundingInitializationError, GroundingError),
        (GroundingShutdownError, GroundingError),
        (GroundingRequestError, GroundingError),
        (UnsupportedCapabilityError, GroundingRequestError),
        (GroundingProviderError, GroundingError),
        (GroundingConnectionError, GroundingProviderError),
        (GroundingTimeoutError, GroundingProviderError),
        (GroundingCancelledError, GroundingProviderError),
        (GroundingResponseError, GroundingError),
        (NoGroundingResultError, GroundingResponseError),
        (InvalidGroundingResultError, GroundingResponseError),
        (UnknownGroundingProviderError, GroundingError),
        (GroundingFallbackError, GroundingError),
        (NoDefaultGroundingProviderError, GroundingError),
    ],
)
def test_exception_inheritance(
    exception_type,
    base_type,
):
    assert issubclass(exception_type, base_type)


# ---------------------------------------------------------------------------
# Message preservation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exception_type",
    [
        GroundingInitializationError,
        GroundingShutdownError,
        GroundingRequestError,
        UnsupportedCapabilityError,
        GroundingProviderError,
        GroundingConnectionError,
        GroundingTimeoutError,
        GroundingCancelledError,
        GroundingResponseError,
        NoGroundingResultError,
        InvalidGroundingResultError,
        UnknownGroundingProviderError,
        NoDefaultGroundingProviderError,
    ],
)
def test_exception_message_preserved(
    exception_type,
):
    exc = exception_type("custom message")

    assert str(exc) == "custom message"


# ---------------------------------------------------------------------------
# Catching
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exception",
    [
        GroundingInitializationError(),
        GroundingShutdownError(),
        GroundingRequestError(),
        UnsupportedCapabilityError(),
        GroundingProviderError(),
        GroundingConnectionError(),
        GroundingTimeoutError(),
        GroundingCancelledError(),
        GroundingResponseError(),
        NoGroundingResultError(),
        InvalidGroundingResultError(),
        UnknownGroundingProviderError(),
        NoDefaultGroundingProviderError(),
    ],
)
def test_all_exceptions_are_grounding_errors(
    exception,
):
    with pytest.raises(GroundingError):
        raise exception


# ---------------------------------------------------------------------------
# GroundingFallbackError
# ---------------------------------------------------------------------------


def test_fallback_error_stores_failures():
    failures = {
        "vlmrun": GroundingProviderError("provider failed"),
        "omniparser": GroundingTimeoutError("timeout"),
    }

    exc = GroundingFallbackError(
        "All providers failed.",
        failures,
    )

    assert exc.message == "All providers failed."
    assert exc.failures == failures


def test_fallback_error_failures_are_read_only():
    exc = GroundingFallbackError(
        "failure",
        {
            "vlmrun": GroundingProviderError(),
        },
    )

    with pytest.raises(TypeError):
        exc.failures["new"] = GroundingProviderError()


def test_fallback_error_copies_mapping():
    failures = {
        "vlmrun": GroundingProviderError(),
    }

    exc = GroundingFallbackError(
        "failure",
        failures,
    )

    failures["groundingdino"] = GroundingProviderError()

    assert "groundingdino" not in exc.failures


def test_fallback_error_string_contains_provider_names():
    exc = GroundingFallbackError(
        "All providers failed.",
        {
            "vlmrun": GroundingProviderError(),
            "omniparser": GroundingProviderError(),
        },
    )

    text = str(exc)

    assert "All providers failed." in text
    assert "vlmrun" in text
    assert "omniparser" in text


def test_fallback_error_provider_names_are_sorted():
    exc = GroundingFallbackError(
        "failure",
        {
            "zeta": GroundingProviderError(),
            "alpha": GroundingProviderError(),
        },
    )

    text = str(exc)

    assert text.index("alpha") < text.index("zeta")


def test_fallback_error_is_grounding_error():
    exc = GroundingFallbackError(
        "failure",
        {},
    )

    with pytest.raises(GroundingError):
        raise exc


# ---------------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------------


def test_exception_repr_contains_class_name():
    exc = GroundingProviderError("backend failure")

    assert "GroundingProviderError" in repr(exc)
